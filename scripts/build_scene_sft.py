#!/usr/bin/env python
"""Build scene-level SFT warmup data for the 2B policy (Phase 2).

Parallel to ``build_format_sft.py`` (pair-level format SFT): before GRPO the
policy should already speak the scene-level tool grammar — ``retrieve``,
``match`` / ``crop_and_match``, ``doppelganger_check``, ``sfm_run``,
``inspect``, ``done`` — in the exact message format ``run_scene_episode``
produces at rollout time.

Each example is an ORACLE trajectory over a MegaDepth scene loaded through
``SceneDataset`` (``scripts/run_scene_grpo.py``).  The oracle does not run the
real tools: it synthesizes plausible tool results from the scene's GT data —
the ``overlap_matrix`` ranks candidate pairs and drives match/inlier counts,
``gt_recon['poses']`` (+ ``intrinsics`` from the scene_info npz) yield GT
relative poses and GT-projected crop boxes, and the overlap graph determines
how many images ``sfm_run`` registers.  Observations are rendered with the
real ``format_observation`` so the text matches deployment exactly.

No GPU / tool server needed.

Usage:
  python scripts/build_scene_sft.py \
      --scenes 0015 0022 \
      --num-episodes 200 \
      --output data/scene_sft_train.jsonl \
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))  # for `import scripts.run_scene_grpo`

from agentic_sfm.agent.policy import ToolCall, format_observation  # noqa: E402
from agentic_sfm.constants import DEFAULT_MATCHER  # noqa: E402
from agentic_sfm.geometry import ORACLE_CROP_BOXES  # noqa: E402
from agentic_sfm.rewards.pose_rewards import compute_scene_reward  # noqa: E402
from agentic_sfm.rl.scene_episode import SCENE_SYSTEM_PROMPT, _pair_key  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Same defaults as configs/phase2_scene.yaml data section.
DEFAULT_SCENE_INFO_DIR = "/scratch/kcwp264/data/megadepth/scene_info_full/scene_info"
DEFAULT_IMAGE_ROOT = "/scratch/kcwp264/data/megadepth/megadepth_test_1500"

# Tools the scene grammar allows (must stay a subset of what
# run_scene_episode / _execute_scene_tool handle).
ALLOWED_SCENE_TOOLS = frozenset({
    "retrieve", "match", "crop", "crop_and_match",
    "doppelganger_check", "sfm_run", "inspect", "done",
})

# Scene-level reward weights — mirrored from configs/phase2_scene.yaml so the
# stored ``reward``/``reward_components`` metadata is meaningful.
SCENE_REWARD_CFG: dict[str, Any] = {
    "registration_weight": 0.5,
    "pose_weight": 1.0,
    "split_penalty": 0.5,
    "doppelganger_weight": 0.3,
    "tool_cost": 0.0,
    "accumulative_tool_coef": 0.1,
    "use_accumulative_tool_reward": True,
    "use_ntep_rewards": False,
}

_N_THUMBNAILS = 8  # run_scene_episode shows up to 8 scene images


# ---------------------------------------------------------------------------
# Scene subsetting / GT helpers
# ---------------------------------------------------------------------------


def _subset_scene_at(scene: dict[str, Any], positions: list[int]) -> dict[str, Any]:
    """Copy ``scene`` restricted to ``positions`` (remaps indices + GT poses)."""
    pos = sorted({int(p) for p in positions})
    new = dict(scene)
    new["image_paths"] = [scene["image_paths"][i] for i in pos]
    new["num_images"] = len(pos)
    if scene.get("image_indices") is not None:
        new["image_indices"] = [scene["image_indices"][i] for i in pos]
    gt = scene.get("gt_recon")
    if isinstance(gt, dict) and isinstance(gt.get("poses"), dict):
        new["gt_recon"] = {
            **gt,
            "num_images": len(pos),
            "poses": {
                str(j): gt["poses"][str(i)]
                for j, i in enumerate(pos)
                if str(i) in gt["poses"]
            },
        }
    return new


def _episode_subset(
    scene: dict[str, Any], k: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Draw a ``k``-image episode subset; mixes three sampling strategies.

    Evenly-spaced (curriculum-style), contiguous blocks (a photographer's
    burst), and uniform-random subsets keep the ~200 episodes diverse even
    though scenes share a fixed loaded image pool.
    """
    n = scene["num_images"]
    k = max(2, min(int(k), n))
    strategy = rng.random()
    if strategy < 0.5 or k >= n:
        pos = np.linspace(0, n - 1, k).round().astype(int).tolist()
        if len(set(pos)) < k:  # linspace rounding collapsed — pad
            chosen = set(pos)
            for i in range(n):
                if i not in chosen:
                    pos.append(i)
                    chosen.add(i)
                    if len(pos) == k:
                        break
    elif strategy < 0.8:
        start = int(rng.integers(0, n - k + 1))
        pos = list(range(start, start + k))
    else:
        pos = sorted(rng.choice(n, size=k, replace=False).tolist())
    return _subset_scene_at(scene, pos)


def _load_scene_geometry(
    scene_info_dir: str, scene_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Load per-scene ``poses``/``intrinsics`` object arrays from the npz files.

    ``SceneDataset`` keeps only sampled poses in ``gt_recon``; intrinsics are
    needed for GT-projected crop boxes, so we cache the raw arrays here keyed
    by npz index (``scene['image_indices']`` maps episode positions to them).
    """
    geom: dict[str, dict[str, Any]] = {}
    for sid in scene_ids:
        npz = Path(scene_info_dir) / f"{sid}.npz"
        try:
            d = np.load(str(npz), allow_pickle=True)
            geom[sid] = {
                "poses": d["poses"],
                "intrinsics": d["intrinsics"],
            }
        except Exception as e:
            logger.warning(f"No geometry cache for scene {sid}: {e}")
    return geom


def _gt_pose(scene: dict[str, Any], i: int) -> np.ndarray | None:
    """4x4 cam-from-world GT pose for episode position ``i``, or None."""
    gt = scene.get("gt_recon") or {}
    poses = gt.get("poses") or {}
    p = poses.get(str(i))
    if p is None:
        return None
    arr = np.asarray(p, dtype=np.float64)
    return arr if arr.shape == (4, 4) else None


def _gt_relative_pose(
    pose_a: np.ndarray | None, pose_b: np.ndarray | None
) -> dict[str, Any] | None:
    """Relative cam-from-world pose a->b as the matcher's ``pose`` dict."""
    if pose_a is None or pose_b is None:
        return None
    R_a, t_a = pose_a[:3, :3], pose_a[:3, 3]
    R_b, t_b = pose_b[:3, :3], pose_b[:3, 3]
    R = R_b @ R_a.T
    t = t_b - R @ t_a
    return {"R": R.tolist(), "t": t.tolist()}


def _camera_center(pose_c2w: np.ndarray) -> np.ndarray:
    """World-frame camera center of a cam-from-world 4x4."""
    R, t = pose_c2w[:3, :3], pose_c2w[:3, 3]
    return -R.T @ t


def _image_size_wh(
    image_root: str, rel_path: str, K: np.ndarray | None
) -> tuple[int, int]:
    """(W, H) from the real image, else inferred from K, else a default."""
    try:
        from PIL import Image

        full = Path(image_root) / rel_path if image_root else Path(rel_path)
        with Image.open(full) as im:
            return im.size
    except Exception:
        pass
    if K is not None:
        try:
            w = int(round(2.0 * float(K[0, 2])))
            h = int(round(2.0 * float(K[1, 2])))
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
    return 1600, 1200


def _gt_overlap_bbox(
    scene: dict[str, Any],
    geom: dict[str, Any] | None,
    i: int,
    j: int,
    image_root: str,
    rng: np.random.Generator,
) -> list[float] | None:
    """GT-derived normalized bbox: where image ``j``'s view lands in ``i``.

    A pixel grid of image j is unprojected at two depth shells (scaled by the
    i-j baseline), pushed through the GT cam-from-world poses into camera i,
    and the robust bounding box of the in-frame projections is returned in
    normalized [x1, y1, x2, y2].  ``None`` when the projection degenerates
    (cameras facing away, missing GT) so the caller can fall back to the
    heuristic ORACLE_CROP_BOXES.
    """
    pose_i = _gt_pose(scene, i)
    pose_j = _gt_pose(scene, j)
    if pose_i is None or pose_j is None:
        return None

    idx = scene.get("image_indices") or list(range(scene["num_images"]))
    K_i = K_j = None
    if geom is not None:
        intr = geom.get("intrinsics")
        if intr is not None and idx[i] < len(intr) and idx[j] < len(intr):
            try:
                K_i = np.asarray(intr[idx[i]], dtype=np.float64).reshape(3, 3)
                K_j = np.asarray(intr[idx[j]], dtype=np.float64).reshape(3, 3)
            except Exception:
                K_i = K_j = None
    paths = scene["image_paths"]
    W_i, H_i = _image_size_wh(image_root, paths[i], K_i)
    W_j, H_j = _image_size_wh(image_root, paths[j], K_j)
    if K_i is None or K_j is None:
        f_i = 1.2 * max(W_i, H_i)
        f_j = 1.2 * max(W_j, H_j)
        K_i = np.array([[f_i, 0, W_i / 2], [0, f_i, H_i / 2], [0, 0, 1.0]])
        K_j = np.array([[f_j, 0, W_j / 2], [0, f_j, H_j / 2], [0, 0, 1.0]])

    baseline = float(np.linalg.norm(_camera_center(pose_i) - _camera_center(pose_j)))
    if baseline <= 0:
        return None

    us = np.linspace(0.03 * W_j, 0.97 * W_j, 11)
    vs = np.linspace(0.03 * H_j, 0.97 * H_j, 11)
    uu, vv = np.meshgrid(us, vs)
    pix = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)])  # 3 x P
    try:
        rays = np.linalg.inv(K_j) @ pix
    except np.linalg.LinAlgError:
        return None

    R_i, t_i = pose_i[:3, :3], pose_i[:3, 3]
    R_j, t_j = pose_j[:3, :3], pose_j[:3, 3]
    xs, ys = [], []
    for depth in (baseline * 1.0, baseline * 2.5):
        pts_j = rays * depth
        Xw = R_j.T @ (pts_j - t_j[:, None])
        Xi = R_i @ Xw + t_i[:, None]
        z = Xi[2]
        ok = z > 1e-6
        if not ok.any():
            continue
        uv = (K_i @ Xi[:, ok])[:2] / z[ok]
        inside = (
            (uv[0] >= 0) & (uv[0] <= W_i) & (uv[1] >= 0) & (uv[1] <= H_i)
        )
        xs.append(uv[0][inside])
        ys.append(uv[1][inside])
    if not xs:
        return None
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    if xs.size < 4:
        return None

    x1, x2 = np.quantile(xs, 0.02), np.quantile(xs, 0.98)
    y1, y2 = np.quantile(ys, 0.02), np.quantile(ys, 0.98)
    pad_x, pad_y = 0.04 * W_i, 0.04 * H_i
    x1, x2 = x1 - pad_x, x2 + pad_x
    y1, y2 = y1 - pad_y, y2 + pad_y
    # Sensible minimum size; clip to normalized [0, 1].
    if (x2 - x1) < 0.15 * W_i:
        cx = 0.5 * (x1 + x2)
        x1, x2 = cx - 0.075 * W_i, cx + 0.075 * W_i
    if (y2 - y1) < 0.15 * H_i:
        cy = 0.5 * (y1 + y2)
        y1, y2 = cy - 0.075 * H_i, cy + 0.075 * H_i
    bbox = [
        float(np.clip(x1 / W_i, 0.0, 1.0)),
        float(np.clip(y1 / H_i, 0.0, 1.0)),
        float(np.clip(x2 / W_i, 0.0, 1.0)),
        float(np.clip(y2 / H_i, 0.0, 1.0)),
    ]
    if bbox[2] - bbox[0] < 0.05 or bbox[3] - bbox[1] < 0.05:
        return None
    return [round(v, 3) for v in bbox]


# ---------------------------------------------------------------------------
# Synthesized tool results (GT-informed, no tool server)
# ---------------------------------------------------------------------------


def _retr_score(overlap: float, rng: np.random.Generator) -> float:
    """Plausible retrieval similarity for a GT overlap value."""
    s = 0.32 + 0.62 * max(overlap, 0.0) + float(rng.normal(0, 0.05))
    return float(np.clip(s, 0.02, 0.99))


def _synth_match_result(
    overlap: float,
    pose: dict[str, Any] | None,
    rng: np.random.Generator,
    matcher: str,
) -> dict[str, Any]:
    """Match stats consistent with the pair's GT overlap.

    High overlap -> many LoFTR matches, high inlier ratio, GT relative pose.
    ~Zero overlap -> few matches, degenerate verification (no pose).
    """
    if overlap <= 0.01:
        num_matches = int(rng.integers(60, 380))
        ratio = float(rng.uniform(0.005, 0.07))
        pose_out = None
        resid = float(rng.uniform(0.8, 1.4))
    else:
        num_matches = int(np.clip(rng.normal(250 + 750 * overlap, 90), 40, 1400))
        ratio = float(np.clip(0.10 + 0.85 * overlap + rng.normal(0, 0.07), 0.03, 0.92))
        pose_out = pose
        resid = float(rng.uniform(0.2, 0.7))
    num_inliers = int(round(num_matches * ratio))
    return {
        "matcher": matcher,
        "num_matches": num_matches,
        "num_inliers": num_inliers,
        "inlier_ratio": round(num_inliers / max(num_matches, 1), 4),
        "pose": pose_out,
        "mean_residual": round(float(rng.uniform(2e-4, 8e-4)), 6),
        "residual_units": round(resid, 3),
    }


def _synth_crop_result(
    image_id: str,
    bbox: list[float],
    size_wh: tuple[int, int],
    path: str,
) -> dict[str, Any]:
    """Crop result with the server's pixel-coordinate crop id convention."""
    w, h = size_wh
    x1 = max(0, int(bbox[0] * w))
    y1 = max(0, int(bbox[1] * h))
    x2 = min(w, int(bbox[2] * w))
    y2 = min(h, int(bbox[3] * h))
    crop_id = f"{image_id}_crop_{x1}_{y1}_{x2}_{y2}"
    return {
        "cropped_image_id": crop_id,
        "crop_id": crop_id,
        "path": path,
        "origin_xy": [x1, y1],
        "size": [x2 - x1, y2 - y1],
        "crop_size": [x2 - x1, y2 - y1],
    }


def _synth_doppelganger_result(
    flagged: bool, rng: np.random.Generator
) -> dict[str, Any]:
    """DoppelgangerDetector-style result (calibrated confidence + verdict)."""
    if flagged:
        conf = float(rng.uniform(0.55, 0.93))
        sim = float(rng.uniform(0.72, 0.95))
        ratio = float(rng.uniform(0.005, 0.08))
        verdict = "doppelganger"
        geom = float(rng.uniform(0.05, 0.30))
        num_inliers = int(rng.integers(2, 30))
    else:
        conf = float(rng.uniform(0.04, 0.42))
        sim = float(rng.uniform(0.35, 0.75))
        ratio = float(rng.uniform(0.15, 0.55))
        verdict = "match" if rng.random() < 0.6 else "uncertain"
        geom = float(rng.uniform(0.55, 0.9))
        num_inliers = int(rng.integers(40, 300))
    return {
        "is_doppelganger": bool(flagged),
        "confidence": round(conf, 4),
        "score": round(conf, 4),
        "similarity_score": round(sim, 4),
        "patch_consistency": round(float(rng.uniform(0.4, 0.9)), 4),
        "patch_coverage": round(float(rng.uniform(0.3, 0.8)), 4),
        "inlier_ratio": round(ratio, 4),
        "num_matches": int(rng.integers(120, 500)),
        "num_inliers": num_inliers,
        "geom_score": round(geom, 4),
        "residual_units": round(float(rng.uniform(0.3, 1.2)), 3),
        "verdict": verdict,
        "method": "dino_v2+geometry",
        "threshold": 0.5,
    }


def _largest_components(
    n: int, edge_ok: np.ndarray
) -> tuple[list[int], int]:
    """Connected components of the n-node graph given by ``edge_ok`` (N×N bool).

    Returns (component_sizes_sorted_desc, num_components_with_>=2_nodes).
    """
    seen = [False] * n
    sizes: list[int] = []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(n):
                if not seen[v] and (edge_ok[u, v] or edge_ok[v, u]):
                    seen[v] = True
                    stack.append(v)
        sizes.append(len(comp))
    sizes.sort(reverse=True)
    n_multi = sum(1 for s in sizes if s >= 2)
    return sizes, n_multi


def _synth_sfm_result(
    scene: dict[str, Any],
    sub_om: np.ndarray,
    flagged_keys: set[str],
    ids: list[str],
    num_checks: int,
    rng: np.random.Generator,
    output_dir: str,
    stragglers: int = 0,
) -> dict[str, Any]:
    """COLMAP result implied by the GT overlap graph minus flagged pairs.

    Registered count = largest covisible component (edge when GT overlap >=
    0.03, the rough LoFTR/COLMAP covisibility floor) minus ``stragglers``.
    ``num_pairs_matched`` mirrors the server: 0 when the episode ran no
    doppelganger_check (server falls back to exhaustive matching), else the
    size of the exhaustive-minus-flagged pair list.
    """
    n = len(ids)
    edge = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if sub_om[i, j] < 0.03:
                continue
            if _pair_key(ids[i], ids[j]) in flagged_keys:
                continue
            edge[i, j] = edge[j, i] = True
    sizes, n_multi = _largest_components(n, edge)
    registered = sizes[0] if sizes and sizes[0] >= 2 else 0
    registered = max(0, registered - int(stragglers))

    if num_checks > 0:
        num_pairs_matched = n * (n - 1) // 2 - len(flagged_keys)
    else:
        num_pairs_matched = 0  # server-side exhaustive matching

    result: dict[str, Any] = {
        "num_registered": int(registered),
        "num_points3d": int(registered * rng.uniform(300, 900)),
        "output_dir": output_dir,
        "num_pairs_matched": int(max(num_pairs_matched, 0)),
        "mean_reproj_error": round(float(rng.uniform(0.6, 1.8)), 3),
        "num_components": int(n_multi),
        "num_doppelgangers_present": len(flagged_keys),
        "num_doppelgangers_filtered": len(flagged_keys),
    }
    if registered == 0:
        result["status"] = "no_convergence"
    return result


# ---------------------------------------------------------------------------
# Oracle trajectory
# ---------------------------------------------------------------------------


def build_oracle_episode(
    scene: dict[str, Any],
    image_root: str,
    ep_idx: int,
    rng: np.random.Generator,
    matcher: str = DEFAULT_MATCHER,
    geom: dict[str, Any] | None = None,
    output_base: str = "outputs/scene_sft",
) -> dict[str, Any]:
    """Synthesize one oracle scene episode -> SFT example dict.

    Trajectory: retrieve -> match/crop_and_match on top GT-overlap pairs ->
    (match +) doppelganger_check on a low-overlap look-alike -> sfm_run ->
    [extra matches + sfm_run when under-registered] -> inspect -> done.
    """
    scene_id = scene["scene_id"]
    n = scene["num_images"]
    ids = [f"img_{i:04d}" for i in range(n)]
    image_indices = scene.get("image_indices") or list(range(n))
    om = np.asarray(scene["overlap_matrix"], dtype=np.float64)
    sub_om = om[np.ix_(image_indices, image_indices)] if om.ndim == 2 else np.full(
        (n, n), -1.0
    )

    # All candidate pairs ranked by GT overlap (oracle knowledge).
    cand = [
        (float(sub_om[i, j]), i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if sub_om[i, j] > 0
    ]
    cand.sort(key=lambda c: c[0], reverse=True)
    if not cand:  # degenerate scene: fall back to consecutive pairs, ~0 overlap
        cand = [(0.02, i, i + 1) for i in range(n - 1)]

    # A low-GT-overlap "look-alike" pair is the doppelganger candidate.
    low_pairs = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if sub_om[i, j] <= 0.01
    ]
    doppel_ij: tuple[int, int] | None = None
    if low_pairs:
        doppel_ij = low_pairs[int(rng.integers(0, len(low_pairs)))]
    else:
        worst = min(
            ((float(sub_om[i, j]), i, j) for i in range(n) for j in range(i + 1, n)),
            key=lambda c: c[0],
            default=None,
        )
        doppel_ij = (worst[1], worst[2]) if worst else None
    # Whether this episode's check flags a doppelganger (~55%).
    flag_doppel = doppel_ij is not None and rng.random() < 0.55
    run_doppel_check = flag_doppel or (doppel_ij is not None and rng.random() < 0.45)

    steps: list[tuple[ToolCall, dict[str, Any] | None]] = []
    used_pairs: set[str] = set()  # pair keys already matched ("do not repeat")

    def _emit(tool: str, args: dict[str, Any], result: dict[str, Any] | None) -> None:
        if tool in ("match", "crop_and_match", "doppelganger_check"):
            a = args.get("image_a") or args.get("image_id") or "?"
            b = args.get("image_b") or "?"
            used_pairs.add(_pair_key(a, b))
        steps.append((ToolCall(tool=tool, args=args), result))

    # ---- retrieve --------------------------------------------------------
    top_k = int(rng.choice([10, 15, 20, 25]))
    retr = [
        {"image_a": ids[i], "image_b": ids[j], "score": round(_retr_score(o, rng), 4)}
        for o, i, j in cand[:top_k]
    ]
    # Retrieval noise: the look-alike pair leaks into the tail of the list.
    if run_doppel_check and doppel_ij is not None and retr and rng.random() < 0.7:
        di, dj = doppel_ij
        key = _pair_key(ids[di], ids[dj])
        if not any(
            _pair_key(p["image_a"], p["image_b"]) == key for p in retr
        ):
            retr[-1] = {
                "image_a": ids[di],
                "image_b": ids[dj],
                "score": round(float(rng.uniform(0.38, 0.60)), 4),
            }
    _emit("retrieve", {"top_k": top_k}, {"pairs": retr, "num_pairs": len(retr)})

    # ---- match / crop_and_match on the best pairs -------------------------
    # Plain match on the top-overlap pairs; crop_and_match on mid-overlap
    # pairs deeper in the retrieve list (the "crop when overlap is small"
    # branch of the workflow).
    n_match = int(min(rng.integers(4, 10), len(cand)))
    n_top = int(min(rng.integers(3, 7), n_match))

    def _emit_crop_and_match(o: float, i: int, j: int) -> None:
        pose = _gt_relative_pose(_gt_pose(scene, i), _gt_pose(scene, j))
        crop_pos, other = (i, j) if rng.random() < 0.5 else (j, i)
        bbox = _gt_overlap_bbox(scene, geom, crop_pos, other, image_root, rng)
        if bbox is None:
            b = ORACLE_CROP_BOXES[int(rng.integers(0, len(ORACLE_CROP_BOXES)))]
            bbox = [
                round(float(np.clip(v + rng.normal(0, 0.02), 0.0, 1.0)), 3)
                for v in b
            ]
        K = None
        if geom is not None and geom.get("intrinsics") is not None:
            try:
                K = np.asarray(
                    geom["intrinsics"][image_indices[crop_pos]], dtype=np.float64
                )
            except Exception:
                K = None
        size_wh = _image_size_wh(image_root, scene["image_paths"][crop_pos], K)
        crop_res = _synth_crop_result(
            ids[crop_pos], bbox, size_wh,
            str(Path(image_root) / scene["image_paths"][crop_pos])
            if image_root
            else scene["image_paths"][crop_pos],
        )
        # Cropping zooms into the shared overlap -> boosted inlier ratio.
        result = _synth_match_result(
            max(o, 0.05) * rng.uniform(1.1, 1.5), pose, rng, matcher
        )
        result["crop"] = crop_res
        result["cropped_image_id"] = crop_res["cropped_image_id"]
        _emit(
            "crop_and_match",
            {
                "image_id": ids[crop_pos],
                "bbox": bbox,
                "image_b": ids[other],
                "matcher": matcher,
            },
            result,
        )

    for o, i, j in cand[:n_top]:
        pose = _gt_relative_pose(_gt_pose(scene, i), _gt_pose(scene, j))
        if 0.02 <= o < 0.35 and rng.random() < 0.5:
            _emit_crop_and_match(o, i, j)
        else:
            _emit(
                "match",
                {"image_a": ids[i], "image_b": ids[j], "matcher": matcher},
                _synth_match_result(o, pose, rng, matcher),
            )

    # Mid-overlap pairs from deeper in the retrieve list -> crop_and_match.
    mid = [
        c for c in cand[n_top:]
        if 0.02 <= c[0] < 0.4 and _pair_key(ids[c[1]], ids[c[2]]) not in used_pairs
    ]
    n_crop = int(min(rng.integers(1, 3), len(mid), max(0, n_match - n_top)))
    for o, i, j in mid[:n_crop]:
        _emit_crop_and_match(o, i, j)
    # Fill any remaining match budget with plain matches (unused pairs only).
    emitted = sum(
        1 for tc, _ in steps if tc.tool in ("match", "crop_and_match")
    )
    for o, i, j in cand:
        if emitted >= n_match:
            break
        if _pair_key(ids[i], ids[j]) in used_pairs:
            continue
        pose = _gt_relative_pose(_gt_pose(scene, i), _gt_pose(scene, j))
        _emit(
            "match",
            {"image_a": ids[i], "image_b": ids[j], "matcher": matcher},
            _synth_match_result(o, pose, rng, matcher),
        )
        emitted += 1

    # Occasional standalone crop -> match (teaches the two-step grammar).
    if n_match and rng.random() < 0.12:
        unused = [
            c for c in cand
            if _pair_key(ids[c[1]], ids[c[2]]) not in used_pairs
        ]
        pool = unused or cand
        o, i, j = pool[int(rng.integers(0, len(pool)))]
        crop_pos, other = (i, j) if rng.random() < 0.5 else (j, i)
        b = ORACLE_CROP_BOXES[int(rng.integers(0, len(ORACLE_CROP_BOXES)))]
        bbox = [
            round(float(np.clip(v + rng.normal(0, 0.02), 0.0, 1.0)), 3)
            for v in b
        ]
        K = None
        if geom is not None and geom.get("intrinsics") is not None:
            try:
                K = np.asarray(geom["intrinsics"][image_indices[crop_pos]], dtype=np.float64)
            except Exception:
                K = None
        size_wh = _image_size_wh(image_root, scene["image_paths"][crop_pos], K)
        crop_res = _synth_crop_result(
            ids[crop_pos], bbox, size_wh,
            str(Path(image_root) / scene["image_paths"][crop_pos])
            if image_root
            else scene["image_paths"][crop_pos],
        )
        _emit("crop", {"image_id": ids[crop_pos], "bbox": bbox}, crop_res)
        pose = _gt_relative_pose(_gt_pose(scene, crop_pos), _gt_pose(scene, other))
        _emit(
            "match",
            {
                "image_a": crop_res["cropped_image_id"],
                "image_b": ids[other],
                "matcher": matcher,
            },
            _synth_match_result(max(o, 0.05), pose, rng, matcher),
        )

    # ---- match the look-alike, then doppelganger_check it -----------------
    flagged_keys: set[str] = set()
    if run_doppel_check and doppel_ij is not None:
        di, dj = doppel_ij
        key = _pair_key(ids[di], ids[dj])
        if key not in used_pairs:
            _emit(
                "match",
                {"image_a": ids[di], "image_b": ids[dj], "matcher": matcher},
                _synth_match_result(0.0, None, rng, matcher),
            )
        result = _synth_doppelganger_result(flag_doppel, rng)
        _emit(
            "doppelganger_check",
            {"image_a": ids[di], "image_b": ids[dj]},
            result,
        )
        if flag_doppel:
            flagged_keys.add(key)

    # Occasional second check on a genuinely good pair -> clean verdict.
    if cand and rng.random() < 0.15:
        o, i, j = cand[int(rng.integers(0, min(3, len(cand))))]
        _emit(
            "doppelganger_check",
            {"image_a": ids[i], "image_b": ids[j]},
            _synth_doppelganger_result(False, rng),
        )

    # ---- sfm_run -----------------------------------------------------------
    num_checks = sum(1 for tc, _ in steps if tc.tool == "doppelganger_check")
    out_dir = f"{output_base}/{scene_id}_ep{ep_idx:04d}/sparse/0"
    stragglers = int(rng.integers(0, 3)) if n >= 6 and rng.random() < 0.4 else 0
    recon = _synth_sfm_result(
        scene, sub_om, flagged_keys, ids, num_checks, rng, out_dir,
        stragglers=stragglers,
    )
    _emit("sfm_run", {}, recon)

    # Recovery: under-registered (or no convergence) -> match a few more
    # pairs and re-run COLMAP, per the prompt's retry instruction.
    if recon["num_registered"] < 0.8 * n and rng.random() < 0.65:
        extra = [
            c for c in cand
            if _pair_key(ids[c[1]], ids[c[2]]) not in used_pairs
        ][:4]
        for o, i, j in extra:
            pose = _gt_relative_pose(_gt_pose(scene, i), _gt_pose(scene, j))
            _emit(
                "match",
                {"image_a": ids[i], "image_b": ids[j], "matcher": matcher},
                _synth_match_result(o, pose, rng, matcher),
            )
        if extra:
            recon = _synth_sfm_result(
                scene, sub_om, flagged_keys, ids, num_checks, rng, out_dir,
                stragglers=0,
            )
            _emit("sfm_run", {}, recon)

    # ---- inspect + done -----------------------------------------------------
    _emit(
        "inspect",
        {},
        {
            "num_images": recon["num_registered"],
            "num_registered": recon["num_registered"],
            "num_points3d": recon["num_points3d"],
            "num_cameras": 1,
            "mean_reproj_error": recon["mean_reproj_error"],
        },
    )
    _emit("done", {}, None)

    # ---- messages (exactly the run_scene_episode layout) --------------------
    n_show = min(n, _N_THUMBNAILS)
    content: list[dict[str, Any]] = []
    for i in range(n_show):
        full = str(Path(image_root) / scene["image_paths"][i]) if image_root else scene["image_paths"][i]
        content.append({"type": "image", "image": full})
    content.append(
        {
            "type": "text",
            "text": (
                f"Scene {scene_id} has {n} images. "
                f"Showing {n_show} thumbnails. "
                f"Registered as img_0000 through img_{n - 1:04d}. "
                f"Use retrieve to find good pairs, then match them, then run sfm_run."
            ),
        }
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SCENE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    for tc, result in steps:
        messages.append(
            {"role": "assistant", "content": json.dumps({"tool": tc.tool, "args": tc.args})}
        )
        if result is not None:
            messages.append(
                {"role": "user", "content": f"Observation: {format_observation(result)}"}
            )

    # ---- reward metadata (pose error injected post-hoc, like the episode) ---
    recon_final = dict(recon)
    if recon_final.get("num_registered", 0) > 0:
        recon_final["mean_pose_error_deg"] = round(float(rng.uniform(2.0, 9.0)), 3)
        recon_final["num_poses_evaluated"] = recon_final["num_registered"]
    tool_calls = [tc for tc, _ in steps]
    tool_results = [r for _, r in steps if r is not None]
    num_valid = sum(1 for tc in tool_calls if tc.tool != "done")
    rc = compute_scene_reward(
        recon_final,
        gt_recon=scene.get("gt_recon"),
        num_tool_calls=len(tool_calls),
        num_valid_calls=num_valid,
        tool_calls=tool_calls,
        tool_results=tool_results,
        **SCENE_REWARD_CFG,
    )

    abs_paths = [
        str(Path(image_root) / p) if image_root else str(p)
        for p in scene["image_paths"]
    ]
    return {
        "scene_id": scene_id,
        "episode_idx": ep_idx,
        "image_paths": abs_paths,
        "num_images": n,
        "messages": messages,
        "reward": rc["total_reward"],
        "reward_components": {**rc, "source": "scene_oracle"},
        "recon_result": recon_final,
        "num_tool_calls": len(tool_calls),
        "difficulty": "scene",
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_examples(
    scene_info_dir: str,
    image_root: str,
    scenes: list[str] | None = None,
    num_episodes: int = 200,
    seed: int = 42,
    min_images: int = 8,
    max_images: int = 30,
    pool_images: int = 60,
    matcher: str = DEFAULT_MATCHER,
    output_base: str = "outputs/scene_sft",
) -> list[dict[str, Any]]:
    """Generate ``num_episodes`` oracle SFT examples over the loaded scenes."""
    from scripts.run_scene_grpo import SceneDataset

    ds = SceneDataset.from_megadepth(
        scene_info_dir,
        image_root,
        scenes=scenes,
        max_images=pool_images,
    )
    pool = [s for s in ds.scenes if s["num_images"] >= min_images]
    if not pool:
        raise RuntimeError(
            f"No usable scenes in {scene_info_dir} (wanted {scenes}, "
            f">={min_images} existing images)"
        )
    logger.info(
        "Loaded %d scene(s): %s",
        len(pool),
        {s["scene_id"]: s["num_images"] for s in pool},
    )
    geom_cache = _load_scene_geometry(scene_info_dir, [s["scene_id"] for s in pool])

    rng = np.random.default_rng(seed)
    examples: list[dict[str, Any]] = []
    for ep_idx in tqdm(range(num_episodes), desc="Building scene SFT episodes"):
        scene = pool[ep_idx % len(pool)]  # round-robin keeps scenes balanced
        k = int(rng.integers(min_images, min(max_images, scene["num_images"]) + 1))
        sub = _episode_subset(scene, k, rng)
        try:
            ex = build_oracle_episode(
                sub,
                image_root=image_root,
                ep_idx=ep_idx,
                rng=rng,
                matcher=matcher,
                geom=geom_cache.get(scene["scene_id"]),
                output_base=output_base,
            )
            examples.append(ex)
        except Exception as e:
            logger.warning(f"Episode {ep_idx} ({scene['scene_id']}) failed: {e}")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scene-level oracle SFT data")
    parser.add_argument("--scene-info-dir", type=str, default=DEFAULT_SCENE_INFO_DIR)
    parser.add_argument("--image-root", type=str, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--scenes", type=str, nargs="+", default=["0015", "0022"],
                        help="MegaDepth scene ids with images under image-root")
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--output", type=str, default="data/scene_sft_train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-images", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=30)
    parser.add_argument("--pool-images", type=int, default=60,
                        help="Loaded per-scene pool that episode subsets draw from")
    parser.add_argument("--matcher", type=str, default=DEFAULT_MATCHER)
    parser.add_argument("--output-base", type=str, default="outputs/scene_sft",
                        help="Synthetic sfm_run output_dir prefix")
    args = parser.parse_args()

    examples = build_examples(
        scene_info_dir=args.scene_info_dir,
        image_root=args.image_root,
        scenes=args.scenes,
        num_episodes=args.num_episodes,
        seed=args.seed,
        min_images=args.min_images,
        max_images=args.max_images,
        pool_images=args.pool_images,
        matcher=args.matcher,
        output_base=args.output_base,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    # Summary sidecar, mirroring build_oracle_sft.py's oracle_sft_summary.json.
    tool_hist: dict[str, int] = {}
    n_flagged = 0
    for ex in examples:
        seen_flag = False
        for msg in ex["messages"]:
            if msg.get("role") != "assistant":
                continue
            try:
                call = json.loads(msg["content"])
            except Exception:
                continue
            tool = call.get("tool", "?")
            tool_hist[tool] = tool_hist.get(tool, 0) + 1
            if tool == "doppelganger_check":
                seen_flag = True
        n_flagged += int(seen_flag)
    summary = {
        "output": str(out_path),
        "num_episodes": len(examples),
        "scenes": sorted({ex["scene_id"] for ex in examples}),
        "tool_histogram": tool_hist,
        "episodes_with_doppelganger_check": n_flagged,
        "mean_images": float(np.mean([ex["num_images"] for ex in examples])) if examples else 0.0,
        "mean_reward": float(np.mean([ex["reward"] for ex in examples])) if examples else 0.0,
        "seed": args.seed,
    }
    summary_path = out_path.parent / "scene_sft_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved %d scene SFT examples to %s", len(examples), out_path)
    logger.info("Summary: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
