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
    "build_initial_graph",
    "inspect_graph",
    "verify_doppelganger",
    "prune_edges",
    "bridge_components",
    "sfm_run",
    "done",
    "retrieve",
    "match",
    "doppelganger_check",
    "inspect",
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

_N_THUMBNAILS = 8  # SFT trajectories; Phase-2 GRPO prompts now show 4 (vLLM cap)


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


def _synth_initial_graph_result(
    n: int,
    cand_pairs: list[tuple[int, int]],
    verified_pairs: list[tuple[int, int]],
    edge_ok: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    sizes, n_multi = _largest_components(n, edge_ok)
    return {
        "num_images": n,
        "num_candidate_edges": len(cand_pairs),
        "num_verified_edges": len(verified_pairs),
        "num_failed_edges": max(0, len(cand_pairs) - len(verified_pairs)),
        "num_components": max(1, n_multi),
        "component_sizes": sizes,
        "status": "ok",
    }


def _synth_inspect_graph_result(
    n: int,
    ids: list[str],
    verified_pairs: set[tuple[int, int]],
    edge_ok: np.ndarray,
    suspicious_pairs: list[tuple[int, int]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    sizes, n_multi = _largest_components(n, edge_ok)
    deg = [0] * n
    for i, j in verified_pairs:
        deg[i] += 1
        deg[j] += 1

    isolated = [ids[i] for i in range(n) if deg[i] == 0]

    seen = [False] * n
    components = []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(ids[u])
            for v in range(n):
                if not seen[v] and (edge_ok[u, v] or edge_ok[v, u]):
                    seen[v] = True
                    stack.append(v)
        components.append(comp)

    suspicious_cycles = []
    for di, dj in suspicious_pairs:
        k = (dj + 1) % n if (dj + 1) % n != di else (dj + 2) % n
        suspicious_cycles.append({
            "cycle": [ids[di], ids[dj], ids[k]],
            "rotation_error_deg": round(float(rng.uniform(18.0, 32.0)), 2),
            "max_error_edge": [ids[di], ids[dj]],
        })

    return {
        "num_images": n,
        "num_verified_edges": len(verified_pairs),
        "num_pruned_edges": 0,
        "num_components": max(1, n_multi),
        "component_sizes": sizes,
        "degrees": {
            "min": int(min(deg)) if deg else 0,
            "mean": round(float(np.mean(deg)), 2) if deg else 0.0,
            "max": int(max(deg)) if deg else 0,
        },
        "isolated_images": isolated,
        "bottlenecks": [],
        "weak_bridges": [],
        "suspicious_cycles": suspicious_cycles,
        "components": components,
        "status": "ok",
    }


def _synth_verify_doppelganger_result(
    image_a: str,
    image_b: str,
    is_doppelganger: bool,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if is_doppelganger:
        sim = float(rng.uniform(0.75, 0.95))
        ratio = float(rng.uniform(0.01, 0.08))
        inliers = int(rng.integers(5, 30))
        rot_err = float(rng.uniform(18.0, 34.0))
        evidence = f"High cycle rotation residual ({rot_err:.1f} deg > 15.0 deg)"
    else:
        sim = float(rng.uniform(0.50, 0.85))
        ratio = float(rng.uniform(0.20, 0.65))
        inliers = int(rng.integers(60, 350))
        rot_err = float(rng.uniform(1.2, 5.5))
        evidence = f"Cycle rotation residual is consistent ({rot_err:.1f} deg <= 15.0 deg)"

    return {
        "pair": [image_a, image_b],
        "visual_similarity": round(sim, 4),
        "num_inliers": inliers,
        "inlier_ratio": round(ratio, 4),
        "cycle_rotation_error_deg": round(rot_err, 2),
        "is_doppelganger": is_doppelganger,
        "evidence": evidence,
        "status": "ok",
    }


def _synth_prune_edges_result(
    pairs_pruned: list[list[str]],
    remaining_verified: set[tuple[int, int]],
    n: int,
    ids: list[str],
) -> dict[str, Any]:
    edge_ok = np.zeros((n, n), dtype=bool)
    for i, j in remaining_verified:
        edge_ok[i, j] = edge_ok[j, i] = True
    sizes, n_multi = _largest_components(n, edge_ok)

    seen = [False] * n
    components = []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(ids[u])
            for v in range(n):
                if not seen[v] and (edge_ok[u, v] or edge_ok[v, u]):
                    seen[v] = True
                    stack.append(v)
        components.append(comp)

    return {
        "num_pruned": len(pairs_pruned),
        "remaining_edges": len(remaining_verified),
        "num_components": max(1, n_multi),
        "components": components,
        "status": "ok",
    }


def _synth_bridge_components_result(
    comp_a_id: int,
    comp_b_id: int,
    added_edges: list[list[str]],
    remaining_verified: set[tuple[int, int]],
    n: int,
    ids: list[str],
) -> dict[str, Any]:
    edge_ok = np.zeros((n, n), dtype=bool)
    for i, j in remaining_verified:
        edge_ok[i, j] = edge_ok[j, i] = True
    sizes, n_multi = _largest_components(n, edge_ok)

    seen = [False] * n
    components = []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(ids[u])
            for v in range(n):
                if not seen[v] and (edge_ok[u, v] or edge_ok[v, u]):
                    seen[v] = True
                    stack.append(v)
        components.append(comp)

    new_edges_info = [
        {"pair": p, "inliers": 90, "inlier_ratio": 0.45, "status": "verified"}
        for p in added_edges
    ]
    return {
        "candidates_tested": max(3, len(added_edges)),
        "bridges_added": len(added_edges),
        "new_edges": new_edges_info,
        "num_components": max(1, n_multi),
        "components": components,
        "status": "ok",
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
    active_edge_count: int = 0,
) -> dict[str, Any]:
    """COLMAP result implied by the GT overlap graph minus flagged pairs.

    Registered count = largest covisible component (edge when GT overlap >=
    0.03, the rough LoFTR/COLMAP covisibility floor) minus ``stragglers``.
    ``num_pairs_matched`` mirrors the server: active verified edges matched.
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

    if active_edge_count > 0:
        num_pairs_matched = active_edge_count
    elif num_checks > 0:
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

SPECIALIST_MATCHERS: tuple[str, ...] = ("loftr", "mast3r", "vggt", "mapanything")
HEAVY_3D_SPECIALISTS: tuple[str, ...] = ("vggt", "mast3r", "mapanything")


def select_diagnostic_specialist_matcher(
    scene_id: str,
    median_overlap: float,
    ep_idx: int,
    rng: np.random.Generator,
    requested_matcher: str = "mixed",
    stage: str = "initial",
) -> str:
    """Diagnostic-conditioned specialist router for SFT trajectory generation.

    Prevents the 100% LoFTR warm-start collapse observed in run 8237240:
      - If ``requested_matcher`` is an explicit specialist name (not ``"mixed"``),
        returns ``requested_matcher`` directly for backwards compatibility.
      - When ``requested_matcher == "mixed"``:
        * On ``stage == "bridge"`` (repairing disconnected components where the
          initial graph left a gap), escalates to a 3D specialist
          (``"vggt"``, ``"mast3r"``, or ``"mapanything"``).
        * On indoor/low-texture ScanNet scenes (``scene_id`` starting with
          ``"scene"``) or low covisibility (``median_overlap < 0.22``), favors
          feed-forward 3D priors (``"vggt"``, ``"mast3r"``, ``"mapanything"``).
        * Otherwise stratifies across ``SPECIALIST_MATCHERS`` using ``ep_idx``
          so every specialist has non-zero probability mass at the SFT warm start.
    """
    req = str(requested_matcher or DEFAULT_MATCHER).lower()
    if req != "mixed":
        return req
    if stage == "bridge":
        return str(HEAVY_3D_SPECIALISTS[(ep_idx + int(rng.integers(0, len(HEAVY_3D_SPECIALISTS)))) % len(HEAVY_3D_SPECIALISTS)])
    is_indoor = str(scene_id).lower().startswith("scene")
    if is_indoor or median_overlap < 0.22:
        return str(HEAVY_3D_SPECIALISTS[ep_idx % len(HEAVY_3D_SPECIALISTS)])
    # Stratified rotation across all 4 matchers with 40% LoFTR, 20% each 3D specialist
    slot = ep_idx % 5
    if slot in (0, 1):
        return "loftr"
    return str(HEAVY_3D_SPECIALISTS[(slot - 2) % len(HEAVY_3D_SPECIALISTS)])


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

    Trajectory follows Scene-Level Graph Orchestration:
    build_initial_graph -> inspect_graph -> [verify_doppelganger & prune_edges] ->
    [bridge_components] -> sfm_run -> done.
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
    if not cand:
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

    flag_doppel = doppel_ij is not None and rng.random() < 0.55
    run_doppel_check = flag_doppel or (doppel_ij is not None and rng.random() < 0.45)

    steps: list[tuple[ToolCall, dict[str, Any] | None]] = []

    def _emit(tool: str, args: dict[str, Any], result: dict[str, Any] | None) -> None:
        steps.append((ToolCall(tool=tool, args=args), result))

    # ---- 1. build_initial_graph ------------------------------------------
    k_neighbors = int(rng.choice([1, 2, 3, 4, 5]))
    cand_pairs: set[tuple[int, int]] = set()
    for i in range(n):
        neighbors = sorted(
            [j for j in range(n) if j != i],
            key=lambda j: sub_om[min(i, j), max(i, j)],
            reverse=True,
        )[:k_neighbors]
        for j in neighbors:
            cand_pairs.add((min(i, j), max(i, j)))

    if run_doppel_check and doppel_ij is not None:
        cand_pairs.add((min(doppel_ij), max(doppel_ij)))

    verified_pairs: set[tuple[int, int]] = set()
    edge_ok = np.zeros((n, n), dtype=bool)
    for i, j in cand_pairs:
        if sub_om[i, j] >= 0.03:
            verified_pairs.add((i, j))
            edge_ok[i, j] = edge_ok[j, i] = True
        elif run_doppel_check and (i, j) == (min(doppel_ij), max(doppel_ij)):
            # False positive verified edge (look-alike loop closure)
            verified_pairs.add((i, j))
            edge_ok[i, j] = edge_ok[j, i] = True

    cand_list = list(cand_pairs)
    pos_overlaps = [float(sub_om[i, j]) for i, j in cand_list if sub_om[i, j] > 0]
    median_overlap = float(np.median(pos_overlaps)) if pos_overlaps else 0.0
    init_matcher = select_diagnostic_specialist_matcher(
        scene_id=scene_id,
        median_overlap=median_overlap,
        ep_idx=ep_idx,
        rng=rng,
        requested_matcher=matcher,
        stage="initial",
    )
    bridge_matcher = select_diagnostic_specialist_matcher(
        scene_id=scene_id,
        median_overlap=median_overlap,
        ep_idx=ep_idx,
        rng=rng,
        requested_matcher=matcher,
        stage="bridge",
    )
    init_res = _synth_initial_graph_result(n, cand_list, list(verified_pairs), edge_ok, rng)
    init_res["matcher_requested"] = init_matcher
    _emit("build_initial_graph", {"k_neighbors": k_neighbors, "matcher": init_matcher}, init_res)

    # ---- 2. inspect_graph ------------------------------------------------
    suspicious = [doppel_ij] if (run_doppel_check and doppel_ij is not None and (min(doppel_ij), max(doppel_ij)) in verified_pairs) else []
    insp_res = _synth_inspect_graph_result(n, ids, verified_pairs, edge_ok, suspicious, rng)
    _emit("inspect_graph", {}, insp_res)

    # ---- 3. verify_doppelganger & prune_edges ----------------------------
    flagged_keys: set[str] = set()
    if run_doppel_check and doppel_ij is not None and (min(doppel_ij), max(doppel_ij)) in verified_pairs:
        di, dj = doppel_ij
        v_res = _synth_verify_doppelganger_result(ids[di], ids[dj], flag_doppel, rng)
        _emit("verify_doppelganger", {"image_a": ids[di], "image_b": ids[dj]}, v_res)

        if flag_doppel:
            pair_key = (min(di, dj), max(di, dj))
            verified_pairs.discard(pair_key)
            edge_ok[di, dj] = edge_ok[dj, di] = False
            flagged_keys.add(_pair_key(ids[di], ids[dj]))
            p_res = _synth_prune_edges_result([[ids[di], ids[dj]]], verified_pairs, n, ids)
            _emit("prune_edges", {"pairs": [[ids[di], ids[dj]]]}, p_res)

    if verified_pairs and rng.random() < 0.15:
        gi, gj = list(verified_pairs)[0]
        v_good = _synth_verify_doppelganger_result(ids[gi], ids[gj], False, rng)
        _emit("verify_doppelganger", {"image_a": ids[gi], "image_b": ids[gj]}, v_good)

    # ---- 4. bridge_components (if disconnected) --------------------------
    sizes, n_multi = _largest_components(n, edge_ok)
    if n_multi > 1 or (sizes and sizes[0] < n and len(sizes) > 1):
        comp0_nodes: list[int] = []
        seen = [False] * n
        stack = [0]
        seen[0] = True
        while stack:
            u = stack.pop()
            comp0_nodes.append(u)
            for v in range(n):
                if not seen[v] and edge_ok[u, v]:
                    seen[v] = True
                    stack.append(v)
        other_nodes = [v for v in range(n) if v not in comp0_nodes]
        if other_nodes:
            best_bridge = max(
                ((sub_om[min(u, v), max(u, v)], min(u, v), max(u, v)) for u in comp0_nodes for v in other_nodes),
                key=lambda x: x[0],
                default=None,
            )
            if best_bridge and best_bridge[0] > 0.01:
                bu, bv = best_bridge[1], best_bridge[2]
                verified_pairs.add((bu, bv))
                edge_ok[bu, bv] = edge_ok[bv, bu] = True
                b_res = _synth_bridge_components_result(0, 1, [[ids[bu], ids[bv]]], verified_pairs, n, ids)
                _emit("bridge_components", {"comp_a_id": 0, "comp_b_id": 1, "budget": 3, "matcher": bridge_matcher}, b_res)

    # ---- 5. sfm_run ------------------------------------------------------
    out_dir = f"{output_base}/{scene_id}_ep{ep_idx:04d}/sparse/0"
    stragglers = int(rng.integers(0, 2)) if n >= 8 and rng.random() < 0.3 else 0
    num_checks = sum(1 for tc, _ in steps if tc.tool == "verify_doppelganger")
    recon = _synth_sfm_result(
        scene, sub_om, flagged_keys, ids, num_checks, rng, out_dir,
        stragglers=stragglers, active_edge_count=len(verified_pairs),
    )
    _emit("sfm_run", {}, recon)

    # Recovery: under-registered -> bridge or add matches and re-run sfm_run
    if recon["num_registered"] < 0.8 * n and rng.random() < 0.5:
        added = False
        for o, i, j in cand:
            if (i, j) not in verified_pairs and _pair_key(ids[i], ids[j]) not in flagged_keys:
                verified_pairs.add((i, j))
                edge_ok[i, j] = edge_ok[j, i] = True
                added = True
                break
        if added:
            b_res = _synth_bridge_components_result(0, 1, [[ids[i], ids[j]]], verified_pairs, n, ids)
            _emit("bridge_components", {"comp_a_id": 0, "comp_b_id": 1, "budget": 2, "matcher": bridge_matcher}, b_res)
            recon = _synth_sfm_result(
                scene, sub_om, flagged_keys, ids, num_checks, rng, out_dir,
                stragglers=0, active_edge_count=len(verified_pairs),
            )
            _emit("sfm_run", {}, recon)

    # ---- 6. done ---------------------------------------------------------
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
                f"Call build_initial_graph to initialize the view graph, inspect/optimize it, and run sfm_run."
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
    matcher_hist: dict[str, int] = {}
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
            args_dict = call.get("args") or {}
            if isinstance(args_dict, dict) and "matcher" in args_dict:
                m_name = str(args_dict["matcher"]).lower()
                matcher_hist[m_name] = matcher_hist.get(m_name, 0) + 1
            if tool == "verify_doppelganger":
                seen_flag = True
        n_flagged += int(seen_flag)
    summary = {
        "output": str(out_path),
        "num_episodes": len(examples),
        "scenes": sorted({ex["scene_id"] for ex in examples}),
        "tool_histogram": tool_hist,
        "matcher_histogram": matcher_hist,
        "episodes_with_doppelganger_verification": n_flagged,
        "mean_images": float(np.mean([ex["num_images"] for ex in examples])) if examples else 0.0,
        "mean_reward": float(np.mean([ex["reward"] for ex in examples])) if examples else 0.0,
        "seed": args.seed,
    }
    summary_path = out_path.parent / (
        "scene_sft_summary.json" if out_path.name == "scene_sft_train.jsonl"
        else f"{out_path.stem}_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved %d scene SFT examples to %s", len(examples), out_path)
    logger.info("Summary: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
