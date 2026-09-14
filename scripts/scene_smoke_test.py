#!/usr/bin/env python
"""Scene-level smoke test for Phase 2.

Validates the full scene pipeline end-to-end:

  SceneDataset -> register images -> run_scene_episode with a scripted agent
  -> retrieve / match / doppelganger_check / sfm_run / inspect / done
  -> scene reward components + tool-call logging.

Mock mode (default) needs no GPU and no running servers: a MockToolClient
stands in for the HTTP tool server and retrieval can be swapped for cheap
deterministic embeddings.  ``--real`` mode points the same scripted episode
at a live tool server (see ``scripts/start_tool_server.sh``).

Usage:
  # Mock mode on real MegaDepth scene 0015 (10 images, DINOv2 on CPU):
  python scripts/scene_smoke_test.py

  # Fully offline / CI: synthetic scene + hash-based retrieval embeddings:
  python scripts/scene_smoke_test.py --synthetic --mock-retrieval

  # Real tool server (matchers + COLMAP must be running on the host):
  # --coherent picks a high-overlap image subset so COLMAP can register it.
  python scripts/scene_smoke_test.py --real --coherent \
      --tool-server-url http://localhost:8765

Exit code 0 = all checks passed, 1 = one or more checks failed.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

from agentic_sfm.constants import DEFAULT_MATCHER  # noqa: E402
from agentic_sfm.rl.scene_episode import (  # noqa: E402
    SceneRolloutEpisode,
    run_scene_episode,
)
from agentic_sfm.tools.client import ToolClient  # noqa: E402

logger = logging.getLogger(__name__)

# Default MegaDepth locations (same as configs/phase2_scene.yaml data section).
DEFAULT_SCENE_INFO_DIR = "/scratch/kcwp264/data/megadepth/scene_info_full/scene_info"
DEFAULT_IMAGE_ROOT = "/scratch/kcwp264/data/megadepth/megadepth_test_1500"

REQUIRED_REWARD_KEYS = (
    "registration_reward",
    "split_penalty",
    "pose_reward",
    "doppelganger_reward",
    "tool_cost",
    "accumulative_tool_reward",
    "ntep_intent_reward",
    "ntep_redundancy_penalty",
    "total_reward",
)


# ---------------------------------------------------------------------------
# Tool clients
# ---------------------------------------------------------------------------


class MockToolClient:
    """In-memory stand-in for the HTTP tool server — no GPU, no models.

    Records every call in ``call_log`` (``"tool:arg..."`` strings) and every
    registration in ``registered`` (image_id -> path) so the smoke test can
    verify the data flow.

    ``doppelganger_confidence`` is returned for every check; >= 0.5 flags the
    pair so it is filtered out of the sfm_run pair list. ``recon_pose_error_deg``
    is reported as ``mean_pose_error_deg`` on sfm_run — a real server leaves
    that field absent (the episode runner injects it from GT poses); the mock
    sets it so the pose-reward path is exercised without pycolmap.
    """

    def __init__(
        self,
        num_inliers: int = 150,
        doppelganger_confidence: float = 0.12,
        recon_pose_error_deg: float | None = 5.0,
    ):
        self.registered: dict[str, str] = {}
        self.call_log: list[str] = []
        self.sfm_pair_lists: list[Any] = []
        self.num_inliers = num_inliers
        self.doppelganger_confidence = doppelganger_confidence
        self.recon_pose_error_deg = recon_pose_error_deg

    def health(self) -> dict[str, Any]:
        self.call_log.append("health")
        return {"status": "ok", "mock": True}

    def register_image(self, image_id: str, path: str) -> dict[str, Any]:
        self.call_log.append(f"register_image:{image_id}")
        self.registered[image_id] = path
        return {"status": "ok", "image_id": image_id}

    def crop(self, image_id: str, bbox: list[float]) -> dict[str, Any]:
        self.call_log.append(f"crop:{image_id}")
        return {
            "cropped_image_id": f"{image_id}_crop",
            "crop_id": f"{image_id}_crop",
            "path": self.registered.get(image_id, image_id),
        }

    def match(self, image_a: str, image_b: str, matcher: str = DEFAULT_MATCHER,
              **kwargs: Any) -> dict[str, Any]:
        self.call_log.append(f"match:{image_a}:{image_b}")
        return {
            "num_matches": 2 * self.num_inliers,
            "num_inliers": self.num_inliers,
            "inlier_ratio": 0.6,
            "pose": {"R": np.eye(3).tolist(), "t": [0.0, 0.0, 1.0]},
            "matcher": matcher,
        }

    def doppelganger_check(self, image_a: str, image_b: str) -> dict[str, Any]:
        self.call_log.append(f"doppelganger_check:{image_a}:{image_b}")
        conf = float(self.doppelganger_confidence)
        return {
            "is_doppelganger": conf >= 0.5,
            "confidence": conf,
            "score": conf,
            "inlier_ratio": 0.6,
            "num_matches": 2 * self.num_inliers,
            "num_inliers": self.num_inliers,
            "verdict": "doppelganger" if conf >= 0.5 else "match",
            "method": "mock",
        }

    def sfm_run(self, image_dir: str, pair_list: list | None = None,
                output_dir: str = "./outputs/sfm_run") -> dict[str, Any]:
        self.sfm_pair_lists.append(pair_list)
        n_pairs = len(pair_list) if pair_list else 0
        self.call_log.append(f"sfm_run:{image_dir}:{n_pairs}")
        result: dict[str, Any] = {
            "num_registered": len(self.registered),
            "num_points3d": 100 * len(self.registered),
            "mean_reproj_error": 1.1,
            "output_dir": output_dir,
            "num_pairs_matched": n_pairs,
            "num_components": 1,
        }
        if self.recon_pose_error_deg is not None:
            result["mean_pose_error_deg"] = float(self.recon_pose_error_deg)
        return result

    def inspect(self, recon_dir: str) -> dict[str, Any]:
        self.call_log.append(f"inspect:{recon_dir}")
        return {
            "num_images": len(self.registered),
            "num_registered": len(self.registered),
            "num_points3d": 100 * len(self.registered),
            "num_cameras": 1,
            "mean_reproj_error": 1.1,
        }

    def close(self) -> None:
        pass


class InstrumentedToolClient(ToolClient):
    """Real HTTP ToolClient that records calls, for --real mode verification.

    Exposes the same ``registered`` / ``call_log`` / ``sfm_pair_lists``
    attributes as MockToolClient so ``verify_episode`` works on both.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.registered: dict[str, str] = {}
        self.call_log: list[str] = []
        self.sfm_pair_lists: list[Any] = []

    def health(self) -> dict[str, Any]:
        self.call_log.append("health")
        return super().health()

    def register_image(self, image_id: str, path: str) -> dict[str, Any]:
        self.call_log.append(f"register_image:{image_id}")
        self.registered[image_id] = path
        return super().register_image(image_id, path)

    def crop(self, image_id: str, bbox: list[float]) -> dict[str, Any]:
        self.call_log.append(f"crop:{image_id}")
        return super().crop(image_id, bbox)

    def match(self, image_a: str, image_b: str, matcher: str = DEFAULT_MATCHER,
              **kwargs: Any) -> dict[str, Any]:
        self.call_log.append(f"match:{image_a}:{image_b}")
        return super().match(image_a, image_b, matcher=matcher, **kwargs)

    def doppelganger_check(self, image_a: str, image_b: str) -> dict[str, Any]:
        self.call_log.append(f"doppelganger_check:{image_a}:{image_b}")
        return super().doppelganger_check(image_a, image_b)

    def sfm_run(self, image_dir: str, pair_list: list | None = None,
                output_dir: str = "./outputs/sfm_run") -> dict[str, Any]:
        self.sfm_pair_lists.append(pair_list)
        self.call_log.append(
            f"sfm_run:{image_dir}:{len(pair_list) if pair_list else 0}")
        return super().sfm_run(image_dir=image_dir, pair_list=pair_list,
                               output_dir=output_dir)

    def inspect(self, recon_dir: str) -> dict[str, Any]:
        self.call_log.append(f"inspect:{recon_dir}")
        return super().inspect(recon_dir)


# ---------------------------------------------------------------------------
# Scripted agent
# ---------------------------------------------------------------------------


class ScriptedSceneAgent:
    """Deterministic stand-in for the VLM policy.

    Runs a fixed Phase-2 pipeline: retrieve -> match top pairs ->
    doppelganger_check -> sfm_run -> inspect -> done.  Retrieved pairs are
    parsed back out of the observation text (``img_XXXX-img_YYYY``) so the
    agent works against any retrieval backend; if no pairs are visible it
    falls back to consecutive image ids.
    """

    _PAIR_RE = re.compile(r"(img_\d+)\s*-\s*(img_\d+)")
    _NUM_RE = re.compile(r"through\s+img_(\d+)")

    def __init__(self, num_matches: int = 5, matcher: str = DEFAULT_MATCHER,
                 run_doppelganger: bool = True,
                 reward_config: dict[str, Any] | None = None):
        self.num_matches = int(num_matches)
        self.matcher = matcher
        self.run_doppelganger = run_doppelganger
        self.reward_config = reward_config or {}
        self._pairs: list[tuple[str, str]] = []
        self._matched = 0
        self._phase = "retrieve"
        self._num_images: int | None = None

    # Interface expected by run_scene_episode -------------------------------

    def _encode_image(self, path: str) -> str:
        # The episode only stores this in ep.images; a stub keeps the smoke
        # test independent of image codecs / base64 size.
        return "smoke-test-image"

    def _generate_turn(self, messages: list[dict[str, Any]],
                       images: list | None = None) -> str:
        # Each episode starts with just [system, user]; reset so the same
        # agent instance can drive group_size rollouts per scene (the real
        # VLLMRolloutAgent is stateless).
        if len(messages) <= 2:
            self._pairs = []
            self._matched = 0
            self._phase = "retrieve"
            self._num_images = None
        self._harvest(messages)

        if self._phase == "retrieve":
            self._phase = "match"
            return json.dumps({"tool": "retrieve", "args": {"top_k": 20}})

        if self._phase == "match":
            if not self._pairs:
                n = self._num_images or (self.num_matches + 1)
                self._pairs = [
                    (f"img_{i:04d}", f"img_{i + 1:04d}")
                    for i in range(max(n - 1, 1))
                ]
            if self._matched < min(self.num_matches, len(self._pairs)):
                a, b = self._pairs[self._matched]
                self._matched += 1
                return json.dumps({"tool": "match", "args": {
                    "image_a": a, "image_b": b, "matcher": self.matcher}})
            self._phase = "doppelganger"

        if self._phase == "doppelganger":
            self._phase = "sfm"
            if self.run_doppelganger:
                a, b = self._pairs[0] if self._pairs else ("img_0000", "img_0001")
                return json.dumps({"tool": "doppelganger_check",
                                   "args": {"image_a": a, "image_b": b}})
            # fall through to sfm in the same turn

        if self._phase == "sfm":
            self._phase = "inspect"
            return json.dumps({"tool": "sfm_run", "args": {}})

        if self._phase == "inspect":
            self._phase = "done"
            return json.dumps({"tool": "inspect", "args": {}})

        return json.dumps({"tool": "done", "args": {}})

    # Internals ------------------------------------------------------------

    def _harvest(self, messages: list[dict[str, Any]]) -> None:
        """Pull retrieved pairs / scene size out of the conversation so far."""
        for msg in messages:
            content = msg.get("content")
            texts: list[str] = []
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                texts.extend(
                    item.get("text", "") for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            for text in texts:
                m = self._NUM_RE.search(text)
                if m:
                    self._num_images = int(m.group(1)) + 1
                pairs = self._PAIR_RE.findall(text)
                if pairs:
                    self._pairs = [(a, b) for a, b in pairs]

    def expected_tool_sequence(self) -> list[str]:
        """Tool names the agent plans to emit, for log verification."""
        tail = (["doppelganger_check"] if self.run_doppelganger else []) + [
            "sfm_run", "inspect", "done"]
        return ["retrieve"] + ["match"] * self._matched + tail


# ---------------------------------------------------------------------------
# Synthetic scene helper (offline mode / tests)
# ---------------------------------------------------------------------------


def write_synthetic_scene(si_dir: Path, img_root: Path, scene_id: str = "0015",
                          n_images: int = 12, seed: int = 0) -> str:
    """Write a minimal MegaDepth-style scene_info npz + real JPEG images.

    Returns the scene_id so it can be loaded through
    ``SceneDataset.from_megadepth`` (exercises the real loader path).
    """
    from PIL import Image

    si_dir = Path(si_dir)
    img_root = Path(img_root)
    si_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    paths: list[str] = []
    for i in range(n_images):
        rel = f"{scene_id}/img_{i:04d}.jpg"
        p = img_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        arr = (rng.random((128, 128, 3)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(p, "JPEG")
        paths.append(rel)

    poses = np.empty(n_images, dtype=object)
    intrinsics = np.empty(n_images, dtype=object)
    for i in range(n_images):
        M = np.eye(4)
        M[0, 3] = float(i)  # tag translation with index
        poses[i] = M
        intrinsics[i] = np.eye(3)
    overlap = np.eye(n_images)

    np.savez(
        si_dir / f"{scene_id}.npz",
        image_paths=np.array(paths),
        poses=poses,
        intrinsics=intrinsics,
        overlap_matrix=overlap,
    )
    return scene_id


def install_mock_retrieval() -> None:
    """Patch DINOv2 embedding extraction with deterministic hash embeddings.

    Keeps ``retrieve_pairs_from_paths`` fully exercised (cosine similarity,
    ranking, top-k) without torch / transformers / network access.
    """
    import agentic_sfm.rl.retrieval as retrieval

    def fake_embed(image_path: Any) -> np.ndarray:
        seed = zlib.crc32(str(image_path).encode())
        return np.random.default_rng(seed).random(64).astype(np.float32)

    retrieval.compute_image_embedding = fake_embed


def select_coherent_subset(scene: dict[str, Any], k: int) -> dict[str, Any]:
    """Pick the ``k`` loaded images with the highest mutual GT overlap.

    Random ``SceneDataset`` subsamples of a large scene often have too little
    covisibility for COLMAP to register anything (observed on 0015).  Greedy
    selection by ``overlap_matrix`` keeps ``--real`` smoke runs meaningful.
    ``image_indices`` and ``gt_recon`` poses are remapped to the subset.
    """
    n = scene["num_images"]
    k = min(k, n)
    if k >= n:
        return scene
    idx = scene.get("image_indices") or list(range(n))
    om = np.asarray(scene["overlap_matrix"], dtype=np.float64)
    sub = om[np.ix_(idx, idx)].copy()
    np.fill_diagonal(sub, -np.inf)

    i, j = np.unravel_index(int(np.argmax(sub)), sub.shape)
    sel = [int(i), int(j)]
    while len(sel) < k:
        rest = [x for x in range(n) if x not in sel]
        # Bottleneck criterion: maximize the *minimum* overlap with the
        # already-selected images, so every chosen pair stays well-connected.
        scores = sub[np.ix_(rest, sel)].min(axis=1)
        sel.append(rest[int(np.argmax(scores))])
    sel = sorted(sel)

    new = dict(scene)
    new["image_paths"] = [scene["image_paths"][i] for i in sel]
    new["num_images"] = len(sel)
    if scene.get("image_indices") is not None:
        new["image_indices"] = [scene["image_indices"][i] for i in sel]
    gt = scene.get("gt_recon")
    if isinstance(gt, dict) and isinstance(gt.get("poses"), dict):
        new["gt_recon"] = {
            **gt,
            "num_images": len(sel),
            "poses": {
                str(j): gt["poses"][str(i)]
                for j, i in enumerate(sel)
                if str(i) in gt["poses"]
            },
        }
    return new


def stage_images(image_paths: list[str], image_root: str,
                 stage_dir: Path) -> list[str]:
    """Symlink the sampled scene images into one flat dir (original basenames).

    In ``--real`` mode the server's sfm_run runs COLMAP over the *whole*
    parent directory of image_paths[0] — for MegaDepth that is thousands of
    images.  Staging keeps the real run fast while preserving basenames so
    GT-pose name resolution still works.
    """
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    seen: set[str] = set()
    for i, rel in enumerate(image_paths):
        src = Path(image_root) / rel if image_root else Path(rel)
        name = src.name
        if name in seen:  # basename collision: fall back to an indexed name
            name = f"{i:04d}_{name}"
        seen.add(name)
        dst = stage_dir / name
        if not dst.exists():
            dst.symlink_to(src.resolve())
        staged.append(str(dst))
    return staged


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def verify_episode(
    ep: SceneRolloutEpisode,
    client: Any,
    scene: dict[str, Any],
    expected_tools: list[str] | None = None,
) -> list[Check]:
    """Run all smoke-test assertions on a finished episode.

    ``client`` is a MockToolClient or InstrumentedToolClient (both expose
    ``registered`` and ``call_log``). Returns a Check list; nothing raises.
    """
    checks: list[Check] = []
    n_images = len(ep.image_paths)

    # 1. Images registered with the tool server ---------------------------
    expected_ids = {f"img_{i:04d}" for i in range(n_images)}
    got_ids = set(client.registered)
    checks.append(Check(
        "images_registered",
        expected_ids <= got_ids,
        f"{len(got_ids & expected_ids)}/{n_images} scene images registered",
    ))

    # Pair executed (non-done) calls with their results, in order.
    exec_calls = [tc for tc in ep.tool_calls if tc.tool != "done"]
    paired = list(zip(exec_calls, ep.results))
    by_tool: dict[str, list[dict[str, Any]]] = {}
    for tc, res in paired:
        by_tool.setdefault(tc.tool, []).append(res)

    # 2. Retrieval returns pairs ------------------------------------------
    retr = by_tool.get("retrieve", [])
    pairs = retr[0].get("pairs", []) if retr else []
    pairs_ok = all(
        isinstance(p, dict) and "image_a" in p and "image_b" in p for p in pairs
    )
    checks.append(Check(
        "retrieval_pairs",
        bool(retr) and len(pairs) > 0 and pairs_ok,
        f"{len(pairs)} pairs returned",
    ))

    # 3. Match results have num_inliers > 0 --------------------------------
    matches = by_tool.get("match", []) + by_tool.get("crop_and_match", [])
    match_ok = bool(matches) and all(
        not r.get("error") and float(r.get("num_inliers") or 0) > 0
        for r in matches
    )
    checks.append(Check(
        "match_inliers",
        match_ok,
        f"{len(matches)} match calls, "
        f"inliers={[r.get('num_inliers') for r in matches]}",
    ))

    # 4. Doppelganger check returns a confidence ---------------------------
    dops = by_tool.get("doppelganger_check", [])

    def _has_conf(r: dict[str, Any]) -> bool:
        c = r.get("confidence", r.get("score"))
        return isinstance(c, (int, float)) and not isinstance(c, bool)

    checks.append(Check(
        "doppelganger_confidence",
        bool(dops) and all(_has_conf(r) for r in dops),
        f"{len(dops)} checks, confidences="
        f"{[r.get('confidence', r.get('score')) for r in dops]}",
    ))
    checks.append(Check(
        "doppelganger_tracked",
        len(ep.doppelganger_checks) == len(dops),
        f"{len(ep.doppelganger_checks)} pair keys tracked in episode",
    ))

    # 5. sfm_run produces a reconstruction ---------------------------------
    recon = ep.recon_result or {}
    n_reg = recon.get("num_registered")
    checks.append(Check(
        "sfm_reconstruction",
        isinstance(recon, dict) and not recon.get("error")
        and isinstance(n_reg, (int, float)) and n_reg > 0,
        f"num_registered={n_reg}, num_points3d={recon.get('num_points3d')}",
    ))

    # inspect should run against the reconstruction the episode produced.
    inspects = by_tool.get("inspect", [])
    checks.append(Check(
        "inspect_result",
        bool(inspects) and all(not r.get("error") for r in inspects),
        f"{len(inspects)} inspect calls",
    ))

    # 6. Reward components computed correctly ------------------------------
    rc = ep.reward_components or {}
    missing = [k for k in REQUIRED_REWARD_KEYS if k not in rc]
    total = rc.get("total_reward")
    recomputed = sum(
        float(rc.get(k, 0.0)) for k in REQUIRED_REWARD_KEYS if k != "total_reward"
    )
    total_ok = (
        isinstance(total, (int, float))
        and abs(float(total) - recomputed) < 1e-6
        and abs(float(ep.reward) - float(total)) < 1e-9
    )
    checks.append(Check(
        "reward_components",
        not missing and total_ok,
        f"total={total}, recomputed={recomputed:.4f}, missing={missing}",
    ))
    checks.append(Check(
        "reward_positive",
        isinstance(total, (int, float)) and float(total) > 0,
        f"reward={ep.reward:.4f}",
    ))

    # 7. All tool calls logged ---------------------------------------------
    tools_seq = [tc.tool for tc in ep.tool_calls]
    seq_ok = expected_tools is None or tools_seq == expected_tools
    checks.append(Check(
        "tool_calls_logged",
        ep.done and seq_ok
        and len(ep.results) == len(exec_calls)
        and len(ep.assistant_responses) == len(ep.tool_calls),
        f"tools={tools_seq}",
    ))

    # Every executed server-side call shows up in the client call log.
    # ("retrieve" is local — it computes embeddings, no server round-trip.)
    log = client.call_log
    log_ok = True
    log_detail = []
    n_server_calls = 0
    for tc in exec_calls:
        if tc.tool == "retrieve":
            continue
        n_server_calls += 1
        prefix = {"crop_and_match": "match:"}.get(tc.tool, f"{tc.tool}:")
        count = sum(1 for entry in log if entry.startswith(prefix))
        if count < 1:
            log_ok = False
        log_detail.append(f"{tc.tool}x{count}")
    checks.append(Check(
        "client_call_log",
        len(log) >= n_server_calls and log_ok,
        ", ".join(log_detail),
    ))

    # Thumbnails encoded into the prompt.
    checks.append(Check(
        "scene_images_encoded",
        len(ep.images) == min(n_images, 8),
        f"{len(ep.images)} thumbnails",
    ))

    return checks


def print_report(ep: SceneRolloutEpisode, checks: list[Check]) -> int:
    """Print a per-check PASS/FAIL table plus episode summary; return exit code."""
    print("\n=== Scene smoke test ===")
    print(f"scene_id={ep.scene_id}  num_images={ep.num_images}  done={ep.done}")
    print(f"reward={ep.reward:.4f}")
    rc = ep.reward_components or {}
    for k in REQUIRED_REWARD_KEYS:
        if k in rc:
            print(f"  {k:28s} {float(rc[k]):+.4f}")
    print("\nChecks:")
    n_fail = 0
    for c in checks:
        status = "PASS" if c.ok else "FAIL"
        if not c.ok:
            n_fail += 1
        print(f"  [{status}] {c.name:26s} {c.detail}")
    print(f"\n{n_fail} failed / {len(checks)} checks")
    return 1 if n_fail else 0


# ---------------------------------------------------------------------------
# Scene loading + main
# ---------------------------------------------------------------------------


def load_scene(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    """Load one small scene through SceneDataset; return (scene, image_root)."""
    from scripts.run_scene_grpo import SceneDataset

    if args.synthetic:
        tmp = Path(tempfile.mkdtemp(prefix="scene_smoke_"))
        write_synthetic_scene(tmp / "scene_info", tmp / "images",
                              scene_id=args.scene, n_images=args.synthetic_images)
        si_dir, img_root = tmp / "scene_info", tmp / "images"
        logger.info(f"Synthetic scene '{args.scene}' written under {tmp}")
    else:
        si_dir, img_root = Path(args.scene_info_dir), Path(args.image_root)

    load_k = args.pool_size if args.coherent else args.max_images
    ds = SceneDataset.from_megadepth(
        str(si_dir), str(img_root),
        scenes=[args.scene],
        max_images=load_k,
        min_images_per_scene=3,
    )
    if not ds.scenes:
        raise SystemExit(
            f"No scene '{args.scene}' found in {si_dir} "
            f"(image_root={img_root}). Use --synthetic for an offline run, or "
            f"pass --scene-info-dir/--image-root."
        )
    scene = ds.scenes[0]
    if args.coherent:
        scene = select_coherent_subset(scene, args.max_images)
        logger.info(
            f"Coherent subset: {scene['num_images']}/{load_k} images by "
            f"GT overlap (indices {scene.get('image_indices')})"
        )
    return scene, str(img_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="0015",
                        help="Scene id to load (default: 0015)")
    parser.add_argument("--max-images", type=int, default=10,
                        help="Images sampled per scene (default: 10)")
    parser.add_argument("--num-matches", type=int, default=5,
                        help="Top retrieved pairs to match (default: 5)")
    parser.add_argument("--matcher", default=DEFAULT_MATCHER)
    parser.add_argument("--scene-info-dir", default=DEFAULT_SCENE_INFO_DIR)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--synthetic", action="store_true",
                        help="Write a synthetic scene to a temp dir and load "
                             "it through SceneDataset (fully offline)")
    parser.add_argument("--synthetic-images", type=int, default=12,
                        help="Images in the synthetic scene (default: 12)")
    parser.add_argument("--coherent", action="store_true",
                        help="Pick the --max-images subset with the highest "
                             "mutual GT overlap (recommended for --real so "
                             "COLMAP can actually register the images)")
    parser.add_argument("--pool-size", type=int, default=60,
                        help="Candidate pool size for --coherent (default: 60)")
    parser.add_argument("--real", action="store_true",
                        help="Use a real tool server instead of MockToolClient")
    parser.add_argument("--tool-server-url", default="http://localhost:8765")
    parser.add_argument("--mock-retrieval", action="store_true",
                        help="Replace DINOv2 embeddings with deterministic "
                             "hash embeddings (no torch / network)")
    parser.add_argument("--flag-doppelganger", action="store_true",
                        help="Mock mode only: report the checked pair as a "
                             "doppelganger so it is filtered from sfm_run")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 1. Load a small scene through the real SceneDataset loader.
    scene, image_root = load_scene(args)
    print(f"Loaded scene {scene['scene_id']}: {scene['num_images']} images")

    # 2. Tool client: mock or real (instrumented either way for log checks).
    if args.real:
        client = InstrumentedToolClient(base_url=args.tool_server_url)
        try:
            print(f"Tool server health: {client.health()}")
        except Exception as e:
            raise SystemExit(
                f"Tool server not reachable at {args.tool_server_url}: {e}\n"
                f"Start one with scripts/start_tool_server.sh, or drop --real."
            )
        # Stage sampled images so COLMAP only sees this subset.
        stage_dir = Path(tempfile.mkdtemp(prefix="scene_smoke_real_"))
        image_paths = stage_images(scene["image_paths"], image_root, stage_dir)
        image_root_arg = ""
        logger.info(f"Staged {len(image_paths)} images under {stage_dir}")
    else:
        client = MockToolClient(
            doppelganger_confidence=0.9 if args.flag_doppelganger else 0.12,
        )
        image_paths = scene["image_paths"]
        image_root_arg = image_root

    if args.mock_retrieval:
        install_mock_retrieval()
        logger.info("Retrieval embeddings patched to deterministic hashes")

    # 3. Run the scene episode with the scripted agent.
    agent = ScriptedSceneAgent(num_matches=args.num_matches, matcher=args.matcher)
    ep = run_scene_episode(
        agent=agent,
        scene_id=scene["scene_id"],
        image_paths=image_paths,
        tool_client=client,
        gt_recon=scene.get("gt_recon"),
        overlap_matrix=scene.get("overlap_matrix"),
        image_indices=scene.get("image_indices"),
        max_tool_calls=20,
        max_turns=30,
        image_root=image_root_arg,
    )

    # 4. Verify everything.
    checks = verify_episode(
        ep, client, scene, expected_tools=agent.expected_tool_sequence())
    return print_report(ep, checks)


if __name__ == "__main__":
    sys.exit(main())
