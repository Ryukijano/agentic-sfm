#!/usr/bin/env python
"""Phase 2: Scene-level GRPO RL training for agentic SfM.

Trains the agent to orchestrate full scene reconstruction: retrieve pairs,
match them, filter doppelgangers, run COLMAP, and inspect the result.

Architecture:
  - vLLM server (GPU 0): fast rollout sampling
  - Training model (GPU 1): LoRA gradient updates
  - Tool server (GPU 2): matchers + COLMAP (pycolmap)

Scene episodes differ from pair episodes:
  - Multiple images per scene (not just a pair)
  - More tool types (retrieve, sfm_run, inspect)
  - Scene-level reward (registration + pose + doppelganger + split)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))  # for `import scripts.run_grpo`

from agentic_sfm.constants import DEFAULT_MATCHER, assert_qwen35_runtime
from agentic_sfm.rl.scene_episode import (
    SceneRolloutEpisode,
    run_scene_episode,
    run_scene_oracle_episode,
)
from agentic_sfm.tools.client import ToolClient
from scripts.run_grpo import GRPOTrainer, VLLMRolloutAgent

logger = logging.getLogger(__name__)

# Scene-level reward components surfaced in logs / wandb (see
# rewards.pose_rewards.compute_scene_reward).
SCENE_REWARD_KEYS = (
    "registration_reward",
    "pose_reward",
    "doppelganger_reward",
    "split_penalty",
    "tool_cost",
    "accumulative_tool_reward",
    "ntep_intent_reward",
    "ntep_redundancy_penalty",
)


def _normalize_scene_ids(scene_ids: list[str] | None) -> set[str] | None:
    """Normalize scene ids for matching (accepts '15' for '0015' etc.)."""
    if scene_ids is None:
        return None
    out: set[str] = set()
    for sid in scene_ids:
        s = str(sid)
        out.add(s)
        out.add(s.zfill(4))
        out.add(s.lstrip("0") or "0")
    return out


def _subsample_scene(scene: dict[str, Any], max_images: int) -> dict[str, Any]:
    """Return a copy of ``scene`` with at most ``max_images`` images.

    Positions are chosen deterministically (evenly spaced over the loaded
    image list) so curriculum stage transitions are reproducible.
    ``image_indices`` and ``gt_recon['poses']`` are remapped consistently.
    """
    n = scene["num_images"]
    k = min(n, max(1, int(max_images)))
    pos = sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))
    if len(pos) < k:  # linspace rounding collapsed some points — pad to k
        chosen = set(pos)
        for i in range(n):
            if i not in chosen:
                pos.append(i)
                chosen.add(i)
                if len(pos) == k:
                    break
        pos.sort()

    new = dict(scene)
    new["image_paths"] = [scene["image_paths"][i] for i in pos]
    new["num_images"] = len(pos)
    if scene.get("image_indices") is not None:
        new["image_indices"] = [scene["image_indices"][i] for i in pos]
    # Keep per-image auxiliary lists (ScanNet depth/pose paths, frame ids)
    # aligned with the subsampled image_paths.
    for key in ("depth_paths", "pose_paths", "frame_ids"):
        val = scene.get(key)
        if isinstance(val, (list, tuple)) and len(val) == n:
            new[key] = [val[i] for i in pos]
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


def _read_mat(path: Path) -> np.ndarray:
    """Read a whitespace-separated float matrix from a .txt file."""
    return np.loadtxt(str(path), dtype=np.float64)


def _normalize_scannet_ids(scene_ids: list[str] | None) -> set[str] | None:
    """Normalize ScanNet scene ids for matching.

    Accepts 'scene0772_00', '0772_00', '772_00' or '772' — all resolve to the
    canonical ``sceneNNNN_MM`` name components.
    """
    if scene_ids is None:
        return None
    out: set[str] = set()
    for sid in scene_ids:
        s = str(sid)
        out.add(s)
        if s.startswith("scene"):
            s = s[len("scene"):]
        out.add(s)
        head = s.split("_")[0]
        out.add(head)
        out.add(head.lstrip("0") or "0")
    return out


def _scannet_id_matches(scene_id: str, wanted: set[str]) -> bool:
    """True if ``sceneNNNN_MM`` matches any normalized wanted form."""
    stem = scene_id[len("scene"):] if scene_id.startswith("scene") else scene_id
    head = stem.split("_")[0]
    return (
        scene_id in wanted
        or stem in wanted
        or head in wanted
        or (head.lstrip("0") or "0") in wanted
    )


def _scannet_overlap_proxy(c2w_poses: np.ndarray) -> np.ndarray:
    """Pose-similarity proxy overlap matrix for ScanNet frames.

    ``c2w_poses`` is (N, 4, 4) camera-to-world.  ScanNet ships no
    covisibility counts, so overlap is approximated as the mean of:

      - view-direction agreement ``|dot(f_i, f_j)|`` where ``f`` is the
        camera +Z axis in world coords (3rd column of the c2w rotation), and
      - translation proximity ``exp(-||c_i - c_j|| / tau)`` with ``tau`` the
        median pairwise camera distance (scale-free).

    Returns an (N, N) matrix in [0, 1] with a unit diagonal.
    """
    n = len(c2w_poses)
    om = np.eye(n, dtype=np.float64)
    if n < 2:
        return om
    fwds = c2w_poses[:, :3, 2]   # camera +Z in world = forward view dir
    centers = c2w_poses[:, :3, 3]
    f_norm = fwds / (np.linalg.norm(fwds, axis=1, keepdims=True) + 1e-12)
    ang = np.abs(f_norm @ f_norm.T)                       # (N, N) in [0, 1]
    d = np.linalg.norm(centers[None, :, :] - centers[:, None, :], axis=-1)
    iu = np.triu_indices(n, 1)
    tau = float(np.median(d[iu])) if iu[0].size else 1.0
    tau = max(tau, 1e-6)
    prox = np.exp(-d / tau)
    np.fill_diagonal(prox, 1.0)
    om = 0.5 * np.clip(ang, 0.0, 1.0) + 0.5 * prox
    np.fill_diagonal(om, 1.0)
    return om


def _load_scannet_scene(scene_dir: Path,
                        max_images_per_scene: int = 20,
                        min_images_per_scene: int = 5) -> dict[str, Any] | None:
    """Build one scene dict from an extracted ``sceneNNNN_MM`` directory.

    Returns None when fewer than ``min_images_per_scene`` frames have all of
    color/depth/pose on disk.  See ``SceneDataset.from_scannet`` for the dict
    contract and pose-convention notes.
    """
    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"
    intrinsic_dir = scene_dir / "intrinsic"
    if not color_dir.is_dir() or not pose_dir.is_dir():
        return None

    # Full valid frame list: color jpg + pose txt (+ depth png when present).
    frames: list[tuple[int, Path, Path | None, Path]] = []
    jpgs = [p for p in color_dir.glob("*.jpg") if p.stem.isdigit()]
    for img in sorted(jpgs, key=lambda p: int(p.stem)):
        fid = int(img.stem)
        pose_f = pose_dir / f"{fid}.txt"
        if not pose_f.exists():
            continue
        depth_f = depth_dir / f"{fid}.png"
        frames.append((fid, img, depth_f if depth_f.exists() else None, pose_f))

    if len(frames) < min_images_per_scene:
        return None

    # Intrinsics (both are 4x4 in the ScanNet export; keep the top-left 3x3
    # for the per-image "intrinsics" slot used elsewhere).
    k_color = None
    k_depth = None
    try:
        k_color = _read_mat(intrinsic_dir / "intrinsic_color.txt")
    except Exception:
        pass
    try:
        k_depth = _read_mat(intrinsic_dir / "intrinsic_depth.txt")
    except Exception:
        pass

    # Full-list c2w poses -> overlap proxy + cam-from-world GT.
    all_c2w = np.stack([_read_mat(f[3]) for f in frames])       # (N,4,4)
    overlap = _scannet_overlap_proxy(all_c2w)

    # Deterministic even-spaced subsample (video frames are temporally
    # ordered, so even spacing gives scene coverage).
    n = len(frames)
    k = min(n, max(1, int(max_images_per_scene)))
    indices = sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))
    if len(indices) < min_images_per_scene:
        return None

    image_paths = [str(frames[i][1].resolve()) for i in indices]
    depth_paths = [
        str(frames[i][2].resolve()) if frames[i][2] is not None else None
        for i in indices
    ]
    pose_paths = [str(frames[i][3].resolve()) for i in indices]
    frame_ids = [frames[i][0] for i in indices]

    # GT poses keyed by str(position in image_paths), matching
    # scene_eval._extract_gt_poses.  ScanNet pose files are camera-to-world;
    # invert to the cam-from-world convention used by MegaDepth scene_info.
    gt_poses = {
        str(j): np.linalg.inv(all_c2w[i]).tolist()
        for j, i in enumerate(indices)
    }

    return {
        "scene_id": scene_dir.name,
        "dataset": "scannet",
        "image_paths": image_paths,
        "num_images": len(image_paths),
        "overlap_matrix": overlap,
        "image_indices": indices,
        "depth_paths": depth_paths,
        "pose_paths": pose_paths,
        "frame_ids": frame_ids,
        "intrinsics": (
            k_color[:3, :3].tolist() if k_color is not None else None
        ),
        "intrinsic_color": k_color.tolist() if k_color is not None else None,
        "intrinsic_depth": k_depth.tolist() if k_depth is not None else None,
        "scene_dir": str(scene_dir.resolve()),
        "gt_recon": {
            "num_images": len(image_paths),
            "poses": gt_poses,
        },
    }


@dataclass
class SceneDataset:
    """Dataset of scenes for Phase 2 training."""
    scenes: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_megadepth(cls, scene_info_dir: str, image_root: str,
                       scene_ids: list[str] | None = None,
                       max_images_per_scene: int = 20,
                       min_images_per_scene: int = 5,
                       scenes: list[str] | None = None,
                       max_images: int | None = None) -> "SceneDataset":
        """Build scene dataset from MegaDepth scene_info.

        Each scene has N images with known poses. We sample a subset of images
        per scene (for training efficiency) and store the GT poses for reward
        computation.

        ``scenes``/``max_images`` are curriculum-style aliases that take
        precedence over ``scene_ids``/``max_images_per_scene`` when given.
        """
        import numpy as np
        from pathlib import Path

        if scenes is not None:
            scene_ids = scenes
        if max_images is not None:
            max_images_per_scene = max_images
        wanted = _normalize_scene_ids(scene_ids)

        scenes_out = []
        si_dir = Path(scene_info_dir)
        img_root = Path(image_root)

        for npz_file in sorted(si_dir.glob("*.npz")):
            scene_id = npz_file.stem
            if "_" in scene_id:
                continue
            if wanted and scene_id not in wanted:
                continue

            try:
                d = np.load(str(npz_file), allow_pickle=True)
                image_paths = [str(p) for p in d["image_paths"]]
                poses = d["poses"]
                intrinsics = d["intrinsics"]

                # Filter to images that exist on disk
                existing = []
                for i, p in enumerate(image_paths):
                    p_str = p.decode() if isinstance(p, bytes) else str(p)
                    if (img_root / p_str).exists():
                        existing.append(i)

                if len(existing) < min_images_per_scene:
                    continue

                # Sample a diverse subset (uniform + coverage)
                n_sample = min(len(existing), max_images_per_scene)
                rng = np.random.default_rng(42)
                indices = rng.choice(existing, size=n_sample, replace=False)
                indices = sorted(indices.tolist())

                scene_images = [image_paths[i] for i in indices]
                # gt poses keyed by position in scene_images (0..n-1), which is
                # what scene_eval._extract_gt_poses resolves via the episode
                # image_paths lookup.
                gt_poses = {}
                for k, i in enumerate(indices):
                    p = poses[i]
                    gt_poses[str(k)] = p.tolist() if hasattr(p, "tolist") else p

                scenes_out.append({
                    "scene_id": scene_id,
                    "image_paths": scene_images,
                    "num_images": len(scene_images),
                    "overlap_matrix": d["overlap_matrix"],
                    "image_indices": indices,
                    "gt_recon": {
                        "num_images": len(scene_images),
                        "poses": gt_poses,
                    },
                })
            except Exception as e:
                logger.warning(f"Failed to load scene {scene_id}: {e}")
                continue

        return cls(scenes=scenes_out)

    @classmethod
    def from_scannet(cls, scannet_root: str,
                     scene_ids: list[str] | None = None,
                     max_images_per_scene: int = 20,
                     min_images_per_scene: int = 5) -> "SceneDataset":
        """Build scene dataset from extracted ScanNet test scenes.

        Expects ``scannet_root`` to contain ``sceneNNNN_MM/`` directories in
        the ``scannet_test_1500`` layout::

            sceneNNNN_MM/
                color/*.jpg            RGB frames (e.g. 105.jpg)
                depth/*.png            uint16 depth, millimetres, 480x640
                pose/*.txt             4x4 camera-to-world per frame
                intrinsic/intrinsic_color.txt   (+ intrinsic_depth.txt,
                                        extrinsic_color/depth.txt)

        The produced scene dicts match the MegaDepth contract (``image_paths``
        absolute, ``gt_recon.poses`` keyed by str(position-in-image_paths))
        plus ScanNet extras: ``depth_paths``, ``pose_paths``, ``frame_ids``,
        ``intrinsics`` (3x3 colour K), ``intrinsic_color``/``intrinsic_depth``
        (full 4x4), ``scene_dir`` and ``dataset="scannet"``.

        Pose-convention note: ScanNet ``pose/*.txt`` are **camera-to-world**;
        the scene_eval/reward path expects **cam-from-world**, so the stored
        ``gt_recon.poses`` are the *inverses* of the file matrices.

        ``overlap_matrix`` is a pose-similarity proxy (no true covisibility
        is shipped): for each frame pair it averages
        ``|dot(forward_i, forward_j)|`` (forward = camera +Z in world) with a
        translation-proximity term ``exp(-dist / median_pairwise_dist)``.
        It is built over the full valid frame list; ``image_indices`` indexes
        into it, matching the MegaDepth convention.
        """
        import numpy as np
        from pathlib import Path

        root = Path(scannet_root).expanduser()
        # Tolerate being pointed one level above the nested
        # ``scannet_test_1500/`` directory the tar creates.
        if not list(root.glob("scene*_*")) and (root / "scannet_test_1500").is_dir():
            root = root / "scannet_test_1500"

        wanted = _normalize_scannet_ids(scene_ids)
        scenes_out: list[dict[str, Any]] = []

        for scene_dir in sorted(root.glob("scene*_*")):
            if not scene_dir.is_dir():
                continue
            scene_id = scene_dir.name
            if wanted and not _scannet_id_matches(scene_id, wanted):
                continue

            try:
                scene = _load_scannet_scene(
                    scene_dir,
                    max_images_per_scene=max_images_per_scene,
                    min_images_per_scene=min_images_per_scene,
                )
            except Exception as e:
                logger.warning(f"Failed to load ScanNet scene {scene_id}: {e}")
                continue
            if scene is not None:
                scenes_out.append(scene)

        return cls(scenes=scenes_out)

    def filter(self, scenes: list[str] | None = None,
               max_images: int | None = None) -> "SceneDataset":
        """Return a filtered view: subset of scenes and/or per-scene images.

        Used by the scene curriculum at stage boundaries. ``scenes=None``
        keeps all scenes; ``max_images=None`` keeps each scene's full loaded
        image list. Subsampling is deterministic (evenly spaced positions).
        """
        wanted = _normalize_scene_ids(scenes)
        out: list[dict[str, Any]] = []
        for s in self.scenes:
            if wanted is not None and s["scene_id"] not in wanted:
                continue
            if max_images is not None and s["num_images"] > max_images:
                s = _subsample_scene(s, max_images)
            out.append(s)
        return SceneDataset(scenes=out)


@dataclass
class CurriculumStage:
    """One stage of the scene curriculum."""
    epochs: int
    max_images: int | None = None
    scenes: list[str] | None = None


class SceneCurriculum:
    """Epoch-indexed curriculum over scene difficulty.

    Stages are cumulative: stage ``i`` is active for epochs
    ``[sum(epochs[:i]), sum(epochs[:i+1]))``. Past the end of the last stage
    the final stage remains active, so ``total_epochs`` in the config may
    exceed the sum of stage epochs.
    """

    def __init__(self, stages: list[dict[str, Any]]):
        self.stages = [
            CurriculumStage(
                epochs=int(s.get("epochs", 0)),
                max_images=s.get("max_images"),
                scenes=s.get("scenes"),
            )
            for s in stages
        ]
        self.boundaries: list[int] = []  # boundaries[i] = first epoch of stage i
        acc = 0
        for st in self.stages:
            self.boundaries.append(acc)
            acc += max(st.epochs, 0)
        self.total_epochs = acc

    def stage_for_epoch(self, epoch: int) -> int:
        """Index of the stage active at ``epoch`` (last stage wins past the end)."""
        idx = 0
        for i, start in enumerate(self.boundaries):
            if epoch >= start:
                idx = i
        return idx

    def describe(self, idx: int) -> str:
        st = self.stages[idx]
        start = self.boundaries[idx]
        end = start + max(st.epochs, 0) - 1
        scenes = "all" if st.scenes is None else ",".join(str(s) for s in st.scenes)
        max_images = "all" if st.max_images is None else str(st.max_images)
        return (f"stage {idx + 1}/{len(self.stages)} (epochs {start}-{end}): "
                f"scenes={scenes}, max_images={max_images}")


class SceneGRPOTrainer(GRPOTrainer):
    """Scene-level GRPO trainer for Phase 2.

    Subclasses the pair-level ``GRPOTrainer`` and overrides everything that
    touches pair-specific fields:

      - ``_group_key`` groups episodes by ``scene_id`` (not ``pair_id``), which
        fixes both ``compute_advantages`` and DAPO zero-variance filtering.
      - ``collect_scene_rollouts`` runs ``run_scene_episode`` per scene ×
        ``group_size`` (plus S-GRPO CGI oracle injection), replacing the
        pair-level ``collect_rollouts``.  The inherited ``train_step`` calls
        ``self.collect_rollouts`` so the whole policy-gradient path
        (logprobs, clipped surrogate, grad accumulation) is reused.
      - ``evaluate`` runs one rollout per held-out (or sampled) scene.
      - ``save_lora_checkpoint`` additionally writes ``trainer_state.json``
        (epoch + global step) so ``--resume`` can restore both.
      - ``resume`` loads a saved LoRA adapter and returns the epoch to
        restart from.

    ``image_root``/``max_turns``/scene reward weights come from the config
    sections ``data``, ``rl`` and ``reward``.
    """

    def __init__(
        self,
        config: dict,
        tool_client: ToolClient,
        vllm_url: str = "http://localhost:8000",
        output_dir: str = "outputs/phase2",
        image_root: str | None = None,
        max_turns: int | None = None,
        scene_dataset: "SceneDataset | None" = None,
        val_scenes: list[dict[str, Any]] | None = None,
    ):
        super().__init__(
            config=config,
            tool_client=tool_client,
            vllm_url=vllm_url,
            output_dir=output_dir,
            load_pair_datasets=False,
        )
        data_cfg = config.get("data", {})
        self.image_root = (
            image_root if image_root is not None else data_cfg.get("image_root", "")
        )
        self.max_turns = (
            max_turns if max_turns is not None else config["rl"].get("max_turns", 30)
        )
        self.scene_dataset = scene_dataset
        self.val_scenes = list(val_scenes or [])
        # Epoch index currently being trained — written to trainer_state.json.
        self.current_epoch = 0
        # Episodes from the most recent collect_scene_rollouts call (pre
        # dynamic-sampling filter), used for scene-level metric logging.
        self._last_collected_episodes: list[SceneRolloutEpisode] = []
        # Scene-level reward weights passed to run_scene_episode through
        # agent.reward_config. Keys match compute_scene_reward's kwargs.
        self._base_reward_config: dict[str, Any] = dict(config.get("reward", {}) or {})

    # ------------------------------------------------------------------
    # Grouping / rollout collection
    # ------------------------------------------------------------------

    def _group_key(self, ep: SceneRolloutEpisode) -> str:
        return ep.scene_id

    def _scheduled_reward_config(self) -> dict[str, Any]:
        """Reward config for the current ``_global_step`` under the schedule.

        Mirrors the pair-level dynamic schedule: during warmup
        (``_global_step < reward_warmup_steps``) the pose term is scaled down
        so registration/format signal dominates while the policy is cold.
        """
        cfg = dict(self._base_reward_config)
        if self.reward_schedule == "dynamic" and self._global_step < self.reward_warmup_steps:
            cfg["pose_weight"] = cfg.get("pose_weight", 1.0) * (1.0 / 3.0)
        return cfg

    def collect_scene_rollouts(
        self,
        scenes: list[dict[str, Any]],
        rollout_agent: VLLMRolloutAgent,
    ) -> list[SceneRolloutEpisode]:
        """Collect ``group_size`` scene rollouts per scene, with S-GRPO CGI.

        When every rollout in a scene's group fails, inject the oracle
        trajectory (GT-overlap pairs -> match -> sfm_run) so the group still
        yields a positive learning signal.
        """
        rollout_agent.reward_config = self._scheduled_reward_config()
        episodes: list[SceneRolloutEpisode] = []
        cgi_injections = 0
        for scene in scenes:
            group_episodes: list[SceneRolloutEpisode] = []
            all_failed = True
            for _ in range(self.group_size):
                ep = run_scene_episode(
                    agent=rollout_agent,
                    scene_id=scene["scene_id"],
                    image_paths=scene["image_paths"],
                    tool_client=self.tool_client,
                    gt_recon=scene.get("gt_recon"),
                    overlap_matrix=scene.get("overlap_matrix"),
                    image_indices=scene.get("image_indices"),
                    max_tool_calls=self.max_tool_calls,
                    max_turns=self.max_turns,
                    image_root=self.image_root,
                )
                group_episodes.append(ep)
                if ep.reward > 0:
                    all_failed = False

            # S-GRPO CGI: inject oracle trajectory when all rollouts fail
            if self.sgrpo_cgi and all_failed and group_episodes:
                failed_max = max(e.reward for e in group_episodes)
                oracle_ep = run_scene_oracle_episode(
                    agent=rollout_agent,
                    scene_id=scene["scene_id"],
                    image_paths=scene["image_paths"],
                    tool_client=self.tool_client,
                    gt_recon=scene.get("gt_recon"),
                    overlap_matrix=scene.get("overlap_matrix"),
                    image_indices=scene.get("image_indices"),
                    failed_group_max_reward=failed_max,
                    image_root=self.image_root,
                )
                if oracle_ep.reward > failed_max:
                    if len(group_episodes) == 1:
                        # Keep the group >1 member so the zero-variance
                        # filter doesn't drop it entirely.
                        group_episodes.append(oracle_ep)
                    else:
                        worst_idx = min(
                            range(len(group_episodes)),
                            key=lambda i: group_episodes[i].reward,
                        )
                        group_episodes[worst_idx] = oracle_ep
                    cgi_injections += 1
                    logger.info(
                        f"  S-GRPO CGI: injected oracle for {scene['scene_id']} "
                        f"(reward={oracle_ep.reward:.3f})"
                    )

            rewards = [e.reward for e in group_episodes]
            logger.info(
                f"  {scene['scene_id']}: rewards={[f'{r:.3f}' for r in rewards]}, "
                f"mean={np.mean(rewards):.3f}"
            )
            episodes.extend(group_episodes)

        if cgi_injections > 0:
            logger.info(
                f"  S-GRPO CGI: injected {cgi_injections} oracle trajectories "
                f"({cgi_injections}/{len(scenes)} scenes)"
            )
        self._last_collected_episodes = episodes
        return episodes

    def collect_rollouts(self, scenes: list, rollout_agent: VLLMRolloutAgent) -> list:
        """Override so the inherited ``train_step`` collects scene rollouts."""
        return self.collect_scene_rollouts(scenes, rollout_agent)

    # ------------------------------------------------------------------
    # Training step / metrics
    # ------------------------------------------------------------------

    def train_step(self, scenes: list, rollout_agent: VLLMRolloutAgent,
                   accum_step: int = 0, is_last_accum: bool = True) -> dict:
        """One scene-level GRPO step (inherited update + scene metrics)."""
        stats = super().train_step(
            scenes, rollout_agent, accum_step=accum_step, is_last_accum=is_last_accum
        )
        stats.update(self._scene_metrics(self._last_collected_episodes))
        return stats

    def _scene_metrics(self, episodes: list[SceneRolloutEpisode]) -> dict[str, float]:
        """Mean scene-level reward components + reconstruction stats."""
        if not episodes:
            return {}
        metrics: dict[str, float] = {}
        for key in SCENE_REWARD_KEYS:
            vals = [
                float(ep.reward_components[key])
                for ep in episodes
                if isinstance(ep.reward_components.get(key), (int, float))
            ]
            metrics[f"scene/{key}"] = float(np.mean(vals)) if vals else 0.0
        registered = [
            float((ep.recon_result or {}).get("num_registered", 0) or 0)
            for ep in episodes
        ]
        metrics["scene/mean_registered"] = float(np.mean(registered))
        metrics["scene/mean_registered_frac"] = float(
            np.mean([r / max(ep.num_images or len(ep.image_paths), 1)
                     for r, ep in zip(registered, episodes)])
        )
        pose_errs = [
            float(ep.recon_result["mean_pose_error_deg"])
            for ep in episodes
            if isinstance((ep.recon_result or {}).get("mean_pose_error_deg"), (int, float))
        ]
        metrics["scene/mean_pose_error_deg"] = (
            float(np.mean(pose_errs)) if pose_errs else 0.0
        )
        metrics["scene/done_rate"] = float(
            np.mean([1.0 if ep.done else 0.0 for ep in episodes])
        )
        metrics["scene/mean_num_images"] = float(
            np.mean([ep.num_images or len(ep.image_paths) for ep in episodes])
        )
        # Doppelganger bookkeeping: checks issued, pairs the sfm_run saw as
        # flagged, and pairs actually filtered out of the COLMAP pair list.
        metrics["scene/mean_doppelganger_checks"] = float(
            np.mean([len(ep.doppelganger_checks) for ep in episodes])
        )
        metrics["scene/mean_doppelgangers_present"] = float(np.mean([
            float((ep.recon_result or {}).get("num_doppelgangers_present", 0) or 0)
            for ep in episodes
        ]))
        metrics["scene/mean_doppelgangers_filtered"] = float(np.mean([
            float((ep.recon_result or {}).get("num_doppelgangers_filtered", 0) or 0)
            for ep in episodes
        ]))
        return metrics

    # ------------------------------------------------------------------
    # Qualitative wandb media (3D recon + match viz)
    # ------------------------------------------------------------------

    def _log_scene_media(self, episodes: list, step: int, tag: str = "eval") -> None:
        """Render each scene's COLMAP reconstruction + a match figure to wandb.

        - The sparse reconstruction is logged as an interactive ``wandb.Object3D``
          point cloud (x,y,z + RGB) AND a static rendered figure.
        - A best-match correspondence figure is logged when the episode carried
          keypoints.
        """
        if not self._wandb:
            return
        try:
            import wandb
            from agentic_sfm.eval.viz import (
                episode_qualitative_media, render_point_cloud,
            )
            import pycolmap
        except Exception as e:
            logger.warning(f"scene media unavailable: {e}")
            return

        for ep in episodes:
            sid = getattr(ep, "scene_id", "scene")
            out_dir = (getattr(ep, "recon_result", None) or {}).get("output_dir")
            if out_dir:
                try:
                    recon = pycolmap.Reconstruction(str(out_dir))
                    # interactive 3D point cloud
                    pts = np.asarray([p.xyz for p in recon.points3D.values()])
                    cols = np.asarray([p.color for p in recon.points3D.values()])
                    if len(pts):
                        pc = np.concatenate([pts, cols], axis=1)  # (N,6)
                        self._wandb.log(
                            {f"{tag}/{sid}_pointcloud": wandb.Object3D(pc)},
                            step=step)
                    # static render
                    fig = render_point_cloud(
                        reconstruction=recon, title=f"{sid} — {len(recon.images)} imgs")
                    self._wandb.log(
                        {f"{tag}/{sid}_recon": wandb.Image(fig)}, step=step)
                except Exception as e:
                    logger.warning(f"recon render failed for {sid}: {e}")
            # match viz for the episode's best match
            try:
                media = episode_qualitative_media(ep, prefix=f"{tag}/{sid}")
                if media:
                    self._wandb.log(
                        {k: wandb.Image(v) for k, v in media.items()}, step=step)
            except Exception as e:
                logger.warning(f"match viz failed for {sid}: {e}")

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def _eval_scenes(self) -> list[dict[str, Any]]:
        """Scenes to evaluate on: held-out val scenes, else a train subset."""
        if self.val_scenes:
            return list(self.val_scenes)
        if self.scene_dataset is not None:
            return list(self.scene_dataset.scenes)
        return []

    def evaluate(self, rollout_agent: VLLMRolloutAgent,
                 max_scenes: int | None = None,
                 scenes: list[dict[str, Any]] | None = None) -> dict:
        eval_scenes = scenes if scenes is not None else self._eval_scenes()
        if not eval_scenes:
            return {"mean_reward": 0.0, "num_scenes": 0}
        if max_scenes is None:
            max_scenes = self.config["training"].get("eval_num_scenes", 4)
        eval_scenes = eval_scenes[:max_scenes]

        # Eval uses the unscheduled reward config.
        rollout_agent.reward_config = dict(self._base_reward_config)
        episodes = []
        for scene in eval_scenes:
            ep = run_scene_episode(
                agent=rollout_agent,
                scene_id=scene["scene_id"],
                image_paths=scene["image_paths"],
                tool_client=self.tool_client,
                gt_recon=scene.get("gt_recon"),
                overlap_matrix=scene.get("overlap_matrix"),
                image_indices=scene.get("image_indices"),
                max_tool_calls=self.max_tool_calls,
                max_turns=self.max_turns,
                image_root=self.image_root,
            )
            episodes.append(ep)

        rewards = [ep.reward for ep in episodes]
        tool_calls = [len(ep.tool_calls) for ep in episodes]
        registered = [
            float((ep.recon_result or {}).get("num_registered", 0) or 0)
            for ep in episodes
        ]
        stats = {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "num_scenes": len(episodes),
            "mean_tool_calls": float(np.mean(tool_calls)) if tool_calls else 0.0,
            "success_rate": float(np.mean([1.0 if r > 0 else 0.0 for r in registered])),
            "mean_registered": float(np.mean(registered)) if registered else 0.0,
        }
        stats.update(self._scene_metrics(episodes))
        if self._wandb:
            self._wandb.log({
                f"eval/{k}": v for k, v in stats.items()
                if isinstance(v, (int, float))
            }, step=self._global_step)
            self._log_scene_media(episodes, step=self._global_step, tag="eval")
        return stats

    # ------------------------------------------------------------------
    # Checkpointing / resume
    # ------------------------------------------------------------------

    def save_lora_checkpoint(self, epoch: int | str,
                             rollout_agent: VLLMRolloutAgent | None = None):
        """Save LoRA + trainer_state.json, then hot-reload the vLLM adapter."""
        super().save_lora_checkpoint(epoch, rollout_agent)
        if self._model is None:
            return
        tag = epoch if isinstance(epoch, str) else f"epoch_{epoch}"
        state = {
            "tag": tag,
            # Next epoch index to resume from (epochs are 0-based; the
            # checkpoint at tag epoch_N is written after epoch N-1 finishes).
            "epoch": self.current_epoch + 1,
            "global_step": self._global_step,
        }
        try:
            (self.ckpt_dir / tag / "trainer_state.json").write_text(
                json.dumps(state, indent=2)
            )
        except Exception as e:
            logger.warning(f"Failed to write trainer_state.json: {e}")

    def resume(self, ckpt_path: str | Path) -> int:
        """Resume LoRA weights from a checkpoint dir. Returns the start epoch.

        Sets ``self.sft_adapter`` so ``_load_training_model`` initialises the
        PEFT model from the saved adapter, and restores ``_global_step`` /
        ``current_epoch`` from ``trainer_state.json`` (falling back to an
        ``epoch_N`` suffix in the directory name).
        """
        p = Path(ckpt_path)
        if not p.is_dir():
            raise FileNotFoundError(f"--resume checkpoint not found: {p}")
        self.sft_adapter = str(p)

        start_epoch = 0
        state_path = p / "trainer_state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
                self._global_step = int(state.get("global_step", 0))
                start_epoch = int(state.get("epoch", 0) or 0)
            except Exception as e:
                logger.warning(f"Could not parse {state_path}: {e}")
        if start_epoch == 0:
            m = re.search(r"epoch_(\d+)", p.name)
            if m:
                start_epoch = int(m.group(1))
        self.current_epoch = start_epoch
        logger.info(
            f"Resuming from {p} (start_epoch={start_epoch}, "
            f"global_step={self._global_step})"
        )
        return start_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--tool-server-url", default="http://localhost:8765")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--output-dir", default="outputs/phase2")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    assert_qwen35_runtime()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(f"Config: {args.config}")
    logger.info(f"vLLM: {args.vllm_url}")
    logger.info(f"Tool server: {args.tool_server_url}")

    # Parse curriculum (if any) before loading so we can size the load once
    data_cfg = config.get("data", {})
    curriculum_cfg = config.get("curriculum", {}) or {}
    curriculum: SceneCurriculum | None = None
    load_max_images = data_cfg.get("max_images_per_scene", 20)
    load_scene_ids = data_cfg.get("scene_ids")

    if curriculum_cfg.get("enabled") and curriculum_cfg.get("stages"):
        curriculum = SceneCurriculum(curriculum_cfg["stages"])
        # Load once at the largest image budget any stage needs.
        stage_maxes = [s.max_images for s in curriculum.stages if s.max_images]
        if stage_maxes:
            load_max_images = max(load_max_images, max(stage_maxes))
        # If every stage names its scenes, only load the union of them.
        named = [s.scenes for s in curriculum.stages if s.scenes]
        if named and len(named) == len(curriculum.stages):
            union = {str(sid) for sc in named for sid in sc}
            if load_scene_ids is not None:
                keep = _normalize_scene_ids(sorted(union)) or set()
                load_scene_ids = [s for s in load_scene_ids if str(s) in keep]
            else:
                load_scene_ids = sorted(union)
        logger.info(f"Curriculum enabled: {len(curriculum.stages)} stages, "
                    f"{curriculum.total_epochs} stage-epochs total")
        for i in range(len(curriculum.stages)):
            logger.info(f"  {curriculum.describe(i)}")

    # Build scene dataset.  MegaDepth (outdoor landmarks) is always loaded
    # from scene_info npz files; when ``data.scannet_root`` is set, the
    # extracted ScanNet test scenes (indoor RGB-D) are concatenated so both
    # domains train together.
    scene_dataset = SceneDataset.from_megadepth(
        scene_info_dir=data_cfg.get("scene_info_dir", "data/megadepth/scene_info_full/scene_info"),
        image_root=data_cfg.get("image_root", "data/megadepth/megadepth_test_1500"),
        scene_ids=load_scene_ids,
        max_images_per_scene=load_max_images,
        min_images_per_scene=data_cfg.get("min_images_per_scene", 5),
    )
    scannet_root = data_cfg.get("scannet_root")
    if scannet_root:
        # ``data.scannet_scene_ids`` scopes which sceneNNNN_MM dirs load.
        # When unset it falls back to ``scene_ids`` so single-scene smoke
        # configs stay small across both sources; a literal "all"/"*" value
        # also loads every extracted scene.
        scannet_ids = data_cfg.get("scannet_scene_ids")
        if scannet_ids is None:
            scannet_ids = load_scene_ids
        if isinstance(scannet_ids, str) and scannet_ids.lower() in ("all", "*"):
            scannet_ids = None
        scannet_dataset = SceneDataset.from_scannet(
            scannet_root=scannet_root,
            scene_ids=scannet_ids,
            max_images_per_scene=data_cfg.get(
                "scannet_max_images_per_scene", load_max_images),
            min_images_per_scene=data_cfg.get("min_images_per_scene", 5),
        )
        logger.info(f"ScanNet dataset: {len(scannet_dataset.scenes)} scenes")
        scene_dataset = SceneDataset(
            scenes=scene_dataset.scenes + scannet_dataset.scenes
        )
    logger.info(f"Scene dataset: {len(scene_dataset.scenes)} scenes")
    for s in scene_dataset.scenes:
        logger.info(f"  {s['scene_id']}: {s['num_images']} images")

    # Held-out val scenes (data.val_scene_ids) are excluded from training and
    # used by the periodic eval.  When unset, eval falls back to a small
    # deterministic subset of the training scenes.
    val_ids = _normalize_scene_ids(data_cfg.get("val_scene_ids"))
    val_scenes: list[dict[str, Any]] = []
    train_dataset = scene_dataset
    if val_ids:
        val_scenes = [s for s in scene_dataset.scenes if s["scene_id"] in val_ids]
        train_dataset = SceneDataset(scenes=[
            s for s in scene_dataset.scenes if s["scene_id"] not in val_ids
        ])
        logger.info(
            f"Held-out val scenes: {[s['scene_id'] for s in val_scenes]} "
            f"({len(train_dataset.scenes)} train scenes remain)"
        )

    # Initialize tool client
    tool_client = ToolClient(base_url=args.tool_server_url)
    try:
        health = tool_client.health()
        logger.info(f"Tool server health: {health}")
    except Exception as e:
        logger.error(f"Tool server not reachable: {e}")
        sys.exit(1)

    # Initialize rollout agent (scene reward weights flow via reward_config)
    rollout_agent = VLLMRolloutAgent(
        model_name=config["model"]["name"],
        vllm_url=args.vllm_url,
        temperature=config["rl"].get("temperature", 1.0),
        top_p=config["rl"].get("top_p", 0.95),
        max_new_tokens=config["model"].get("max_new_tokens", 512),
        max_tool_calls=config["rl"].get("max_tool_calls", 20),
        matcher=data_cfg.get("matcher", DEFAULT_MATCHER),
        sft_adapter=config["model"].get("sft_adapter"),
        reward_config=config.get("reward", {}),
    )

    # Scene-level GRPO trainer
    trainer = SceneGRPOTrainer(
        config=config,
        tool_client=tool_client,
        vllm_url=args.vllm_url,
        output_dir=args.output_dir,
        image_root=data_cfg.get("image_root", ""),
        max_turns=config["rl"].get("max_turns", 30),
        scene_dataset=train_dataset,
        val_scenes=val_scenes,
    )

    # --resume: restore LoRA weights + trainer state, start at saved epoch.
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.resume(args.resume)
    else:
        # No explicit resume: warm-start the training LoRA from the Phase 1
        # checkpoint, falling back to the SFT adapter when the Phase 1
        # checkpoint isn't a real PEFT adapter dir.
        for warm in (config["model"].get("checkpoint"),
                     config["model"].get("sft_adapter")):
            if warm and os.path.isdir(str(warm)) and \
                    (Path(warm) / "adapter_config.json").exists():
                trainer.sft_adapter = str(warm)
                logger.info(f"Warm-starting training LoRA from {warm}")
                break

    # Hot-load the initial adapter into vLLM so rollouts are on-policy.
    init_adapter = args.resume or trainer.sft_adapter
    if init_adapter and os.path.isdir(str(init_adapter)):
        trainer._reload_vllm_lora(Path(init_adapter), rollout_agent)

    # Phase 2 training loop: scenes -> micro-batches with gradient accumulation.
    total_epochs = config["training"].get("total_epochs", 50)
    batch_size = config.get("rollout", {}).get("batch_size", 2)
    save_freq = config["training"].get("save_freq", 5)
    eval_freq = config["training"].get("eval_freq", 5)
    log_freq = config["training"].get("log_freq", 10)
    grad_accum = trainer.grad_accum

    active_dataset = train_dataset
    current_stage = -1
    global_step = trainer._global_step

    for epoch in range(start_epoch, total_epochs):
        trainer.current_epoch = epoch

        # Curriculum: re-filter the dataset at stage boundaries.
        if curriculum is not None:
            stage_idx = curriculum.stage_for_epoch(epoch)
            if stage_idx != current_stage:
                current_stage = stage_idx
                stage = curriculum.stages[stage_idx]
                active_dataset = train_dataset.filter(
                    scenes=stage.scenes, max_images=stage.max_images)
                logger.info(f"=== Curriculum {curriculum.describe(stage_idx)} ===")
                if active_dataset.scenes:
                    for s in active_dataset.scenes:
                        logger.info(f"    {s['scene_id']}: {s['num_images']} images")
                else:
                    logger.warning(
                        "Curriculum stage matched 0 scenes — "
                        "falling back to the full dataset for this stage")
                    active_dataset = train_dataset

        stage_tag = (f" [curriculum stage {current_stage + 1}/"
                     f"{len(curriculum.stages)}]" if curriculum is not None else "")
        logger.info(f"=== Epoch {epoch}{stage_tag} ===")

        scenes = list(active_dataset.scenes)
        if not scenes:
            logger.warning("No scenes in the active dataset — skipping epoch.")
            continue
        rng = np.random.default_rng(epoch)
        rng.shuffle(scenes)

        epoch_stats = []
        micro_step = 0

        for batch_start in range(0, len(scenes), batch_size):
            batch = scenes[batch_start:batch_start + batch_size]
            micro_step += 1
            is_last_accum = (
                (micro_step % grad_accum == 0)
                or (batch_start + batch_size >= len(scenes))
            )

            stats = trainer.train_step(
                batch, rollout_agent,
                accum_step=micro_step, is_last_accum=is_last_accum,
            )
            epoch_stats.append(stats)

            if is_last_accum:
                global_step += 1
                trainer._global_step = global_step
                rollout_agent._global_step = global_step

                if global_step % log_freq == 0:
                    logger.info(
                        f"  Step {global_step}: "
                        f"reward={stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}, "
                        f"tool_calls={stats['mean_tool_calls']:.1f}, "
                        f"registered={stats.get('scene/mean_registered', 0.0):.1f}, "
                        f"loss={stats.get('loss', 0.0):.4f}"
                    )
                    if trainer._wandb:
                        payload = {
                            f"train/{k}": v for k, v in stats.items()
                            if isinstance(v, (int, float))
                        }
                        payload["train/epoch"] = epoch + 1
                        payload["train/global_step"] = global_step
                        trainer._wandb.log(payload)

        if epoch_stats:
            mean_reward = np.mean([s.get("mean_reward", 0.0) for s in epoch_stats])
            logger.info(f"Epoch {epoch} mean reward: {mean_reward:.3f}")

        # Eval
        if (epoch + 1) % eval_freq == 0:
            logger.info(f"  Running eval at epoch {epoch}...")
            eval_stats = trainer.evaluate(rollout_agent)
            logger.info(
                f"  Eval: reward={eval_stats['mean_reward']:.3f}, "
                f"success_rate={eval_stats.get('success_rate', 0.0):.3f}, "
                f"registered={eval_stats.get('mean_registered', 0.0):.1f}, "
                f"pose_err={eval_stats.get('scene/mean_pose_error_deg', 0.0):.2f}°"
            )

        # Always dump `latest` so vLLM rollouts track the LoRA policy;
        # numbered checkpoints at save_freq.
        trainer.save_lora_checkpoint("latest", rollout_agent)
        if (epoch + 1) % save_freq == 0:
            trainer.save_lora_checkpoint(epoch + 1, rollout_agent)
            logger.info(f"  Saved checkpoint at epoch {epoch + 1}")

    trainer.save_lora_checkpoint(total_epochs, rollout_agent)
    if trainer._wandb:
        trainer._wandb.finish()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
