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
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.agent.policy import parse_tool_call, format_observation
from agentic_sfm.constants import DEFAULT_MATCHER, DEFAULT_POLICY_MODEL
from agentic_sfm.rl.scene_episode import SceneRolloutEpisode, run_scene_episode
from agentic_sfm.tools.client import ToolClient

logger = logging.getLogger(__name__)


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

    # Build scene dataset
    scene_dataset = SceneDataset.from_megadepth(
        scene_info_dir=data_cfg.get("scene_info_dir", "data/megadepth/scene_info_full/scene_info"),
        image_root=data_cfg.get("image_root", "data/megadepth/megadepth_test_1500"),
        scene_ids=load_scene_ids,
        max_images_per_scene=load_max_images,
        min_images_per_scene=data_cfg.get("min_images_per_scene", 5),
    )
    logger.info(f"Scene dataset: {len(scene_dataset.scenes)} scenes")
    for s in scene_dataset.scenes:
        logger.info(f"  {s['scene_id']}: {s['num_images']} images")

    # Initialize tool client
    tool_client = ToolClient(base_url=args.tool_server_url)
    try:
        health = tool_client.health()
        logger.info(f"Tool server health: {health}")
    except Exception as e:
        logger.error(f"Tool server not reachable: {e}")
        sys.exit(1)

    # Initialize rollout agent
    from scripts.run_grpo import VLLMRolloutAgent, GRPOTrainer
    rollout_agent = VLLMRolloutAgent(
        model_name=config["model"]["name"],
        vllm_url=args.vllm_url,
        temperature=config["rl"].get("temperature", 1.0),
        top_p=config["rl"].get("top_p", 0.95),
        max_new_tokens=config["model"].get("max_new_tokens", 512),
        max_tool_calls=config["rl"].get("max_tool_calls", 20),
        matcher=data_cfg.get("matcher", DEFAULT_MATCHER),
        sft_adapter=config["model"].get("sft_adapter"),
    )

    # Initialize GRPO trainer
    trainer = GRPOTrainer(
        config=config,
        tool_client=tool_client,
        vllm_url=args.vllm_url,
        output_dir=args.output_dir,
    )

    # Phase 2 training loop
    total_epochs = config["training"].get("total_epochs", 50)
    group_size = config["rl"].get("group_size", 6)
    save_freq = config["training"].get("save_freq", 5)
    eval_freq = config["training"].get("eval_freq", 5)

    active_dataset = scene_dataset
    current_stage = -1

    for epoch in range(total_epochs):
        # Curriculum: re-filter the dataset at stage boundaries.
        if curriculum is not None:
            stage_idx = curriculum.stage_for_epoch(epoch)
            if stage_idx != current_stage:
                current_stage = stage_idx
                stage = curriculum.stages[stage_idx]
                active_dataset = scene_dataset.filter(
                    scenes=stage.scenes, max_images=stage.max_images)
                logger.info(f"=== Curriculum {curriculum.describe(stage_idx)} ===")
                if active_dataset.scenes:
                    for s in active_dataset.scenes:
                        logger.info(f"    {s['scene_id']}: {s['num_images']} images")
                else:
                    logger.warning(
                        "Curriculum stage matched 0 scenes — "
                        "falling back to the full dataset for this stage")
                    active_dataset = scene_dataset

        stage_tag = (f" [curriculum stage {current_stage + 1}/"
                     f"{len(curriculum.stages)}]" if curriculum is not None else "")
        logger.info(f"=== Epoch {epoch}{stage_tag} ===")
        epoch_rewards = []
        epoch_episodes = []

        for scene in active_dataset.scenes:
            # Sample group_size rollouts per scene
            scene_episodes = []
            all_failed = True
            for g in range(group_size):
                ep = run_scene_episode(
                    agent=rollout_agent,
                    scene_id=scene["scene_id"],
                    image_paths=scene["image_paths"],
                    tool_client=tool_client,
                    gt_recon=scene.get("gt_recon"),
                    overlap_matrix=scene.get("overlap_matrix"),
                    image_indices=scene.get("image_indices"),
                    max_tool_calls=config["rl"].get("max_tool_calls", 20),
                    max_turns=config["rl"].get("max_turns", 30),
                    image_root=data_cfg.get("image_root", ""),
                )
                scene_episodes.append(ep)
                if ep.reward > 0:
                    all_failed = False

            # S-GRPO CGI: inject oracle trajectory when all rollouts fail
            if config["rl"].get("sgrpo_cgi", True) and all_failed and scene_episodes:
                from agentic_sfm.rl.scene_episode import run_scene_oracle_episode

                failed_max = max(e.reward for e in scene_episodes)
                oracle_ep = run_scene_oracle_episode(
                    agent=rollout_agent,
                    scene_id=scene["scene_id"],
                    image_paths=scene["image_paths"],
                    tool_client=tool_client,
                    gt_recon=scene.get("gt_recon"),
                    overlap_matrix=scene.get("overlap_matrix"),
                    image_indices=scene.get("image_indices"),
                    failed_group_max_reward=failed_max,
                    image_root=data_cfg.get("image_root", ""),
                )
                if oracle_ep.reward > failed_max:
                    if len(scene_episodes) == 1:
                        scene_episodes.append(oracle_ep)
                    else:
                        worst_idx = min(range(len(scene_episodes)),
                                        key=lambda i: scene_episodes[i].reward)
                        scene_episodes[worst_idx] = oracle_ep
                    logger.info(f"  S-GRPO CGI: injected oracle for {scene['scene_id']} (reward={oracle_ep.reward:.3f})")

            epoch_episodes.extend(scene_episodes)
            rewards = [e.reward for e in scene_episodes]
            logger.info(
                f"  {scene['scene_id']}: rewards={[f'{r:.3f}' for r in rewards]}, "
                f"mean={np.mean(rewards):.3f}"
            )
            epoch_rewards.extend(rewards)

        # Compute advantages and update
        if epoch_episodes:
            metrics = trainer.train_step(epoch_episodes, rollout_agent)
            logger.info(f"  Epoch {epoch} metrics: {metrics}")
            logger.info(f"  Mean reward: {np.mean(epoch_rewards):.4f}")

        # Save checkpoint
        if (epoch + 1) % save_freq == 0:
            trainer.save_checkpoint(epoch)
            logger.info(f"  Saved checkpoint at epoch {epoch}")

        # Eval
        if (epoch + 1) % eval_freq == 0:
            logger.info(f"  Running eval at epoch {epoch}...")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
