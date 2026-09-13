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


@dataclass
class SceneDataset:
    """Dataset of scenes for Phase 2 training."""
    scenes: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_megadepth(cls, scene_info_dir: str, image_root: str,
                       scene_ids: list[str] | None = None,
                       max_images_per_scene: int = 20,
                       min_images_per_scene: int = 5) -> "SceneDataset":
        """Build scene dataset from MegaDepth scene_info.

        Each scene has N images with known poses. We sample a subset of images
        per scene (for training efficiency) and store the GT poses for reward
        computation.
        """
        import numpy as np
        from pathlib import Path

        scenes = []
        si_dir = Path(scene_info_dir)
        img_root = Path(image_root)

        for npz_file in sorted(si_dir.glob("*.npz")):
            scene_id = npz_file.stem
            if "_" in scene_id:
                continue
            if scene_ids and scene_id not in scene_ids:
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
                scene_poses = {i: poses[i] for i in indices}
                scene_Ks = {i: intrinsics[i] for i in indices}

                scenes.append({
                    "scene_id": scene_id,
                    "image_paths": scene_images,
                    "num_images": len(scene_images),
                    "gt_recon": {
                        "num_images": len(scene_images),
                        "poses": {str(i): scene_poses[i].tolist() if hasattr(scene_poses[i], "tolist") else scene_poses[i] for i in range(len(scene_images))},
                    },
                })
            except Exception as e:
                logger.warning(f"Failed to load scene {scene_id}: {e}")
                continue

        return cls(scenes=scenes)


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

    # Build scene dataset
    data_cfg = config.get("data", {})
    scene_dataset = SceneDataset.from_megadepth(
        scene_info_dir=data_cfg.get("scene_info_dir", "data/megadepth/scene_info_full/scene_info"),
        image_root=data_cfg.get("image_root", "data/megadepth/megadepth_test_1500"),
        scene_ids=data_cfg.get("scene_ids"),
        max_images_per_scene=data_cfg.get("max_images_per_scene", 20),
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

    for epoch in range(total_epochs):
        logger.info(f"=== Epoch {epoch} ===")
        epoch_rewards = []
        epoch_episodes = []

        for scene in scene_dataset.scenes:
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
                import numpy as np
                from pathlib import Path

                # Load overlap matrix for oracle pair selection
                om = None
                indices = None
                si_path = Path(data_cfg.get("scene_info_dir", "")) / f"{scene['scene_id']}.npz"
                if si_path.exists():
                    d = np.load(str(si_path), allow_pickle=True)
                    om = d["overlap_matrix"]
                    # Map image_paths to indices in the full scene
                    all_paths = [str(p) for p in d["image_paths"]]
                    indices = [all_paths.index(p) for p in scene["image_paths"] if p in all_paths]

                failed_max = max(e.reward for e in scene_episodes)
                oracle_ep = run_scene_oracle_episode(
                    agent=rollout_agent,
                    scene_id=scene["scene_id"],
                    image_paths=scene["image_paths"],
                    tool_client=tool_client,
                    gt_recon=scene.get("gt_recon"),
                    overlap_matrix=om,
                    image_indices=indices,
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
