#!/usr/bin/env python
"""Phase 1a: Collect zero-shot trajectories from stock Qwen3.5-2B.

Runs the stock model (no LoRA) on all training pairs with temperature=1.0
and group_size=N to collect diverse trajectories for SFT warmup data.

Usage:
  python scripts/collect_trajectories.py \
      --config configs/phase1_grpo.yaml \
      --tool-server-url http://localhost:8765 \
      --vllm-url http://localhost:8000 \
      --group-size 8 \
      --output outputs/phase0/trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.data.hard_pairs import HardPairDataset
from agentic_sfm.tools.client import ToolClient

# Reuse VLLMRolloutAgent and RolloutEpisode from run_grpo
from scripts.run_grpo import VLLMRolloutAgent, RolloutEpisode, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def episode_to_dict(ep: RolloutEpisode) -> dict:
    """Serialize a RolloutEpisode to a JSON-compatible dict."""
    return {
        "pair_id": ep.pair_id,
        "image_a": ep.image_a,
        "image_b": ep.image_b,
        "tool_calls": [
            {"tool": tc.tool, "args": tc.args} for tc in ep.tool_calls
        ],
        "results": ep.results,
        "final_match": ep.final_match,
        "reward": ep.reward,
        "reward_components": ep.reward_components,
        "assistant_responses": ep.assistant_responses,
        "messages": ep.messages,
        "num_tool_calls": len(ep.tool_calls),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect zero-shot trajectories")
    parser.add_argument("--config", type=str, default="configs/phase1_grpo.yaml")
    parser.add_argument("--tool-server-url", type=str, default="http://localhost:8765")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000")
    parser.add_argument("--group-size", type=int, default=8,
                        help="Number of rollouts per pair")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Limit number of pairs (for testing)")
    parser.add_argument("--output", type=str, default="outputs/phase0/trajectories.jsonl")
    parser.add_argument("--difficulties", type=str, nargs="+", default=None,
                        help="Filter by difficulty (e.g., easy medium hard extreme)")
    args = parser.parse_args()

    config = load_config(args.config)
    os.chdir(Path(args.config).parent.parent)

    tool_client = ToolClient(args.tool_server_url)
    try:
        health = tool_client.health()
        logger.info(f"Tool server: {health}")
    except Exception as e:
        logger.error(f"Cannot connect to tool server: {e}")
        return

    # Load dataset
    train_path = config["data"].get("train_pairs", "data/hard_pairs_train.json")
    dataset = HardPairDataset.load(train_path)
    logger.info(f"Loaded {len(dataset)} training pairs from {train_path}")

    # Filter by difficulty
    pairs = dataset.pairs
    if args.difficulties:
        pairs = [p for p in pairs if p.difficulty in args.difficulties]
        logger.info(f"Filtered to {len(pairs)} pairs (difficulties: {args.difficulties})")
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
        logger.info(f"Limited to {len(pairs)} pairs")

    # Create rollout agent with stock model
    rollout_agent = VLLMRolloutAgent(
        vllm_url=args.vllm_url,
        model_name=config["model"]["name"],
        max_new_tokens=config["model"].get("max_new_tokens", 512),
        max_tool_calls=config["rl"]["max_tool_calls"],
        temperature=1.0,  # high temperature for diversity
        top_p=0.95,
        pose_weight=config.get("reward", {}).get("pose_weight", 1.0),
        inlier_weight=config.get("reward", {}).get("inlier_weight", 0.1),
        tool_cost=config.get("reward", {}).get("tool_cost", 0.02),
        format_weight=config.get("reward", {}).get("format_weight", 0.1),
        invalid_penalty=config.get("reward", {}).get("invalid_penalty", 0.2),
        reward_schedule="static",  # no dynamic scaling during collection
        reward_warmup_steps=0,
        matcher=config.get("data", {}).get("matcher", "loftr"),
    )

    # Collect trajectories
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    successful_episodes = 0
    rewards_all = []

    with open(output_path, "w") as f:
        for pair in tqdm(pairs, desc="Pairs"):
            gt_pose = {"R": pair.gt_R.tolist(), "t": pair.gt_t.tolist()} if pair.gt_R is not None else None

            for rollout_idx in range(args.group_size):
                ep = rollout_agent.run_episode(
                    pair_id=pair.pair_id,
                    image_a_path=pair.image_a,
                    image_b_path=pair.image_b,
                    tool_client=tool_client,
                    gt_pose=gt_pose,
                    K_a=pair.K_a.tolist() if pair.K_a is not None else None,
                    K_b=pair.K_b.tolist() if pair.K_b is not None else None,
                )

                ep_dict = episode_to_dict(ep)
                ep_dict["rollout_idx"] = rollout_idx
                ep_dict["difficulty"] = pair.difficulty
                f.write(json.dumps(ep_dict) + "\n")
                f.flush()

                total_episodes += 1
                rewards_all.append(ep.reward)
                if ep.reward > 0:
                    successful_episodes += 1

    success_rate = successful_episodes / max(total_episodes, 1)
    mean_reward = float(np.mean(rewards_all)) if rewards_all else 0.0
    median_reward = float(np.median(rewards_all)) if rewards_all else 0.0

    logger.info(f"\n=== Trajectory Collection Complete ===")
    logger.info(f"Total episodes: {total_episodes}")
    logger.info(f"Successful (reward > 0): {successful_episodes} ({success_rate:.1%})")
    logger.info(f"Mean reward: {mean_reward:.4f}")
    logger.info(f"Median reward: {median_reward:.4f}")
    logger.info(f"Saved to: {output_path}")

    # Save summary
    summary = {
        "total_episodes": total_episodes,
        "successful_episodes": successful_episodes,
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "median_reward": median_reward,
        "group_size": args.group_size,
        "num_pairs": len(pairs),
        "output_file": str(output_path),
    }
    summary_path = output_path.parent / "trajectory_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
