#!/usr/bin/env python
"""Phase 1b: Filter and prepare SFT training data from collected trajectories.

Reads trajectories.jsonl, filters by reward threshold, and formats as
SFT-ready chat messages for supervised fine-tuning.

Usage:
  python scripts/prepare_sft_data.py \
      --trajectories outputs/phase0/trajectories.jsonl \
      --output data/sft_train.jsonl \
      --reward-threshold 0.3 \
      --max-trajectories 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_trajectories(path: str) -> list[dict]:
    """Load trajectories from JSONL file."""
    trajectories = []
    with open(path) as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    return trajectories


def filter_trajectories(
    trajectories: list[dict],
    reward_threshold: float = 0.3,
    max_trajectories: int | None = None,
) -> list[dict]:
    """Filter trajectories by reward and deduplicate by pair_id."""
    # Sort by reward descending
    sorted_trajs = sorted(trajectories, key=lambda t: t.get("reward", 0.0), reverse=True)

    # Filter by threshold
    filtered = [t for t in sorted_trajs if t.get("reward", 0.0) >= reward_threshold]

    # Deduplicate by pair_id — keep best trajectory per pair
    seen_pairs = set()
    deduped = []
    for t in filtered:
        pid = t["pair_id"]
        if pid not in seen_pairs:
            deduped.append(t)
            seen_pairs.add(pid)

    # Limit count
    if max_trajectories:
        deduped = deduped[:max_trajectories]

    return deduped


def _used_crop(traj: dict) -> bool:
    return any(
        tc.get("tool") in ("crop", "crop_and_match")
        for tc in traj.get("tool_calls") or []
    )


def _traj_inliers(traj: dict) -> float:
    fm = traj.get("final_match") or {}
    return float(fm.get("num_inliers") or 0)


def filter_beats_fullframe(all_trajs: list[dict], successful: list[dict]) -> list[dict]:
    """Keep successes that beat a no-crop (full-frame) baseline for that pair."""
    by_pair: dict[str, list[dict]] = {}
    for t in all_trajs:
        by_pair.setdefault(t["pair_id"], []).append(t)

    kept: list[dict] = []
    skipped = 0
    for t in successful:
        peers = by_pair.get(t["pair_id"], [])
        baselines = [_traj_inliers(p) for p in peers if not _used_crop(p)]
        if not baselines:
            kept.append(t)
            continue
        baseline = max(baselines)
        if _traj_inliers(t) > baseline:
            kept.append(t)
        else:
            skipped += 1
    logger.info(
        "Beats-fullframe filter: kept %d / %d (dropped %d at or below full-frame inliers)",
        len(kept), len(successful), skipped,
    )
    return kept


def _crop_paths_from_traj(traj: dict) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    blobs = list(traj.get("results") or [])
    fm = traj.get("final_match")
    if isinstance(fm, dict):
        blobs.append(fm)
    for result in blobs:
        candidates = [result]
        nested = result.get("crop") if isinstance(result, dict) else None
        if isinstance(nested, dict):
            candidates.append(nested)
        for blob in candidates:
            if not isinstance(blob, dict):
                continue
            path = blob.get("path")
            if path and os.path.exists(path) and path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def trajectory_to_sft_example(traj: dict) -> dict:
    """Convert a trajectory to an SFT training example.

    The SFT example is a list of chat messages (system, user, assistant, user, assistant, ...)
    that the model will be trained on with cross-entropy loss over assistant tokens.
    """
    messages = traj.get("messages", [])

    # Ensure messages have the right format for the processor
    # The messages should already be in chat format from the rollout
    # We just need to clean up any image placeholders
    crop_paths = _crop_paths_from_traj(traj)
    extra_allowed = len(crop_paths)
    pair_images_seen = 0
    cleaned_messages = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, list):
            parts = []
            for item in content:
                if item.get("type") in ("image", "image_url"):
                    if pair_images_seen < 2:
                        parts.append({"type": "image"})
                        pair_images_seen += 1
                    elif extra_allowed > 0:
                        parts.append({"type": "image"})
                        extra_allowed -= 1
                    # else drop unmatched crop placeholders so SFT token counts match
                elif item.get("type") == "text":
                    parts.append({"type": "text", "text": item.get("text", "")})
            cleaned_messages.append({"role": role, "content": parts})
        else:
            cleaned_messages.append({"role": role, "content": content})

    return {
        "pair_id": traj["pair_id"],
        "image_a": traj["image_a"],
        "image_b": traj["image_b"],
        "crop_image_paths": crop_paths,
        "messages": cleaned_messages,
        "reward": traj.get("reward", 0.0),
        "reward_components": traj.get("reward_components", {}),
        "difficulty": traj.get("difficulty", "unknown"),
        "num_tool_calls": traj.get("num_tool_calls", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT training data")
    parser.add_argument("--trajectories", type=str, required=True,
                        help="Path to trajectories.jsonl")
    parser.add_argument("--output", type=str, default="data/sft_train.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--reward-threshold", type=float, default=0.3,
                        help="Minimum reward to include in SFT data")
    parser.add_argument("--max-trajectories", type=int, default=None,
                        help="Maximum number of trajectories to include")
    parser.add_argument("--include-near-miss", action="store_true",
                        help="Include near-miss trajectories (inlier_ratio > 0.1 but low pose)")
    parser.add_argument("--near-miss-threshold", type=float, default=0.1,
                        help="Inlier ratio threshold for near-miss inclusion")
    parser.add_argument(
        "--prefer-beats-fullframe",
        action="store_true",
        default=True,
        help="Keep only successes that beat a no-crop baseline for that pair",
    )
    parser.add_argument(
        "--no-prefer-beats-fullframe",
        action="store_false",
        dest="prefer_beats_fullframe",
        help="Disable the full-frame baseline filter",
    )
    args = parser.parse_args()

    # Load trajectories
    trajectories = load_trajectories(args.trajectories)
    logger.info(f"Loaded {len(trajectories)} trajectories from {args.trajectories}")

    # Filter by reward
    successful = filter_trajectories(
        trajectories,
        reward_threshold=args.reward_threshold,
        max_trajectories=args.max_trajectories,
    )
    logger.info(f"Filtered to {len(successful)} successful trajectories (reward >= {args.reward_threshold})")

    # Optionally include near-miss trajectories
    if args.include_near_miss:
        near_miss = []
        for t in trajectories:
            if t.get("reward", 0.0) >= args.reward_threshold:
                continue
            components = t.get("reward_components", {})
            inlier_ratio = components.get("inlier_reward", 0.0) / 0.1  # undo weight
            if inlier_ratio > args.near_miss_threshold:
                near_miss.append(t)
        # Deduplicate near-miss by pair_id
        seen_pairs = {t["pair_id"] for t in successful}
        near_miss = [t for t in near_miss if t["pair_id"] not in seen_pairs]
        if args.max_trajectories:
            remaining = args.max_trajectories - len(successful)
            near_miss = near_miss[:remaining]
        logger.info(f"Added {len(near_miss)} near-miss trajectories (inlier_ratio > {args.near_miss_threshold})")
        successful.extend(near_miss)

    if args.prefer_beats_fullframe:
        successful = filter_beats_fullframe(trajectories, successful)

    # Convert to SFT format
    sft_examples = [trajectory_to_sft_example(t) for t in successful]

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex) + "\n")

    # Stats
    rewards = [ex["reward"] for ex in sft_examples]
    difficulties = {}
    for ex in sft_examples:
        d = ex["difficulty"]
        difficulties[d] = difficulties.get(d, 0) + 1

    logger.info(f"\n=== SFT Data Preparation Complete ===")
    logger.info(f"Total examples: {len(sft_examples)}")
    logger.info(f"Mean reward: {np.mean(rewards):.4f}" if rewards else "No examples")
    logger.info(f"Min reward: {np.min(rewards):.4f}" if rewards else "")
    logger.info(f"Max reward: {np.max(rewards):.4f}" if rewards else "")
    logger.info(f"Difficulty distribution: {difficulties}")
    logger.info(f"Saved to: {output_path}")

    # Save summary
    summary = {
        "total_examples": len(sft_examples),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "min_reward": float(np.min(rewards)) if rewards else 0.0,
        "max_reward": float(np.max(rewards)) if rewards else 0.0,
        "difficulty_distribution": difficulties,
        "reward_threshold": args.reward_threshold,
        "include_near_miss": args.include_near_miss,
        "output_file": str(output_path),
    }
    summary_path = output_path.parent / "sft_data_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
