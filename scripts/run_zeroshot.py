#!/usr/bin/env python
"""Phase 0: Zero-shot crop feasibility evaluation.

Evaluates whether stock Qwen3-VL-4B proposing crops improves matching
on hard image pairs, compared to direct matching baselines.

Usage:
  python scripts/run_zeroshot.py --config configs/phase0_zeroshot.yaml
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.data.hard_pairs import HardPairDataset, build_dataset, difficulty_bin
from agentic_sfm.eval.evaluate import (
    compare_methods,
    evaluate_agent,
    evaluate_direct_matching,
    save_results,
)
from agentic_sfm.tools.client import ToolClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Zero-shot feasibility")
    parser.add_argument("--config", type=str, default="configs/phase0_zeroshot.yaml")
    parser.add_argument("--tool-server-url", type=str, default="http://localhost:8765")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip direct matching baselines")
    parser.add_argument("--skip-agent", action="store_true", help="Skip agent evaluation")
    args = parser.parse_args()

    config = load_config(args.config)
    os.chdir(Path(args.config).parent.parent)

    # Setup output dirs
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config["output"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 0: Zero-shot crop feasibility evaluation")
    logger.info("=" * 60)

    # --- Build or load dataset ---
    data_path = config["data"]["output_path"]
    if os.path.exists(data_path):
        logger.info(f"Loading existing dataset from {data_path}")
        dataset = HardPairDataset.load(data_path)
    else:
        logger.info("Building hard-pair dataset...")
        megadepth_dir = config["data"].get("megadepth_dir")
        if megadepth_dir and os.path.exists(megadepth_dir):
            dataset = build_dataset(
                megadepth_dir=megadepth_dir,
                output_path=data_path,
                max_pairs_per_scene=config["data"]["max_pairs_per_scene"],
                min_overlap=config["data"]["min_overlap"],
            )
        else:
            logger.warning(f"MegaDepth not found at {megadepth_dir}")
            logger.info("Creating synthetic test dataset with sample images...")
            dataset = _build_synthetic_dataset(data_path)

    stats = dataset.stats()
    logger.info(f"Dataset: {stats}")

    # Subsample for feasibility
    max_per_diff = config["eval"]["max_pairs_per_difficulty"]
    difficulties = config["eval"]["difficulties"]
    subset_pairs = []
    for diff in difficulties:
        diff_pairs = [p for p in dataset.pairs if p.difficulty == diff]
        subset_pairs.extend(diff_pairs[:max_per_diff])
    eval_dataset = HardPairDataset(subset_pairs)
    logger.info(f"Eval subset: {eval_dataset.stats()}")

    # Connect to tool server
    tool_client = ToolClient(args.tool_server_url)
    try:
        health = tool_client.health()
        logger.info(f"Tool server: {health}")
    except Exception as e:
        logger.error(f"Cannot connect to tool server at {args.tool_server_url}: {e}")
        logger.info("Start it with: python tools_server/server.py")
        return

    all_results = []

    # --- Baselines: direct matching ---
    if not args.skip_baselines:
        for matcher in ["loftr", "mast3r"]:
            logger.info(f"\n--- Baseline: direct {matcher} ---")
            result = evaluate_direct_matching(eval_dataset, tool_client, matcher=matcher)
            save_results(result, str(results_dir / f"direct_{matcher}.json"))
            all_results.append(result)
            logger.info(f"Direct {matcher}: pose_auc={result['mean_pose_auc']:.3f}, "
                        f"inliers={result['mean_inliers']:.1f}")

    # --- Agent: zero-shot tool calling ---
    if not args.skip_agent:
        from agentic_sfm.agent.policy import AgenticSfMAgent

        logger.info("\n--- Agent: Qwen3-VL-4B zero-shot ---")
        agent = AgenticSfMAgent(
            model_name=config["model"]["name"],
            device=config["model"]["device"],
            max_new_tokens=config["model"]["max_new_tokens"],
            max_tool_calls=config["model"]["max_tool_calls"],
        )
        result = evaluate_agent(eval_dataset, agent, tool_client)
        save_results(result, str(results_dir / "agent_zeroshot.json"))
        all_results.append(result)
        logger.info(f"Agent zero-shot: pose_auc={result['mean_pose_auc']:.3f}, "
                    f"inliers={result['mean_inliers']:.1f}, "
                    f"tool_calls={result.get('mean_tool_calls', 0):.1f}")

    # --- Comparison ---
    if len(all_results) > 1:
        comparison = compare_methods(all_results)
        logger.info("\n" + "=" * 60)
        logger.info("Comparison:")
        for method, metrics in comparison.items():
            logger.info(f"  {method}: {metrics}")
        save_results(comparison, str(results_dir / "comparison.json"))

    logger.info(f"\nResults saved to {results_dir}/")


def _build_synthetic_dataset(data_path: str) -> HardPairDataset:
    """Build a small synthetic dataset for testing without MegaDepth."""
    from agentic_sfm.data.hard_pairs import ImagePair
    import cv2

    # Create synthetic image pairs with known poses
    data_dir = Path(data_path).parent / "synthetic"
    data_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for i in range(20):
        # Generate two random images
        img_a = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_b = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        path_a = str(data_dir / f"pair_{i}_a.png")
        path_b = str(data_dir / f"pair_{i}_b.png")
        cv2.imwrite(path_a, img_a)
        cv2.imwrite(path_b, img_b)

        # Random GT pose
        angle = np.random.uniform(0, 60)  # 0-60 degrees
        R = np.array([
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
            [0, 0, 1],
        ])
        t = np.array([np.random.uniform(-1, 1), 0, np.random.uniform(0.5, 2)])

        overlap = np.random.uniform(0.05, 0.95)
        pairs.append(ImagePair(
            pair_id=f"synthetic_{i}",
            image_a=path_a,
            image_b=path_b,
            gt_R=R,
            gt_t=t,
            overlap_score=overlap,
            difficulty=difficulty_bin(overlap),
            dataset="synthetic",
            scene="synthetic",
        ))

    dataset = HardPairDataset(pairs)
    dataset.save(data_path)
    return dataset


if __name__ == "__main__":
    main()
