#!/usr/bin/env python
"""Phase 5a: Full evaluation suite for agentic SfM.

Evaluates a trained model (LoRA checkpoint or merged model) on the validation
set, computing pair-level metrics: pose AUC, inlier ratio, tool call efficiency,
success rate.

Supports multiple baselines and model variants for comparison.

Usage:
  python scripts/run_evaluation.py \
      --config configs/phase1_grpo.yaml \
      --tool-server-url http://localhost:8765 \
      --vllm-url http://localhost:8000 \
      --output-dir outputs/eval

  # Evaluate specific checkpoint
  python scripts/run_evaluation.py \
      --config configs/phase1_grpo.yaml \
      --vllm-url http://localhost:8000 \
      --tool-server-url http://localhost:8765 \
      --lora-checkpoint outputs/phase1/checkpoints/epoch_25 \
      --output-dir outputs/eval_epoch25
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
from agentic_sfm.rewards.pose_rewards import compute_pose_error, pose_auc_score
from agentic_sfm.tools.client import ToolClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_direct_matching(
    dataset: HardPairDataset,
    tool_client: ToolClient,
    matcher: str = "loftr",
    max_pairs: int = 50,
) -> dict:
    """Evaluate direct matching (no agent) as a baseline."""
    pairs = dataset.pairs[:max_pairs]
    results = []

    for pair in tqdm(pairs, desc=f"Direct {matcher}"):
        tool_client.register_image("img_a", pair.image_a)
        tool_client.register_image("img_b", pair.image_b)

        try:
            match_result = tool_client.match("img_a", "img_b", matcher)
            gt_pose = {"R": pair.gt_R.tolist(), "t": pair.gt_t.tolist()} if pair.gt_R is not None else None

            if gt_pose and match_result.get("pose"):
                pred_R = np.array(match_result["pose"]["R"])
                pred_t = np.array(match_result["pose"]["t"])
                pe = compute_pose_error(pred_R, pred_t, np.array(gt_pose["R"]), np.array(gt_pose["t"]))
                auc = pose_auc_score(pe)
            else:
                auc = 0.0

            results.append({
                "pair_id": pair.pair_id,
                "difficulty": pair.difficulty,
                "num_inliers": match_result.get("num_inliers", 0),
                "inlier_ratio": match_result.get("inlier_ratio", 0.0),
                "pose_auc": auc,
                "rotation_error_deg": pe.rotation_error_deg if gt_pose else None,
                "translation_error_deg": pe.translation_error_deg if gt_pose else None,
            })
        except Exception as e:
            logger.warning(f"Failed for {pair.pair_id}: {e}")
            results.append({
                "pair_id": pair.pair_id,
                "difficulty": pair.difficulty,
                "error": str(e),
                "pose_auc": 0.0,
            })

    return _aggregate_results(results, method=f"direct_{matcher}")


def evaluate_agent(
    dataset: HardPairDataset,
    rollout_agent,
    tool_client: ToolClient,
    max_pairs: int = 50,
) -> dict:
    """Evaluate agent-based matching."""
    pairs = dataset.pairs[:max_pairs]
    results = []

    for pair in tqdm(pairs, desc="Agent eval"):
        gt_pose = {"R": pair.gt_R.tolist(), "t": pair.gt_t.tolist()} if pair.gt_R is not None else None
        ep = rollout_agent.run_episode(
            pair_id=pair.pair_id,
            image_a_path=pair.image_a,
            image_b_path=pair.image_b,
            tool_client=tool_client,
            gt_pose=gt_pose,
        )

        components = ep.reward_components
        results.append({
            "pair_id": pair.pair_id,
            "difficulty": pair.difficulty,
            "reward": ep.reward,
            "pose_auc": components.get("pose_reward", 0.0),
            "num_inliers": ep.final_match.get("num_inliers", 0) if ep.final_match else 0,
            "inlier_ratio": ep.final_match.get("inlier_ratio", 0.0) if ep.final_match else 0.0,
            "num_tool_calls": len(ep.tool_calls),
            "rotation_error_deg": components.get("rotation_error_deg"),
            "translation_error_deg": components.get("translation_error_deg"),
            "success": ep.reward > 0,
        })

    return _aggregate_results(results, method="agent")


def _aggregate_results(results: list[dict], method: str) -> dict:
    """Aggregate per-pair results into summary metrics."""
    pose_aucs = [r.get("pose_auc", 0.0) for r in results if r.get("pose_auc") is not None]
    inlier_ratios = [r.get("inlier_ratio", 0.0) for r in results]
    n_inliers = [r.get("num_inliers", 0) for r in results]
    tool_calls = [r.get("num_tool_calls", 1) for r in results]
    successes = [r for r in results if r.get("success", r.get("pose_auc", 0) > 0)]

    # Per-difficulty breakdown
    by_difficulty = {}
    for r in results:
        d = r.get("difficulty", "unknown")
        by_difficulty.setdefault(d, []).append(r)

    difficulty_stats = {}
    for d, d_results in by_difficulty.items():
        d_aucs = [r.get("pose_auc", 0.0) for r in d_results]
        d_successes = sum(1 for r in d_results if r.get("success", r.get("pose_auc", 0) > 0))
        difficulty_stats[d] = {
            "count": len(d_results),
            "mean_pose_auc": float(np.mean(d_aucs)) if d_aucs else 0.0,
            "success_rate": d_successes / max(len(d_results), 1),
        }

    return {
        "method": method,
        "num_pairs": len(results),
        "mean_pose_auc": float(np.mean(pose_aucs)) if pose_aucs else 0.0,
        "median_pose_auc": float(np.median(pose_aucs)) if pose_aucs else 0.0,
        "mean_inlier_ratio": float(np.mean(inlier_ratios)) if inlier_ratios else 0.0,
        "mean_num_inliers": float(np.mean(n_inliers)) if n_inliers else 0.0,
        "mean_tool_calls": float(np.mean(tool_calls)) if tool_calls else 0.0,
        "success_rate": len(successes) / max(len(results), 1),
        "by_difficulty": difficulty_stats,
        "per_pair": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Full evaluation suite")
    parser.add_argument("--config", type=str, default="configs/phase1_grpo.yaml")
    parser.add_argument("--tool-server-url", type=str, default="http://localhost:8765")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000")
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    parser.add_argument("--max-pairs", type=int, default=50)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-agent", action="store_true")
    parser.add_argument("--matchers", type=str, nargs="+", default=["loftr"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.chdir(Path(args.config).parent.parent)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    val_path = config["data"].get("val_pairs", "data/hard_pairs_val.json")
    dataset = HardPairDataset.load(val_path)
    logger.info(f"Loaded {len(dataset)} validation pairs")

    tool_client = ToolClient(args.tool_server_url)
    try:
        health = tool_client.health()
        logger.info(f"Tool server: {health}")
    except Exception as e:
        logger.error(f"Cannot connect to tool server: {e}")
        return

    all_results = {}

    # Baselines: direct matching
    if not args.skip_baselines:
        for matcher in args.matchers:
            logger.info(f"\n=== Evaluating direct {matcher} matching ===")
            result = evaluate_direct_matching(dataset, tool_client, matcher, args.max_pairs)
            all_results[f"direct_{matcher}"] = result
            logger.info(f"  Mean pose AUC: {result['mean_pose_auc']:.4f}")
            logger.info(f"  Success rate: {result['success_rate']:.1%}")

    # Agent evaluation
    if not args.skip_agent:
        logger.info("\n=== Evaluating agent ===")
        from scripts.run_grpo import VLLMRolloutAgent

        rollout_agent = VLLMRolloutAgent(
            vllm_url=args.vllm_url,
            model_name=config["model"]["name"],
            max_new_tokens=config["model"].get("max_new_tokens", 512),
            max_tool_calls=config["rl"]["max_tool_calls"],
            temperature=0.1,  # low temperature for eval
            top_p=0.95,
            pose_weight=config.get("reward", {}).get("pose_weight", 1.0),
            inlier_weight=config.get("reward", {}).get("inlier_weight", 0.1),
            tool_cost=config.get("reward", {}).get("tool_cost", 0.02),
            format_weight=config.get("reward", {}).get("format_weight", 0.1),
            invalid_penalty=config.get("reward", {}).get("invalid_penalty", 0.2),
            reward_schedule="static",
            reward_warmup_steps=0,
        )

        result = evaluate_agent(dataset, rollout_agent, tool_client, args.max_pairs)
        all_results["agent"] = result
        logger.info(f"  Mean pose AUC: {result['mean_pose_auc']:.4f}")
        logger.info(f"  Success rate: {result['success_rate']:.1%}")
        logger.info(f"  Mean tool calls: {result['mean_tool_calls']:.1f}")

    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Print comparison table
    logger.info("\n=== Evaluation Summary ===")
    logger.info(f"{'Method':20s} {'Pose AUC':>10s} {'Success':>10s} {'Inliers':>10s} {'Tool Calls':>12s}")
    logger.info("-" * 65)
    for method, result in all_results.items():
        logger.info(
            f"{method:20s} {result['mean_pose_auc']:>10.4f} "
            f"{result['success_rate']:>9.1%} "
            f"{result['mean_num_inliers']:>10.1f} "
            f"{result.get('mean_tool_calls', 0):>12.1f}"
        )


if __name__ == "__main__":
    main()
