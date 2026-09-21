"""Evaluation: zero-shot and RL policy evaluation on hard pairs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from agentic_sfm.agent.policy import AgenticSfMAgent, Episode
from agentic_sfm.constants import DEFAULT_MATCHER
from agentic_sfm.data.hard_pairs import HardPairDataset, ImagePair
from agentic_sfm.rewards.pose_rewards import (
    compute_pair_reward,
    compute_pose_error,
    pose_auc_score,
)
from agentic_sfm.tools.client import ToolClient

logger = logging.getLogger(__name__)


def evaluate_direct_matching(
    dataset: HardPairDataset,
    tool_client: ToolClient,
    matcher: str = DEFAULT_MATCHER,
) -> dict[str, Any]:
    """Baseline: direct matching without any cropping.

    This is the no-agent baseline — just run the matcher on full images.
    """
    results = []
    for pair in tqdm(dataset, desc=f"Direct matching ({matcher})"):
        tool_client.register_image("img_a", pair.image_a)
        tool_client.register_image("img_b", pair.image_b)

        match_result = tool_client.match(
            "img_a",
            "img_b",
            matcher=matcher,
            K_a=pair.K_a.tolist() if pair.K_a is not None else None,
            K_b=pair.K_b.tolist() if pair.K_b is not None else None,
        )
        gt_pose = {"R": pair.gt_R.tolist(), "t": pair.gt_t.tolist()} if pair.gt_R is not None else None

        reward_components = compute_pair_reward(
            match_result, gt_pose=gt_pose, num_tool_calls=1, num_valid_calls=1
        )

        results.append({
            "pair_id": pair.pair_id,
            "difficulty": pair.difficulty,
            "overlap": pair.overlap_score,
            "num_matches": match_result.get("num_matches", 0),
            "num_inliers": match_result.get("num_inliers", 0),
            "inlier_ratio": match_result.get("inlier_ratio", 0.0),
            "reward": reward_components["total_reward"],
            "rotation_error": reward_components.get("rotation_error_deg", -1),
            "translation_error": reward_components.get("translation_error_deg", -1),
            "pose_auc": reward_components.get("pose_reward", 0),
        })

    return _summarize_results(results, f"direct_{matcher}")


def evaluate_agent(
    dataset: HardPairDataset,
    agent: AgenticSfMAgent,
    tool_client: ToolClient,
    tool_server_url: str = "http://localhost:8765",
) -> dict[str, Any]:
    """Evaluate the MLLM agent on the dataset."""
    results = []
    for pair in tqdm(dataset, desc="Agent rollout"):
        gt_pose = {"R": pair.gt_R.tolist(), "t": pair.gt_t.tolist()} if pair.gt_R is not None else None

        episode = agent.run_episode(
            pair_id=pair.pair_id,
            image_a_path=pair.image_a,
            image_b_path=pair.image_b,
            tool_client=tool_client,
            gt_pose=gt_pose,
            K_a=pair.K_a.tolist() if pair.K_a is not None else None,
            K_b=pair.K_b.tolist() if pair.K_b is not None else None,
        )

        results.append({
            "pair_id": pair.pair_id,
            "difficulty": pair.difficulty,
            "overlap": pair.overlap_score,
            "num_tool_calls": len(episode.tool_calls),
            "num_matches": episode.final_match.get("num_matches", 0) if episode.final_match else 0,
            "num_inliers": episode.final_match.get("num_inliers", 0) if episode.final_match else 0,
            "inlier_ratio": episode.final_match.get("inlier_ratio", 0.0) if episode.final_match else 0.0,
            "reward": episode.reward,
            "rotation_error": episode.reward_components.get("rotation_error_deg", -1),
            "translation_error": episode.reward_components.get("translation_error_deg", -1),
            "pose_auc": episode.reward_components.get("pose_reward", 0),
            "tool_sequence": [tc.tool for tc in episode.tool_calls],
        })

    return _summarize_results(results, "agent")


def select_eval_pairs(
    pairs: list[ImagePair],
    max_pairs: int | None = None,
    seed: int = 0,
) -> list[ImagePair]:
    """Shuffle then optionally cap so ``[:N]`` is not a single difficulty bin.

    The val JSON is difficulty-ordered (hard, then medium, …). Taking the
    first 50 pairs therefore evaluated only hard pairs. A fixed seed keeps
    the subset reproducible across evals.
    """
    if not pairs:
        return []
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(pairs))
    selected = [pairs[int(i)] for i in order]
    if max_pairs is not None and int(max_pairs) > 0:
        selected = selected[: int(max_pairs)]
    return selected


def summarize_pair_eval(results: list[dict[str, Any]], method: str = "eval") -> dict[str, Any]:
    """Summarize pair eval rows, including a per-difficulty breakdown.

    Optional keys: ``geometric_reward`` (pose+inlier, comparable to matcher
    baselines that do not get the first-call bonus) and ``success``.
    """
    return _summarize_results(results, method)


def _summarize_results(results: list[dict[str, Any]], method: str) -> dict[str, Any]:
    """Summarize evaluation results."""
    if not results:
        return {"method": method, "total": 0, "mean_reward": 0.0, "num_pairs": 0}

    summary = {
        "method": method,
        "total": len(results),
        "num_pairs": len(results),
        "mean_reward": float(np.mean([r["reward"] for r in results])),
        "mean_inliers": float(np.mean([r.get("num_inliers", 0) for r in results])),
        "mean_inlier_ratio": float(np.mean([r.get("inlier_ratio", 0.0) for r in results])),
        "mean_pose_auc": float(np.mean([r.get("pose_auc", 0.0) for r in results])),
        "mean_rot_error": float(np.mean([r["rotation_error"] for r in results if r.get("rotation_error", -1) >= 0] or [0.0])),
        "mean_trans_error": float(np.mean([r["translation_error"] for r in results if r.get("translation_error", -1) >= 0] or [0.0])),
    }
    if any("geometric_reward" in r for r in results):
        summary["mean_geometric_reward"] = float(
            np.mean([r.get("geometric_reward", 0.0) for r in results])
        )
    if any("success" in r for r in results):
        summary["success_rate"] = float(np.mean([1.0 if r.get("success") else 0.0 for r in results]))
    if any("num_tool_calls" in r for r in results):
        summary["mean_tool_calls"] = float(np.mean([r.get("num_tool_calls", 0) for r in results]))

    for diff in ["easy", "medium", "hard", "extreme"]:
        subset = [r for r in results if r.get("difficulty") == diff]
        if subset:
            summary[f"{diff}_count"] = len(subset)
            summary[f"{diff}_mean_reward"] = float(np.mean([r["reward"] for r in subset]))
            summary[f"{diff}_mean_inliers"] = float(np.mean([r.get("num_inliers", 0) for r in subset]))
            summary[f"{diff}_mean_pose_auc"] = float(np.mean([r.get("pose_auc", 0.0) for r in subset]))
            if any("geometric_reward" in r for r in subset):
                summary[f"{diff}_mean_geometric_reward"] = float(
                    np.mean([r.get("geometric_reward", 0.0) for r in subset])
                )
            if any("success" in r for r in subset):
                summary[f"{diff}_success_rate"] = float(
                    np.mean([1.0 if r.get("success") else 0.0 for r in subset])
                )

    if method == "agent" and "mean_tool_calls" not in summary and any(
        "num_tool_calls" in r for r in results
    ):
        summary["mean_tool_calls"] = float(np.mean([r["num_tool_calls"] for r in results]))

    summary["raw_results"] = results
    return summary


def save_results(results: dict[str, Any], path: str) -> None:
    """Save evaluation results to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results to {path}")


def compare_methods(results_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare multiple methods."""
    comparison = {}
    for r in results_list:
        method = r["method"]
        comparison[method] = {
            "mean_reward": r["mean_reward"],
            "mean_inliers": r["mean_inliers"],
            "mean_inlier_ratio": r["mean_inlier_ratio"],
            "mean_pose_auc": r["mean_pose_auc"],
        }
        for diff in ["easy", "medium", "hard", "extreme"]:
            if f"{diff}_mean_reward" in r:
                comparison[method][f"{diff}_pose_auc"] = r[f"{diff}_mean_pose_auc"]

    return comparison
