"""Verifiable geometric rewards for agentic SfM RL training.

Reward components:
  - pose_auc: relative pose AUC@{5°,10°,20°} against GT
  - inlier_shaping: inlier count as shaping reward
  - tool_cost: per-tool-call cost penalty
  - doppelganger_correctness: binary reward for correct doppelganger classification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PoseError:
    """Relative pose error metrics."""

    rotation_error_deg: float
    translation_error_deg: float
    pose_auc_5: float
    pose_auc_10: float
    pose_auc_20: float


def compute_pose_error(
    pred_R: np.ndarray, pred_t: np.ndarray,
    gt_R: np.ndarray, gt_t: np.ndarray,
) -> PoseError:
    """Compute rotation and translation errors."""
    # Rotation error in degrees
    R_rel = pred_R @ gt_R.T
    trace = np.clip(np.trace(R_rel), -1.0, 3.0)
    rot_err = np.degrees(np.arccos((trace - 1) / 2))

    # Translation error in degrees (angle between vectors)
    pred_t_n = pred_t / (np.linalg.norm(pred_t) + 1e-8)
    gt_t_n = gt_t / (np.linalg.norm(gt_t) + 1e-8)
    cos_angle = np.clip(np.dot(pred_t_n, gt_t_n), -1.0, 1.0)
    trans_err = np.degrees(np.arccos(cos_angle))

    # AUC thresholds
    thresholds = [5, 10, 20]
    aucs = []
    for thresh in thresholds:
        if rot_err < thresh and trans_err < thresh:
            aucs.append(1.0)
        else:
            aucs.append(0.0)

    return PoseError(
        rotation_error_deg=rot_err,
        translation_error_deg=trans_err,
        pose_auc_5=aucs[0],
        pose_auc_10=aucs[1],
        pose_auc_20=aucs[2],
    )


def pose_auc_score(pose_error: PoseError) -> float:
    """AUC@{5,10,20} — standard pose AUC metric."""
    thresholds = np.array([5, 10, 20])
    rot_pass = pose_error.rotation_error_deg < thresholds
    trans_pass = pose_error.translation_error_deg < thresholds
    both_pass = rot_pass & trans_pass
    # AUC = mean of pass rates at each threshold
    return float(np.mean(both_pass.astype(float)))


def compute_pair_reward(
    match_result: dict[str, Any],
    gt_pose: dict[str, Any] | None = None,
    num_tool_calls: int = 1,
    num_invalid_calls: int = 0,
    num_valid_calls: int = 0,
    tool_cost: float = 0.02,
    inlier_weight: float = 0.1,
    pose_weight: float = 1.0,
    format_weight: float = 0.1,
    invalid_penalty: float = 0.2,
) -> dict[str, Any]:
    """Compute reward for a pair-level matching episode.

    Fine-grained reward decomposition based on ToolRL findings:
    - format_reward: binary, did the model produce valid tool calls?
    - inlier_reward: dense shaping from match quality
    - pose_reward: final outcome (dominant term)
    - tool_cost: per-step efficiency penalty
    - invalid_penalty: penalty for syntactically invalid tool calls

    Args:
        match_result: Output from tool_match()
        gt_pose: Ground truth pose {"R": [...], "t": [...]}
        num_tool_calls: Total number of tool calls made by the agent
        num_invalid_calls: Number of invalid/unparseable tool calls
        num_valid_calls: Number of valid tool calls
        tool_cost: Per-tool-call cost penalty
        inlier_weight: Weight for inlier-count shaping
        pose_weight: Weight for pose-error reward (should dominate)
        format_weight: Weight for format compliance reward
        invalid_penalty: Per-invalid-call penalty

    Returns:
        Dict with total reward and components.
    """
    components = {}

    # Format reward: binary, did the model produce valid tool calls?
    if num_tool_calls > 0:
        components["format_reward"] = format_weight * (num_valid_calls / max(num_tool_calls, 1))
    else:
        components["format_reward"] = 0.0

    # Invalid tool call penalty
    components["invalid_penalty"] = -invalid_penalty * num_invalid_calls

    # Inlier shaping (dense intermediate signal)
    num_inliers = match_result.get("num_inliers", 0)
    inlier_ratio = match_result.get("inlier_ratio", 0.0)
    components["inlier_reward"] = inlier_weight * min(inlier_ratio, 1.0)

    # Pose reward (final outcome — must dominate)
    if gt_pose is not None and match_result.get("pose") is not None:
        pred_R = np.array(match_result["pose"]["R"])
        pred_t = np.array(match_result["pose"]["t"])
        gt_R = np.array(gt_pose["R"])
        gt_t = np.array(gt_pose["t"])
        pe = compute_pose_error(pred_R, pred_t, gt_R, gt_t)
        components["pose_reward"] = pose_weight * pose_auc_score(pe)
        components["rotation_error_deg"] = pe.rotation_error_deg
        components["translation_error_deg"] = pe.translation_error_deg
    else:
        components["pose_reward"] = 0.0

    # Tool cost penalty (step penalty for efficiency)
    components["tool_cost"] = -tool_cost * num_tool_calls

    # Total
    total = sum(v for k, v in components.items() if isinstance(v, (int, float)))
    components["total_reward"] = total

    return components


def compute_scene_reward(
    recon_result: dict[str, Any],
    gt_recon: dict[str, Any] | None = None,
    num_tool_calls: int = 1,
    tool_cost: float = 0.05,
    registration_weight: float = 0.5,
    pose_weight: float = 1.0,
    split_penalty: float = 0.5,
) -> dict[str, Any]:
    """Compute reward for a scene-level SfM episode.

    Args:
        recon_result: Output from tool_sfm_run() or tool_inspect()
        gt_recon: Ground truth reconstruction stats
        num_tool_calls: Number of tool calls
        tool_cost: Per-tool-call cost
        registration_weight: Weight for registered image ratio
        pose_weight: Weight for pose accuracy
        split_penalty: Penalty for split/corrupt models

    Returns:
        Dict with total reward and components.
    """
    components = {}

    num_registered = recon_result.get("num_registered", 0)
    num_points = recon_result.get("num_points3d", 0)

    if gt_recon:
        gt_images = gt_recon.get("num_images", 1)
        reg_ratio = num_registered / max(gt_images, 1)
        components["registration_reward"] = registration_weight * reg_ratio
    else:
        components["registration_reward"] = registration_weight * min(num_registered / 100, 1.0)

    # Penalize split models (0 registered = total failure)
    if num_registered == 0:
        components["split_penalty"] = -split_penalty
    else:
        components["split_penalty"] = 0.0

    # Tool cost
    components["tool_cost"] = -tool_cost * num_tool_calls

    total = sum(v for k, v in components.items() if isinstance(v, (int, float)))
    components["total_reward"] = total

    return components


def compute_doppelganger_reward(
    pred_is_doppelganger: bool,
    gt_is_doppelganger: bool,
) -> float:
    """Binary reward for doppelganger classification."""
    return 1.0 if pred_is_doppelganger == gt_is_doppelganger else -1.0
