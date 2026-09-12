"""Verifiable geometric rewards for agentic SfM RL training.

Reward components that enter the total (nothing else):
  - pose_reward: AUC@{5°,10°,20°} against GT
  - inlier_reward: ratio + log inlier-count shaping
  - format_reward / invalid_penalty
  - accumulative_tool_reward: PyVision-RL style — reward productive tool calls
    only when the outcome is correct (prevents interaction collapse)

Replaced the old per-call tool_cost penalty (which caused interaction collapse
per PyVision-RL, ICML 2026) with an accumulative tool reward.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from agentic_sfm.constants import REWARD_TOTAL_KEYS

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
    """Compute rotation and translation direction errors in degrees."""
    R_rel = pred_R @ gt_R.T
    trace = np.clip(np.trace(R_rel), -1.0, 3.0)
    rot_err = np.degrees(np.arccos((trace - 1) / 2))

    pred_t_n = pred_t / (np.linalg.norm(pred_t) + 1e-8)
    gt_t_n = gt_t / (np.linalg.norm(gt_t) + 1e-8)
    cos_angle = np.clip(np.dot(pred_t_n, gt_t_n), -1.0, 1.0)
    trans_err = np.degrees(np.arccos(cos_angle))

    thresholds = [5, 10, 20]
    aucs = []
    for thresh in thresholds:
        if rot_err < thresh and trans_err < thresh:
            aucs.append(1.0)
        else:
            aucs.append(0.0)

    return PoseError(
        rotation_error_deg=float(rot_err),
        translation_error_deg=float(trans_err),
        pose_auc_5=aucs[0],
        pose_auc_10=aucs[1],
        pose_auc_20=aucs[2],
    )


def pose_auc_score(pose_error: PoseError) -> float:
    """Mean of pass/fail at 5/10/20° (standard coarse pose AUC proxy)."""
    thresholds = np.array([5, 10, 20])
    rot_pass = pose_error.rotation_error_deg < thresholds
    trans_pass = pose_error.translation_error_deg < thresholds
    both_pass = rot_pass & trans_pass
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
    accumulative_tool_coef: float = 0.1,
    use_accumulative_tool_reward: bool = True,
    **_ignored: Any,
) -> dict[str, Any]:
    """Compute reward for a pair-level matching episode.

    Diagnostics (rotation_error_deg, translation_error_deg) are recorded but
    NEVER added into total_reward.

    Accumulative tool reward (PyVision-RL, ICML 2026):
        R_tool = coef * n_tool_calls * 1[outcome_is_correct]
    This replaces the old per-call tool_cost penalty which caused interaction
    collapse (models learn to reduce tool usage to minimize penalty).
    """
    components: dict[str, Any] = {}

    if num_tool_calls > 0:
        components["format_reward"] = format_weight * (
            num_valid_calls / max(num_tool_calls, 1)
        )
    else:
        components["format_reward"] = 0.0

    components["invalid_penalty"] = -invalid_penalty * num_invalid_calls

    num_inliers = float(match_result.get("num_inliers") or 0)
    inlier_ratio = float(match_result.get("inlier_ratio") or 0.0)
    count_term = min(np.log1p(num_inliers) / np.log1p(200.0), 1.0)
    ratio_term = min(inlier_ratio, 1.0)
    components["inlier_reward"] = inlier_weight * (0.5 * ratio_term + 0.5 * count_term)

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

    if use_accumulative_tool_reward:
        # PyVision-RL: reward tool calls only when outcome is correct.
        # "Correct" = pose_reward > 0 (at least one AUC threshold passed).
        outcome_correct = float(components["pose_reward"]) > 0.0
        components["accumulative_tool_reward"] = (
            accumulative_tool_coef * num_valid_calls * (1.0 if outcome_correct else 0.0)
        )
        components["tool_cost"] = 0.0  # kept for compatibility, no longer penalizes
    else:
        # Legacy per-call penalty (causes interaction collapse — not recommended)
        components["tool_cost"] = -tool_cost * num_tool_calls
        components["accumulative_tool_reward"] = 0.0

    components["total_reward"] = float(
        sum(float(components[k]) for k in REWARD_TOTAL_KEYS if k in components)
    )
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
    """Compute reward for a scene-level SfM episode."""
    components: dict[str, Any] = {}

    num_registered = recon_result.get("num_registered", 0)

    if gt_recon:
        gt_images = gt_recon.get("num_images", 1)
        reg_ratio = num_registered / max(gt_images, 1)
        components["registration_reward"] = registration_weight * reg_ratio
    else:
        components["registration_reward"] = registration_weight * min(num_registered / 100, 1.0)

    if num_registered == 0:
        components["split_penalty"] = -split_penalty
    else:
        components["split_penalty"] = 0.0

    pose_err = recon_result.get("mean_pose_error_deg")
    if pose_err is not None and pose_weight:
        components["pose_reward"] = pose_weight * max(0.0, 1.0 - float(pose_err) / 20.0)
    else:
        components["pose_reward"] = 0.0

    components["tool_cost"] = -tool_cost * num_tool_calls
    components["total_reward"] = float(
        components["registration_reward"]
        + components["split_penalty"]
        + components["pose_reward"]
        + components["tool_cost"]
    )
    return components


def compute_doppelganger_reward(
    pred_is_doppelganger: bool,
    gt_is_doppelganger: bool,
) -> float:
    """Binary reward for doppelganger classification."""
    return 1.0 if pred_is_doppelganger == gt_is_doppelganger else -1.0
