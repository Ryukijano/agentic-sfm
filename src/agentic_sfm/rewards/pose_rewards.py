"""Verifiable geometric rewards for agentic SfM RL training.

Reward components that enter the total (nothing else):
  - pose_reward: AUC@{5°,10°,20°} against GT
  - inlier_reward: ratio + log inlier-count shaping
  - format_reward / invalid_penalty
  - accumulative_tool_reward: PyVision-RL style — reward productive tool calls
    only when the outcome is correct (prevents interaction collapse)
  - ntep_intent_reward: NTEP (arXiv 2609.03493) process reward — per-call
    bonus when the tool call's intent aligns with a valid evidence-seeking
    goal and the result is non-error
  - ntep_redundancy_penalty: NTEP non-repeated-goal regularizer — per-call
    penalty when a call revisits an evidence goal already pursued earlier
    in the episode (same tool + same target + overlapping bbox / same pair)

Replaced the old per-call tool_cost penalty (which caused interaction collapse
per PyVision-RL, ICML 2026) with an accumulative tool reward.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Hashable

import numpy as np

from agentic_sfm.constants import REWARD_TOTAL_KEYS

if TYPE_CHECKING:
    from agentic_sfm.agent.policy import ToolCall

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


# ---------------------------------------------------------------------------
# NTEP-style process rewards (Necessary Tool-Evidence Path, arXiv 2609.03493)
#
# Two per-call shaping terms over the episode's tool trajectory:
#   1. Intent-evidence alignment: +coef for each call whose intent matches the
#      tool's evidence-seeking capability AND whose result is non-error.
#   2. Non-repeated-goal regularizer: -penalty for each call that revisits an
#      evidence goal already pursued earlier in the episode.
# ---------------------------------------------------------------------------

_NTEP_IOU_THRESHOLD = 0.5


def _first_arg(args: dict[str, Any], *names: str) -> Any:
    """First non-None value among alternative arg spellings."""
    for name in names:
        value = args.get(name)
        if value is not None:
            return value
    return None


def _bbox_from_args(args: dict[str, Any]) -> list[float] | None:
    """Extract a normalized [x1, y1, x2, y2] bbox from tool args, or None."""
    bbox = args.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        return [float(x) for x in bbox[:4]]
    except (TypeError, ValueError):
        return None


def _bbox_iou(a: list[float], b: list[float]) -> float:
    """IoU between two normalized [x1, y1, x2, y2] bboxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _ntep_goal(tool: str, args: dict[str, Any]) -> tuple[Hashable | None, list[float] | None]:
    """Evidence-goal signature for the non-repeated-goal regularizer.

    Returns (goal_key, bbox). ``goal_key`` groups calls pursuing the same
    evidence goal; ``bbox`` (when present) is compared via IoU so that only
    overlapping regions count as repeats. ``None`` key = no trackable goal.
    """
    if tool in ("crop", "crop_and_match"):
        image_id = _first_arg(args, "image_id", "crop_image_id")
        return (tool, image_id), _bbox_from_args(args)
    if tool == "match":
        a = _first_arg(args, "image_a", "image_a_id")
        b = _first_arg(args, "image_b", "image_b_id")
        # Different matcher = different evidence source → different goal.
        return (tool, frozenset(x for x in (a, b) if x is not None), args.get("matcher")), None
    if tool == "doppelganger_check":
        a = _first_arg(args, "image_a", "image_a_id")
        b = _first_arg(args, "image_b", "image_b_id")
        return (tool, frozenset(x for x in (a, b) if x is not None)), None
    return None, None


def _ntep_intent_satisfied(tool: str, args: dict[str, Any], result: Any) -> bool:
    """True when the call's intent aligns with a valid evidence goal.

    Requires a non-error result (a missing result is treated as non-error —
    the caller simply did not track results). Intent checks:
      - crop_and_match with bbox → "find overlap region" → needs inliers > 0
      - match (no crop) → "full-frame baseline" → always valid intent
      - doppelganger_check → "verify scene identity" → needs is_doppelganger
    """
    if isinstance(result, dict):
        if result.get("error"):
            return False
        res = result
    else:
        res = {}
    if tool == "crop_and_match":
        return _bbox_from_args(args) is not None and float(res.get("num_inliers") or 0) > 0
    if tool == "match":
        return True
    if tool == "doppelganger_check":
        return "is_doppelganger" in res
    return False


def _ntep_process_reward_counts(
    tool_calls: list[ToolCall],
    tool_results: list[dict[str, Any]] | None,
    iou_threshold: float = _NTEP_IOU_THRESHOLD,
) -> tuple[int, int]:
    """Walk the trajectory once; return (n_aligned_intents, n_redundant_calls).

    ``tool_results`` are paired with non-``done`` calls in order (callers only
    record results for executed calls; the terminal ``done`` produces none).
    """
    n_aligned = 0
    n_redundant = 0
    seen_goals: dict[Hashable, list[list[float] | None]] = {}

    results_iter = iter(tool_results or [])
    for tc in tool_calls:
        tool = getattr(tc, "tool", None)
        if tool is None or tool == "done":
            continue
        args = getattr(tc, "args", None) or {}
        result = next(results_iter, None)

        if _ntep_intent_satisfied(tool, args, result):
            n_aligned += 1

        goal_key, bbox = _ntep_goal(tool, args)
        if goal_key is not None:
            history = seen_goals.setdefault(goal_key, [])
            if bbox is None:
                # Goal has no spatial extent — any exact-goal revisit is redundant.
                if history:
                    n_redundant += 1
            elif any(prev is not None and _bbox_iou(bbox, prev) > iou_threshold for prev in history):
                n_redundant += 1
            history.append(bbox)

    return n_aligned, n_redundant


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
    tool_calls: list[ToolCall] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    ntep_intent_coef: float = 0.05,
    ntep_redundancy_penalty: float = 0.05,
    use_ntep_rewards: bool = False,
    **_ignored: Any,
) -> dict[str, Any]:
    """Compute reward for a pair-level matching episode.

    Diagnostics (rotation_error_deg, translation_error_deg) are recorded but
    NEVER added into total_reward.

    Accumulative tool reward (PyVision-RL, ICML 2026):
        R_tool = coef * n_tool_calls * 1[outcome_is_correct]
    This replaces the old per-call tool_cost penalty which caused interaction
    collapse (models learn to reduce tool usage to minimize penalty).

    NTEP process rewards (arXiv 2609.03493, Phase 2):
        ntep_intent_reward      = +ntep_intent_coef per call whose intent
                                  aligns with a valid evidence-seeking goal
                                  and whose result is non-error.
        ntep_redundancy_penalty = -ntep_redundancy_penalty per call revisiting
                                  an earlier evidence goal (same tool + same
                                  image + bbox IoU > 0.5, or same image pair).
    Both default to 0 unless ``use_ntep_rewards=True`` and ``tool_calls`` is
    provided; all new params are optional so existing callers are unaffected.
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

    # NTEP process rewards: per-call intent-evidence alignment bonus and
    # non-repeated-goal penalty over the episode's tool trajectory.
    if use_ntep_rewards and tool_calls:
        n_aligned, n_redundant = _ntep_process_reward_counts(tool_calls, tool_results)
        components["ntep_intent_reward"] = ntep_intent_coef * n_aligned
        components["ntep_redundancy_penalty"] = -ntep_redundancy_penalty * n_redundant
    else:
        components["ntep_intent_reward"] = 0.0
        components["ntep_redundancy_penalty"] = 0.0

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
