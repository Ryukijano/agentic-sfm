#!/usr/bin/env python
"""Fallback/custom reward function for the verl reward loop.

The primary reward is computed inside :class:`AgenticSfmAgentLoop` (see
:mod:`agentic_sfm.verl_agent_loop`) because the reward depends on the full
multi-turn trajectory and the ground-truth pose.  This module is kept as a thin
verl-compatible fallback that reuses a precomputed score when it is available
in ``extra_info``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return a verl-compatible reward dict for a completed trajectory.

    Prefer a reward already computed by the agent loop.  If none is present,
    fall back to ``0.0`` with a warning so training can still proceed.
    """
    extra_info = extra_info or {}

    # If the agent loop already computed and stored the reward, reuse it.
    if isinstance(extra_info, dict):
        rollout_scores = extra_info.get("rollout_reward_scores", {})
        if isinstance(rollout_scores, dict) and "reward_score" in rollout_scores:
            score = float(rollout_scores["reward_score"])
            return {"score": score, "reward_score": score, "source": "agent_loop"}
        precomputed = extra_info.get("reward_score")
        if precomputed is not None:
            score = float(precomputed)
            return {"score": score, "reward_score": score, "source": "extra_info"}

    logger.warning("No precomputed reward found for %s; returning 0.0 fallback.", data_source)
    return {"score": 0.0, "reward_score": 0.0, "source": "fallback"}
