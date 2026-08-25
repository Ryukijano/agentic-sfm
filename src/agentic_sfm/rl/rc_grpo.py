"""RC-GRPO: Reward-Conditioned GRPO to prevent group variance collapse.

Source: arXiv:2602.03025

After SFT warmup, the policy can become too peaked, producing near-identical
rollouts within a group. This causes zero variance → zero gradient (DAPO dynamic
sampling skips them all → no training signal).

RC-GRPO solves this by conditioning rollouts on reward tokens:
- Inject <|high_reward|> or <|low_reward|> tokens into the prompt
- Sample G/2 with each conditioning token
- This forces diversity even when the policy is peaked

Two-phase approach:
1. Train Reward-Conditioned Trajectory Policy (RCTP) via SFT on mixed-quality trajectories
2. Use reward-conditioned rollouts during GRPO training
"""

from __future__ import annotations

import logging
import random
from typing import Any

logger = logging.getLogger(__name__)

# Reward conditioning tokens
HIGH_REWARD_TOKEN = "<|high_reward|>"
LOW_REWARD_TOKEN = "<|low_reward|>"


def inject_reward_condition(
    messages: list[dict],
    reward_level: str = "high",
    position: str = "system_suffix",
) -> list[dict]:
    """Inject reward conditioning token into messages.

    Args:
        messages: Original conversation messages
        reward_level: "high" or "low"
        position: Where to inject — "system_suffix" (append to system msg) or "user_prefix"

    Returns:
        New messages list with reward conditioning injected
    """
    token = HIGH_REWARD_TOKEN if reward_level == "high" else LOW_REWARD_TOKEN
    new_messages = [msg.copy() for msg in messages]

    if position == "system_suffix":
        # Append to system message
        for msg in new_messages:
            if msg.get("role") == "system":
                if isinstance(msg.get("content"), str):
                    msg["content"] = msg["content"] + f"\n\nReward target: {token}"
                break
    elif position == "user_prefix":
        # Prepend to first user message
        for msg in new_messages:
            if msg.get("role") == "user":
                if isinstance(msg.get("content"), str):
                    msg["content"] = f"{token} " + msg["content"]
                elif isinstance(msg.get("content"), list):
                    msg["content"] = [{"type": "text", "text": token}] + msg["content"]
                break

    return new_messages


def sample_reward_conditioned_group(
    messages: list[dict],
    group_size: int = 8,
    high_reward_ratio: float = 0.5,
) -> list[tuple[list[dict], str]]:
    """Generate a group of reward-conditioned message variants.

    Args:
        messages: Original conversation messages
        group_size: Total number of rollouts in the group
        high_reward_ratio: Fraction of rollouts with high_reward conditioning

    Returns:
        List of (conditioned_messages, reward_level) tuples
    """
    n_high = int(group_size * high_reward_ratio)
    n_low = group_size - n_high

    group = []
    for _ in range(n_high):
        group.append((inject_reward_condition(messages, "high"), "high"))
    for _ in range(n_low):
        group.append((inject_reward_condition(messages, "low"), "low"))

    # Shuffle so the model can't infer position
    random.shuffle(group)
    return group


def compute_rc_advantages(
    episodes: list,
    reward_levels: list[str],
) -> list[float]:
    """Compute advantages for RC-GRPO.

    Instead of standard group-relative advantages, we compute:
    1. Within-group advantage (standard GRPO)
    2. Cross-condition advantage (high vs low reward difference)

    This encourages the model to produce better outputs when conditioned on high_reward.
    """
    import numpy as np

    # Standard group-relative advantages
    groups: dict[str, list] = {}
    for ep, level in zip(episodes, reward_levels):
        groups.setdefault(ep.pair_id, []).append(ep)

    advantages = []
    for ep, level in zip(episodes, reward_levels):
        group = groups[ep.pair_id]
        rewards = [e.reward for e in group]
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-8
        group_adv = (ep.reward - mean_r) / std_r

        # Cross-condition bonus: if high_reward episode gets higher reward
        # than low_reward episodes, give it extra advantage
        pair_episodes = [(e, l) for e, l in zip(episodes, reward_levels) if e.pair_id == ep.pair_id]
        high_rewards = [e.reward for e, l in pair_episodes if l == "high"]
        low_rewards = [e.reward for e, l in pair_episodes if l == "low"]

        if high_rewards and low_rewards:
            condition_diff = np.mean(high_rewards) - np.mean(low_rewards)
            # Scale condition bonus by how well this episode matches its conditioning
            if level == "high":
                condition_bonus = condition_diff * 0.1
            else:
                condition_bonus = -condition_diff * 0.1
        else:
            condition_bonus = 0.0

        advantages.append(group_adv + condition_bonus)

    return advantages


def prepare_rctp_training_data(
    trajectories: list[dict],
    reward_threshold: float = 0.3,
) -> list[dict]:
    """Prepare training data for Reward-Conditioned Trajectory Policy (RCTP).

    Takes mixed-quality trajectories and adds reward conditioning tokens.

    Args:
        trajectories: List of trajectory dicts with 'messages' and 'reward'
        reward_threshold: Reward above which is "high", below is "low"

    Returns:
        List of SFT examples with reward conditioning injected
    """
    examples = []
    for traj in trajectories:
        reward = traj.get("reward", 0.0)
        level = "high" if reward >= reward_threshold else "low"
        messages = traj.get("messages", [])

        if not messages:
            continue

        conditioned_messages = inject_reward_condition(messages, level)

        examples.append({
            "pair_id": traj.get("pair_id", ""),
            "image_a": traj.get("image_a", ""),
            "image_b": traj.get("image_b", ""),
            "messages": conditioned_messages,
            "reward": reward,
            "reward_level": level,
            "difficulty": traj.get("difficulty", "unknown"),
            "num_tool_calls": traj.get("num_tool_calls", 0),
        })

    return examples
