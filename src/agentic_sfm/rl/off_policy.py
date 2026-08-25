"""Off-policy GRPO: Reuse rollout samples for μ iterations to reduce vLLM serving cost.

Source: arXiv:2505.22257

Instead of collecting fresh vLLM rollouts every step, store (episodes, advantages,
old_logps) in a replay buffer and recompute current policy logprobs on stored
episodes for μ-1 additional GRPO updates without new vLLM calls.

Integration:
    from agentic_sfm.rl.off_policy import OffPolicyReplayBuffer
    buffer = OffPolicyReplayBuffer(max_size=1000, reuse_iterations=2)
    buffer.add(episodes, advantages)
    # On reuse steps, sample from buffer instead of collecting new rollouts
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class OffPolicyReplayBuffer:
    """FIFO replay buffer for off-policy GRPO sample reuse.

    Stores (episodes, advantages) tuples. On each training step:
    - If buffer has enough samples and we're on a reuse step: sample from buffer
    - Otherwise: collect fresh rollouts and add to buffer

    The reuse count tracks how many times each sample has been used.
    Samples exceeding max_reuse are discarded.
    """

    def __init__(
        self,
        max_size: int = 1000,
        reuse_iterations: int = 2,
        min_samples: int = 8,
    ):
        self.max_size = max_size
        self.reuse_iterations = reuse_iterations  # μ: total uses per sample (1 = no reuse)
        self.min_samples = min_samples
        self._buffer: deque = deque(maxlen=max_size)
        self._reuse_counts: deque = deque(maxlen=max_size)
        self._current_step = 0

    def add(self, episodes: list, advantages: list[float]):
        """Add new episodes and advantages to the buffer."""
        for ep, adv in zip(episodes, advantages):
            self._buffer.append(ep)
            self._reuse_counts.append(0)

    def should_reuse(self) -> bool:
        """Determine if this step should reuse samples from the buffer."""
        self._current_step += 1
        # Reuse on every μ-th step (i.e., step % μ != 0 means reuse)
        if self.reuse_iterations <= 1:
            return False
        return (self._current_step % self.reuse_iterations) != 1

    def sample(self, batch_size: int | None = None) -> tuple[list, list[float]] | None:
        """Sample episodes from the buffer for off-policy update.

        Returns (episodes, advantages) or None if buffer is empty.
        Only returns samples that haven't exceeded reuse_iterations.
        """
        if len(self._buffer) < self.min_samples:
            return None

        # Filter samples that haven't been reused too many times
        valid_indices = [
            i for i, count in enumerate(self._reuse_counts)
            if count < self.reuse_iterations
        ]

        if not valid_indices:
            # All samples exhausted, clear buffer
            self._buffer.clear()
            self._reuse_counts.clear()
            return None

        n = min(batch_size or len(valid_indices), len(valid_indices))
        sampled_indices = np.random.choice(valid_indices, size=n, replace=False)

        episodes = []
        advantages = []
        for idx in sampled_indices:
            ep = self._buffer[idx]
            # Recompute advantage from stored reward using group statistics
            episodes.append(ep)
            self._reuse_counts[idx] += 1

        # Compute group-relative advantages from sampled episodes
        advantages = self._compute_group_advantages(episodes)

        # Clean up exhausted samples
        while self._reuse_counts and self._reuse_counts[0] >= self.reuse_iterations:
            self._buffer.popleft()
            self._reuse_counts.popleft()

        logger.info(
            f"Off-policy reuse: sampled {n} episodes from buffer "
            f"({len(self._buffer)} remaining)"
        )
        return episodes, advantages

    def _compute_group_advantages(self, episodes: list) -> list[float]:
        """Compute group-relative advantages from episodes."""
        groups: dict[str, list] = {}
        for ep in episodes:
            groups.setdefault(ep.pair_id, []).append(ep)

        advantages = []
        for ep in episodes:
            group = groups[ep.pair_id]
            rewards = [e.reward for e in group]
            mean_r = np.mean(rewards)
            std_r = np.std(rewards) + 1e-8
            advantages.append((ep.reward - mean_r) / std_r)

        return advantages

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self):
        """Clear the buffer."""
        self._buffer.clear()
        self._reuse_counts.clear()
        self._current_step = 0
