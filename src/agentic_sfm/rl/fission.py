"""FISSION-GRPO: Error recovery via corrective training instances.

After each GRPO step, failed trajectories are "fissioned" into corrective
training instances. For each failed trajectory, the error point is identified
(first failed tool call), a diagnostic feedback message is injected, and
recovery rollouts are sampled from the corrective context.

Reference: ACL 2026 — improves error recovery by 5.7% absolute.

Integration:
    from agentic_sfm.rl.fission import FissionGRPO
    fission = FissionGRPO(vllm_agent, tool_client, config)
    recovery_episodes = fission.fission_failed_episodes(failed_episodes, gt_pose_map)
    # Then compute GRPO update on recovery episodes with adjusted advantages
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agentic_sfm.constants import DEFAULT_MATCHER

logger = logging.getLogger(__name__)


@dataclass
class FissionConfig:
    """Configuration for FISSION-GRPO."""
    enabled: bool = True
    reward_threshold: float = 0.1  # Episodes below this are "failed"
    group_size: int = 4  # Number of recovery rollouts per failed episode
    max_recovery_steps: int = 5  # Max additional tool calls in recovery
    diagnostic_feedback: bool = True  # Inject diagnostic hints


# Diagnostic feedback templates for common failure modes
DIAGNOSTIC_TEMPLATES = {
    "no_inliers": (
        "The previous match returned 0 inliers. This suggests the images "
        "have little overlap at the current scale. Try cropping to a "
        "region of visual overlap and matching again."
    ),
    "low_inliers": (
        "The previous match returned only {n_inliers} inliers. "
        "Consider cropping to the shared region or trying a different matcher."
    ),
    "invalid_tool": (
        "The last tool call was invalid. Please output valid JSON: "
        '{{"tool": "...", "args": {{...}}}}'
    ),
    "unknown_tool": (
        "Unknown tool called. Available tools: crop, match, doppelganger_check, done."
    ),
    "generic": (
        "The previous approach did not produce a good result. "
        "Try a different strategy: crop to overlap region, use a different matcher, "
        "or check for doppelgangers."
    ),
}


def _diagnose_failure(episode) -> str:
    """Diagnose why an episode failed and return appropriate feedback."""
    results = episode.results
    if not results:
        return DIAGNOSTIC_TEMPLATES["generic"]

    last_result = results[-1]
    if "error" in str(last_result.get("error", "")):
        if "Unknown tool" in str(last_result.get("error", "")):
            return DIAGNOSTIC_TEMPLATES["unknown_tool"]
        return DIAGNOSTIC_TEMPLATES["invalid_tool"]

    # Check match results
    if episode.final_match:
        n_inliers = episode.final_match.get("num_inliers", 0)
        if n_inliers == 0:
            return DIAGNOSTIC_TEMPLATES["no_inliers"]
        elif n_inliers < 20:
            return DIAGNOSTIC_TEMPLATES["low_inliers"].format(n_inliers=n_inliers)

    return DIAGNOSTIC_TEMPLATES["generic"]


def _find_error_point(episode) -> int:
    """Find the index of the first failed tool call in the episode."""
    for i, result in enumerate(episode.results):
        if "error" in result:
            return i
    # If no explicit error, find the last match with low inliers
    if episode.final_match:
        n_inliers = episode.final_match.get("num_inliers", 0)
        if n_inliers < 10:
            return len(episode.results) - 1
    return len(episode.results)  # No specific error point found


class FissionGRPO:
    """FISSION-GRPO: Fission failed trajectories into corrective training instances.

    For each failed episode:
    1. Identify the error point (first failed tool call)
    2. Truncate the conversation up to the error point
    3. Inject diagnostic feedback as a user message
    4. Sample G' recovery rollouts from the corrective context
    5. Compute GRPO update on recovery trajectories

    The advantage for recovery episodes is:
        adv = recovery_reward - original_failed_reward
    (positive if recovery improves over the original failure)
    """

    def __init__(
        self,
        vllm_agent,  # VLLMRolloutAgent
        tool_client,  # ToolClient
        config: FissionConfig | dict | None = None,
    ):
        if config is None:
            config = FissionConfig()
        elif isinstance(config, dict):
            config = FissionConfig(**config)

        self.agent = vllm_agent
        self.tool_client = tool_client
        self.config = config

    def fission_failed_episodes(
        self,
        failed_episodes: list,
        gt_pose_map: dict[str, dict | None] | None = None,
    ) -> list:
        """Generate recovery episodes from failed trajectories.

        Args:
            failed_episodes: List of RolloutEpisode objects that failed (reward < threshold)
            gt_pose_map: Mapping from pair_id to ground truth pose dict

        Returns:
            List of recovery RolloutEpisode objects
        """
        if not self.config.enabled or not failed_episodes:
            return []

        recovery_episodes = []

        for ep in failed_episodes:
            if ep.reward >= self.config.reward_threshold:
                continue

            # Find error point
            error_idx = _find_error_point(ep)

            # Build corrective context: messages up to error point + diagnostic feedback
            corrective_messages = ep.messages[:_find_message_cutoff(ep.messages, error_idx)]

            if self.config.diagnostic_feedback:
                feedback = _diagnose_failure(ep)
                corrective_messages.append({
                    "role": "user",
                    "content": f"Correction: {feedback}",
                })

            # Sample recovery rollouts
            gt_pose = gt_pose_map.get(ep.pair_id) if gt_pose_map else None

            for _ in range(self.config.group_size):
                recovery_ep = self._run_recovery_episode(
                    ep.pair_id,
                    ep.image_a,
                    ep.image_b,
                    corrective_messages,
                    gt_pose,
                    original_reward=ep.reward,
                    seed_images=list(ep.images) if ep.images else None,
                    K_a=getattr(ep, "K_a", None),
                    K_b=getattr(ep, "K_b", None),
                )
                if recovery_ep is not None:
                    recovery_episodes.append(recovery_ep)

        logger.info(
            f"FISSION: {len(failed_episodes)} failed episodes → "
            f"{len(recovery_episodes)} recovery episodes"
        )
        return recovery_episodes

    def _run_recovery_episode(
        self,
        pair_id: str,
        image_a: str,
        image_b: str,
        corrective_messages: list[dict],
        gt_pose: dict | None,
        original_reward: float,
        seed_images: list | None = None,
        K_a=None,
        K_b=None,
    ):
        """Run a single recovery rollout from the corrective context."""
        import os

        from agentic_sfm.agent.policy import execute_sfm_tool, format_observation, parse_tool_call
        from agentic_sfm.geometry import keep_best_match
        from scripts.run_grpo import RolloutEpisode

        ep = RolloutEpisode(
            pair_id=pair_id,
            image_a=image_a,
            image_b=image_b,
        )

        if seed_images:
            image_b64s = list(seed_images)
        else:
            image_b64s = [
                self.agent._encode_image(image_a),
                self.agent._encode_image(image_b),
            ]
        ep.images = image_b64s

        match_kwargs: dict[str, Any] = {
            "matcher": getattr(self.agent, "matcher", DEFAULT_MATCHER)
        }
        if K_a is not None:
            match_kwargs["K_a"] = K_a if not hasattr(K_a, "tolist") else K_a.tolist()
        if K_b is not None:
            match_kwargs["K_b"] = K_b if not hasattr(K_b, "tolist") else K_b.tolist()
        ep.K_a = match_kwargs.get("K_a")
        ep.K_b = match_kwargs.get("K_b")

        messages = list(corrective_messages)
        ep.messages = messages

        self.tool_client.register_image("img_a", image_a)
        self.tool_client.register_image("img_b", image_b)

        for _step in range(self.config.max_recovery_steps):
            texts, token_lps = self.agent._vllm_chat(messages, image_b64s, n=1)
            response = texts[0]

            ep.assistant_responses.append(response)
            if token_lps and token_lps[0] is not None:
                ep.vllm_token_logprobs.append(token_lps[0])
            else:
                ep.vllm_token_logprobs.append([])

            tc = parse_tool_call(response)
            if tc is None:
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": 'Please call a tool using JSON format: {"tool": "...", "args": {...}}'
                })
                continue

            ep.tool_calls.append(tc)
            if tc.tool == "done":
                messages.append({"role": "assistant", "content": response})
                break

            try:
                result = execute_sfm_tool(self.tool_client, tc, match_kwargs)
            except Exception as e:
                result = {"error": str(e)}

            ep.results.append(result)
            if tc.tool in ("match", "crop_and_match") or result.get("pose") is not None:
                ep.final_match = keep_best_match(ep.final_match, result)

            messages.append({"role": "assistant", "content": response})
            obs_text = f"Observation: {format_observation(result)}"
            obs_content: list | str = obs_text
            crop_b64 = result.get("image_b64") or (result.get("crop") or {}).get("image_b64")
            crop_path = result.get("path") or (result.get("crop") or {}).get("path")
            if not crop_b64 and crop_path and os.path.exists(crop_path):
                crop_b64 = self.agent._encode_image(crop_path)
            if crop_b64:
                image_b64s.append(crop_b64)
                ep.images = image_b64s
                obs_content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{crop_b64}"}},
                    {"type": "text", "text": obs_text},
                ]
            messages.append({"role": "user", "content": obs_content})

        ep.messages = messages

        from agentic_sfm.rewards.pose_rewards import compute_pair_reward
        num_valid = len(ep.tool_calls)
        num_invalid = sum(
            1 for r in ep.results
            if "error" in r and "Unknown tool" not in str(r.get("error", ""))
        )
        ep.reward_components = compute_pair_reward(
            ep.final_match or {}, gt_pose=gt_pose,
            num_tool_calls=len(ep.tool_calls) + num_invalid,
            num_invalid_calls=num_invalid,
            num_valid_calls=num_valid,
            tool_cost=self.agent.tool_cost,
            inlier_weight=self.agent.inlier_weight,
            pose_weight=self.agent.pose_weight,
            format_weight=self.agent.format_weight,
            invalid_penalty=self.agent.invalid_penalty,
        )
        ep.reward = ep.reward_components["total_reward"]

        # Store original reward for advantage computation
        ep.reward_components["original_reward"] = original_reward
        ep.reward_components["recovery_advantage"] = ep.reward - original_reward

        return ep

    def compute_recovery_advantages(self, recovery_episodes: list) -> list[float]:
        """Compute advantages for recovery episodes.

        Recovery advantage = recovery_reward - original_failed_reward
        (positive if recovery improves over the original failure)

        Then normalize within the recovery group (same pair_id).
        """
        groups: dict[str, list] = {}
        for ep in recovery_episodes:
            groups.setdefault(ep.pair_id, []).append(ep)

        advantages = []
        for ep in recovery_episodes:
            group = groups[ep.pair_id]
            recovery_rewards = [e.reward for e in group]
            original = ep.reward_components.get("original_reward", 0.0)

            # Advantage relative to original failure
            raw_adv = ep.reward - original

            # Normalize within recovery group
            mean_r = np.mean(recovery_rewards)
            std_r = np.std(recovery_rewards) + 1e-8
            normalized = (ep.reward - mean_r) / std_r

            # Blend: 50% absolute recovery, 50% group-relative
            advantages.append(0.5 * raw_adv + 0.5 * normalized)

        return advantages


def _find_message_cutoff(messages: list[dict], error_idx: int) -> int:
    """Find the message index corresponding to the error point.

    The conversation alternates: user, assistant, user (observation), assistant, ...
    Each tool call produces: assistant message + user observation.
    So error_idx tool calls → cutoff at message 2*error_idx + 1 (after the error observation).
    """
    # Count assistant messages to find the cutoff
    assistant_count = 0
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            if assistant_count == error_idx:
                # Include up to this assistant message + the following observation
                return min(i + 2, len(messages))
            assistant_count += 1
    return len(messages)
