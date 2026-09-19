#!/usr/bin/env python
"""Phase 1: GRPO RL training for agentic matching.

Trains Qwen3.5-2B with LoRA using GRPO (Group Relative Policy Optimization)
to learn crop/match tool-calling policies for hard image pairs.

Architecture:
  - vLLM server (GPU 0): fast rollout sampling for episode generation
  - Training model (GPU 1): LoRA-adapted Qwen3.5-2B VLM for policy gradient updates
    - Tool server (GPU 2): matcher inference (LoFTR/MASt3R)

GRPO: For each prompt, sample N trajectories via vLLM, compute group-relative
advantages, update LoRA policy with clipped objective (PPO-style).

Usage:
  python scripts/run_grpo.py --config configs/phase1_grpo.yaml
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.agent.policy import (
    MAX_IDENTICAL_TOOL_CALLS,
    Episode,
    SYSTEM_PROMPT,
    ToolCall,
    apply_policy_chat_template,
    execute_sfm_tool,
    format_observation,
    parse_tool_call,
    repeat_breaker_message,
    tool_call_signature,
)
from agentic_sfm.constants import DEFAULT_MATCHER, DEFAULT_POLICY_MODEL
from agentic_sfm.data.hard_pairs import HardPairDataset
from agentic_sfm.geometry import keep_best_match
from agentic_sfm.rewards.pose_rewards import compute_pair_reward
from agentic_sfm.tools.client import ToolClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _n_images_in_messages(messages: list[dict[str, Any]]) -> int:
    n = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") in ("image", "image_url"):
                n += 1
    return n


@dataclass
class RolloutEpisode:
    """Episode with token-level log-probs for GRPO training."""
    pair_id: str
    image_a: str
    image_b: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    final_match: dict[str, Any] | None = None
    reward: float = 0.0
    reward_components: dict[str, Any] = field(default_factory=dict)
    # Per-turn assistant response texts and vLLM logprobs
    assistant_responses: list[str] = field(default_factory=list)
    vllm_token_logprobs: list[list[float]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    images: list[Any] = field(default_factory=list)
    K_a: Any = None
    K_b: Any = None
    # True for injected S-GRPO oracle trajectories — excluded from media/metrics
    # picks so wandb shows real policy rollouts, not synthetic demos.
    is_oracle: bool = False


class VLLMRolloutAgent:
    """Agent that uses vLLM server for fast rollout sampling."""

    def __init__(self, vllm_url: str, model_name: str, max_new_tokens: int = 512,
                 max_tool_calls: int = 10, temperature: float = 1.0, top_p: float = 0.95,
                 min_p: float = 0.05, top_k: int = 50,
                 pose_weight: float = 1.0, inlier_weight: float = 0.1,
                 tool_cost: float = 0.02, format_weight: float = 0.1,
                 invalid_penalty: float = 0.2, reward_schedule: str = "static",
                 reward_warmup_steps: int = 30, matcher: str = DEFAULT_MATCHER,
                 accumulative_tool_coef: float = 0.1,
                 use_accumulative_tool_reward: bool = True,
                 ntep_intent_coef: float = 0.05,
                 ntep_redundancy_penalty: float = 0.05,
                 use_ntep_rewards: bool = False,
                 sft_adapter: str | None = None,
                 reward_config: dict[str, Any] | None = None):
        self.vllm_url = vllm_url.rstrip("/")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_tool_calls = max_tool_calls
        self.temperature = temperature
        self.top_p = top_p
        self.min_p = min_p
        self.top_k = top_k
        # Reward config (needed in run_episode)
        self.pose_weight = pose_weight
        self.inlier_weight = inlier_weight
        self.tool_cost = tool_cost
        self.format_weight = format_weight
        self.invalid_penalty = invalid_penalty
        self.reward_schedule = reward_schedule
        self.reward_warmup_steps = reward_warmup_steps
        self.matcher = matcher
        # PyVision-RL accumulative tool reward (prevents interaction collapse)
        self.accumulative_tool_coef = accumulative_tool_coef
        self.use_accumulative_tool_reward = use_accumulative_tool_reward
        # NTEP process rewards (per-call intent alignment + non-repeated-goal)
        self.ntep_intent_coef = ntep_intent_coef
        self.ntep_redundancy_penalty = ntep_redundancy_penalty
        self.use_ntep_rewards = use_ntep_rewards
        self._global_step = 0
        self.vllm_model = model_name
        self._lora_loaded = False
        # Optional LoRA adapter to serve rollouts from (hot-loaded into vLLM by
        # the trainer via _reload_vllm_lora). Stored here for reference.
        self.sft_adapter = sft_adapter
        # Scene-level reward weights — read by run_scene_episode via
        # getattr(agent, "reward_config", {}).
        self.reward_config = dict(reward_config or {})

    def _encode_image(self, image_path: str) -> str:
        img = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _generate_turn(self, messages: list[dict], images_b64: list[str] | None = None) -> str:
        """Generate a single assistant turn via vLLM.

        Used by scene-level episodes (run_scene_episode). Returns the raw
        response text. For pair-level episodes, use run_episode() instead.
        """
        texts, _ = self._vllm_chat(messages, images_b64 or [], n=1)
        return texts[0]

    def _vllm_chat(self, messages: list[dict], images_b64: list[str],
                   n: int = 1, temperature: float | None = None) -> tuple[list[str], list[list[float] | None]]:
        """Returns (response_texts, per_token_logprobs_list)."""
        import openai
        client = openai.OpenAI(base_url=f"{self.vllm_url}/v1", api_key="EMPTY")
        temp = temperature if temperature is not None else self.temperature

        oai_messages = []
        img_idx = 0
        for msg in messages:
            if isinstance(msg.get("content"), list):
                parts = []
                for item in msg["content"]:
                    if item.get("type") == "image":
                        if img_idx < len(images_b64):
                            parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{images_b64[img_idx]}"}
                            })
                            img_idx += 1
                    elif item.get("type") == "text":
                        parts.append({"type": "text", "text": item["text"]})
                    elif item.get("type") == "image_url":
                        parts.append(item)
                oai_messages.append({"role": msg["role"], "content": parts})
            else:
                oai_messages.append(msg)

        sampling_kwargs = {}
        if self.min_p is not None:
            sampling_kwargs["min_p"] = self.min_p
        if self.top_k is not None:
            sampling_kwargs["top_k"] = self.top_k

        def _create():
            return client.chat.completions.create(
                model=self.vllm_model,
                messages=oai_messages,
                max_tokens=self.max_new_tokens,
                temperature=temp,
                top_p=self.top_p,
                n=n,
                logprobs=True,
                **sampling_kwargs,
            )

        try:
            try:
                resp = _create()
            except TypeError:
                # Older vLLM without min_p/top_k — retry without the extras.
                sampling_kwargs.clear()
                resp = _create()
            texts = []
            all_logprobs = []
            for choice in resp.choices:
                texts.append(choice.message.content)
                if choice.logprobs and choice.logprobs.content:
                    token_lps = [lp.logprob for lp in choice.logprobs.content if lp.logprob is not None]
                    all_logprobs.append(token_lps)
                else:
                    all_logprobs.append(None)
            return texts, all_logprobs
        except Exception as e:
            logger.error(f"vLLM chat error: {e}")
            return [""] * n, [None] * n

    def run_episode(self, pair_id: str, image_a_path: str, image_b_path: str,
                    tool_client: ToolClient, gt_pose: dict | None = None,
                    K_a=None, K_b=None) -> RolloutEpisode:
        ep = RolloutEpisode(pair_id=pair_id, image_a=image_a_path, image_b=image_b_path)

        tool_client.register_image("img_a", image_a_path)
        tool_client.register_image("img_b", image_b_path)

        img_a_b64 = self._encode_image(image_a_path)
        img_b_b64 = self._encode_image(image_b_path)
        image_b64s = [img_a_b64, img_b_b64]
        ep.images = image_b64s
        match_kwargs = {"matcher": self.matcher}
        if K_a is not None:
            match_kwargs["K_a"] = K_a if not hasattr(K_a, "tolist") else K_a.tolist()
        if K_b is not None:
            match_kwargs["K_b"] = K_b if not hasattr(K_b, "tolist") else K_b.tolist()
        ep.K_a = match_kwargs.get("K_a")
        ep.K_b = match_kwargs.get("K_b")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": "placeholder_a"},
                {"type": "image", "image": "placeholder_b"},
                {"type": "text", "text": "Match these two images. Call tools to achieve the best matching result, then output {\"tool\": \"done\"}."},
            ]},
        ]
        ep.messages = messages

        num_invalid = 0
        repeat_counts: dict[tuple, int] = {}
        for step in range(self.max_tool_calls):
            texts, token_lps = self._vllm_chat(messages, image_b64s, n=1)
            response = texts[0]
            logger.info(f"[{pair_id}] Step {step}: {response[:200]}")

            ep.assistant_responses.append(response)
            if token_lps and token_lps[0] is not None:
                ep.vllm_token_logprobs.append(token_lps[0])
            else:
                ep.vllm_token_logprobs.append([])

            tc = parse_tool_call(response)
            if tc is None:
                num_invalid += 1
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "Please call a tool using JSON format: {\"tool\": \"...\", \"args\": {...}}"})
                continue

            ep.tool_calls.append(tc)
            if tc.tool == "done":
                messages.append({"role": "assistant", "content": response})
                break

            # Identical-call loop guard: the model re-emitting the same rejected
            # call burns the whole budget, makes every episode in the group fail
            # identically, and leaves GRPO no contrast to learn from.
            sig = tool_call_signature(tc)
            repeat_counts[sig] = repeat_counts.get(sig, 0) + 1
            if repeat_counts[sig] > MAX_IDENTICAL_TOOL_CALLS:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": repeat_breaker_message(sig, repeat_counts[sig])})
                break

            try:
                result = execute_sfm_tool(tool_client, tc, match_kwargs)
            except Exception as e:
                result = {"error": str(e)}

            ep.results.append(result)
            if (not result.get("error")
                    and (tc.tool in ("match", "crop_and_match")
                         or result.get("pose") is not None)):
                ep.final_match = keep_best_match(ep.final_match, result)

            messages.append({"role": "assistant", "content": response})
            obs_text = f"Observation: {format_observation(result)}"
            obs_content: list | str = obs_text
            crop_b64 = result.get("image_b64") or (result.get("crop") or {}).get("image_b64")
            crop_path = result.get("path") or (result.get("crop") or {}).get("path")
            if not crop_b64 and crop_path and os.path.exists(crop_path):
                crop_b64 = self._encode_image(crop_path)
            # vLLM is launched with --limit-mm-per-prompt image:6. The prompt
            # already carries img_a + img_b (2 images), so keep at most 4 crop
            # images and drop the oldest once we exceed the cap — the policy
            # only needs the most recent crop to decide the next call.
            if crop_b64:
                n_crops = max(0, len(image_b64s) - 2)
                if n_crops >= 4:
                    # Too many crop images already in context; keep text only.
                    crop_b64 = None
                else:
                    image_b64s.append(crop_b64)
                    ep.images = image_b64s
                    obs_content = [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{crop_b64}"}},
                        {"type": "text", "text": obs_text},
                    ]
            messages.append({"role": "user", "content": obs_content})

        ep.messages = messages

        # "Valid" = executed call whose result is non-error. Error results
        # (bad args, server rejection) must NOT count — otherwise rejection
        # is reward-invisible and spam calls inflate format/accumulative
        # reward. ep.results parallels non-done tool calls.
        num_valid = sum(
            1 for res in ep.results
            if not (isinstance(res, dict) and res.get("error"))
        )
        pose_w = self.pose_weight
        format_w = self.format_weight
        if self.reward_schedule == "dynamic" and self._global_step < self.reward_warmup_steps:
            pose_w = pose_w * (1.0 / 3.0)
        elif self.reward_schedule == "dynamic":
            format_w = format_w * 0.5

        ep.reward_components = compute_pair_reward(
            ep.final_match or {}, gt_pose=gt_pose,
            num_tool_calls=len(ep.tool_calls) + num_invalid,
            num_invalid_calls=num_invalid,
            num_valid_calls=num_valid,
            tool_cost=self.tool_cost,
            inlier_weight=self.inlier_weight,
            pose_weight=pose_w,
            format_weight=format_w,
            invalid_penalty=self.invalid_penalty,
            accumulative_tool_coef=self.accumulative_tool_coef,
            use_accumulative_tool_reward=self.use_accumulative_tool_reward,
            tool_calls=ep.tool_calls,
            tool_results=ep.results,
            ntep_intent_coef=self.ntep_intent_coef,
            ntep_redundancy_penalty=self.ntep_redundancy_penalty,
            use_ntep_rewards=self.use_ntep_rewards,
        )
        ep.reward = ep.reward_components["total_reward"]

        return ep

    def run_oracle_episode(self, pair_id: str, image_a_path: str, image_b_path: str,
                           tool_client: ToolClient, gt_pose: dict | None = None,
                           K_a=None, K_b=None,
                           failed_group_max_reward: float = 0.0) -> RolloutEpisode:
        """S-GRPO CGI: generate an oracle trajectory using heuristic crop boxes.

        Tries each ORACLE_CROP_BOX on both images, runs crop_and_match, keeps
        the best result. Constructs a synthetic episode with the oracle
        trajectory and guarantees reward >= failed_group_max_reward so the
        injected trajectory always yields positive advantage in the group.

        ``failed_group_max_reward`` is the highest reward among the failed
        rollouts in this group (usually 0 or negative). The oracle's final
        reward is max(computed_reward, failed_group_max_reward + epsilon).
        """
        from agentic_sfm.geometry import iter_oracle_crops, keep_best_match

        ep = RolloutEpisode(pair_id=pair_id, image_a=image_a_path, image_b=image_b_path)
        tool_client.register_image("img_a", image_a_path)
        tool_client.register_image("img_b", image_b_path)

        img_a_b64 = self._encode_image(image_a_path)
        img_b_b64 = self._encode_image(image_b_path)
        ep.images = [img_a_b64, img_b_b64]

        match_kwargs = {"matcher": self.matcher}
        if K_a is not None:
            match_kwargs["K_a"] = K_a if not hasattr(K_a, "tolist") else K_a.tolist()
        if K_b is not None:
            match_kwargs["K_b"] = K_b if not hasattr(K_b, "tolist") else K_b.tolist()

        best_match = None
        best_crop_info: tuple[str, list[float], str] | None = None

        for crop_img_id, bbox, other_img_id in iter_oracle_crops():
            try:
                # execute_sfm_tool expects: image_id (crop target), bbox, image_b (other side)
                tc = ToolCall(
                    tool="crop_and_match",
                    args={
                        "image_id": crop_img_id,
                        "bbox": bbox,
                        "image_b": other_img_id,
                        "matcher": self.matcher,
                    },
                )
                result = execute_sfm_tool(tool_client, tc, match_kwargs)
                if result and not result.get("error"):
                    best_match = keep_best_match(best_match, result)
                    if best_match is result:
                        best_crop_info = (crop_img_id, bbox, other_img_id)
            except Exception as e:
                logger.debug(f"Oracle crop failed for {crop_img_id} {bbox}: {e}")
                continue

        # Also try full-frame match as baseline
        try:
            tc_full = ToolCall(tool="match", args={
                "image_a": "img_a", "image_b": "img_b",
                "matcher": self.matcher,
            })
            full_result = execute_sfm_tool(tool_client, tc_full, match_kwargs)
            best_match = keep_best_match(best_match, full_result)
            if best_match is full_result:
                best_crop_info = None  # full-frame is best
        except Exception:
            pass

        # Construct synthetic oracle trajectory with correct arg names
        oracle_tool_calls: list[ToolCall] = []
        if best_crop_info:
            crop_img_id, bbox, other_img_id = best_crop_info
            oracle_tc = ToolCall(
                tool="crop_and_match",
                args={
                    "image_id": crop_img_id,
                    "bbox": bbox,
                    "image_b": other_img_id,
                    "matcher": self.matcher,
                },
            )
            oracle_response = json.dumps({"tool": "crop_and_match", "args": oracle_tc.args})
        else:
            oracle_tc = ToolCall(
                tool="match",
                args={"image_a": "img_a", "image_b": "img_b", "matcher": self.matcher},
            )
            oracle_response = json.dumps({"tool": "match", "args": oracle_tc.args})

        done_tc = ToolCall(tool="done", args={})
        ep.tool_calls = [oracle_tc, done_tc]
        ep.assistant_responses = [oracle_response, json.dumps({"tool": "done", "args": {}})]
        ep.final_match = best_match
        ep.results = [best_match or {}]

        # Build messages matching run_episode's structure (with crop image if available)
        obs_text = f"Observation: {format_observation(best_match or {})}"
        obs_content: list | str = obs_text
        crop_b64 = None
        if best_match:
            crop_b64 = (best_match.get("image_b64")
                        or (best_match.get("crop") or {}).get("image_b64"))
            crop_path = (best_match.get("path")
                         or (best_match.get("crop") or {}).get("path"))
            if not crop_b64 and crop_path and os.path.exists(crop_path):
                crop_b64 = self._encode_image(crop_path)
        if crop_b64:
            ep.images.append(crop_b64)
            obs_content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{crop_b64}"}},
                {"type": "text", "text": obs_text},
            ]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": "placeholder_a"},
                {"type": "image", "image": "placeholder_b"},
                {"type": "text", "text": "Match these two images. Call tools to achieve the best matching result, then output {\"tool\": \"done\"}."},
            ]},
            {"role": "assistant", "content": oracle_response},
            {"role": "user", "content": obs_content},
            {"role": "assistant", "content": json.dumps({"tool": "done", "args": {}})},
        ]
        ep.messages = messages

        # Apply same reward schedule as normal episodes
        pose_w = self.pose_weight
        format_w = self.format_weight
        if self.reward_schedule == "dynamic" and self._global_step < self.reward_warmup_steps:
            pose_w = pose_w * (1.0 / 3.0)
        elif self.reward_schedule == "dynamic":
            format_w = format_w * 0.5

        ep.reward_components = compute_pair_reward(
            ep.final_match or {}, gt_pose=gt_pose,
            num_tool_calls=2, num_invalid_calls=0, num_valid_calls=1,
            tool_cost=self.tool_cost,
            inlier_weight=self.inlier_weight,
            pose_weight=pose_w,
            format_weight=format_w,
            invalid_penalty=self.invalid_penalty,
            accumulative_tool_coef=self.accumulative_tool_coef,
            use_accumulative_tool_reward=self.use_accumulative_tool_reward,
            tool_calls=ep.tool_calls,
            tool_results=ep.results,
            ntep_intent_coef=self.ntep_intent_coef,
            ntep_redundancy_penalty=self.ntep_redundancy_penalty,
            use_ntep_rewards=self.use_ntep_rewards,
        )
        ep.reward = ep.reward_components["total_reward"]
        ep.is_oracle = True

        # S-GRPO: guarantee oracle reward > all failed rollouts in the group
        # so it always yields positive advantage. Small epsilon ensures strictly
        # greater even when computed reward happens to tie.
        # Only crown an oracle that actually produced geometry — otherwise the
        # floor would pay positive advantage for imitating a trajectory that
        # produced nothing (e.g. during a tool-server outage).
        oracle_has_geometry = bool(
            ep.final_match and (ep.final_match.get("num_inliers") or 0) > 0
        )
        if oracle_has_geometry and ep.reward <= failed_group_max_reward:
            ep.reward = failed_group_max_reward + 0.1
            ep.reward_components["oracle_floor_boost"] = ep.reward - ep.reward_components["total_reward"]

        return ep


class GRPOTrainer:
    """GRPO trainer with LoRA policy gradient updates.

    GRPO: For each prompt, sample N trajectories via vLLM, compute group-relative
    advantages, update LoRA policy with clipped objective (PPO-style).
    """

    def __init__(
        self,
        config: dict,
        tool_client: ToolClient,
        vllm_url: str = "http://localhost:8000",
        output_dir: str = "outputs/phase1",
        load_pair_datasets: bool = True,
    ):
        self.config = config
        self.tool_client = tool_client
        self.vllm_url = vllm_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.group_size = config["rl"]["group_size"]
        self.max_tool_calls = config["rl"]["max_tool_calls"]
        self.temperature = config["rl"]["temperature"]
        self.top_p = config["rl"]["top_p"]
        self.lr = config["training"]["lr"]
        self.total_epochs = config["training"]["total_epochs"]
        self.save_freq = config["training"]["save_freq"]
        self.eval_freq = config["training"]["eval_freq"]
        self.grad_accum = config["training"]["gradient_accumulation_steps"]
        self.max_grad_norm = config["training"]["max_grad_norm"]
        # DAPO-style asymmetric clipping (Clip-Higher)
        self.clip_low = config["rl"].get("clip_low", 0.2)
        self.clip_high = config["rl"].get("clip_high", 0.28)
        # Dynamic sampling: skip groups with zero reward variance
        self.dynamic_sampling = config["rl"].get("dynamic_sampling", True)
        # Dynamic reward scaling: format-heavy early, correctness-heavy later
        self.reward_schedule = config["rl"].get("reward_schedule", "static")
        self.reward_warmup_steps = config["rl"].get("reward_warmup_steps", 30)
        # Reward weights from config
        self.pose_weight = config.get("reward", {}).get("pose_weight", 1.0)
        self.inlier_weight = config.get("reward", {}).get("inlier_weight", 0.1)
        self.tool_cost = config.get("reward", {}).get("tool_cost", 0.02)
        self.format_weight = config.get("reward", {}).get("format_weight", 0.1)
        self.invalid_penalty = config.get("reward", {}).get("invalid_penalty", 0.2)
        # PyVision-RL accumulative tool reward
        self.accumulative_tool_coef = config.get("reward", {}).get("accumulative_tool_coef", 0.1)
        self.use_accumulative_tool_reward = config.get("reward", {}).get("use_accumulative_tool_reward", True)
        # NTEP process rewards (arXiv 2609.03493) — off by default for Phase 1
        self.ntep_intent_coef = config.get("reward", {}).get("ntep_intent_coef", 0.05)
        self.ntep_redundancy_penalty = config.get("reward", {}).get("ntep_redundancy_penalty", 0.05)
        self.use_ntep_rewards = config.get("reward", {}).get("use_ntep_rewards", False)
        # S-GRPO: Conditional Ground-Truth Trajectory Injection
        self.sgrpo_cgi = config.get("rl", {}).get("sgrpo_cgi", True)
        # KL penalty to the frozen SFT reference (anchors the policy to the
        # distribution that writes clean tool JSON). 0 disables.
        self.kl_coef = config.get("training", {}).get("kl_coef", 0.0)
        # μ-GRPO off-support veto: for negative advantages, skip tokens the
        # policy already finds implausible (old_logp < threshold). None disables.
        self.off_support_threshold = config.get("training", {}).get("off_support_threshold")
        # On-policy rollouts: hot-reload the LoRA into vLLM after every global
        # step so rollouts track the policy instead of lagging a whole epoch.
        self.on_policy_reload = config.get("training", {}).get("on_policy_reload", False)
        # Where to stage the per-step LoRA snapshots (fast local storage).
        # Empty -> /dev/shm (node-local RAM).
        self.lora_staging_dir = config.get("training", {}).get("lora_staging_dir", "")
        # Rollout sampling hardening — cut the long tail that seeds repetition.
        self.min_p = config.get("rollout", {}).get("min_p", 0.05)
        self.top_k = config.get("rollout", {}).get("top_k", 50)
        self._global_step = 0
        self._ref_model = None
        self._adapter_version = 0

        self.curriculum_stages = config.get("curriculum", {}).get("stages", [])

        self.matcher = config.get("data", {}).get("matcher", DEFAULT_MATCHER)

        if load_pair_datasets:
            train_path = config["data"].get("train_pairs", "data/hard_pairs_train.json")
            val_path = config["data"].get("val_pairs", "data/hard_pairs_val.json")
            self.train_dataset = HardPairDataset.load(train_path) if os.path.exists(train_path) else HardPairDataset()
            self.val_dataset = HardPairDataset.load(val_path) if os.path.exists(val_path) else HardPairDataset()
            logger.info(f"Train: {len(self.train_dataset)} pairs | Val: {len(self.val_dataset)} pairs")
        else:
            # Scene-level trainers never touch the pair datasets — skip the load.
            self.train_dataset = HardPairDataset()
            self.val_dataset = HardPairDataset()

        self.model_name = config["model"].get("name", DEFAULT_POLICY_MODEL)
        self.lora_config = config["model"]["lora"]
        self.sft_adapter = config["model"].get("sft_adapter")
        # Use cuda:0 if CUDA_VISIBLE_DEVICES is set (only 1 GPU visible)
        # Otherwise use the configured training GPU (default cuda:1)
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            self.training_device = "cuda:0"
        else:
            self.training_device = config.get("rollout", {}).get("training_gpu", "cuda:1")

        self._model = None
        self._processor = None
        self._lora_params = None
        self._optimizer = None
        self._wandb = None
        self._best_epoch_reward = float("-inf")
        self._best_eval_reward = float("-inf")
        self._global_step = 0

        wandb_cfg = config.get("output", {})
        if wandb_cfg.get("wandb_enabled", True):
            try:
                import wandb
                self._wandb = wandb.init(
                    project=wandb_cfg.get("wandb_project", "agentic-sfm"),
                    entity=wandb_cfg.get("wandb_entity"),
                    name=wandb_cfg.get("wandb_run_name", "phase1-grpo"),
                    config=config,
                )
                logger.info("W&B logging enabled.")
            except Exception as e:
                logger.warning(f"W&B init failed: {e}")
                self._wandb = None

    def _load_training_model(self):
        """Load model + processor + LoRA on training GPU."""
        if self._model is not None:
            return

        from peft import LoraConfig, get_peft_model

        from agentic_sfm.agent.policy import load_policy_processor_and_model
        from agentic_sfm.constants import DEFAULT_LORA_TARGET_MODULES

        logger.info(f"Loading training model {self.model_name} on {self.training_device}...")
        self._processor, self._model = load_policy_processor_and_model(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.training_device,
        )

        lora_cfg = LoraConfig(
            r=self.lora_config["rank"],
            lora_alpha=self.lora_config["alpha"],
            lora_dropout=self.lora_config["dropout"],
            target_modules=self.lora_config.get("target_modules") or DEFAULT_LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        if self.sft_adapter and os.path.isdir(self.sft_adapter):
            logger.info(f"Loading SFT adapter warmup from {self.sft_adapter}")
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(self._model, self.sft_adapter, is_trainable=True)
        else:
            self._model = get_peft_model(self._model, lora_cfg)

        # Gradient checkpointing: recompute activations during backward instead of
        # storing them. Essential for 8192-token multimodal sequences — the eval
        # forward alone is ~9GB, and the grad pass would otherwise need ~40GB+.
        if hasattr(self._model, "gradient_checkpointing_enable"):
            self._model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        # use_cache must be off when checkpointing is on.
        if hasattr(self._model, "config") and hasattr(self._model.config, "use_cache"):
            self._model.config.use_cache = False
        self._model.print_trainable_parameters()

        trainable = [p for p in self._model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=self.config["training"]["weight_decay"])
        logger.info("Training model ready with LoRA.")

        # Frozen SFT reference for the KL anchor: the same warm-start adapter
        # the policy began from. Prevents the policy drifting into the
        # degenerate/repetitive modes RL without any anchor discovers.
        if self.kl_coef > 0 and self.sft_adapter and os.path.isdir(self.sft_adapter):
            try:
                from peft import PeftModel
                ref = PeftModel.from_pretrained(self._model.get_base_model(), self.sft_adapter, is_trainable=False)
                ref.eval()
                for p in ref.parameters():
                    p.requires_grad = False
                self._ref_model = ref
                logger.info(f"Reference (KL anchor) model loaded from {self.sft_adapter}")
            except Exception as e:
                logger.warning(f"Reference model load failed — KL disabled: {e}")
                self._ref_model = None

    def get_curriculum_difficulties(self, epoch: int) -> list[str]:
        if not self.curriculum_stages:
            return ["easy", "medium", "hard", "extreme"]
        cumulative = 0
        for stage in self.curriculum_stages:
            cumulative += stage["epochs"]
            if epoch < cumulative:
                return stage["difficulties"]
        return self.curriculum_stages[-1]["difficulties"]

    def collect_rollouts(self, pairs: list, rollout_agent: VLLMRolloutAgent) -> list[RolloutEpisode]:
        """Collect N rollouts per pair, with S-GRPO CGI injection.

        S-GRPO (arXiv 2604.16557): when all rollouts in a group fail (zero
        reward or no valid trajectory), inject the oracle trajectory with
        max reward. This provides a positive learning signal during cold-start
        without a separate SFT stage.
        """
        episodes = []
        cgi_injections = 0
        for pair in pairs:
            group_episodes = []
            all_failed = True
            for _ in range(self.group_size):
                gt_pose = {"R": pair.gt_R.tolist(), "t": pair.gt_t.tolist()} if pair.gt_R is not None else None
                ep = rollout_agent.run_episode(
                    pair_id=pair.pair_id,
                    image_a_path=pair.image_a,
                    image_b_path=pair.image_b,
                    tool_client=self.tool_client,
                    gt_pose=gt_pose,
                    K_a=pair.K_a,
                    K_b=pair.K_b,
                )
                group_episodes.append(ep)
                if ep.reward > 0:
                    all_failed = False

            # S-GRPO CGI: if all rollouts failed, inject oracle trajectory
            if self.sgrpo_cgi and all_failed and group_episodes:
                gt_pose = {"R": pair.gt_R.tolist(), "t": pair.gt_t.tolist()} if pair.gt_R is not None else None
                failed_max = max(e.reward for e in group_episodes)
                oracle_ep = rollout_agent.run_oracle_episode(
                    pair_id=pair.pair_id,
                    image_a_path=pair.image_a,
                    image_b_path=pair.image_b,
                    tool_client=self.tool_client,
                    gt_pose=gt_pose,
                    K_a=pair.K_a,
                    K_b=pair.K_b,
                    failed_group_max_reward=failed_max,
                )
                if oracle_ep.reward > failed_max:
                    if len(group_episodes) == 1:
                        # Append so the group has >1 member (zero-variance filter
                        # would drop a single-episode group).
                        group_episodes.append(oracle_ep)
                    else:
                        # Replace the worst rollout with the oracle trajectory.
                        worst_idx = min(range(len(group_episodes)),
                                        key=lambda i: group_episodes[i].reward)
                        group_episodes[worst_idx] = oracle_ep
                    cgi_injections += 1
                    logger.debug(f"  S-GRPO CGI: injected oracle for {pair.pair_id} (reward={oracle_ep.reward:.3f})")

            episodes.extend(group_episodes)

        if cgi_injections > 0:
            logger.info(f"  S-GRPO CGI: injected {cgi_injections} oracle trajectories ({cgi_injections}/{len(pairs)} pairs)")
        return episodes

    def _group_key(self, ep) -> str:
        """Grouping key for group-relative advantages.

        Pair-level episodes group by ``pair_id``; SceneGRPOTrainer overrides
        this to group scene-level episodes by ``scene_id``.
        """
        return ep.pair_id

    def compute_advantages(self, episodes: list[RolloutEpisode]) -> list[float]:
        groups: dict[str, list[RolloutEpisode]] = {}
        for ep in episodes:
            groups.setdefault(self._group_key(ep), []).append(ep)
        advantages = []
        for ep in episodes:
            group = groups[self._group_key(ep)]
            rewards = [e.reward for e in group]
            mean_r = np.mean(rewards)
            std_r = np.std(rewards) + 1e-8
            advantages.append((ep.reward - mean_r) / std_r)
        return advantages

    def _filter_zero_variance_groups(self, episodes: list[RolloutEpisode]) -> list[RolloutEpisode]:
        """DAPO Dynamic Sampling: remove groups where all rollouts have identical reward.
        These produce zero advantage = zero gradient, wasting compute.
        """
        if not self.dynamic_sampling:
            return episodes
        groups: dict[str, list[RolloutEpisode]] = {}
        for ep in episodes:
            groups.setdefault(self._group_key(ep), []).append(ep)
        filtered = []
        skipped = 0
        for group_key, group in groups.items():
            rewards = [e.reward for e in group]
            if np.std(rewards) < 1e-8:
                skipped += len(group)
            else:
                filtered.extend(group)
        if skipped > 0:
            logger.info(f"  Dynamic sampling: skipped {skipped} episodes (zero variance)")
        return filtered

    def _compute_assistant_mask(self, messages: list[dict], images: list, 
                                  processor) -> list[bool]:
        """Build a boolean mask over tokenized conversation: True for assistant tokens.
        
        For each assistant message at index k, we tokenize messages[:k] with
        add_generation_prompt=True to get the prefix length, and messages[:k+1]
        with add_generation_prompt=False to get the end. Tokens in [start, end)
        are assistant tokens.
        """
        def _imgs_for(msgs: list[dict[str, Any]]) -> list[Image.Image] | None:
            n = min(_n_images_in_messages(msgs), len(images))
            sliced = images[:n]
            return sliced or None

        full_text = apply_policy_chat_template(
            processor,
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_ids = processor(text=[full_text], images=_imgs_for(messages), return_tensors="pt")["input_ids"][0]
        n_tokens = len(full_ids)
        mask = [False] * n_tokens

        for k, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            try:
                prefix_text = apply_policy_chat_template(
                    processor,
                    messages[:k],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prefix_ids = processor(
                    text=[prefix_text], images=_imgs_for(messages[:k]), return_tensors="pt"
                )["input_ids"][0]
                start = len(prefix_ids)

                through_text = apply_policy_chat_template(
                    processor,
                    messages[:k + 1],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                through_ids = processor(
                    text=[through_text], images=_imgs_for(messages[:k + 1]), return_tensors="pt"
                )["input_ids"][0]
                end = len(through_ids)

                for i in range(start, min(end, n_tokens)):
                    mask[i] = True
            except Exception as e:
                logger.warning(f"Mask computation failed for msg {k}: {e}")

        return mask

    def _episode_pil_images(self, ep: RolloutEpisode) -> list[Image.Image]:
        """Decode rollout images (pair + crops) so logprobs see the same pixels as vLLM."""
        images: list[Image.Image] = []
        for item in ep.images or []:
            if isinstance(item, Image.Image):
                images.append(item.convert("RGB"))
                continue
            if not isinstance(item, str):
                continue
            try:
                raw = base64.b64decode(item)
                images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
            except Exception:
                continue
        if len(images) < 2:
            images = []
            for p in self._episode_fallback_image_paths(ep):
                try:
                    images.append(Image.open(p).convert("RGB"))
                except Exception:
                    continue
        return images

    def _episode_fallback_image_paths(self, ep) -> list[str]:
        """On-disk image paths to load when an episode's stored images can't be decoded.

        Pair episodes carry ``image_a``/``image_b``; scene episodes carry
        ``image_paths`` (resolved against ``image_root`` when relative).
        """
        image_a = getattr(ep, "image_a", None)
        image_b = getattr(ep, "image_b", None)
        if image_a and image_b:
            return [image_a, image_b]
        root = getattr(ep, "image_root", "") or ""
        out: list[str] = []
        for p in list(getattr(ep, "image_paths", None) or [])[:8]:
            pp = Path(p)
            if not pp.is_absolute() and root:
                pp = Path(root) / pp
            out.append(str(pp))
        return out

    def _policy_messages_and_images(
        self, ep: RolloutEpisode
    ) -> tuple[list[dict[str, Any]], list[Image.Image]]:
        """Rewrite vLLM image_url / placeholders to PIL ``type=image`` for the LoRA model."""
        images = self._episode_pil_images(ep)
        idx = 0
        out: list[dict[str, Any]] = []
        used: list[Image.Image] = []
        for msg in ep.messages:
            content = msg.get("content")
            if not isinstance(content, list):
                out.append(msg)
                continue
            new_content: list[dict[str, Any]] = []
            for item in content:
                if item.get("type") in ("image", "image_url"):
                    if idx < len(images):
                        used.append(images[idx])
                        new_content.append({"type": "image", "image": images[idx]})
                        idx += 1
                    continue
                new_content.append(item)
            out.append({**msg, "content": new_content})
        return out, used or images[:2]

    def compute_logprobs(self, episodes: list[RolloutEpisode], 
                          requires_grad: bool = False,
                          model: torch.nn.Module | None = None) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compute per-episode (token_logps, assistant_mask) under a LoRA policy.

        ``model`` defaults to the current trainable model; pass ``self._ref_model``
        to score the same episodes under the frozen SFT reference for the KL term.
        Returns list of (per_token_logps_tensor, mask_bool_tensor) for each episode.
        Only assistant token positions are masked True.
        """
        if model is None:
            self._load_training_model()
            model = self._model
        results = []

        for ep in episodes:
            if not ep.messages:
                results.append((torch.tensor(0.0, device=self.training_device), torch.tensor([], device=self.training_device, dtype=torch.bool)))
                continue

            try:
                messages, images = self._policy_messages_and_images(ep)
                text = apply_policy_chat_template(
                    self._processor,
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                inputs = self._processor(
                    text=[text], images=images, return_tensors="pt", padding=True
                ).to(self.training_device)

                mask = self._compute_assistant_mask(messages, images, self._processor)
                mask_tensor = torch.tensor(mask, device=self.training_device, dtype=torch.bool)

                ctx_manager = torch.enable_grad() if requires_grad else torch.no_grad()
                with ctx_manager:
                    outputs = model(**inputs)

                logits = outputs.logits[:, :-1, :]
                target_ids = inputs["input_ids"][:, 1:]
                # Memory-efficient logprob: log p(target) = logit_target - logsumexp(logits).
                # Avoids materializing the full [seq_len, vocab] log_softmax tensor, which
                # OOMs at seq_len~8192 x vocab~150k (two ~5GB fp32 copies on the train GPU).
                lse = torch.logsumexp(logits, dim=-1)  # [1, seq_len-1]
                target_logit = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                per_token_logps = target_logit - lse  # [1, seq_len-1]
                del logits, outputs

                # Align mask: mask[i] corresponds to input_ids[i], logps[i] predicts input_ids[i+1]
                # So logps at position i predicts token i+1; we need mask shifted by 1
                assistant_mask = mask_tensor[1:].to(per_token_logps.device)  # [seq_len-1]

                results.append((per_token_logps.squeeze(0), assistant_mask))
            except Exception as e:
                logger.warning(f"Logprob computation failed for {self._group_key(ep)}: {e}")
                results.append((torch.tensor(0.0, device=self.training_device), torch.tensor([], device=self.training_device, dtype=torch.bool)))

        return results

    @staticmethod
    def _episode_diagnostics(episodes) -> dict:
        """Tool-health metrics so reward hacking is visible in wandb.

        A tolerant tool server can silently fix malformed calls and make
        reward look clean; these stats expose the underlying failure mode
        (tool errors, missing 'done', degraded format compliance).
        """
        results = [r for ep in episodes for r in ep.results]
        n_err = sum(1 for r in results if isinstance(r, dict) and r.get("error"))
        raw_rewards = [ep.reward for ep in episodes]
        return {
            "num_episodes_raw": len(episodes),
            "mean_reward_raw": float(np.mean(raw_rewards)) if raw_rewards else 0.0,
            "cgi_injections": float(sum(1 for ep in episodes if ep.is_oracle)),
            "oracle_floor_boost_mean": (
                float(np.mean([
                    float((ep.reward_components or {}).get("oracle_floor_boost", 0.0))
                    for ep in episodes if ep.is_oracle
                ])) if any(ep.is_oracle for ep in episodes) else 0.0
            ),
            "tool_error_rate": n_err / len(results) if results else 0.0,
            "done_rate": (
                sum(1 for ep in episodes
                    if ep.tool_calls and ep.tool_calls[-1].tool == "done")
                / len(episodes) if episodes else 0.0
            ),
            "format_reward_mean": (
                float(np.mean([
                    float((ep.reward_components or {}).get("format_reward", 0.0))
                    for ep in episodes
                ])) if episodes else 0.0
            ),
        }

    def train_step(self, pairs: list, rollout_agent: VLLMRolloutAgent,
                   accum_step: int = 0, is_last_accum: bool = True) -> dict:
        """One GRPO step: collect rollouts, compute advantages, update LoRA.
        
        Gradient accumulation: if is_last_accum is False, gradients are accumulated
        but optimizer.step() is not called. When is_last_accum is True, optimizer
        steps and gradients are zeroed.
        """
        episodes = self.collect_rollouts(pairs, rollout_agent)
        self._last_episodes = episodes  # kept for wandb media logging
        diag = self._episode_diagnostics(episodes)
        # DAPO Dynamic Sampling: filter zero-variance groups
        episodes = self._filter_zero_variance_groups(episodes)
        if not episodes:
            logger.warning("All groups had zero variance after dynamic sampling — skipping.")
            if is_last_accum and self._optimizer is not None:
                # Drop stale accumulated grads — no step ran this window.
                self._optimizer.zero_grad(set_to_none=True)
            return {"num_episodes": 0, "mean_reward": 0.0, "std_reward": 0.0,
                    "mean_advantage": 0.0, "mean_tool_calls": 0.0,
                    "max_reward": 0.0, "min_reward": 0.0, "loss": 0.0, **diag}
        advantages = self.compute_advantages(episodes)

        rewards = [ep.reward for ep in episodes]
        tool_calls = [len(ep.tool_calls) for ep in episodes]

        stats = {
            "num_episodes": len(episodes),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "mean_advantage": float(np.mean(advantages)) if advantages else 0.0,
            "mean_tool_calls": float(np.mean(tool_calls)) if tool_calls else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "min_reward": float(np.min(rewards)) if rewards else 0.0,
            **diag,
        }

        if not episodes or all(ep.reward == 0 for ep in episodes):
            logger.warning("All episodes have zero reward — skipping update.")
            if is_last_accum and self._optimizer is not None:
                self._optimizer.zero_grad(set_to_none=True)
            return stats

        # Token-level GRPO on the LoRA model (same tokenizer both sides).
        # Mixing vLLM token logprobs with Transformers tokens is invalid.
        # NOTE: `old` is the current training model under no-grad — NOT the
        # rollout-time adapter (vLLM serves the adapter saved at last epoch
        # end, while `old` has already taken optimizer steps this epoch).
        # So ratio = pi_now / pi_few-steps-ago, an approximation of the true
        # PPO correction vs the sampling policy; the bias grows with
        # steps-per-epoch. A proper fix would reuse ep.vllm_token_logprobs,
        # but that needs careful vLLM<->HF token alignment.
        self._load_training_model()
        self._model.eval()
        old_results = self.compute_logprobs(episodes, requires_grad=False)
        # Reference (SFT) logprobs for the KL anchor — frozen, no grad.
        ref_results = None
        if self.kl_coef > 0 and self._ref_model is not None:
            ref_results = self.compute_logprobs(
                episodes, requires_grad=False, model=self._ref_model
            )
        self._model.train()

        total_loss = 0.0  # Python float accumulator — avoids holding GPU tensors across episodes
        valid_count = 0

        for i, (ep, adv) in enumerate(zip(episodes, advantages)):
            if not ep.messages or not ep.assistant_responses:
                continue

            old_logps, old_mask = old_results[i]
            if old_logps.ndim == 0 or old_mask.numel() == 0:
                continue

            # Compute the grad-pass logprob for THIS episode only, so at most one
            # computation graph is live. Calling compute_logprobs on the whole
            # batch upfront would hold N forward graphs simultaneously and OOM.
            try:
                new_logps, new_mask = self.compute_logprobs([ep], requires_grad=True)[0]
            except Exception as e:
                logger.warning(f"Logprob (grad) failed for {self._group_key(ep)}: {e}")
                continue
            if new_logps.ndim == 0 or new_mask.numel() == 0:
                continue

            try:
                n = min(new_logps.shape[0], new_mask.shape[0], old_logps.shape[0], old_mask.shape[0])
                mask = new_mask[:n] & old_mask[:n]
                if int(mask.sum()) == 0:
                    continue

                new_tok = new_logps[:n][mask].to(torch.float32)
                old_tok = old_logps[:n][mask].to(torch.float32).detach()

                # μ-GRPO off-support veto: with a negative advantage, do not
                # keep pushing down tokens the policy already finds implausible
                # (old_logp very negative) — that pressure is what shoves the
                # model into narrow degenerate modes.
                keep = None
                if adv < 0 and self.off_support_threshold is not None:
                    keep = old_tok >= self.off_support_threshold
                    if int(keep.sum()) == 0:
                        continue
                    new_tok = new_tok[keep]
                    old_tok = old_tok[keep]

                ratio = torch.exp(new_tok - old_tok)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_low, 1.0 + self.clip_high)
                adv_tensor = torch.tensor(adv, device=ratio.device, dtype=ratio.dtype)
                loss = -torch.min(ratio * adv_tensor, clipped_ratio * adv_tensor).mean()

                # KL(pi_theta || pi_SFT) per token (k3 estimator, >= 0). Anchors
                # the policy to the distribution that writes clean tool JSON so
                # RL cannot drift into degenerate/repetitive modes. The same
                # veto subset is applied to the reference tokens so all three
                # sequences stay aligned.
                if ref_results is not None:
                    ref_logps, ref_mask = ref_results[i]
                    if ref_logps.ndim > 0 and ref_mask.numel() > 0:
                        nr = min(ref_logps.shape[0], ref_mask.shape[0])
                        ref_full = ref_logps[:nr][ref_mask[:nr]].to(torch.float32).detach()
                        if keep is not None:
                            ref_full = ref_full[: keep.shape[0]][keep]
                        k = min(new_tok.shape[0], ref_full.shape[0])
                        if k > 0:
                            kl = torch.exp(ref_full[:k] - new_tok[:k]) - (ref_full[:k] - new_tok[:k]) - 1.0
                            loss = loss + self.kl_coef * kl.mean()

                # Scale by 1/grad_accum for gradient accumulation
                loss = loss / self.grad_accum
                loss.backward()

                total_loss += float(loss.detach().item())
                del new_logps, new_mask, ratio, clipped_ratio, loss
                valid_count += 1
            except Exception as e:
                logger.warning(f"Policy gradient failed for episode {i}: {e}")
                continue

        stats["loss"] = (total_loss / valid_count) if valid_count > 0 else 0.0
        if is_last_accum:
            if valid_count > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self._model.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self._optimizer.step()
                # On-policy rollouts: push the fresh LoRA into vLLM NOW so the
                # next batch samples the policy that just updated — not a
                # one-epoch-stale snapshot. This is what makes the 0.399-style
                # measurement artifact impossible: the measured policy IS the
                # trained policy.
                if self.on_policy_reload:
                    self._push_policy_to_vllm(rollout_agent)
            # Always clear grads at the window boundary — a skipped step must
            # not leak accumulated grads into the next window.
            self._optimizer.zero_grad(set_to_none=True)

        if is_last_accum:
            self._model.eval()
        return stats

    def _push_policy_to_vllm(self, rollout_agent: VLLMRolloutAgent) -> None:
        """Save the current LoRA to fast local storage and hot-reload vLLM.

        Uses a unique adapter name per push + a prefix/encoder cache flush so
        vLLM cannot serve stale KV blocks from a same-name reload (vLLM #42125,
        #44939). Staging on /dev/shm keeps the per-step write off Lustre.
        """
        if self._model is None:
            return
        staging = self.lora_staging_dir or "/dev/shm"
        ckpt_dir = Path(staging) / "agentic-sfm-live"
        try:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            self._model.save_pretrained(str(ckpt_dir))
        except Exception as e:
            logger.warning(f"Per-step LoRA staging failed (staying on last adapter): {e}")
            return
        self._adapter_version += 1
        self._reload_vllm_lora(ckpt_dir, rollout_agent, version=self._adapter_version)

    def _log_qualitative_media(self, step: int, tag: str = "qual") -> None:
        """Render the best episode's match figure + log to wandb.

        Picks the highest-reward episode from the last train_step, builds the
        inlier/outlier correspondence figure, and logs it as a wandb.Image so
        the dashboard shows live match quality during training.
        """
        if not self._wandb:
            return
        eps = getattr(self, "_last_episodes", None) or []
        if not eps:
            return
        try:
            from agentic_sfm.eval.viz import episode_qualitative_media
            import wandb
            # Prefer real policy rollouts — a CGI-injected oracle is a synthetic
            # demo, not evidence of what the policy does.
            policy_eps = [e for e in eps if not getattr(e, "is_oracle", False)]
            best = max(policy_eps or eps, key=lambda e: getattr(e, "reward", 0.0))
            media = episode_qualitative_media(best, prefix=tag)
            if media:
                self._wandb.log({k: wandb.Image(v) for k, v in media.items()},
                                step=step)
        except Exception as e:
            logger.warning(f"qualitative media log failed: {e}")

    def evaluate(self, rollout_agent: VLLMRolloutAgent, max_pairs: int = 50) -> dict:
        if len(self.val_dataset) == 0:
            return {"mean_reward": 0.0, "num_pairs": 0}
        eval_pairs = self.val_dataset.pairs[:max_pairs]
        episodes = []
        for pair in eval_pairs:
            gt_pose = {"R": pair.gt_R.tolist(), "t": pair.gt_t.tolist()} if pair.gt_R is not None else None
            ep = rollout_agent.run_episode(
                pair_id=pair.pair_id, image_a_path=pair.image_a, image_b_path=pair.image_b,
                tool_client=self.tool_client, gt_pose=gt_pose,
                K_a=pair.K_a, K_b=pair.K_b,
            )
            episodes.append(ep)
        rewards = [ep.reward for ep in episodes]
        tool_calls = [len(ep.tool_calls) for ep in episodes]
        # Success = the episode produced real geometry, not merely positive
        # reward (format-only episodes can be > 0).
        success_count = sum(
            1 for ep in episodes
            if ep.final_match and (ep.final_match.get("num_inliers") or 0) > 0
        )
        stats = {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "num_pairs": len(episodes),
            "mean_tool_calls": float(np.mean(tool_calls)) if tool_calls else 0.0,
            "success_rate": success_count / len(episodes) if episodes else 0.0,
        }
        if self._wandb:
            self._wandb.log({f"eval/{k}": v for k, v in stats.items()},
                            step=self._global_step)
        return stats

    def save_lora_checkpoint(self, epoch: int | str, rollout_agent: VLLMRolloutAgent | None = None):
        tag = epoch if isinstance(epoch, str) else f"epoch_{epoch}"
        ckpt_path = self.ckpt_dir / tag
        ckpt_path.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            self._model.save_pretrained(str(ckpt_path))
            logger.info(f"  Saved LoRA checkpoint: {ckpt_path}")
            if rollout_agent is not None:
                self._reload_vllm_lora(ckpt_path, rollout_agent)
        else:
            logger.warning("  No model loaded — skipping checkpoint save.")

    def _reload_vllm_lora(self, ckpt_path: Path, rollout_agent: VLLMRolloutAgent,
                          version: int | None = None) -> None:
        """Hot-load the latest LoRA into vLLM so the next rollouts are on-policy.

        Each load uses a UNIQUE adapter name and flushes the prefix/encoder
        caches afterwards — a same-name reload can reuse stale KV blocks
        (vLLM #42125, #44939) and silently serve the previous policy.
        """
        import urllib.error
        import urllib.request

        if version is None:
            # Every reload needs a fresh name — reusing a name that is already
            # loaded (e.g. the epoch-end "latest" reload right after a per-step
            # push with the same version) makes vLLM 400.
            self._adapter_version += 1
            version = self._adapter_version
        adapter = f"agentic-sfm-policy-v{version}"
        base = rollout_agent.vllm_url.rstrip("/")
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
        }
        path = str(ckpt_path.resolve())

        def _post(route: str, payload: dict, timeout: int = 120) -> None:
            req = urllib.request.Request(
                f"{base}{route}",
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp.read()

        try:
            _post("/v1/load_lora_adapter", {"lora_name": adapter, "lora_path": path})
            rollout_agent.vllm_model = adapter
            rollout_agent._lora_loaded = True
            logger.info(f"vLLM serving LoRA adapter {adapter} from {path}")
        except Exception as e:
            logger.warning(f"vLLM load LoRA failed (rollouts stay on {rollout_agent.vllm_model}): {e}")
            return

        # A reloaded adapter must not reuse KV blocks from the previous policy.
        for route in ("/v1/reset_prefix_cache", "/v1/reset_encoder_cache"):
            try:
                _post(route, {}, timeout=60)
            except Exception as e:
                logger.debug(f"vLLM {route} failed (non-fatal): {e}")

    def train(self):
        """Main GRPO training loop with gradient accumulation."""
        logger.info("Starting GRPO training...")
        rollout_agent = VLLMRolloutAgent(
            vllm_url=self.vllm_url,
            model_name=self.model_name,
            max_new_tokens=self.config["model"].get("max_new_tokens", 512),
            max_tool_calls=self.max_tool_calls,
            temperature=self.temperature,
            top_p=self.top_p,
            min_p=self.min_p,
            top_k=self.top_k,
            pose_weight=self.pose_weight,
            inlier_weight=self.inlier_weight,
            tool_cost=self.tool_cost,
            format_weight=self.format_weight,
            invalid_penalty=self.invalid_penalty,
            reward_schedule=self.reward_schedule,
            reward_warmup_steps=self.reward_warmup_steps,
            matcher=self.matcher,
            accumulative_tool_coef=self.accumulative_tool_coef,
            use_accumulative_tool_reward=self.use_accumulative_tool_reward,
            ntep_intent_coef=self.ntep_intent_coef,
            ntep_redundancy_penalty=self.ntep_redundancy_penalty,
            use_ntep_rewards=self.use_ntep_rewards,
        )
        if self.sft_adapter and os.path.isdir(self.sft_adapter):
            self._reload_vllm_lora(Path(self.sft_adapter), rollout_agent)

        global_step = 0

        for epoch in range(self.total_epochs):
            difficulties = self.get_curriculum_difficulties(epoch)
            train_pairs = [p for p in self.train_dataset.pairs if p.difficulty in difficulties]

            if not train_pairs:
                logger.warning(f"No training pairs for difficulties {difficulties}")
                continue

            logger.info(f"\nEpoch {epoch+1}/{self.total_epochs} | "
                        f"Difficulties: {difficulties} | "
                        f"Pairs: {len(train_pairs)}")

            rng = np.random.default_rng(epoch)
            rng.shuffle(train_pairs)

            batch_size = self.config["rollout"]["batch_size"]
            epoch_stats = []
            micro_step = 0

            for batch_start in range(0, len(train_pairs), batch_size):
                batch = train_pairs[batch_start:batch_start + batch_size]
                micro_step += 1
                is_last_accum = (micro_step % self.grad_accum == 0) or (batch_start + batch_size >= len(train_pairs))

                stats = self.train_step(batch, rollout_agent, 
                                        accum_step=micro_step, is_last_accum=is_last_accum)
                epoch_stats.append(stats)

                if is_last_accum:
                    global_step += 1
                    self._global_step = global_step
                    rollout_agent._global_step = global_step

                    if global_step % self.config["training"]["log_freq"] == 0:
                        logger.info(
                            f"  Step {global_step}: "
                            f"reward={stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}, "
                            f"tool_calls={stats['mean_tool_calls']:.1f}, "
                            f"loss={stats.get('loss', 0.0):.4f}"
                        )
                        if self._wandb:
                            self._wandb.log({
                                "train/reward_mean": stats["mean_reward"],
                                "train/reward_std": stats["std_reward"],
                                "train/reward_max": stats["max_reward"],
                                "train/reward_min": stats["min_reward"],
                                "train/tool_calls_mean": stats["mean_tool_calls"],
                                "train/tool_error_rate": stats.get("tool_error_rate", 0.0),
                                "train/done_rate": stats.get("done_rate", 0.0),
                                "train/format_reward_mean": stats.get("format_reward_mean", 0.0),
                                "train/num_episodes": stats.get("num_episodes", 0),
                                "train/num_episodes_raw": stats.get("num_episodes_raw", 0),
                                "train/mean_reward_raw": stats.get("mean_reward_raw", 0.0),
                                "train/mean_advantage": stats.get("mean_advantage", 0.0),
                                "train/cgi_injections": stats.get("cgi_injections", 0.0),
                                "train/oracle_floor_boost_mean": stats.get("oracle_floor_boost_mean", 0.0),
                                "train/loss": stats.get("loss", 0.0),
                                "train/epoch": epoch + 1,
                                "train/global_step": global_step,
                            }, step=global_step)
                            self._log_qualitative_media(global_step, tag="train")

            # Gate `best` on the PRE-filter reward mean — post-filter stats
            # exclude failed groups and are inflated exactly when the policy
            # is degenerating.
            mean_reward = np.mean(
                [s.get("mean_reward_raw", s["mean_reward"]) for s in epoch_stats]
            ) if epoch_stats else 0.0
            logger.info(f"Epoch {epoch+1} mean reward: {mean_reward:.3f}")

            if (epoch + 1) % self.eval_freq == 0:
                eval_stats = self.evaluate(rollout_agent)
                logger.info(f"  Eval: reward={eval_stats['mean_reward']:.3f}, "
                          f"success_rate={eval_stats.get('success_rate', 0.0):.3f}")
                if eval_stats["mean_reward"] > self._best_eval_reward:
                    self._best_eval_reward = eval_stats["mean_reward"]
                    self.save_lora_checkpoint("best_eval")
                    logger.info(f"  New best eval reward {eval_stats['mean_reward']:.3f} -> checkpoints/best_eval")

            # Always dump `latest` so vLLM rollouts track the LoRA policy.
            self.save_lora_checkpoint("latest", rollout_agent)
            if (epoch + 1) % self.save_freq == 0:
                self.save_lora_checkpoint(epoch + 1, rollout_agent)

            # `latest` tracks the *current* policy (needed for on-policy
            # rollouts) — a regressed epoch will overwrite it. `best` (train,
            # raw mean) and `best_eval` (held-out verifier) are the
            # quality-gated resume points.
            if mean_reward > self._best_epoch_reward:
                self._best_epoch_reward = mean_reward
                self.save_lora_checkpoint("best")
                logger.info(f"  New best epoch mean reward {mean_reward:.3f} -> checkpoints/best")

        self.save_lora_checkpoint(self.total_epochs, rollout_agent)
        if self._wandb:
            self._wandb.finish()
        logger.info("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: GRPO RL training")
    parser.add_argument("--config", type=str, default="configs/phase1_grpo.yaml")
    parser.add_argument("--tool-server-url", type=str, default="http://localhost:8765")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000")
    parser.add_argument("--output-dir", type=str, default="outputs/phase1")
    args = parser.parse_args()

    config = load_config(args.config)
    os.chdir(Path(args.config).parent.parent)

    from agentic_sfm.constants import assert_qwen35_runtime

    assert_qwen35_runtime()

    tool_client = ToolClient(args.tool_server_url)
    try:
        health = tool_client.health()
        logger.info(f"Tool server: {health}")
    except Exception as e:
        logger.error(f"Cannot connect to tool server: {e}")
        return

    trainer = GRPOTrainer(
        config=config,
        tool_client=tool_client,
        vllm_url=args.vllm_url,
        output_dir=args.output_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
