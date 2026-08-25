#!/usr/bin/env python
"""Phase 1: GRPO RL training for agentic matching.

Trains Qwen3-VL-8B with LoRA using GRPO (Group Relative Policy Optimization)
to learn crop/match tool-calling policies for hard image pairs.

Architecture:
  - vLLM server (GPU 0): fast rollout sampling for episode generation
  - Training model (GPU 1): LoRA-adapted Qwen3-VL for policy gradient updates
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
    Episode,
    SYSTEM_PROMPT,
    ToolCall,
    format_observation,
    parse_tool_call,
)
from agentic_sfm.data.hard_pairs import HardPairDataset
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


class VLLMRolloutAgent:
    """Agent that uses vLLM server for fast rollout sampling."""

    def __init__(self, vllm_url: str, model_name: str, max_new_tokens: int = 512,
                 max_tool_calls: int = 10, temperature: float = 1.0, top_p: float = 0.95,
                 pose_weight: float = 1.0, inlier_weight: float = 0.1,
                 tool_cost: float = 0.02, format_weight: float = 0.1,
                 invalid_penalty: float = 0.2, reward_schedule: str = "static",
                 reward_warmup_steps: int = 30):
        self.vllm_url = vllm_url.rstrip("/")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_tool_calls = max_tool_calls
        self.temperature = temperature
        self.top_p = top_p
        # Reward config (needed in run_episode)
        self.pose_weight = pose_weight
        self.inlier_weight = inlier_weight
        self.tool_cost = tool_cost
        self.format_weight = format_weight
        self.invalid_penalty = invalid_penalty
        self.reward_schedule = reward_schedule
        self.reward_warmup_steps = reward_warmup_steps
        self._global_step = 0

    def _encode_image(self, image_path: str) -> str:
        img = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

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
                        img_idx += 1
                oai_messages.append({"role": msg["role"], "content": parts})
            else:
                oai_messages.append(msg)

        try:
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=oai_messages,
                max_tokens=self.max_new_tokens,
                temperature=temp,
                top_p=self.top_p,
                n=n,
                logprobs=True,
            )
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
                    tool_client: ToolClient, gt_pose: dict | None = None) -> RolloutEpisode:
        ep = RolloutEpisode(pair_id=pair_id, image_a=image_a_path, image_b=image_b_path)

        tool_client.register_image("img_a", image_a_path)
        tool_client.register_image("img_b", image_b_path)

        img_a_b64 = self._encode_image(image_a_path)
        img_b_b64 = self._encode_image(image_b_path)
        ep.images = [img_a_b64, img_b_b64]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": "placeholder_a"},
                {"type": "image", "image": "placeholder_b"},
                {"type": "text", "text": "Match these two images. Call tools to achieve the best matching result, then output {\"tool\": \"done\"}."},
            ]},
        ]
        ep.messages = messages

        for step in range(self.max_tool_calls):
            texts, token_lps = self._vllm_chat(messages, [img_a_b64, img_b_b64], n=1)
            response = texts[0]
            logger.info(f"[{pair_id}] Step {step}: {response[:200]}")

            ep.assistant_responses.append(response)
            if token_lps and token_lps[0] is not None:
                ep.vllm_token_logprobs.append(token_lps[0])
            else:
                ep.vllm_token_logprobs.append([])

            tc = parse_tool_call(response)
            if tc is None:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "Please call a tool using JSON format: {\"tool\": \"...\", \"args\": {...}}"})
                continue

            ep.tool_calls.append(tc)
            if tc.tool == "done":
                messages.append({"role": "assistant", "content": response})
                break

            try:
                if tc.tool == "crop":
                    result = tool_client.crop(tc.args["image_id"], tc.args["bbox"])
                elif tc.tool == "match":
                    result = tool_client.match(tc.args["image_a"], tc.args["image_b"], tc.args.get("matcher", "loftr"))
                    ep.final_match = result
                elif tc.tool == "doppelganger_check":
                    result = tool_client.doppelganger_check(tc.args["image_a"], tc.args["image_b"])
                else:
                    result = {"error": f"Unknown tool: {tc.tool}"}
            except Exception as e:
                result = {"error": str(e)}

            ep.results.append(result)
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {format_observation(result)}"})

        ep.messages = messages

        if ep.final_match:
            # Count valid vs invalid tool calls
            num_valid = len(ep.tool_calls)
            num_invalid = sum(1 for r in ep.results if "error" in r and "Unknown tool" not in str(r.get("error", "")))
            # Apply dynamic reward scaling if configured
            pose_w = self.pose_weight
            format_w = self.format_weight
            if self.reward_schedule == "dynamic" and self._global_step < self.reward_warmup_steps:
                # Early training: emphasize format, de-emphasize correctness
                pose_w = pose_w * (1.0 / 3.0)
                format_w = format_w * 1.0
            elif self.reward_schedule == "dynamic":
                # Later training: full correctness, reduced format
                format_w = format_w * 0.5

            ep.reward_components = compute_pair_reward(
                ep.final_match, gt_pose=gt_pose,
                num_tool_calls=len(ep.tool_calls),
                num_invalid_calls=num_invalid,
                num_valid_calls=num_valid,
                tool_cost=self.tool_cost,
                inlier_weight=self.inlier_weight,
                pose_weight=pose_w,
                format_weight=format_w,
                invalid_penalty=self.invalid_penalty,
            )
            ep.reward = ep.reward_components["total_reward"]

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
        self._global_step = 0

        self.curriculum_stages = config.get("curriculum", {}).get("stages", [])

        train_path = config["data"].get("train_pairs", "data/hard_pairs_train.json")
        val_path = config["data"].get("val_pairs", "data/hard_pairs_val.json")
        self.train_dataset = HardPairDataset.load(train_path) if os.path.exists(train_path) else HardPairDataset()
        self.val_dataset = HardPairDataset.load(val_path) if os.path.exists(val_path) else HardPairDataset()
        logger.info(f"Train: {len(self.train_dataset)} pairs | Val: {len(self.val_dataset)} pairs")

        self.model_name = config["model"]["name"]
        self.lora_config = config["model"]["lora"]
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
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info(f"Loading training model {self.model_name} on {self.training_device}...")
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.training_device,
        )

        lora_cfg = LoraConfig(
            r=self.lora_config["rank"],
            lora_alpha=self.lora_config["alpha"],
            lora_dropout=self.lora_config["dropout"],
            target_modules=self.lora_config["target_modules"],
            task_type="CAUSAL_LM",
        )
        self._model = get_peft_model(self._model, lora_cfg)
        self._model.print_trainable_parameters()

        trainable = [p for p in self._model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=self.config["training"]["weight_decay"])
        logger.info("Training model ready with LoRA.")

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
        episodes = []
        for pair in pairs:
            for _ in range(self.group_size):
                gt_pose = {"R": pair.gt_R.tolist(), "t": pair.gt_t.tolist()} if pair.gt_R is not None else None
                ep = rollout_agent.run_episode(
                    pair_id=pair.pair_id,
                    image_a_path=pair.image_a,
                    image_b_path=pair.image_b,
                    tool_client=self.tool_client,
                    gt_pose=gt_pose,
                )
                episodes.append(ep)
        return episodes

    def compute_advantages(self, episodes: list[RolloutEpisode]) -> list[float]:
        groups: dict[str, list[RolloutEpisode]] = {}
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

    def _filter_zero_variance_groups(self, episodes: list[RolloutEpisode]) -> list[RolloutEpisode]:
        """DAPO Dynamic Sampling: remove groups where all rollouts have identical reward.
        These produce zero advantage = zero gradient, wasting compute.
        """
        if not self.dynamic_sampling:
            return episodes
        groups: dict[str, list[RolloutEpisode]] = {}
        for ep in episodes:
            groups.setdefault(ep.pair_id, []).append(ep)
        filtered = []
        skipped = 0
        for pair_id, group in groups.items():
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
        full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        full_ids = processor(text=[full_text], images=images, return_tensors="pt")["input_ids"][0]
        n_tokens = len(full_ids)
        mask = [False] * n_tokens

        for k, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            try:
                prefix_text = processor.apply_chat_template(
                    messages[:k], tokenize=False, add_generation_prompt=True
                )
                prefix_ids = processor(text=[prefix_text], images=images, return_tensors="pt")["input_ids"][0]
                start = len(prefix_ids)

                through_text = processor.apply_chat_template(
                    messages[:k+1], tokenize=False, add_generation_prompt=False
                )
                through_ids = processor(text=[through_text], images=images, return_tensors="pt")["input_ids"][0]
                end = len(through_ids)

                for i in range(start, min(end, n_tokens)):
                    mask[i] = True
            except Exception as e:
                logger.warning(f"Mask computation failed for msg {k}: {e}")

        return mask

    def compute_logprobs(self, episodes: list[RolloutEpisode], 
                          requires_grad: bool = False) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Compute per-episode (token_logps, assistant_mask) under current LoRA policy.
        
        Returns list of (per_token_logps_tensor, mask_bool_tensor) for each episode.
        Only assistant token positions are masked True.
        """
        self._load_training_model()
        results = []

        for ep in episodes:
            if not ep.messages:
                results.append((torch.tensor(0.0, device=self.training_device), torch.tensor([], device=self.training_device, dtype=torch.bool)))
                continue

            try:
                text = self._processor.apply_chat_template(
                    ep.messages, tokenize=False, add_generation_prompt=False
                )
                images = [Image.open(ep.image_a).convert("RGB"), Image.open(ep.image_b).convert("RGB")]
                inputs = self._processor(
                    text=[text], images=images, return_tensors="pt", padding=True
                ).to(self.training_device)

                mask = self._compute_assistant_mask(ep.messages, images, self._processor)
                mask_tensor = torch.tensor(mask, device=self.training_device, dtype=torch.bool)

                ctx_manager = torch.enable_grad() if requires_grad else torch.no_grad()
                with ctx_manager:
                    outputs = self._model(**inputs)

                logits = outputs.logits[:, :-1, :]
                target_ids = inputs["input_ids"][:, 1:]
                token_logps = F.log_softmax(logits, dim=-1)
                per_token_logps = token_logps.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # [1, seq_len-1]

                # Align mask: mask[i] corresponds to input_ids[i], logps[i] predicts input_ids[i+1]
                # So logps at position i predicts token i+1; we need mask shifted by 1
                assistant_mask = mask_tensor[1:].to(per_token_logps.device)  # [seq_len-1]

                results.append((per_token_logps.squeeze(0), assistant_mask))
            except Exception as e:
                logger.warning(f"Logprob computation failed for {ep.pair_id}: {e}")
                results.append((torch.tensor(0.0, device=self.training_device), torch.tensor([], device=self.training_device, dtype=torch.bool)))

        return results

    def train_step(self, pairs: list, rollout_agent: VLLMRolloutAgent,
                   accum_step: int = 0, is_last_accum: bool = True) -> dict:
        """One GRPO step: collect rollouts, compute advantages, update LoRA.
        
        Gradient accumulation: if is_last_accum is False, gradients are accumulated
        but optimizer.step() is not called. When is_last_accum is True, optimizer
        steps and gradients are zeroed.
        """
        episodes = self.collect_rollouts(pairs, rollout_agent)
        # DAPO Dynamic Sampling: filter zero-variance groups
        episodes = self._filter_zero_variance_groups(episodes)
        if not episodes:
            logger.warning("All groups had zero variance after dynamic sampling — skipping.")
            return {"num_episodes": 0, "mean_reward": 0.0, "std_reward": 0.0,
                    "mean_tool_calls": 0.0, "loss": 0.0}
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
        }

        if not episodes or all(ep.reward == 0 for ep in episodes):
            logger.warning("All episodes have zero reward — skipping update.")
            return stats

        self._load_training_model()
        self._model.train()

        # Compute old log-probs from vLLM (stored during rollout)
        # Fall back to computing from model if vLLM logprobs unavailable
        old_logps_data = []
        for ep in episodes:
            if ep.vllm_token_logprobs:
                # Sum per-token logprobs across all assistant turns
                total = sum(sum(turn_lps) for turn_lps in ep.vllm_token_logprobs if turn_lps)
                old_logps_data.append(total)
            else:
                old_logps_data.append(None)

        # Compute current policy log-probs with gradient
        logprob_results = self.compute_logprobs(episodes, requires_grad=True)

        total_loss = torch.tensor(0.0, device=self.training_device, dtype=torch.float32)
        valid_count = 0

        for i, (ep, adv) in enumerate(zip(episodes, advantages)):
            if not ep.messages or not ep.assistant_responses:
                continue

            token_logps, assistant_mask = logprob_results[i]
            if assistant_mask.numel() == 0 or assistant_mask.sum() == 0:
                continue

            try:
                # Sum log-probs only over assistant tokens
                masked_logps = token_logps[assistant_mask]
                new_logp = masked_logps.sum()

                # Old log-prob: from vLLM or from non-grad model pass
                if old_logps_data[i] is not None:
                    old_logp = torch.tensor(old_logps_data[i], device=self.training_device, dtype=torch.float32)
                else:
                    # Fallback: compute without grad (will be ~same as new for first step)
                    with torch.no_grad():
                        old_logp = masked_logps.detach().sum()

                # DAPO-style Clip-Higher: asymmetric clipping
                ratio = torch.exp(new_logp - old_logp)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_low, 1.0 + self.clip_high)
                adv_tensor = torch.tensor(adv, device=self.training_device, dtype=torch.float32)
                loss = -torch.min(ratio * adv_tensor, clipped_ratio * adv_tensor)

                # Scale by 1/grad_accum for gradient accumulation
                loss = loss / self.grad_accum
                loss.backward()

                total_loss = total_loss + loss.detach()
                valid_count += 1
            except Exception as e:
                logger.warning(f"Policy gradient failed for episode {i}: {e}")
                continue

        if valid_count > 0 and is_last_accum:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self._model.parameters() if p.requires_grad],
                self.max_grad_norm,
            )
            self._optimizer.step()
            self._optimizer.zero_grad()
            stats["loss"] = (total_loss / valid_count).item()
        elif valid_count > 0:
            stats["loss"] = (total_loss / valid_count).item()
        else:
            stats["loss"] = 0.0

        if is_last_accum:
            self._model.eval()
        return stats

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
            )
            episodes.append(ep)
        rewards = [ep.reward for ep in episodes]
        tool_calls = [len(ep.tool_calls) for ep in episodes]
        # Count successful matches (non-zero reward)
        success_count = sum(1 for r in rewards if r > 0)
        stats = {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "num_pairs": len(episodes),
            "mean_tool_calls": float(np.mean(tool_calls)) if tool_calls else 0.0,
            "success_rate": success_count / len(episodes) if episodes else 0.0,
        }
        if self._wandb:
            self._wandb.log({f"eval/{k}": v for k, v in stats.items()})
        return stats

    def save_lora_checkpoint(self, epoch: int):
        ckpt_path = self.ckpt_dir / f"epoch_{epoch}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            self._model.save_pretrained(str(ckpt_path))
            logger.info(f"  Saved LoRA checkpoint: {ckpt_path}")
        else:
            logger.warning("  No model loaded — skipping checkpoint save.")

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
            pose_weight=self.pose_weight,
            inlier_weight=self.inlier_weight,
            tool_cost=self.tool_cost,
            format_weight=self.format_weight,
            invalid_penalty=self.invalid_penalty,
            reward_schedule=self.reward_schedule,
            reward_warmup_steps=self.reward_warmup_steps,
        )

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
                                "train/loss": stats.get("loss", 0.0),
                                "train/epoch": epoch + 1,
                                "train/global_step": global_step,
                            })

            mean_reward = np.mean([s["mean_reward"] for s in epoch_stats]) if epoch_stats else 0.0
            logger.info(f"Epoch {epoch+1} mean reward: {mean_reward:.3f}")

            if (epoch + 1) % self.eval_freq == 0:
                eval_stats = self.evaluate(rollout_agent)
                logger.info(f"  Eval: reward={eval_stats['mean_reward']:.3f}, "
                          f"success_rate={eval_stats.get('success_rate', 0.0):.3f}")

            if (epoch + 1) % self.save_freq == 0:
                self.save_lora_checkpoint(epoch + 1)

        self.save_lora_checkpoint(self.total_epochs)
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
