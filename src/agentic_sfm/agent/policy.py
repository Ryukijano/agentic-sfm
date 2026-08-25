"""MLLM agent: Qwen3-VL policy for agentic SfM tool orchestration.

The agent receives image pairs, decides which tools to call (crop, match,
doppelganger_check), and produces final match results. Trained with GRPO
via verl in Phase 1.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# System prompt for the agentic SfM agent
SYSTEM_PROMPT = """\
You are an agentic 3D reconstruction assistant. Given two images, your goal is to \
produce the best possible image matching result by calling tools.

Available tools:
1. crop(image_id, bbox) — Crop image to [x1,y1,x2,y2] (normalized 0-1). Use this to \
   zoom into regions of overlap when the images have little shared content.
2. match(image_a, image_b, matcher) — Match two images. Matcher options: "mast3r", \
   "loftr", "lightglue". Returns correspondences, inlier count, and relative pose.
3. doppelganger_check(image_a, image_b) — Check if two visually similar images are \
   actually the same scene (true match) or doppelgangers (distinct but similar).

Strategy:
- If the images have little overlap, crop to the shared region first, then match.
- If matching fails (few inliers), try cropping to different regions.
- If images look very similar but matching fails, check for doppelgangers.
- You can call match directly if images have good overlap.
- Minimize tool calls — each call has a cost.

Output format: Call a tool by outputting JSON:
{"tool": "crop", "args": {"image_id": "img_a", "bbox": [0.2, 0.3, 0.8, 0.9]}}
{"tool": "match", "args": {"image_a": "img_a", "image_b": "img_b", "matcher": "mast3r"}}

When you are satisfied with the result, output:
{"tool": "done", "args": {}}
"""


@dataclass
class ToolCall:
    """A single tool call from the agent."""

    tool: str
    args: dict[str, Any]


@dataclass
class Episode:
    """A full episode of agent-tool interaction."""

    pair_id: str
    image_a: str
    image_b: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    final_match: dict[str, Any] | None = None
    reward: float = 0.0
    reward_components: dict[str, Any] = field(default_factory=dict)


def parse_tool_call(text: str) -> ToolCall | None:
    """Parse a JSON tool call from model output text."""
    # Find JSON tool call in the text — try multiple patterns
    # Pattern 1: standalone JSON object with "tool" key
    # Pattern 2: JSON embedded in text (greedy brace matching)
    candidates = re.findall(r'\{.*?"tool".*?\}', text, re.DOTALL)
    for candidate in reversed(candidates):  # try longest match first
        try:
            data = json.loads(candidate)
            if "tool" in data:
                return ToolCall(tool=data["tool"], args=data.get("args", {}))
        except (json.JSONDecodeError, KeyError):
            continue
    # Fallback: try to find a balanced JSON object
    brace_depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and start >= 0:
                try:
                    data = json.loads(text[start:i+1])
                    if "tool" in data:
                        return ToolCall(tool=data["tool"], args=data.get("args", {}))
                except (json.JSONDecodeError, KeyError):
                    pass
                start = -1
    return None


def format_observation(result: dict[str, Any]) -> str:
    """Format a tool result as a text observation for the MLLM."""
    if "error" in result:
        return f"Error: {result['error']}"

    parts = []
    if "num_matches" in result:
        parts.append(f"Matches: {result['num_matches']}")
    if "num_inliers" in result:
        parts.append(f"Inliers: {result['num_inliers']}")
    if "inlier_ratio" in result:
        parts.append(f"Inlier ratio: {result['inlier_ratio']:.3f}")
    if "is_doppelganger" in result:
        score = result.get("score", 0.0)
        parts.append(f"Doppelganger: {result['is_doppelganger']} (score: {score:.3f})")
    if "cropped_image_id" in result:
        parts.append(f"Cropped image: {result['cropped_image_id']}")

    return " | ".join(parts) if parts else json.dumps(result)


class AgenticSfMAgent:
    """Qwen3-VL agent for agentic SfM.

    In Phase 0 (zero-shot): uses stock model with tool-calling prompt.
    In Phase 1 (RL): LoRA-tuned model trained with GRPO.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 512,
        max_tool_calls: int = 10,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_tool_calls = max_tool_calls
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy load the model and processor."""
        if self._model is not None:
            return

        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info(f"Loading {self.model_name}...")
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self._model.eval()

    def generate(
        self,
        messages: list[dict[str, Any]],
        images: list[Image.Image] | None = None,
    ) -> str:
        """Generate a response from the MLLM."""
        self._load_model()

        # Prepare inputs
        if images:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text], images=images, return_tensors="pt", padding=True
            ).to(self.device)
        else:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text], return_tensors="pt", padding=True
            ).to(self.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        # Decode only new tokens
        input_len = inputs["input_ids"].shape[1]
        response = self._processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )
        return response

    def run_episode(
        self,
        pair_id: str,
        image_a_path: str,
        image_b_path: str,
        tool_client,
        gt_pose: dict[str, Any] | None = None,
    ) -> Episode:
        """Run a full episode: agent calls tools until done or max calls."""
        from agentic_sfm.rewards.pose_rewards import compute_pair_reward

        episode = Episode(pair_id=pair_id, image_a=image_a_path, image_b=image_b_path)

        # Register images with tool server
        tool_client.register_image("img_a", image_a_path)
        tool_client.register_image("img_b", image_b_path)

        # Build conversation
        img_a = Image.open(image_a_path).convert("RGB")
        img_b = Image.open(image_b_path).convert("RGB")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_a},
                    {"type": "image", "image": img_b},
                    {
                        "type": "text",
                        "text": "Match these two images. Call tools to achieve the best matching result, then output {\"tool\": \"done\"}.",
                    },
                ],
            },
        ]

        for step in range(self.max_tool_calls):
            # Generate tool call
            response = self.generate(messages, images=[img_a, img_b])
            logger.info(f"[{pair_id}] Step {step}: {response[:200]}")

            tc = parse_tool_call(response)
            if tc is None:
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "Please call a tool using JSON format: {\"tool\": \"...\", \"args\": {...}}",
                })
                continue

            episode.tool_calls.append(tc)

            if tc.tool == "done":
                break

            # Execute tool call
            try:
                if tc.tool == "crop":
                    result = tool_client.crop(tc.args["image_id"], tc.args["bbox"])
                elif tc.tool == "match":
                    result = tool_client.match(
                        tc.args["image_a"],
                        tc.args["image_b"],
                        tc.args.get("matcher", "mast3r"),
                    )
                    episode.final_match = result
                elif tc.tool == "doppelganger_check":
                    result = tool_client.doppelganger_check(
                        tc.args["image_a"], tc.args["image_b"]
                    )
                else:
                    result = {"error": f"Unknown tool: {tc.tool}"}
            except Exception as e:
                result = {"error": str(e)}

            episode.results.append(result)

            # Add observation to conversation
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Observation: {format_observation(result)}",
            })

        # Compute reward
        if episode.final_match:
            episode.reward_components = compute_pair_reward(
                episode.final_match,
                gt_pose=gt_pose,
                num_tool_calls=len(episode.tool_calls),
            )
            episode.reward = episode.reward_components["total_reward"]

        return episode
