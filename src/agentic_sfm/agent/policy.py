"""MLLM agent: Qwen 4B VLM policy for agentic SfM tool orchestration."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

from agentic_sfm.constants import DEFAULT_MATCHER, DEFAULT_POLICY_MODEL, assert_qwen35_runtime
from agentic_sfm.geometry import crop_image_id, crop_pil_from_result, keep_best_match


def load_policy_processor_and_model(
    model_name: str,
    *,
    torch_dtype=None,
    device_map=None,
    quantization_config=None,
):
    """Load Qwen3-VL-2B-Instruct processor + VLM (AutoModelForImageTextToText)."""
    import torch
    from transformers import AutoProcessor

    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    if model_name == DEFAULT_POLICY_MODEL or "Qwen3-VL" in model_name:
        assert_qwen35_runtime()

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype, "trust_remote_code": True}
    if device_map is not None:
        kwargs["device_map"] = device_map
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    try:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load policy {model_name}. Qwen3-VL-2B-Instruct needs Transformers 5.x "
            f"(model_type qwen3_vl) and vLLM >= 0.17. Original error: {e}"
        ) from e
    return processor, model

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an agentic 3D reconstruction assistant. Given two images, produce the \
best matching result by calling tools.

Available tools:
1. crop(image_id, bbox) — Crop to [x1,y1,x2,y2] normalized 0-1. Use this to zoom \
   into the shared overlap. The observation returns cropped_image_id; use that id \
   in the next match call.
2. match(image_a, image_b, matcher) — Match two image ids. Matcher: "mast3r", \
   "loftr", "lightglue". Returns correspondences, inliers, and relative pose.
3. crop_and_match(image_id, bbox, image_b, matcher) — Crop one image then match \
   against the other in one step. Prefer this when you already know the overlap box.
4. doppelganger_check(image_a, image_b) — True match vs look-alike (doppelganger).

Strategy:
- If overlap is small, crop (or crop_and_match) first, then match the cropped ids.
- If a match has few inliers, try a different crop or matcher. Keep going until \
  inliers improve or you have tried 2-3 regions.
- Call match on the original img_a/img_b if overlap looks large.
- Minimize tool calls — each call has a cost.

Output exactly one JSON object per turn:
{"tool": "crop", "args": {"image_id": "img_a", "bbox": [0.2, 0.3, 0.8, 0.9]}}
{"tool": "match", "args": {"image_a": "img_a", "image_b": "img_b", "matcher": "loftr"}}
{"tool": "crop_and_match", "args": {"image_id": "img_a", "bbox": [0.1, 0.2, 0.9, 0.8], "image_b": "img_b", "matcher": "loftr"}}
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


def apply_policy_chat_template(processor, messages, **kwargs):
    """Chat template with thinking disabled; drop the kwarg if unsupported."""
    kwargs.setdefault("enable_thinking", False)
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return processor.apply_chat_template(messages, **kwargs)


def parse_tool_call(text: str) -> ToolCall | None:
    """Parse a JSON tool call from model output text."""
    if not text:
        return None
    stripped = text.strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, dict) and "tool" in data:
            return ToolCall(tool=data["tool"], args=data.get("args") or {})
    except (json.JSONDecodeError, TypeError):
        pass

    brace_depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start >= 0:
                try:
                    data = json.loads(text[start : i + 1])
                    if isinstance(data, dict) and "tool" in data:
                        return ToolCall(tool=data["tool"], args=data.get("args") or {})
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass
                start = -1

    for candidate in reversed(re.findall(r'\{[^{}]*"tool"[^{}]*\}', text, re.DOTALL)):
        try:
            data = json.loads(candidate)
            if "tool" in data:
                return ToolCall(tool=data["tool"], args=data.get("args") or {})
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def format_observation(result: dict[str, Any]) -> str:
    """Format a tool result as a text observation for the MLLM."""
    if "error" in result:
        return f"Error: {result['error']}"

    parts = []
    cid = crop_image_id(result)
    if cid:
        parts.append(f"cropped_image_id: {cid} (use this id in match)")
    if "crop_size" in result:
        parts.append(f"crop_size: {result['crop_size']}")
    if "size" in result:
        parts.append(f"size: {result['size']}")
    if "num_matches" in result:
        parts.append(f"Matches: {result['num_matches']}")
    if "num_inliers" in result:
        parts.append(f"Inliers: {result['num_inliers']}")
    if "inlier_ratio" in result:
        parts.append(f"Inlier ratio: {result['inlier_ratio']:.3f}")
    if result.get("pose") is not None:
        parts.append("Pose: estimated")
    if "is_doppelganger" in result:
        score = result.get("score", 0.0)
        parts.append(f"Doppelganger: {result['is_doppelganger']} (score: {score:.3f})")
    return " | ".join(parts) if parts else json.dumps(result)


def execute_sfm_tool(
    tool_client: Any,
    tc: ToolCall,
    match_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch crop / match / crop_and_match / doppelganger to the tool server."""
    match_kwargs = match_kwargs or {}
    args = tc.args
    if tc.tool == "crop":
        return tool_client.crop(args["image_id"], args["bbox"])
    if tc.tool == "match":
        return tool_client.match(
            args.get("image_a", "img_a"),
            args.get("image_b", "img_b"),
            args.get("matcher", match_kwargs.get("matcher", DEFAULT_MATCHER)),
            **{k: v for k, v in match_kwargs.items() if k != "matcher"},
        )
    if tc.tool == "crop_and_match":
        crop_res = tool_client.crop(args.get("image_id", "img_a"), args["bbox"])
        cid = crop_image_id(crop_res) or args.get("image_id", "img_a")
        match_res = tool_client.match(
            cid,
            args.get("image_b", "img_b"),
            args.get("matcher", match_kwargs.get("matcher", DEFAULT_MATCHER)),
            **{k: v for k, v in match_kwargs.items() if k != "matcher"},
        )
        match_res["crop"] = crop_res
        if crop_image_id(crop_res):
            match_res["cropped_image_id"] = crop_image_id(crop_res)
        return match_res
    if tc.tool == "doppelganger_check":
        return tool_client.doppelganger_check(
            args.get("image_a", "img_a"), args.get("image_b", "img_b")
        )
    return {"error": f"Unknown tool: {tc.tool}"}


def _collect_pil_images(messages: list[dict[str, Any]]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "image" and isinstance(item.get("image"), Image.Image):
                images.append(item["image"])
    return images


class AgenticSfMAgent:
    """Qwen 4B VLM agent for agentic SfM.

    Phase 0 (zero-shot): stock model with tool-calling prompt.
    Phase 1 (RL): LoRA-tuned model trained with GRPO.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_POLICY_MODEL,
        device: str = "cuda",
        max_new_tokens: int = 512,
        max_tool_calls: int = 10,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.95,
        matcher: str = DEFAULT_MATCHER,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_tool_calls = max_tool_calls
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.matcher = matcher
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy load the model and processor."""
        if self._model is not None:
            return

        logger.info(f"Loading {self.model_name}...")
        self._processor, self._model = load_policy_processor_and_model(
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
        if images is None:
            images = _collect_pil_images(messages)

        text = apply_policy_chat_template(
            self._processor,
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        proc_kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt", "padding": True}
        if images:
            proc_kwargs["images"] = images
        inputs = self._processor(**proc_kwargs).to(self.device)

        gen_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if self.do_sample:
            gen_kwargs.update(
                do_sample=True, temperature=self.temperature, top_p=self.top_p
            )
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        return self._processor.decode(output_ids[0][input_len:], skip_special_tokens=True)

    def _execute_tool(self, tool_client, tc: ToolCall, match_kwargs: dict[str, Any]) -> dict[str, Any]:
        return execute_sfm_tool(tool_client, tc, match_kwargs)

    def run_episode(
        self,
        pair_id: str,
        image_a_path: str,
        image_b_path: str,
        tool_client,
        gt_pose: dict[str, Any] | None = None,
        K_a: list | None = None,
        K_b: list | None = None,
    ) -> Episode:
        """Run a full episode: agent calls tools until done or max calls."""
        from agentic_sfm.rewards.pose_rewards import compute_pair_reward

        episode = Episode(pair_id=pair_id, image_a=image_a_path, image_b=image_b_path)
        tool_client.register_image("img_a", image_a_path)
        tool_client.register_image("img_b", image_b_path)

        img_a = Image.open(image_a_path).convert("RGB")
        img_b = Image.open(image_b_path).convert("RGB")
        match_kwargs: dict[str, Any] = {"matcher": self.matcher}
        if K_a is not None:
            match_kwargs["K_a"] = K_a
        if K_b is not None:
            match_kwargs["K_b"] = K_b

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_a},
                    {"type": "image", "image": img_b},
                    {
                        "type": "text",
                        "text": 'Match these two images. Call tools to achieve the best matching result, then output {"tool": "done"}.',
                    },
                ],
            },
        ]

        num_invalid = 0
        for step in range(self.max_tool_calls):
            response = self.generate(messages)
            logger.info(f"[{pair_id}] Step {step}: {response[:200]}")

            tc = parse_tool_call(response)
            if tc is None:
                num_invalid += 1
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": 'Please call a tool using JSON format: {"tool": "...", "args": {...}}',
                })
                continue

            episode.tool_calls.append(tc)
            if tc.tool == "done":
                break

            try:
                result = self._execute_tool(tool_client, tc, match_kwargs)
            except Exception as e:
                result = {"error": str(e)}

            episode.results.append(result)
            if tc.tool in ("match", "crop_and_match") or result.get("pose") is not None:
                episode.final_match = keep_best_match(episode.final_match, result)

            messages.append({"role": "assistant", "content": response})
            obs_text = f"Observation: {format_observation(result)}"
            cid = crop_image_id(result)
            crop_content: list[dict[str, Any]] = [{"type": "text", "text": obs_text}]
            crop_im = crop_pil_from_result(result)
            if crop_im is not None:
                crop_content.insert(0, {"type": "image", "image": crop_im})
            if cid and "cropped_image_id" not in obs_text:
                crop_content[-1]["text"] += f" Use id {cid} in match."
            messages.append({"role": "user", "content": crop_content})

        num_valid = len(episode.tool_calls)
        episode.reward_components = compute_pair_reward(
            episode.final_match or {},
            gt_pose=gt_pose,
            num_tool_calls=len(episode.tool_calls) + num_invalid,
            num_invalid_calls=num_invalid,
            num_valid_calls=num_valid,
        )
        episode.reward = episode.reward_components["total_reward"]

        return episode
