#!/usr/bin/env python
"""Custom verl multi-turn agent loop for Agentic-SFM hard image pair matching.

This agent loop interacts with the existing tool server (MASt3R / LoFTR) and
computes a reward based on the final match quality and pose accuracy.  It is
registered with verl's agent loop registry via the hydra ``_target_`` mechanism
in ``configs/verl/agent_loop.yaml``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

import httpx
import numpy as np
import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics
from verl.utils.chat_template import extract_system_prompt_and_generation

from agentic_sfm.agent.policy import parse_tool_call, format_observation, ToolCall
from agentic_sfm.rewards.pose_rewards import compute_pair_reward

logger = logging.getLogger(__name__)


class AsyncToolClient:
    """Async HTTP client for the tool server."""

    def __init__(self, base_url: str = "http://localhost:8765", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def _get(self, path: str) -> dict[str, Any]:
        r = await self.client.get(path)
        r.raise_for_status()
        return r.json()

    async def _post(self, path: str, json_data: dict[str, Any]) -> dict[str, Any]:
        r = await self.client.post(path, json=json_data)
        r.raise_for_status()
        return r.json()

    async def health(self) -> dict[str, Any]:
        return await self._get("/health")

    async def register_image(self, image_id: str, path: str) -> dict[str, Any]:
        return await self._post("/register_image", {"image_id": image_id, "path": path})

    async def crop(self, image_id: str, bbox: list[float]) -> dict[str, Any]:
        return await self._post("/crop", {"image_id": image_id, "bbox": bbox})

    async def match(
        self,
        image_a: str,
        image_b: str,
        matcher: str = "mast3r",
        max_size: int = 512,
    ) -> dict[str, Any]:
        return await self._post(
            "/match",
            {"image_a": image_a, "image_b": image_b, "matcher": matcher, "max_size": max_size},
        )

    async def doppelganger_check(self, image_a: str, image_b: str) -> dict[str, Any]:
        return await self._post("/doppelganger_check", {"image_a": image_a, "image_b": image_b})

    async def retrieve(self, query_image: str, k: int = 5) -> dict[str, Any]:
        return await self._post("/retrieve", {"query_image": query_image, "k": k})

    async def sfm_run(
        self,
        image_dir: str,
        pair_list: list[tuple[str, str]] | None = None,
        output_dir: str = "./outputs/sfm_run",
    ) -> dict[str, Any]:
        return await self._post(
            "/sfm_run",
            {"image_dir": image_dir, "pair_list": pair_list, "output_dir": output_dir},
        )

    async def inspect(self, recon_dir: str) -> dict[str, Any]:
        return await self._post("/inspect", {"recon_dir": recon_dir})

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class AgenticSfmAgentLoop(AgentLoopBase):
    """Multi-turn tool-calling agent loop for image pair matching.

    The loop repeatedly samples an assistant response from the vLLM rollout
    server, parses the tool call, executes it against the tool server, appends
    the formatted observation, and terminates when the model emits
    ``{\"tool\": \"done\"}`` or the turn limit is reached.  A reward is computed
    from the final match result and ground-truth pose.
    """

    def __init__(
        self,
        trainer_config: Any,
        server_manager: Any,
        tokenizer: Any,
        processor: Any,
        dataset_cls: Any,
        data_config: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(trainer_config, server_manager, tokenizer, processor, dataset_cls, data_config)

        self.tool_server_url: str = kwargs.get("tool_server_url", "http://localhost:8765")
        self.max_tool_calls: int = kwargs.get("max_tool_calls", 10)
        rc = kwargs.get("reward_config") or {}
        self.reward_config: dict[str, Any] = dict(rc) if not isinstance(rc, dict) else rc
        self.matcher: str = kwargs.get("matcher", "mast3r")
        self.match_max_size: int = kwargs.get("match_max_size", 512)

        # Generate the "assistant" generation prompt tokens once so we can
        # insert them between observation messages without re-tokenising the
        # whole conversation each turn.
        processing_class = processor if processor is not None else tokenizer
        try:
            _, gen_prompt_ids = extract_system_prompt_and_generation(
                processing_class, **self.apply_chat_template_kwargs
            )
            self._gen_prompt_ids: list[int] = [int(x) for x in gen_prompt_ids]
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to extract assistant generation prompt ids: %s", exc)
            self._gen_prompt_ids = []

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        extra_info: dict[str, Any] = kwargs.get("extra_info") or {}
        raw_prompt: list[dict[str, Any]] | None = kwargs.get("raw_prompt")
        if raw_prompt is None:
            raise ValueError("raw_prompt is required for AgenticSfmAgentLoop")

        # Build a mutable copy of the conversation.  Images are already placed
        # in the user message content by ``RLHFDataset._build_messages``.
        messages: list[dict[str, Any]] = [dict(m) for m in raw_prompt]

        multi_modal_data = await self.process_multi_modal_info(messages)
        images: Optional[list[Image.Image]] = multi_modal_data.get("images")
        videos: Optional[list[Any]] = multi_modal_data.get("videos")
        audios: Optional[list[Any]] = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        pair_id: str = extra_info.get("pair_id", kwargs.get("index", "unknown"))
        image_a: str | None = extra_info.get("image_a")
        image_b: str | None = extra_info.get("image_b")

        gt_pose: Optional[dict[str, Any]] = None
        if "gt_R" in extra_info and "gt_t" in extra_info:
            gt_pose = {"R": extra_info["gt_R"], "t": extra_info["gt_t"]}

        tool_client = AsyncToolClient(self.tool_server_url)
        try:
            if image_a:
                await tool_client.register_image("img_a", image_a)
            if image_b:
                await tool_client.register_image("img_b", image_b)
        except Exception as exc:
            logger.warning("[%s] Failed to register images: %s", pair_id, exc)

        # Tokenise the initial prompt (system + user + generation prompt).
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
            remove_system_prompt=False,
        )

        # ``all_ids`` tracks the full token sequence fed to the model.  It starts
        # with the initial prompt and grows with assistant tokens, observation
        # tokens, and inter-turn generation prompts.
        all_ids: list[int] = list(prompt_ids)
        response_mask: list[int] = []

        final_match_result: dict[str, Any] = {}
        num_tool_calls = 0
        num_valid_calls = 0
        num_invalid_calls = 0
        terminated = False

        start_time = time.time()
        for turn in range(self.max_tool_calls):
            try:
                output = await self.server_manager.generate(
                    request_id=f"{pair_id}_{turn}",
                    prompt_ids=all_ids,
                    sampling_params=sampling_params,
                    image_data=images,
                    video_data=videos,
                    audio_data=audios,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
            except Exception as exc:
                logger.error("[%s] LLM generation failed at turn %d: %s", pair_id, turn, exc)
                break

            assistant_ids = list(output.token_ids)
            all_ids += assistant_ids
            response_mask += [1] * len(assistant_ids)

            # Decode and parse the tool call.
            assistant_text = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(assistant_ids, skip_special_tokens=True),
            )
            tc = parse_tool_call(assistant_text)
            num_tool_calls += 1

            if tc is None:
                num_invalid_calls += 1
                logger.warning("[%s] Turn %d produced invalid tool call: %r", pair_id, turn, assistant_text)
                terminated = True
                break

            if tc.tool == "done":
                terminated = True
                break

            # Execute the tool and store the result.
            try:
                result = await self._execute_tool(tool_client, tc)
            except Exception as exc:
                logger.warning("[%s] Tool execution failed for %s: %s", pair_id, tc.tool, exc)
                result = {"error": str(exc)}

            if tc.tool in ("match", "crop_and_match"):
                final_match_result = result
            elif "match_result" in result:
                final_match_result = result["match_result"]

            num_valid_calls += 1

            # Format observation as a user message and tokenise it.  We strip
            # the system prompt because the conversation history already
            # contains it; we do not add a generation prompt here either.
            obs_text = format_observation(result)
            obs_message = {"role": "user", "content": f"Observation: {obs_text}"}
            try:
                obs_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.encode(
                        f"Observation: {obs_text}", add_special_tokens=False
                    ),
                )
            except Exception as exc:
                logger.warning("[%s] Failed to tokenise observation: %s", pair_id, exc)
                obs_ids = []

            all_ids += list(obs_ids)
            response_mask += [0] * len(obs_ids)

            # Add the assistant generation prompt for the next turn.
            if self._gen_prompt_ids:
                all_ids += self._gen_prompt_ids
                response_mask += [0] * len(self._gen_prompt_ids)

        generate_time = time.time() - start_time

        # Compute reward from the final match result and the ground-truth pose.
        reward_components = compute_pair_reward(
            final_match_result,
            gt_pose=gt_pose,
            num_tool_calls=num_tool_calls,
            num_invalid_calls=num_invalid_calls,
            num_valid_calls=num_valid_calls,
            **self.reward_config,
        )
        total_reward = float(reward_components.get("total_reward", 0.0))

        response_ids = all_ids[len(prompt_ids):]
        response_length = self.rollout_config.get("response_length", len(response_ids))
        if len(response_ids) > response_length:
            response_ids = response_ids[:response_length]
            response_mask = response_mask[:response_length]

        # Clean numeric reward components for ``extra_fields``.
        extra_reward_info = {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in reward_components.items()
            if isinstance(v, (int, float, np.floating))
        }

        await tool_client.close()

        return AgentLoopOutput(
            prompt_ids=list(prompt_ids),
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=None,
            multi_modal_data=multi_modal_data,
            reward_score=total_reward,
            num_turns=num_tool_calls,
            metrics=AgentLoopMetrics(
                generate_sequences=generate_time,
                tool_calls=float(num_tool_calls),
            ),
            extra_fields={"reward_extra_info": extra_reward_info},
            mm_processor_kwargs=mm_processor_kwargs,
        )

    async def _execute_tool(self, tool_client: AsyncToolClient, tc: ToolCall) -> dict[str, Any]:
        """Dispatch a parsed tool call to the tool server."""
        args = tc.args

        if tc.tool == "match":
            image_a = args.get("image_a", "img_a")
            image_b = args.get("image_b", "img_b")
            matcher = args.get("matcher", self.matcher)
            max_size = args.get("max_size", self.match_max_size)
            return await tool_client.match(image_a, image_b, matcher=matcher, max_size=max_size)

        if tc.tool == "crop":
            image_id = args.get("image_id", "img_a")
            bbox = args.get("bbox")
            if bbox is None:
                raise ValueError("crop requires 'bbox'")
            return await tool_client.crop(image_id, bbox)

        if tc.tool == "crop_and_match":
            # Convenience compound operation: crop img_a and then match against img_b.
            crop_image_id = args.get("image_id", "img_a")
            bbox = args.get("bbox")
            image_b = args.get("image_b", "img_b")
            if bbox is None:
                raise ValueError("crop_and_match requires 'bbox'")
            crop_res = await tool_client.crop(crop_image_id, bbox)
            cropped_id = crop_res.get("cropped_image_id", crop_image_id)
            matcher = args.get("matcher", self.matcher)
            max_size = args.get("max_size", self.match_max_size)
            return await tool_client.match(cropped_id, image_b, matcher=matcher, max_size=max_size)

        if tc.tool == "doppelganger_check":
            image_a = args.get("image_a", "img_a")
            image_b = args.get("image_b", "img_b")
            return await tool_client.doppelganger_check(image_a, image_b)

        if tc.tool == "retrieve":
            query_image = args.get("query_image", "img_a")
            k = args.get("k", 5)
            return await tool_client.retrieve(query_image, k=k)

        if tc.tool == "sfm_run":
            image_dir = args.get("image_dir")
            pair_list = args.get("pair_list")
            output_dir = args.get("output_dir", "./outputs/sfm_run")
            if image_dir is None:
                raise ValueError("sfm_run requires 'image_dir'")
            return await tool_client.sfm_run(image_dir, pair_list=pair_list, output_dir=output_dir)

        if tc.tool == "inspect":
            recon_dir = args.get("recon_dir")
            if recon_dir is None:
                raise ValueError("inspect requires 'recon_dir'")
            return await tool_client.inspect(recon_dir)

        raise ValueError(f"Unknown tool: {tc.tool}")
