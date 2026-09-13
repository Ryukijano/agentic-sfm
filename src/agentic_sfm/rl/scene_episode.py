"""Scene-level SfM episode runner for Phase 2.

Each episode takes a scene (set of images) and lets the agent orchestrate the
full reconstruction pipeline:
  1. retrieve   — select which image pairs to match
  2. match / crop_and_match — match selected pairs
  3. doppelganger_check — filter visually-similar but wrong pairs
  4. sfm_run    — run COLMAP incremental reconstruction
  5. inspect    — check reconstruction quality
  6. done       — end episode

The scene-level reward is computed by ``compute_scene_reward`` from
``rewards/pose_rewards.py`` (registration + pose + doppelganger + accumulative
tool + NTEP).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agentic_sfm.agent.policy import (
    ToolCall,
    execute_sfm_tool,
    format_observation,
    parse_tool_call,
)
from agentic_sfm.constants import DEFAULT_MATCHER
from agentic_sfm.geometry import keep_best_match
from agentic_sfm.rewards.pose_rewards import compute_scene_reward
from agentic_sfm.tools.client import ToolClient

logger = logging.getLogger(__name__)

# System prompt for scene-level episodes
SCENE_SYSTEM_PROMPT = """You are an SfM reconstruction agent. You are given a set of images from a scene.
Your goal is to reconstruct the scene by matching image pairs and running COLMAP.

Available tools:
- retrieve: find candidate image pairs to match
- match: match a specific image pair
- crop_and_match: crop one image then match to the other
- doppelganger_check: check if a pair is a doppelganger (looks similar but different scene)
- sfm_run: run COLMAP reconstruction on registered images
- inspect: inspect the current reconstruction
- done: end the episode

Output format: {"tool": "<name>", "args": {...}}

Strategy:
1. Call retrieve to find good candidate pairs
2. For each pair, call match or crop_and_match
3. If a pair looks suspicious, call doppelganger_check
4. After matching enough pairs, call sfm_run
5. Call inspect to check the result
6. Call done when satisfied"""


@dataclass
class SceneRolloutEpisode:
    """A scene-level rollout episode."""

    scene_id: str = ""
    image_paths: list[str] = field(default_factory=list)
    num_images: int = 0

    # Trajectory
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    assistant_responses: list[str] = field(default_factory=list)

    # Pair-level matches collected during the episode
    pair_matches: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Final reconstruction result
    recon_result: dict[str, Any] = field(default_factory=dict)
    final_match: dict[str, Any] = field(default_factory=dict)

    # Reward
    reward: float = 0.0
    reward_components: dict[str, Any] = field(default_factory=dict)

    # For training
    images: list[str] = field(default_factory=list)  # base64 images for vLLM
    done: bool = False

    # Ground truth
    gt_recon: dict[str, Any] | None = None


def run_scene_episode(
    agent: Any,
    scene_id: str,
    image_paths: list[str],
    tool_client: ToolClient,
    gt_recon: dict[str, Any] | None = None,
    max_tool_calls: int = 20,
    max_turns: int = 30,
    image_root: str = "",
) -> SceneRolloutEpisode:
    """Run a scene-level SfM episode.

    The agent sees all scene images and orchestrates the full reconstruction
    pipeline via tool calls. Returns a SceneRolloutEpisode with the trajectory
    and scene-level reward.
    """
    ep = SceneRolloutEpisode(
        scene_id=scene_id,
        image_paths=image_paths,
        num_images=len(image_paths),
        gt_recon=gt_recon,
    )

    # Register all images with the tool server
    registered_ids: dict[str, str] = {}  # path -> server_id
    for i, path in enumerate(image_paths):
        img_id = f"img_{i:04d}"
        try:
            tool_client.register_image(img_id, str(Path(image_root) / path) if image_root else path)
            registered_ids[path] = img_id
        except Exception as e:
            logger.warning(f"Failed to register {path}: {e}")

    # Build initial message with scene images (first N as thumbnails)
    n_show = min(len(image_paths), 8)  # show up to 8 images
    ep.images = []
    content: list[dict[str, Any]] = []
    for i in range(n_show):
        path = image_paths[i]
        full_path = str(Path(image_root) / path) if image_root else path
        try:
            b64 = agent._encode_image(full_path)
            ep.images.append(b64)
            content.append({"type": "image", "image": f"placeholder_{i}"})
        except Exception:
            pass
    content.append({
        "type": "text",
        "text": (
            f"Scene {scene_id} has {len(image_paths)} images. "
            f"Showing {n_show} thumbnails. "
            f"Registered as img_0000 through img_{len(image_paths)-1:04d}. "
            f"Use retrieve to find good pairs, then match them, then run sfm_run."
        ),
    })

    ep.messages = [
        {"role": "system", "content": SCENE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    # Scene-level rollout loop
    num_tool_calls = 0
    num_invalid_calls = 0
    num_valid_calls = 0
    terminated = False

    for turn in range(max_turns):
        if terminated:
            break

        # Generate via vLLM
        try:
            response = agent._generate_turn(ep.messages, ep.images)
        except Exception as e:
            logger.error(f"[{scene_id}] Generation failed at turn {turn}: {e}")
            break

        ep.assistant_responses.append(response)
        ep.messages.append({"role": "assistant", "content": response})

        tc = parse_tool_call(response)
        if tc is None:
            num_invalid_calls += 1
            num_tool_calls += 1
            ep.messages.append({
                "role": "user",
                "content": 'Please call a tool: {"tool": "...", "args": {...}}',
            })
            continue

        num_tool_calls += 1
        ep.tool_calls.append(tc)

        if tc.tool == "done":
            terminated = True
            ep.done = True
            continue

        # Execute the tool
        result = _execute_scene_tool(tool_client, tc, registered_ids, ep)
        ep.results.append(result)
        num_valid_calls += 1

        # Track pair-level matches for keep_best_match
        if tc.tool in ("match", "crop_and_match") and not result.get("error"):
            pair_key = f"{tc.args.get('image_a', '?')}__{tc.args.get('image_b', tc.args.get('image_id', '?'))}"
            ep.pair_matches[pair_key] = result
            kept = keep_best_match(ep.final_match or None, result)
            if kept:
                ep.final_match = kept

        # Track reconstruction result
        if tc.tool == "sfm_run":
            ep.recon_result = result

        # Format observation
        obs_text = format_observation(result)
        ep.messages.append({"role": "user", "content": f"Observation: {obs_text}"})

    # Compute scene-level reward
    agent_reward_cfg = getattr(agent, "reward_config", {})
    ep.reward_components = compute_scene_reward(
        ep.recon_result or {},
        gt_recon=gt_recon,
        num_tool_calls=num_tool_calls,
        tool_cost=agent_reward_cfg.get("tool_cost", 0.05),
        registration_weight=agent_reward_cfg.get("registration_weight", 0.5),
        pose_weight=agent_reward_cfg.get("pose_weight", 1.0),
        split_penalty=agent_reward_cfg.get("split_penalty", 0.5),
        doppelganger_weight=agent_reward_cfg.get("doppelganger_weight", 0.3),
        accumulative_tool_coef=agent_reward_cfg.get("accumulative_tool_coef", 0.1),
        use_accumulative_tool_reward=agent_reward_cfg.get("use_accumulative_tool_reward", True),
        num_valid_calls=num_valid_calls,
        tool_calls=ep.tool_calls,
        tool_results=ep.results,
        ntep_intent_coef=agent_reward_cfg.get("ntep_intent_coef", 0.05),
        ntep_redundancy_penalty=agent_reward_cfg.get("ntep_redundancy_penalty", 0.05),
        use_ntep_rewards=agent_reward_cfg.get("use_ntep_rewards", False),
    )
    ep.reward = ep.reward_components["total_reward"]

    logger.info(
        f"[{scene_id}] Episode done: {num_tool_calls} calls, "
        f"reward={ep.reward:.3f}, registered={ep.recon_result.get('num_registered', 0)}"
    )
    return ep


def run_scene_oracle_episode(
    agent: Any,
    scene_id: str,
    image_paths: list[str],
    tool_client: ToolClient,
    gt_recon: dict[str, Any] | None = None,
    overlap_matrix: np.ndarray | None = None,
    image_indices: list[int] | None = None,
    failed_group_max_reward: float = 0.0,
    image_root: str = "",
) -> SceneRolloutEpisode:
    """S-GRPO CGI for scene-level episodes.

    When all rollouts in a scene group fail to produce a reconstruction,
    generate an oracle trajectory: select top-overlap pairs from the GT
    overlap matrix, match them, run sfm_run, and inject the result.

    ``overlap_matrix`` is the full N×N GT overlap matrix from scene_info.
    ``image_indices`` maps image_paths indices to the full scene's indices.
    ``failed_group_max_reward`` is the highest reward among failed rollouts.
    """
    ep = SceneRolloutEpisode(
        scene_id=scene_id,
        image_paths=image_paths,
        num_images=len(image_paths),
        gt_recon=gt_recon,
    )

    # Register images
    registered_ids: dict[str, str] = {}
    for i, path in enumerate(image_paths):
        img_id = f"img_{i:04d}"
        try:
            tool_client.register_image(img_id, str(Path(image_root) / path) if image_root else path)
            registered_ids[path] = img_id
        except Exception:
            pass

    # Encode images for the prompt
    n_show = min(len(image_paths), 8)
    ep.images = []
    for i in range(n_show):
        try:
            full_path = str(Path(image_root) / image_paths[i]) if image_root else image_paths[i]
            ep.images.append(agent._encode_image(full_path))
        except Exception:
            pass

    # Build oracle trajectory: retrieve → match top pairs → sfm_run → done
    oracle_calls: list[ToolCall] = []
    oracle_responses: list[str] = []
    oracle_results: list[dict[str, Any]] = []
    best_match = None
    recon_result: dict[str, Any] = {}

    # Step 1: retrieve (select top-overlap pairs)
    ids = list(registered_ids.values())
    if overlap_matrix is not None and image_indices is not None:
        # Use GT overlap to pick best pairs
        sub_om = overlap_matrix[np.ix_(image_indices, image_indices)]
        pair_scores = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pair_scores.append((sub_om[i, j], ids[i], ids[j]))
        pair_scores.sort(reverse=True)
        top_pairs = pair_scores[:10]  # top 10 pairs
        pairs_str = ", ".join(f"({p[1]}, {p[2]}) overlap={p[0]:.2f}" for p in top_pairs[:5])
        oracle_response = json.dumps({"tool": "retrieve", "args": {}})
        oracle_calls.append(ToolCall(tool="retrieve", args={}))
        oracle_responses.append(oracle_response)
        oracle_results.append({"pairs": [{"image_a": p[1], "image_b": p[2], "score": float(p[0])} for p in top_pairs]})
    else:
        # Fallback: use consecutive pairs
        top_pairs = [(1.0, ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
        oracle_response = json.dumps({"tool": "retrieve", "args": {}})
        oracle_calls.append(ToolCall(tool="retrieve", args={}))
        oracle_responses.append(oracle_response)
        oracle_results.append({"pairs": [{"image_a": p[1], "image_b": p[2], "score": p[0]} for p in top_pairs]})

    # Step 2: match top pairs (up to 5)
    for _, img_a_id, img_b_id in top_pairs[:5]:
        tc = ToolCall(tool="match", args={
            "image_a": img_a_id,
            "image_b": img_b_id,
            "matcher": getattr(agent, "matcher", DEFAULT_MATCHER),
        })
        oracle_calls.append(tc)
        response = json.dumps({"tool": "match", "args": tc.args})
        oracle_responses.append(response)
        try:
            result = execute_sfm_tool(tool_client, tc, {"matcher": getattr(agent, "matcher", DEFAULT_MATCHER)})
            oracle_results.append(result)
            if not result.get("error"):
                kept = keep_best_match(best_match, result)
                if kept:
                    best_match = kept
        except Exception as e:
            oracle_results.append({"error": str(e)})

    # Step 3: sfm_run
    oracle_calls.append(ToolCall(tool="sfm_run", args={}))
    oracle_responses.append(json.dumps({"tool": "sfm_run", "args": {}}))
    try:
        image_dir = str(Path(image_paths[0]).parent) if image_paths else ""
        recon_result = tool_client.sfm_run(image_dir=image_dir)
        oracle_results.append(recon_result)
    except Exception as e:
        recon_result = {"error": str(e), "num_registered": 0}
        oracle_results.append(recon_result)

    # Step 4: done
    oracle_calls.append(ToolCall(tool="done", args={}))
    oracle_responses.append(json.dumps({"tool": "done", "args": {}}))

    # Build messages
    ep.tool_calls = oracle_calls
    ep.assistant_responses = oracle_responses
    ep.results = oracle_results
    ep.final_match = best_match
    ep.recon_result = recon_result
    ep.done = True

    # Build message list (simplified — just the key turns)
    content: list[dict[str, Any]] = [{"type": "text", "text": f"Scene {scene_id}: {len(image_paths)} images"}]
    for i in range(min(len(ep.images), 4)):
        content.insert(i, {"type": "image", "image": f"placeholder_{i}"})
    ep.messages = [
        {"role": "system", "content": SCENE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
        {"role": "assistant", "content": oracle_responses[0]},  # retrieve
        {"role": "user", "content": f"Found {len(top_pairs)} candidate pairs"},
        {"role": "assistant", "content": oracle_responses[1]},  # first match
        {"role": "user", "content": f"Observation: {format_observation(oracle_results[1])}"},
        {"role": "assistant", "content": oracle_responses[-2]},  # sfm_run
        {"role": "user", "content": f"Observation: {format_observation(recon_result)}"},
        {"role": "assistant", "content": oracle_responses[-1]},  # done
    ]

    # Compute reward
    agent_reward_cfg = getattr(agent, "reward_config", {})
    n_valid = len(oracle_calls) - 1  # exclude done
    ep.reward_components = compute_scene_reward(
        recon_result, gt_recon=gt_recon,
        num_tool_calls=len(oracle_calls), num_valid_calls=n_valid,
        tool_cost=agent_reward_cfg.get("tool_cost", 0.05),
        registration_weight=agent_reward_cfg.get("registration_weight", 0.5),
        pose_weight=agent_reward_cfg.get("pose_weight", 1.0),
        split_penalty=agent_reward_cfg.get("split_penalty", 0.5),
        doppelganger_weight=agent_reward_cfg.get("doppelganger_weight", 0.3),
        accumulative_tool_coef=agent_reward_cfg.get("accumulative_tool_coef", 0.1),
        use_accumulative_tool_reward=agent_reward_cfg.get("use_accumulative_tool_reward", True),
        tool_calls=oracle_calls,
        tool_results=oracle_results,
        ntep_intent_coef=agent_reward_cfg.get("ntep_intent_coef", 0.05),
        ntep_redundancy_penalty=agent_reward_cfg.get("ntep_redundancy_penalty", 0.05),
        use_ntep_rewards=agent_reward_cfg.get("use_ntep_rewards", False),
    )
    ep.reward = ep.reward_components["total_reward"]

    # Guarantee oracle reward > failed group max
    if ep.reward <= failed_group_max_reward:
        ep.reward = failed_group_max_reward + 0.1
        ep.reward_components["oracle_floor_boost"] = ep.reward - ep.reward_components["total_reward"]

    return ep


def _execute_scene_tool(
    tool_client: ToolClient,
    tc: ToolCall,
    registered_ids: dict[str, str],
    ep: SceneRolloutEpisode,
) -> dict[str, Any]:
    """Execute a scene-level tool call."""
    args = tc.args

    if tc.tool == "retrieve":
        # Simple retrieval: return all pairs sorted by overlap (if GT available)
        # In practice, this would use a retrieval model (e.g., NetVLAD, DINO)
        pairs = []
        ids = list(registered_ids.values())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append({
                    "image_a": ids[i],
                    "image_b": ids[j],
                    "score": 1.0 - abs(i - j) / max(len(ids), 1),  # proximity heuristic
                })
        pairs.sort(key=lambda p: p["score"], reverse=True)
        return {"pairs": pairs[:20], "num_pairs": len(pairs[:20])}

    if tc.tool == "sfm_run":
        image_dir = str(Path(ep.image_paths[0]).parent) if ep.image_paths else ""
        return tool_client.sfm_run(image_dir=image_dir)

    if tc.tool == "inspect":
        recon_dir = args.get("recon_dir", ep.recon_result.get("output_dir", ""))
        if not recon_dir:
            return {"error": "No reconstruction to inspect"}
        return tool_client.inspect(recon_dir)

    # Pair-level tools: delegate to execute_sfm_tool
    return execute_sfm_tool(tool_client, tc)
