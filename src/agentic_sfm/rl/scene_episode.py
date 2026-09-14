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
import re
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

# Confidence at/above which a doppelganger_check flags a pair.
DOPPELGANGER_CONFIDENCE_THRESHOLD = 0.5

# Crop ids look like "<image_id>_crop_<x1>_<y1>_<x2>_<y2>".
_CROP_SUFFIX_RE = re.compile(r"_crop_\d+_\d+_\d+_\d+$")


def _base_image_id(image_id: str) -> str:
    """Strip a server crop suffix so crop ids map back to their parent."""
    return _CROP_SUFFIX_RE.sub("", str(image_id))


def _pair_key(image_a: str, image_b: str) -> str:
    """Canonical pair key: sorted base ids joined with '__'."""
    return "__".join(sorted((_base_image_id(image_a), _base_image_id(image_b))))


def _is_flagged_doppelganger(result: dict[str, Any]) -> bool:
    """True when a doppelganger_check result flags the pair.

    Uses the calibrated ``confidence`` when present (new response format)
    and falls back to the raw ``is_doppelganger`` bool.
    """
    if not isinstance(result, dict) or result.get("error"):
        return False
    conf = result.get("confidence", result.get("score"))
    if conf is not None:
        try:
            return float(conf) >= DOPPELGANGER_CONFIDENCE_THRESHOLD
        except (TypeError, ValueError):
            pass
    return bool(result.get("is_doppelganger"))


def _sfm_pair_list(
    ep: SceneRolloutEpisode, registered_ids: dict[str, str]
) -> tuple[list[list[str]] | None, int, int]:
    """Build the COLMAP pair list for sfm_run minus flagged doppelgangers.

    Returns ``(pair_list, n_doppelgangers_present, n_doppelgangers_filtered)``.
    When the agent never called ``doppelganger_check`` the pair list is
    ``None`` so the server falls back to exhaustive matching (unchanged
    behaviour).  Otherwise all registered-image pairs are passed except
    flagged doppelgangers — i.e. "exhaustive minus doppelgangers".
    """
    if not ep.doppelganger_checks:
        return None, 0, 0

    flagged = {
        key for key, res in ep.doppelganger_checks.items()
        if _is_flagged_doppelganger(res)
    }
    ids = sorted(set(registered_ids.values()))
    pair_list: list[list[str]] = []
    n_filtered = 0
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if _pair_key(a, b) in flagged:
                n_filtered += 1
                continue
            pair_list.append([a, b])
    return pair_list, len(flagged), n_filtered


def _inject_pose_error(
    ep: SceneRolloutEpisode, gt_recon: dict[str, Any] | None
) -> None:
    """Compute ``mean_pose_error_deg`` from the persisted COLMAP model and
    inject it into ``ep.recon_result`` so ``compute_scene_reward`` can score
    pose accuracy.

    The tool server cannot compute this itself — it has no GT poses — so we
    load the reconstruction it wrote to ``output_dir``, align it to the GT
    cam-from-world poses (Sim(3) on camera centers with a rotational gauge
    fix), and record the mean geodesic rotation error.  When fewer than 3
    images are matched we fall back to the gauge-free mean pairwise relative
    rotation error.  Best-effort: any failure leaves the field absent.
    """
    recon = ep.recon_result
    if not isinstance(recon, dict) or recon.get("error"):
        return
    if recon.get("mean_pose_error_deg") is not None:
        return
    if not gt_recon or not gt_recon.get("poses"):
        return
    output_dir = recon.get("output_dir")
    if not output_dir:
        return

    try:
        from agentic_sfm.eval.scene_eval import (
            _camera_center,
            _extract_gt_poses,
            _index_lookup,
            _poses_from_colmap_dir,
            _resolve_image_index,
            _rot_geodesic_deg,
            _rotation_gauge,
            _umeyama,
            _apply_sim3_to_pose,
        )

        lookup = _index_lookup(ep)
        named = _poses_from_colmap_dir(str(output_dir))
        pred_poses: dict[int, np.ndarray] = {}
        for name, M in named.items():
            idx = _resolve_image_index(name, lookup)
            if idx is not None:
                pred_poses[idx] = M
        gt_poses = _extract_gt_poses(gt_recon, lookup)
        matched = sorted(set(pred_poses) & set(gt_poses))
        if not matched:
            return

        if len(matched) >= 3:
            W = _rotation_gauge(pred_poses, gt_poses, matched)
            pred_w = {
                i: _apply_sim3_to_pose(pred_poses[i], 1.0, W, np.zeros(3))
                for i in matched
            }
            src = np.stack([_camera_center(pred_w[i]) for i in matched])
            dst = np.stack([_camera_center(gt_poses[i]) for i in matched])
            s, R_res, ts = _umeyama(src, dst)
            Rs = R_res @ W
            errs = [
                _rot_geodesic_deg(
                    _apply_sim3_to_pose(pred_poses[i], s, Rs, ts)[:3, :3],
                    gt_poses[i][:3, :3],
                )
                for i in matched
            ]
        else:
            # Gauge-free fallback: mean pairwise relative rotation error.
            errs = []
            for a_i, i in enumerate(matched):
                for j in matched[a_i + 1:]:
                    R_rel_p = pred_poses[j][:3, :3] @ pred_poses[i][:3, :3].T
                    R_rel_g = gt_poses[j][:3, :3] @ gt_poses[i][:3, :3].T
                    errs.append(_rot_geodesic_deg(R_rel_p, R_rel_g))

        if errs:
            recon["mean_pose_error_deg"] = float(np.mean(errs))
            recon["num_poses_evaluated"] = len(matched)
    except Exception as e:
        logger.debug(f"Pose-error injection skipped: {e}")


# System prompt for scene-level episodes
SCENE_SYSTEM_PROMPT = """\
You are an SfM agent. Reconstruct the scene from its images (img_0000, \
img_0001, ...). Output exactly one JSON tool call per turn:
{"tool": "name", "args": {...}}

Tools:
- retrieve {"top_k": 20} -> candidate pairs [{"image_a","image_b","score"}]
- match {"image_a","image_b","matcher":"loftr"} -> num_inliers, inlier_ratio, \
pose
- crop {"image_id","bbox":[x1,y1,x2,y2]} -> cropped_image_id; bbox coords 0-1
- crop_and_match {"image_id","bbox":[...],"image_b","matcher"} -> crop + match \
in one step; use when overlap is small
- doppelganger_check {"image_a","image_b"} -> is_doppelganger, confidence, \
inlier_ratio; detects look-alike pairs of a different place
- sfm_run {} -> COLMAP reconstruction: num_registered, num_points3d, \
mean_reproj_error
- inspect {} -> stats of the last reconstruction
- done {} -> end episode

Workflow: retrieve once; match the best 5-10 pairs (crop_and_match for small \
overlap); doppelganger_check suspicious pairs — similar look but low \
inlier_ratio — BEFORE sfm_run (flagged pairs are filtered out); then sfm_run \
and inspect; if few images registered, match more and rerun; done when most \
are registered.

Reward: higher for more registered images, accurate poses, and filtered \
doppelgangers. sfm_run with too few or bad matches scores low. Do not repeat \
calls.

Examples:
{"tool": "retrieve", "args": {"top_k": 15}}
{"tool": "match", "args": {"image_a": "img_0000", "image_b": "img_0001", "matcher": "loftr"}}
{"tool": "doppelganger_check", "args": {"image_a": "img_0002", "image_b": "img_0007"}}
{"tool": "sfm_run", "args": {}}
{"tool": "done", "args": {}}"""


@dataclass
class SceneRolloutEpisode:
    """A scene-level rollout episode."""

    scene_id: str = ""
    image_paths: list[str] = field(default_factory=list)
    image_root: str = ""
    num_images: int = 0

    # Trajectory
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    assistant_responses: list[str] = field(default_factory=list)

    # Pair-level matches collected during the episode
    pair_matches: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Doppelganger checks: canonical pair key ("idA__idB", sorted, crop
    # suffixes stripped) -> raw /doppelganger_check response.  Pairs whose
    # confidence exceeds the flag threshold are excluded from sfm_run.
    doppelganger_checks: dict[str, dict[str, Any]] = field(default_factory=dict)

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
    overlap_matrix: np.ndarray | None = None,
    image_indices: list[int] | None = None,
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
        image_root=image_root,
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
        if terminated or num_tool_calls >= max_tool_calls:
            break

        # Trim the rolling context to stay under the model's context budget
        # before generating (long scenes + many tool calls would otherwise
        # overflow vLLM's max_model_len and the whole episode would fail).
        _trim_scene_context(ep)
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
        result = _execute_scene_tool(
            tool_client, tc, registered_ids, ep,
            overlap_matrix=overlap_matrix,
            image_indices=image_indices,
        )
        ep.results.append(result)
        num_valid_calls += 1

        # Track pair-level matches for keep_best_match
        if tc.tool in ("match", "crop_and_match") and not result.get("error"):
            pair_key = f"{tc.args.get('image_a', '?')}__{tc.args.get('image_b', tc.args.get('image_id', '?'))}"
            ep.pair_matches[pair_key] = result
            kept = keep_best_match(ep.final_match or None, result)
            if kept:
                ep.final_match = kept

        # Track doppelganger checks: flagged pairs are excluded from the
        # sfm_run pair list and counted for the doppelganger reward.
        if tc.tool == "doppelganger_check" and not result.get("error"):
            key = _pair_key(
                tc.args.get("image_a", "?"), tc.args.get("image_b", "?")
            )
            ep.doppelganger_checks[key] = result

        # Track reconstruction result
        if tc.tool == "sfm_run":
            ep.recon_result = result

        # Format observation
        obs_text = format_observation(result)
        ep.messages.append({"role": "user", "content": f"Observation: {obs_text}"})

    # Inject GT pose error into recon_result before reward computation
    _inject_pose_error(ep, gt_recon)

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


# Conservative context budget for scene rollouts. The model supports ~262k
# positions but vLLM is launched with a smaller --max-model-len; we keep the
# rolling conversation well under it so long episodes never hit the 400
# "input exceeds max context length" errors seen in early Phase-2 runs.
SCENE_CONTEXT_BUDGET = 24000  # tokens (chars/4 estimate + image cost)
_IMG_TOKEN_COST = 300         # rough token cost per image/thumbnail


def _estimate_context_tokens(messages: list[dict[str, Any]], num_images: int) -> int:
    """Rough token estimate: ~4 chars/token of text + per-image cost."""
    chars = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text":
                    chars += len(str(part.get("text", "")))
    return chars // 4 + num_images * _IMG_TOKEN_COST


def _trim_scene_context(ep: Any, max_tokens: int = SCENE_CONTEXT_BUDGET) -> None:
    """Drop the oldest assistant/observation turns when over the token budget.

    Keeps ``messages[0]`` (system) and ``messages[1]`` (initial user with the
    scene thumbnails), then retains only the most recent turns so the rolling
    prompt stays under ``max_tokens``. Mutates ``ep.messages``; also bounds
    ``ep.images`` to the thumbnail count actually referenced.
    """
    num_images = len(getattr(ep, "images", []) or [])
    if _estimate_context_tokens(ep.messages, num_images) <= max_tokens:
        return
    # messages[0]=system, [1]=user+images; tail = [assistant, user-obs] turns.
    head = ep.messages[:2]
    tail = ep.messages[2:]
    # Drop oldest turns until under budget (each tail element is one message).
    while tail and _estimate_context_tokens(head + tail, num_images) > max_tokens:
        tail = tail[1:]
    ep.messages = head + tail


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
        image_root=image_root,
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
        if image_paths:
            p0 = Path(image_paths[0])
            if not p0.is_absolute() and image_root:
                p0 = Path(image_root) / p0
            image_dir = str(p0.parent)
        elif image_root:
            image_dir = str(image_root)
        else:
            image_dir = ""
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

    # Inject GT pose error before reward computation
    _inject_pose_error(ep, gt_recon)
    recon_result = ep.recon_result  # may now contain mean_pose_error_deg

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
    overlap_matrix: np.ndarray | None = None,
    image_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Execute a scene-level tool call."""
    args = tc.args

    if tc.tool == "retrieve":
        # Use learned retrieval (DINOv2 embeddings) for agent rollouts.
        # The S-GRPO oracle uses GT overlap_matrix instead — that's the
        # difference between what the agent sees and what the oracle knows.
        from agentic_sfm.rl.retrieval import retrieve_pairs_from_paths
        pairs = retrieve_pairs_from_paths(
            ep.image_paths, registered_ids,
            top_k=args.get("top_k", 20),
            image_root=ep.image_root,
        )
        return {"pairs": pairs, "num_pairs": len(pairs)}

    if tc.tool == "sfm_run":
        # Resolve image_dir via image_root when image_paths are relative.
        if ep.image_paths:
            p0 = Path(ep.image_paths[0])
            if not p0.is_absolute() and ep.image_root:
                p0 = Path(ep.image_root) / p0
            image_dir = str(p0.parent)
        elif ep.image_root:
            image_dir = str(ep.image_root)
        else:
            image_dir = ""
        pair_list, n_present, n_filtered = _sfm_pair_list(ep, registered_ids)
        result = tool_client.sfm_run(image_dir=image_dir, pair_list=pair_list)
        if isinstance(result, dict):
            result.setdefault("num_doppelgangers_present", n_present)
            result.setdefault("num_doppelgangers_filtered", n_filtered)
        return result

    if tc.tool == "inspect":
        recon_dir = args.get("recon_dir", ep.recon_result.get("output_dir", ""))
        if not recon_dir:
            return {"error": "No reconstruction to inspect"}
        return tool_client.inspect(recon_dir)

    # Pair-level tools: delegate to execute_sfm_tool
    return execute_sfm_tool(tool_client, tc)
