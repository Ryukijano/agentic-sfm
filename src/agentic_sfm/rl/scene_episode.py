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
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agentic_sfm.agent.policy import (
    MAX_IDENTICAL_TOOL_CALLS,
    ToolCall,
    execute_sfm_tool,
    format_observation,
    parse_tool_call,
    repeat_breaker_message,
    tool_call_signature,
)
from agentic_sfm.constants import DEFAULT_MATCHER
from agentic_sfm.geometry import keep_best_match
from agentic_sfm.rewards.pose_rewards import compute_scene_reward
from agentic_sfm.tools.client import ToolClient

logger = logging.getLogger(__name__)


class ToolServerDown(RuntimeError):
    """Tool server is unreachable (connection refused / dead process)."""


def _is_connection_error(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, ConnectionRefusedError, TimeoutError, OSError)):
        if isinstance(exc, OSError) and getattr(exc, "errno", None) not in (111, 61, 104, None):
            # Keep generic OSError only when the message looks like a transport fail.
            msg = str(exc).lower()
            return any(s in msg for s in ("connection refused", "errno 111", "connecterror", "timed out"))
        if isinstance(exc, OSError) and getattr(exc, "errno", None) in (111, 61, 104):
            return True
        if isinstance(exc, (ConnectionError, ConnectionRefusedError, TimeoutError)):
            return True
    try:
        import httpx
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.TransportError)):
            return True
    except ImportError:
        pass
    msg = str(exc).lower()
    return any(
        s in msg
        for s in ("connection refused", "errno 111", "connecterror", "all connection attempts failed")
    )


def stage_episode_images(
    image_paths: list[str],
    image_root: str = "",
    session_id: str = "",
) -> str:
    """Stage exactly the N episode images into an isolated directory via symlinks.

    Guarantees COLMAP sees ONLY the images in this episode, preventing
    discovery of non-sampled images in the dataset directory.
    """
    import shutil
    import tempfile

    candidates = [
        Path("/dev/shm"),
        Path(os.environ.get("TMP_SHARED", "")) if os.environ.get("TMP_SHARED") else None,
        Path(tempfile.gettempdir()),
    ]
    base_dir = None
    for cand in candidates:
        if cand is not None and cand.exists() and os.access(cand, os.W_OK):
            base_dir = cand
            break
    if base_dir is None:
        base_dir = Path(tempfile.gettempdir())

    stage_id = session_id or uuid.uuid4().hex[:8]
    stage_dir = base_dir / f"asfm_stage_{stage_id}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(image_paths):
        full_path = Path(image_root) / path if image_root else Path(path)
        full_path = full_path.resolve()
        ext = full_path.suffix or ".jpg"
        sym_name = f"img_{i:04d}{ext}"
        target = stage_dir / sym_name
        if not target.exists():
            try:
                target.symlink_to(full_path)
            except Exception:
                shutil.copy2(full_path, target)
    return str(stage_dir)


def cleanup_stage_dir(stage_dir: str) -> None:
    """Clean up the isolated staging directory and symlinks."""
    if not stage_dir or not os.path.exists(stage_dir):
        return
    import shutil
    try:
        shutil.rmtree(stage_dir)
    except Exception as e:
        logger.debug(f"Failed to clean up staging dir {stage_dir}: {e}")


def _register_scene_images(
    tool_client: ToolClient,
    image_paths: list[str],
    image_root: str = "",
    staged_dir: str = "",
) -> dict[str, str]:
    """Register scene images; abort immediately if the tool server is down."""
    registered_ids: dict[str, str] = {}
    for i, path in enumerate(image_paths):
        img_id = f"img_{i:04d}"
        if staged_dir and os.path.isdir(staged_dir):
            ext = Path(path).suffix or ".jpg"
            staged_file = os.path.join(staged_dir, f"img_{i:04d}{ext}")
            full = staged_file if os.path.exists(staged_file) else (str(Path(image_root) / path) if image_root else path)
        else:
            full = str(Path(image_root) / path) if image_root else path
        try:
            tool_client.register_image(img_id, full)
            registered_ids[path] = img_id
        except Exception as e:
            if _is_connection_error(e):
                raise ToolServerDown(
                    f"tool server unreachable while registering {full}: {e}"
                ) from e
            logger.warning(f"Failed to register {path}: {e}")
    return registered_ids

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


# Matches agent-facing image ids: "img_0012" (crop suffixes are stripped by
# _base_image_id first); bare positional indices are accepted too.
_EPISODE_IMG_ID_RE = re.compile(r"^img_(\d{1,6})$")


def _episode_image_index(image_id: Any, num_images: int) -> int | None:
    """Resolve an agent-facing image id to its episode-local index.

    Agent-facing ids are ``img_0000..img_{N-1:04d}`` — the position in
    ``image_paths`` (exactly how ``_register_scene_images`` names them), which
    is also the row/column of the episode's ``sub_om`` covisibility submatrix.
    """
    s = _base_image_id(str(image_id)).strip()
    m = _EPISODE_IMG_ID_RE.match(s)
    if m:
        i = int(m.group(1))
    elif s.isdigit():
        i = int(s)
    else:
        return None
    return i if 0 <= i < num_images else None


def _in_play_pairs(
    ep: SceneRolloutEpisode,
    tool_client: Any = None,
) -> list[tuple[str, str]]:
    """Collect the episode-local image-id pairs that were actually in play.

    Union of (a) the active edge list from the most recent
    ``inspect_graph``/``build_initial_graph`` result carrying an ``edges``
    field (one extra read-only ``inspect_graph`` call at episode end when the
    episode used graph tools but no result carried edges), (b) pairs sent to
    ``verify_doppelganger``/``doppelganger_check`` (tool-call args plus the
    recorded ``ep.doppelganger_checks`` keys), (c) ``prune_edges`` pair args,
    and (d) the explicit ``sfm_run`` pair list plus retrieve-style ``pairs``
    results.  Ids stay in the agent-facing ``img_XXXX`` namespace.
    """
    pairs: set[tuple[str, str]] = set()

    def _add(a: Any, b: Any) -> None:
        a_s, b_s = _base_image_id(str(a)), _base_image_id(str(b))
        if a_s and b_s and a_s not in ("?", "") and b_s not in ("?", "") and a_s != b_s:
            pairs.add((a_s, b_s) if a_s <= b_s else (b_s, a_s))

    def _add_pairs_list(raw: Any) -> None:
        if not isinstance(raw, list):
            return
        for p in raw:
            if isinstance(p, dict):
                _add(p.get("image_a"), p.get("image_b"))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                _add(p[0], p[1])

    # (a) most recent graph result carrying an active edge list
    edges: list[Any] | None = None
    for res in reversed(ep.results):
        if isinstance(res, dict) and isinstance(res.get("edges"), list):
            edges = res["edges"]
            break
    if edges is None and tool_client is not None:
        used_graph_tools = any(
            tc.tool in ("build_initial_graph", "inspect_graph", "prune_edges",
                        "bridge_components")
            for tc in ep.tool_calls
        )
        if used_graph_tools and hasattr(tool_client, "inspect_graph"):
            # Read-only final read of the session's view graph so the episode
            # knows which edges the reconstruction actually used.
            try:
                res = tool_client.inspect_graph(session=ep.session_id)
                if isinstance(res, dict) and isinstance(res.get("edges"), list):
                    edges = res["edges"]
            except Exception as e:
                logger.debug(f"Episode-end inspect_graph failed: {e}")
    _add_pairs_list(edges)

    # (b) doppelganger checks: recorded pair keys + tool-call args
    for key in ep.doppelganger_checks:
        parts = str(key).split("__")
        if len(parts) == 2:
            _add(parts[0], parts[1])
    for tc in ep.tool_calls:
        if tc.tool in ("verify_doppelganger", "doppelganger_check"):
            _add(
                tc.args.get("image_a"),
                tc.args.get("image_b") or tc.args.get("image_id"),
            )
        elif tc.tool == "prune_edges":
            # (c) prune_edges pair args
            _add_pairs_list(tc.args.get("pairs"))

    # (d) explicit sfm pair list + retrieve-style pair results
    _add_pairs_list(getattr(ep, "sfm_pair_list", None))
    for res in ep.results:
        if isinstance(res, dict):
            _add_pairs_list(res.get("pair_list"))
            if isinstance(res.get("pairs"), list):
                _add_pairs_list(res["pairs"])

    return sorted(pairs)


def _gt_impostor_pairs(
    ep: SceneRolloutEpisode,
    overlap_matrix: np.ndarray | None,
    image_indices: list[int] | None,
    tool_client: Any = None,
) -> list[list[str]] | None:
    """Ground-truth doppelganger labels from the covisibility matrix.

    MegaDepth overlap semantics: > 0 means measured covisibility, 0.0 means
    measured none, and -1.0 means never measured.  Both ``<= 0`` values mean
    "not actually overlapping", so an in-play pair on such a value is a true
    doppelganger.  Returns ``[[img_a, img_b], ...]`` in the agent-facing id
    namespace, or ``None`` when GT overlap data is unavailable (pure mock
    episodes — the reward then uses its gated fallback).
    """
    if overlap_matrix is None or image_indices is None:
        return None
    om = np.asarray(overlap_matrix)
    if om.ndim != 2:
        return None
    n = len(ep.image_paths)
    if n < 2 or len(image_indices) != n:
        return None
    try:
        sub_om = om[np.ix_(image_indices, image_indices)]
    except (IndexError, TypeError, ValueError):
        return None

    impostors: list[list[str]] = []
    for a, b in _in_play_pairs(ep, tool_client=tool_client):
        i = _episode_image_index(a, n)
        j = _episode_image_index(b, n)
        if i is None or j is None or i == j:
            continue
        if float(sub_om[i, j]) <= 0.0:
            impostors.append([a, b])
    return impostors


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
You are a Photogrammetry Director agent. Reconstruct the 3D scene from its images (img_0000, img_0001, ...) by actively constructing, verifying, and optimizing the view graph. Output exactly one JSON tool call per turn:
{"tool": "name", "args": {...}}

Tools:
- build_initial_graph {"k_neighbors": 5, "matcher": "loftr"} -> initializes view graph via visual retrieval and batch matching; returns graph summary (num_components, isolated_images, bottlenecks, suspicious_cycles)
- inspect_graph {} -> analyzes current view graph topology, connected components, articulation bridges, and rotation cycle residuals
- verify_doppelganger {"image_a": "img_0000", "image_b": "img_0001", "matcher": "loftr"} -> checks if a pair is a visual doppelganger / symmetry ambiguity using cycle rotation consistency and appearance similarity
- prune_edges {"pairs": [["img_0000", "img_0001"]]} -> removes incorrect or ambiguous edges from the view graph before reconstruction
- bridge_components {"comp_a_id": 0, "comp_b_id": 1, "budget": 3, "matcher": "loftr"} -> searches for cross-component candidate pairs and attempts to connect disconnected subgraphs
- sfm_run {} -> runs incremental SfM on the verified active view graph; returns num_registered, num_points3d, mean_reproj_error
- inspect {} -> detailed statistics of the reconstruction
- done {} -> end episode

Matchers (the "matcher" argument of build_initial_graph, bridge_components and verify_doppelganger):
- "loftr": dense detector-free matcher (fast, ~0.06s/pair). Best on textured outdoor scenes.
- "mast3r": 3D-aware matcher from a pretrained two-view reconstruction model. Returns correspondences and a pose.
- "vggt": feed-forward multi-view geometry model. Its extrinsics prior is available for every pair, rescuing low-texture indoor pairs where LoFTR fails.
- "mapanything": universal metric 3D reconstruction model (facebook/map-anything-apache). Predicts metric 3D point maps and camera poses across wide baselines.
Each build_initial_graph summary reports matcher_requested, pairs_fallback (pairs that fell back to loftr because the requested matcher failed), pairs_zero_matches and median_inliers, so you can judge a matcher from its results and route to a 3D specialist ("vggt", "mast3r", or "mapanything") when LoFTR inliers are low or components split.

Workflow:
1. Always start by calling build_initial_graph to create the initial view graph.
2. Call inspect_graph or inspect suspicious_cycles in the initial summary.
3. If suspicious cycles or symmetry ambiguities exist, call verify_doppelganger on those pairs.
4. Call prune_edges to eliminate verified doppelganger edges.
5. If the graph has multiple disconnected components (num_components > 1), call bridge_components to connect them.
6. Call sfm_run to reconstruct the scene from the verified graph.
7. Call done when reconstruction succeeds.

Reward: higher for registered images (coverage) and accurate camera poses. Failed or empty graphs incur a severe penalty. Do not repeat calls.

Examples:
{"tool": "build_initial_graph", "args": {"k_neighbors": 5, "matcher": "loftr"}}
{"tool": "inspect_graph", "args": {}}
{"tool": "verify_doppelganger", "args": {"image_a": "img_0000", "image_b": "img_0001"}}
{"tool": "prune_edges", "args": {"pairs": [["img_0000", "img_0001"]]}}
{"tool": "bridge_components", "args": {"comp_a_id": 0, "comp_b_id": 1, "budget": 3}}
{"tool": "sfm_run", "args": {}}
{"tool": "done", "args": {}}"""


# Must match vLLM --limit-mm-per-prompt image:N and stay under --max-model-len.
SCENE_MAX_PROMPT_IMAGES = 4
SCENE_CONTEXT_BUDGET = 7000  # tokens (chars/4 estimate + image cost)
_IMG_TOKEN_COST = 512         # ~vision tokens at 401408 px (SFT pixel cap)


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
    # suffixes stripped) -> raw /doppelganger_check or /verify_doppelganger
    # response.  Pairs whose confidence exceeds the flag threshold are
    # excluded from sfm_run.
    doppelganger_checks: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Explicit pair list handed to the last sfm_run (None = server-side
    # exhaustive/graph-driven matching).  Used for GT doppelganger labelling.
    sfm_pair_list: list[list[str]] | None = None

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

    # True for injected S-GRPO oracle trajectories (excluded from media picks).
    is_oracle: bool = False
    # Unique per-episode id — scopes the server's match store so sibling
    # rollouts / earlier epochs can't subsidize this episode's sfm_run.
    session_id: str = ""
    # Isolated staging directory where symlinks to exact episode images live
    staged_image_dir: str = ""



def _scene_reward_kwargs(agent: Any) -> dict[str, Any]:
    """Keyword args for ``compute_scene_reward`` from the rollout agent's config."""
    cfg = getattr(agent, "reward_config", {}) or {}
    return {
        "tool_cost": cfg.get("tool_cost", 0.05),
        "registration_weight": cfg.get("registration_weight", 0.5),
        "pose_weight": cfg.get("pose_weight", 1.0),
        "split_penalty": cfg.get("split_penalty", 0.5),
        "doppelganger_weight": cfg.get("doppelganger_weight", 0.3),
        "accumulative_tool_coef": cfg.get("accumulative_tool_coef", 0.1),
        "use_accumulative_tool_reward": cfg.get("use_accumulative_tool_reward", True),
        "ntep_intent_coef": cfg.get("ntep_intent_coef", 0.05),
        "ntep_redundancy_penalty": cfg.get("ntep_redundancy_penalty", 0.05),
        "use_ntep_rewards": cfg.get("use_ntep_rewards", False),
        "first_call_validity_weight": cfg.get("first_call_validity_weight", 0.5),
        "first_call_invalid_penalty": cfg.get("first_call_invalid_penalty", 1.0),
        "gate_geometry_on_validity": cfg.get("gate_geometry_on_validity", True),
        # Graph sparsity cost (only applies when coverage >= 0.7; normalized
        # by the exhaustive N(N-1)/2 pair count) and the opt-in gate for the
        # self-certified doppelganger fallback (disabled by default — real
        # scoring is against GT-derived impostor labels).
        "graph_cost_weight": cfg.get("graph_cost_weight", 0.0),
        "allow_self_certified_doppelganger": cfg.get(
            "allow_self_certified_doppelganger", False
        ),
    }


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
    forced_specialist_matcher: str | None = None,
    matcher_explore_eps: float = 0.0,
    allowed_matchers: tuple[str, ...] = ("loftr", "mast3r", "vggt", "mapanything"),
    explore_rng: np.random.Generator | None = None,
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
    ep.session_id = f"{scene_id}-{uuid.uuid4().hex[:8]}"

    # Stage exact N episode images into isolated directory
    ep.staged_image_dir = stage_episode_images(image_paths, image_root=image_root, session_id=ep.session_id)

    # Register all images with the tool server
    registered_ids = _register_scene_images(
        tool_client, image_paths, image_root=image_root, staged_dir=ep.staged_image_dir
    )

    # Build initial message with scene images (first N as thumbnails).
    # Must stay <= vLLM --limit-mm-per-prompt image:4 (Phase 2 r5).
    n_show = min(len(image_paths), SCENE_MAX_PROMPT_IMAGES)
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
            f"Call build_initial_graph to initialize the view graph, inspect/optimize it, and run sfm_run."
        ),
    })

    ep.messages = [
        {"role": "system", "content": SCENE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    # Scene-level rollout loop
    num_tool_calls = 0
    num_invalid_calls = 0
    repeat_counts: dict[tuple, int] = {}
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
        # Anti-collapse specialist exploration: when forced_specialist_matcher
        # is provided for this rollout slot or with probability matcher_explore_eps,
        # rewrite the matcher argument on matcher-accepting tools AND synchronize
        # the assistant JSON text in ep.messages / ep.assistant_responses so GRPO
        # computes policy-gradient updates on the actual explored specialist tokens.
        if tc.tool in ("build_initial_graph", "bridge_components", "verify_doppelganger", "match"):
            chosen_m: str | None = None
            if forced_specialist_matcher:
                chosen_m = str(forced_specialist_matcher).lower()
                # Only force the primary graph-building call once unless random exploration triggers later
                forced_specialist_matcher = None
            elif matcher_explore_eps > 0.0 and allowed_matchers:
                _rng = explore_rng if explore_rng is not None else np.random.default_rng()
                if float(_rng.random()) < float(matcher_explore_eps):
                    non_default = [m for m in allowed_matchers if m != str(tc.args.get("matcher", DEFAULT_MATCHER)).lower()]
                    pool = non_default or list(allowed_matchers)
                    chosen_m = str(pool[int(_rng.integers(0, len(pool)))]).lower()
            if chosen_m:
                tc.args = dict(tc.args or {})
                tc.args["matcher"] = chosen_m
                rewritten_json = json.dumps({"tool": tc.tool, "args": tc.args})
                ep.assistant_responses[-1] = rewritten_json
                ep.messages[-1]["content"] = rewritten_json

        ep.tool_calls.append(tc)

        if tc.tool == "done":
            terminated = True
            ep.done = True
            continue

        # Identical-call loop guard: a repeated rejected call burns the budget
        # and makes every episode in the group fail the same way, leaving GRPO
        # no reward contrast to learn from.
        sig = tool_call_signature(tc)
        repeat_counts[sig] = repeat_counts.get(sig, 0) + 1
        if repeat_counts[sig] > MAX_IDENTICAL_TOOL_CALLS:
            ep.messages.append({
                "role": "user",
                "content": repeat_breaker_message(sig, repeat_counts[sig]),
            })
            break

        # Execute the tool — never let an HTTP error kill the whole run.
        try:
            result = _execute_scene_tool(
                tool_client, tc, registered_ids, ep,
                overlap_matrix=overlap_matrix,
                image_indices=image_indices,
            )
        except Exception as exc:
            result = {"error": str(exc)}
        ep.results.append(result)
        # Errored calls are not "valid" — else rejection is reward-invisible.
        if not (isinstance(result, dict) and result.get("error")):
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

    # Build GT impostor labels from the covisibility matrix.  gt_recon is
    # shared across the whole GRPO group — never mutate it; attach the
    # episode-specific labels to a shallow copy instead.
    gt = gt_recon
    if overlap_matrix is not None and image_indices is not None:
        impostors = _gt_impostor_pairs(
            ep, overlap_matrix, image_indices, tool_client=tool_client
        )
        if impostors is not None:
            gt = dict(gt_recon or {})
            gt["impostors"] = impostors
            ep.gt_recon = gt

    # Inject GT pose error into recon_result before reward computation
    _inject_pose_error(ep, gt)

    # Compute scene-level reward
    ep.reward_components = compute_scene_reward(
        ep.recon_result or {},
        gt_recon=gt,
        num_tool_calls=num_tool_calls,
        num_valid_calls=num_valid_calls,
        tool_calls=ep.tool_calls,
        tool_results=ep.results,
        **_scene_reward_kwargs(agent),
    )
    ep.reward = ep.reward_components["total_reward"]

    logger.info(
        f"[{scene_id}] Episode done: {num_tool_calls} calls, "
        f"reward={ep.reward:.3f}, registered={ep.recon_result.get('num_registered', 0)}"
    )
    cleanup_stage_dir(ep.staged_image_dir)
    return ep


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
    ep.session_id = f"{scene_id}-oracle-{uuid.uuid4().hex[:8]}"

    ep.staged_image_dir = stage_episode_images(image_paths, image_root, session_id=ep.session_id)
    registered_ids = _register_scene_images(
        tool_client, image_paths, image_root, staged_dir=ep.staged_image_dir
    )

    # Encode images for the prompt
    n_show = min(len(image_paths), SCENE_MAX_PROMPT_IMAGES)
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
            result = execute_sfm_tool(
                tool_client, tc,
                {"matcher": getattr(agent, "matcher", DEFAULT_MATCHER)},
                session=ep.session_id,
            )
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
        if ep.staged_image_dir and os.path.isdir(ep.staged_image_dir):
            image_dir = ep.staged_image_dir
        elif image_paths:
            p0 = Path(image_paths[0])
            if not p0.is_absolute() and image_root:
                p0 = Path(image_root) / p0
            image_dir = str(p0.parent)
        elif image_root:
            image_dir = str(image_root)
        else:
            image_dir = ""
        recon_result = tool_client.sfm_run(
            image_dir=image_dir, session=ep.session_id
        )
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

    # Build GT impostor labels from the covisibility matrix (same GT-based
    # doppelganger scoring as the agent rollout path).  gt_recon is shared
    # across the GRPO group — attach labels to a shallow copy only.
    gt = gt_recon
    if overlap_matrix is not None and image_indices is not None:
        impostors = _gt_impostor_pairs(
            ep, overlap_matrix, image_indices, tool_client=tool_client
        )
        if impostors is not None:
            gt = dict(gt_recon or {})
            gt["impostors"] = impostors
            ep.gt_recon = gt

    # Inject GT pose error before reward computation
    _inject_pose_error(ep, gt)
    recon_result = ep.recon_result  # may now contain mean_pose_error_deg

    # Compute reward
    n_valid = len(oracle_calls) - 1  # exclude done
    ep.reward_components = compute_scene_reward(
        recon_result, gt_recon=gt,
        num_tool_calls=len(oracle_calls), num_valid_calls=n_valid,
        tool_calls=oracle_calls,
        tool_results=oracle_results,
        **_scene_reward_kwargs(agent),
    )
    ep.reward = ep.reward_components["total_reward"]
    ep.is_oracle = True

    # Guarantee oracle reward > failed group max — but only when the oracle
    # actually produced a reconstruction. Otherwise the floor pays positive
    # advantage for imitating a trajectory that built nothing.
    oracle_has_geometry = bool(
        ep.recon_result and (ep.recon_result.get("num_registered") or 0) > 0
    )
    if oracle_has_geometry and ep.reward <= failed_group_max_reward:
        ep.reward = failed_group_max_reward + 0.1
        ep.reward_components["oracle_floor_boost"] = ep.reward - ep.reward_components["total_reward"]

    cleanup_stage_dir(ep.staged_image_dir)
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

    if tc.tool == "build_initial_graph":
        k_neighbors = int(args.get("k_neighbors", 5))
        matcher = str(args.get("matcher", DEFAULT_MATCHER))
        return tool_client.build_initial_graph(
            image_ids=list(registered_ids.values()),
            k_neighbors=k_neighbors,
            matcher=matcher,
            session=ep.session_id,
        )

    if tc.tool == "inspect_graph":
        return tool_client.inspect_graph(session=ep.session_id)

    if tc.tool == "verify_doppelganger":
        a = str(args.get("image_a", ""))
        b = str(args.get("image_b", ""))
        res = tool_client.verify_doppelganger(
            image_a=a, image_b=b, session=ep.session_id,
            matcher=str(args.get("matcher", "loftr")),
        )
        if isinstance(res, dict) and not res.get("error"):
            # Record every checked pair (flagged or not) so it stays "in
            # play" for GT doppelganger labelling; _sfm_pair_list still only
            # filters pairs the response actually flagged.
            ep.doppelganger_checks[_pair_key(a, b)] = res
        return res

    if tc.tool == "prune_edges":
        pairs = args.get("pairs", [])
        return tool_client.prune_edges(pairs=pairs, session=ep.session_id)

    if tc.tool == "bridge_components":
        comp_a = int(args.get("comp_a_id", args.get("comp_a", 0)))
        comp_b = int(args.get("comp_b_id", args.get("comp_b", 1)))
        budget = int(args.get("budget", 3))
        matcher = str(args.get("matcher", DEFAULT_MATCHER))
        return tool_client.bridge_components(
            comp_a_id=comp_a,
            comp_b_id=comp_b,
            budget=budget,
            matcher=matcher,
            session=ep.session_id,
        )

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
        # Prioritize isolated staging directory (exact N images)
        if ep.staged_image_dir and os.path.isdir(ep.staged_image_dir):
            image_dir = ep.staged_image_dir
        elif ep.image_paths:
            p0 = Path(ep.image_paths[0])
            if not p0.is_absolute() and ep.image_root:
                p0 = Path(ep.image_root) / p0
            image_dir = str(p0.parent)
        elif ep.image_root:
            image_dir = str(ep.image_root)
        else:
            image_dir = ""
        pair_list, n_present, n_filtered = _sfm_pair_list(ep, registered_ids)
        ep.sfm_pair_list = pair_list
        result = tool_client.sfm_run(
            image_dir=image_dir, pair_list=pair_list, session=ep.session_id
        )
        if isinstance(result, dict):
            result.setdefault("num_doppelgangers_present", n_present)
            result.setdefault("num_doppelgangers_filtered", n_filtered)
        return result

    if tc.tool == "inspect":
        recon_dir = args.get("recon_dir", ep.recon_result.get("output_dir", ""))
        if not recon_dir:
            return {"error": "No reconstruction to inspect"}
        return tool_client.inspect(recon_dir)

    # Pair-level tools: delegate to execute_sfm_tool (session-scoped so the
    # server match cache only credits this episode's own matches)
    return execute_sfm_tool(tool_client, tc, session=ep.session_id)
