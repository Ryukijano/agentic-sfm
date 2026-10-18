"""Tool server: FastAPI service exposing vision tools to the MLLM agent.

Tools:
  - crop(image_id, bbox) → cropped image
  - match(img_a, img_b, matcher) → correspondences + pose
  - doppelganger_check(img_a, img_b) → score
  - retrieve(query_img, k) → candidate pairs
  - sfm_run(pair_list) → reconstruction
  - inspect(recon_id) → stats
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sys
from collections import OrderedDict

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from agentic_sfm.graph.view_graph import EdgeData, ViewGraph
from agentic_sfm.rl.retrieval import compute_image_embedding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class CropRequest(BaseModel):
    image_id: str
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2] in normalized [0,1] coords")


class RegisterRequest(BaseModel):
    image_id: str
    path: str


class MatchRequest(BaseModel):
    image_a: str
    image_b: str
    matcher: str = "loftr"
    max_size: int = 512
    K_a: list[float] | list[list[float]] | None = None
    K_b: list[float] | list[list[float]] | None = None
    # Episode-scoped namespace for the match cache: sfm_run imports only the
    # caller's own session so sibling rollouts can't free-ride on matches
    # they never made.
    session: str = ""


class DoppelgangerRequest(BaseModel):
    image_a: str
    image_b: str


class RetrieveRequest(BaseModel):
    query_image: str
    k: int = 5


class BuildGraphRequest(BaseModel):
    image_ids: list[str] = Field(..., description="Registered image ids for the scene")
    k_neighbors: int = 5
    matcher: str = "loftr"
    session: str = ""


class InspectGraphRequest(BaseModel):
    session: str = ""


class VerifyDoppelgangerRequest(BaseModel):
    image_a: str
    image_b: str
    session: str = ""
    matcher: str = "loftr"


class PruneEdgesRequest(BaseModel):
    pairs: list[list[str]]
    session: str = ""


class BridgeComponentsRequest(BaseModel):
    comp_a_id: int = 0
    comp_b_id: int = 1
    budget: int = 3
    matcher: str = "loftr"
    session: str = ""


class SfMRequest(BaseModel):
    image_dir: str
    pair_list: list[tuple[str, str]] | None = None
    # Empty -> server creates a unique per-call dir. A shared fixed dir lets
    # concurrent rollouts wipe each other's reconstruction.
    output_dir: str = ""
    session: str = ""
    # When true, import the agent's own matcher correspondences (cached by
    # earlier /match calls) into COLMAP instead of SIFT matching, so the
    # reconstruction is driven by the agent's match quality.
    use_agent_matches: bool = True


class InspectRequest(BaseModel):
    recon_dir: str


# ---------------------------------------------------------------------------
# Image store — maps image_id → file path or numpy array
# ---------------------------------------------------------------------------

_IMAGE_STORE: dict[str, str | np.ndarray] = {}
_image_registry = _IMAGE_STORE
_RECON_STORE: dict[str, dict[str, Any]] = {}
_CROP_META: dict[str, dict[str, Any]] = {}

# Cache of the agent's own matcher correspondences per resolved image pair.
# Session-scoped: sfm_run imports only the calling episode's matches — a
# global store let later rollouts free-ride on matches made by earlier
# episodes over the same image paths (reward for work never done).
# Bounded FIFO so long runs can't grow it without limit.
_match_store: "OrderedDict[str, OrderedDict[tuple[str, str], dict[str, Any]]]" = OrderedDict()
_MATCH_STORE_MAX_SESSIONS = 64
_DEFAULT_SESSION = "_default"

# Session-scoped ViewGraph stores for Phase 2 scene-level graph orchestration.
_SESSION_GRAPHS: dict[str, ViewGraph] = {}
# Cap on the active edge list returned by tool_inspect_graph.
_INSPECT_GRAPH_MAX_EDGES = 500
# Cached image embeddings (keyed by resolved image path)
_EMBEDDING_CACHE: dict[str, np.ndarray] = {}


# COLMAP runs in this child process so a native SIGBUS (FAISS matcher) can't
# abort the server. Overridable for tests that simulate a crashing worker.
_SFM_WORKER_PATH = Path(__file__).resolve().parent / "sfm_worker.py"

# Shortest crop side, as a fraction of the image's shorter side. Rejects the
# tiny-crop mode RL drifted into; far below any SFT/oracle box (>= 0.6).
MIN_CROP_FRACTION = float(os.environ.get("AGENTIC_SFM_MIN_CROP_FRACTION", "0.25"))


def colmap_matching_options():
    """FeatureMatchingOptions that avoid COLMAP's FAISS descriptor index.

    pycolmap 4.x's SIFT matcher builds a per-image FAISS ``FeatureDescriptorIndex``
    (IVF clustering) unless ``sift.cpu_brute_force_matcher`` is set. That index
    SIGBUS'd inside ``faiss::Clustering::train_encoded`` on this cluster, killing
    the whole tool server mid-run (job 7882190, 2026-09-16) — a native crash no
    Python-level try/except can contain.

    Brute-force matching sidesteps FAISS entirely. It is slower per pair, but our
    scenes are tens of images, and a dead tool server costs far more than slow
    matching. Threads are pinned to 1 as well: FAISS/COLMAP's OpenMP region
    coexists badly with torch's OpenMP runtime in the same process, and the
    crash is thread-contention sensitive (faiss#5294).
    """
    import pycolmap

    opts = pycolmap.FeatureMatchingOptions()
    try:
        opts.sift.cpu_brute_force_matcher = True
    except AttributeError:
        # Older pycolmap without the FAISS matcher — nothing to disable.
        pass
    try:
        opts.num_threads = 1
    except AttributeError:
        pass
    return opts


def _store_path(id_or_path: str) -> str | None:
    """Resolve an image id (or raw path) to a real file path, or None.

    ``_IMAGE_STORE`` values may be numpy arrays (e.g. in-memory crops); those
    have no on-disk path COLMAP can key on, so they resolve to None and are
    skipped by the match cache / importer.
    """
    entry = _IMAGE_STORE.get(id_or_path)
    if isinstance(entry, str):
        return entry
    if isinstance(id_or_path, str) and os.path.exists(id_or_path):
        return id_or_path
    return None


def _root_path_and_offset(image_id: str) -> tuple[str | None, tuple[float, float]]:
    """Resolve an image id (or crop id / raw path) to its root full-frame path and (ox, oy) offset."""
    meta = _CROP_META.get(image_id)
    if meta is not None:
        origin = meta.get("origin_xy", [0.0, 0.0])
        ox, oy = float(origin[0]), float(origin[1])
        parent_id = meta.get("parent_id")
        if not parent_id and isinstance(image_id, str) and "_crop_" in image_id:
            parent_id = image_id.split("_crop_")[0]
        visited: set[str] = {image_id}
        while parent_id and parent_id in _CROP_META and parent_id not in visited:
            visited.add(parent_id)
            pmeta = _CROP_META[parent_id]
            parent_id = pmeta.get("parent_id")
        root_path = _store_path(parent_id) if parent_id else None
        return root_path, (ox, oy)
    return _store_path(image_id), (0.0, 0.0)


def _record_agent_match_union(
    path_a: str,
    path_b: str,
    result: dict[str, Any],
    session: str = "",
    offset_a: tuple[float, float] | list[float] | None = None,
    offset_b: tuple[float, float] | list[float] | None = None,
) -> None:
    """Remap crop keypoints to root full-frame coords and union-merge into ``_match_store``.

    Correspondences are deduplicated at 0.1 px rounding so subsequent crop matches
    monotonically grow (and never overwrite or shrink) the full-frame correspondence
    set for an image pair.
    """
    if not (result.get("keypoints_a") and result.get("keypoints_b")):
        return

    resolved_a, meta_off_a = _root_path_and_offset(path_a)
    resolved_b, meta_off_b = _root_path_and_offset(path_b)
    actual_path_a = resolved_a or _CROP_META.get(path_a, {}).get("parent_id") or path_a
    actual_path_b = resolved_b or _CROP_META.get(path_b, {}).get("parent_id") or path_b

    if offset_a is not None:
        ox_a, oy_a = float(offset_a[0]), float(offset_a[1])
    elif result.get("offset_a") is not None:
        ox_a, oy_a = float(result["offset_a"][0]), float(result["offset_a"][1])
    else:
        ox_a, oy_a = meta_off_a

    if offset_b is not None:
        ox_b, oy_b = float(offset_b[0]), float(offset_b[1])
    elif result.get("offset_b") is not None:
        ox_b, oy_b = float(result["offset_b"][0]), float(result["offset_b"][1])
    else:
        ox_b, oy_b = meta_off_b

    raw_kpts_a = result["keypoints_a"]
    raw_kpts_b = result["keypoints_b"]
    n_pts = min(len(raw_kpts_a), len(raw_kpts_b))
    if n_pts == 0:
        return

    shifted_a = [
        [float(raw_kpts_a[i][0]) + ox_a, float(raw_kpts_a[i][1]) + oy_a]
        for i in range(n_pts)
    ]
    shifted_b = [
        [float(raw_kpts_b[i][0]) + ox_b, float(raw_kpts_b[i][1]) + oy_b]
        for i in range(n_pts)
    ]
    raw_mask = list(result.get("inlier_mask") or [])

    sess = session or _DEFAULT_SESSION
    bucket = _match_store.setdefault(sess, OrderedDict())
    _match_store.move_to_end(sess)
    while len(_match_store) > _MATCH_STORE_MAX_SESSIONS:
        _match_store.popitem(last=False)

    abs_a = os.path.abspath(actual_path_a)
    abs_b = os.path.abspath(actual_path_b)
    key = tuple(sorted((abs_a, abs_b)))

    existing = bucket.get(key)
    if existing is None:
        dedup_a: list[list[float]] = []
        dedup_b: list[list[float]] = []
        dedup_mask: list[bool] = []
        seen: dict[tuple[float, float, float, float], int] = {}
        has_mask = len(raw_mask) > 0
        for i in range(n_pts):
            ka, kb = shifted_a[i], shifted_b[i]
            inl = bool(raw_mask[i]) if i < len(raw_mask) else True
            ck = (round(ka[0], 1), round(ka[1], 1), round(kb[0], 1), round(kb[1], 1))
            if ck not in seen:
                seen[ck] = len(dedup_a)
                dedup_a.append(ka)
                dedup_b.append(kb)
                if has_mask:
                    dedup_mask.append(inl)
            elif has_mask and inl:
                dedup_mask[seen[ck]] = True
        bucket[key] = {
            "path_a": abs_a,
            "path_b": abs_b,
            "keypoints_a": dedup_a,
            "keypoints_b": dedup_b,
            "inlier_mask": dedup_mask if has_mask else [],
            "matcher": result.get("matcher", "agent"),
        }
    else:
        if abs_a == existing["path_b"] and abs_b == existing["path_a"] and abs_a != abs_b:
            shifted_a, shifted_b = shifted_b, shifted_a
        merged_a = list(existing["keypoints_a"])
        merged_b = list(existing["keypoints_b"])
        existing_mask = list(existing.get("inlier_mask") or [])
        has_mask = bool(existing_mask) or bool(raw_mask)
        if has_mask and len(existing_mask) < len(merged_a):
            existing_mask.extend([True] * (len(merged_a) - len(existing_mask)))

        seen = {}
        for idx, (ka, kb) in enumerate(zip(merged_a, merged_b)):
            ck = (
                round(float(ka[0]), 1),
                round(float(ka[1]), 1),
                round(float(kb[0]), 1),
                round(float(kb[1]), 1),
            )
            seen[ck] = idx

        for i in range(n_pts):
            ka, kb = shifted_a[i], shifted_b[i]
            inl = bool(raw_mask[i]) if i < len(raw_mask) else True
            ck = (round(ka[0], 1), round(ka[1], 1), round(kb[0], 1), round(kb[1], 1))
            if ck not in seen:
                seen[ck] = len(merged_a)
                merged_a.append(ka)
                merged_b.append(kb)
                if has_mask:
                    existing_mask.append(inl)
            elif has_mask and inl:
                existing_mask[seen[ck]] = True

        existing["keypoints_a"] = merged_a
        existing["keypoints_b"] = merged_b
        existing["inlier_mask"] = existing_mask if has_mask else []
        if result.get("matcher"):
            existing["matcher"] = result["matcher"]


def _record_agent_match(
    path_a: str,
    path_b: str,
    result: dict[str, Any],
    session: str = "",
    offset_a: tuple[float, float] | list[float] | None = None,
    offset_b: tuple[float, float] | list[float] | None = None,
) -> None:
    """Cache a match result's keypoints for later COLMAP import (monotonic union)."""
    _record_agent_match_union(
        path_a, path_b, result, session=session, offset_a=offset_a, offset_b=offset_b
    )

# Matcher weights are large; instantiate once per process (GRPO calls match every step).
_LOFTR = None
_MAST3R = None
_LIGHTGLUE = None
_SUPERPOINT = None


def _torch_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_loftr():
    global _LOFTR
    if _LOFTR is not None:
        return _LOFTR
    import kornia.feature as KF

    device = _torch_device()
    _LOFTR = KF.LoFTR(pretrained="outdoor").to(device).eval()
    logger.info("Cached LoFTR on %s", device)
    return _LOFTR


def _get_mast3r():
    global _MAST3R
    if _MAST3R is not None:
        return _MAST3R
    from mast3r.model import AsymmetricMASt3R

    device = _torch_device()
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    _MAST3R = AsymmetricMASt3R.from_pretrained(model_name).to(device).eval()
    logger.info("Cached MASt3R on %s", device)
    return _MAST3R


def _get_lightglue():
    global _LIGHTGLUE, _SUPERPOINT
    if _LIGHTGLUE is not None and _SUPERPOINT is not None:
        return _LIGHTGLUE, _SUPERPOINT
    import kornia.feature as KF

    device = _torch_device()
    _LIGHTGLUE = KF.LightGlueMatcher("superpoint").to(device).eval()
    _SUPERPOINT = KF.SuperPoint().to(device).eval()
    logger.info("Cached LightGlue + SuperPoint on %s", device)
    return _LIGHTGLUE, _SUPERPOINT


_VGGT = None


def _get_vggt():
    global _VGGT
    if _VGGT is not None:
        return _VGGT
    try:
        from vggt.models.vggt import VGGT
    except ImportError:
        raise ImportError("vggt package not available")
    device = _torch_device()
    ckpt = os.environ.get("AGENTIC_SFM_VGGT_CKPT", "facebook/VGGT-1B")
    _VGGT = VGGT.from_pretrained(ckpt).to(device).eval()
    logger.info("Cached VGGT on %s", device)
    return _VGGT


_MAPANYTHING = None


def _get_mapanything():
    global _MAPANYTHING
    if _MAPANYTHING is not None:
        return _MAPANYTHING
    try:
        from mapanything.models import MapAnything
    except ImportError:
        raise ImportError("mapanything package not available")
    device = _torch_device()
    ckpt = os.environ.get("AGENTIC_SFM_MAPANYTHING_CKPT", "facebook/map-anything-apache")
    _MAPANYTHING = MapAnything.from_pretrained(ckpt).to(device).eval()
    logger.info("Cached MapAnything (%s) on %s", ckpt, device)
    return _MAPANYTHING


def register_image(image_id: str, path_or_array: str | np.ndarray) -> None:
    _IMAGE_STORE[image_id] = path_or_array


def get_image(image_id: str) -> np.ndarray:
    entry = _IMAGE_STORE.get(image_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")
    if isinstance(entry, str):
        img = cv2.imread(entry)
        if img is None:
            raise HTTPException(status_code=400, detail=f"Cannot read image: {entry}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return entry


def image_to_base64(img: np.ndarray, max_size: int = 1024) -> str:
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode("utf-8")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _bbox_to_pixels(bbox: list[float], w: int, h: int) -> tuple[int, int, int, int] | str:
    """Convert a model bbox to a valid pixel crop box, or an error string.

    Accepts either an all-normalized [0,1] box or an all-pixel box. Mixed units
    (e.g. [0.2, 0.3, 0.8, 937]) are rejected with a teaching message: silently
    clamping mixed boxes previously made malformed output reward-equivalent to
    a correct crop, and RL drifted into emitting pixel heights.
    """
    if len(bbox) != 4:
        return f"bbox must have 4 values, got {len(bbox)}"
    vals = [float(v) for v in bbox]
    normalized = all(v <= 1.0 for v in vals)
    pixels = all(v > 1.0 for v in vals)
    if not (normalized or pixels):
        return (f"bbox units must be all-normalized [0,1] or all-pixels; got {bbox}. "
                f"Normalized example: [0.2, 0.3, 0.8, 0.9]")
    if normalized:
        px = sorted(max(0, min(w, int(v * w))) for v in (vals[0], vals[2]))
        py = sorted(max(0, min(h, int(v * h))) for v in (vals[1], vals[3]))
    else:
        px = sorted(max(0, min(w, int(v))) for v in (vals[0], vals[2]))
        py = sorted(max(0, min(h, int(v))) for v in (vals[1], vals[3]))
    if px[1] - px[0] < 4 or py[1] - py[0] < 4:
        return f"degenerate bbox {bbox} -> ({px[0]}, {py[0]}, {px[1]}, {py[1]})"
    # Minimum-size floor. Tiny crops (e.g. [0.2,0.2,0.3,0.3]) were a reward-hack
    # attractor: on a hopeless pair they fail slightly less badly than a large
    # crop, so group-relative advantage reinforced them until the policy slid
    # into degenerate boxes everywhere (Phase-1 epoch 2, 2026-09-16). Every SFT
    # box has a min side of 0.6-1.0 and every oracle box >= 0.7, so this floor
    # cannot reject a legitimate crop.
    min_side = MIN_CROP_FRACTION * min(w, h)
    if (px[1] - px[0]) < min_side or (py[1] - py[0]) < min_side:
        return (
            f"crop too small: {bbox} -> {px[1] - px[0]}x{py[1] - py[0]} px; the "
            f"shortest side must be >= {MIN_CROP_FRACTION:.0%} of the image "
            f"({int(min_side)} px). Example: [0.2, 0.3, 0.8, 0.9]"
        )
    return px[0], py[0], px[1], py[1]


def tool_crop(image_id: str, bbox: list[float]) -> dict[str, Any]:
    """Crop image to bbox [x1, y1, x2, y2]; accepts all-normalized or all-pixel."""
    img = get_image(image_id)
    h, w = img.shape[:2]
    box = _bbox_to_pixels(bbox, w, h)
    if isinstance(box, str):
        return {"error": box}
    px1, py1, px2, py2 = box
    cropped = img[py1:py2, px1:px2]
    new_id = f"{image_id}_crop_{uuid.uuid4().hex[:8]}"
    _IMAGE_STORE[new_id] = cropped
    parent = _CROP_META.get(image_id, {})
    ox = int(parent.get("origin_xy", [0, 0])[0]) + px1
    oy = int(parent.get("origin_xy", [0, 0])[1]) + py1
    parent_id = parent.get("parent_id", image_id)
    _CROP_META[new_id] = {"origin_xy": [ox, oy], "parent_id": parent_id}
    crop_path = str(Path(tempfile.gettempdir()) / f"{new_id}.jpg")
    try:
        Image.fromarray(cropped).save(crop_path)
    except Exception:
        crop_path = None
    return {
        "cropped_image_id": new_id,
        "crop_id": new_id,
        "parent_id": parent_id,
        "origin_xy": [ox, oy],
        "crop_size": [px2 - px1, py2 - py1],
        "bbox_normalized": [round(px1 / w, 3), round(py1 / h, 3),
                            round(px2 / w, 3), round(py2 / h, 3)],
        "path": crop_path,
        "image_b64": image_to_base64(cropped, max_size=512),
    }


_VALID_MATCHERS = ("loftr", "lightglue", "mast3r", "vggt", "mapanything")


def tool_match(
    image_a: str,
    image_b: str,
    matcher: str = "loftr",
    max_size: int = 512,
    K_a: list | None = None,
    K_b: list | None = None,
    session: str = "",
) -> dict[str, Any]:
    """Match two images and estimate relative pose."""
    from agentic_sfm.geometry import k_for_image

    if matcher not in _VALID_MATCHERS:
        return {
            "error": f"unknown matcher '{matcher}'; valid: {', '.join(_VALID_MATCHERS)}",
            "matcher": matcher,
        }
    if image_a == image_b:
        return {"error": "image_a and image_b are the same — self-match is not informative"}

    img_a = get_image(image_a)
    img_b = get_image(image_b)
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    try:
        Ka = k_for_image(K_a, _CROP_META.get(image_a, {}).get("origin_xy"), (w_a, h_a))
        Kb = k_for_image(K_b, _CROP_META.get(image_b, {}).get("origin_xy"), (w_b, h_b))
    except ValueError as e:
        return {"error": str(e), "num_inliers": 0, "inlier_ratio": 0.0}
    kwargs = {"K_a": Ka, "K_b": Kb, "session": session}

    # Resolve both images (including crops) to root full-frame file paths for
    # the agent-matches -> COLMAP correspondence union cache.
    path_a, _off_a = _root_path_and_offset(image_a)
    path_b, _off_b = _root_path_and_offset(image_b)
    if path_a and path_b and os.path.abspath(path_a) == os.path.abspath(path_b):
        return {"error": "image_a and image_b resolve to the same file — self-match is not informative"}
    cache_pair = (
        (image_a if image_a in _CROP_META else path_a),
        (image_b if image_b in _CROP_META else path_b),
    ) if (path_a and path_b) else None

    if matcher == "loftr":
        res = _match_loftr(img_a, img_b, max_size, cache_pair=cache_pair, **kwargs)
    elif matcher == "lightglue":
        res = _match_lightglue(img_a, img_b, max_size, cache_pair=cache_pair, **kwargs)
    elif matcher == "vggt":
        res = _match_vggt(img_a, img_b, max_size, cache_pair=cache_pair, **kwargs)
    elif matcher == "mapanything":
        res = _match_mapanything(img_a, img_b, max_size, cache_pair=cache_pair, **kwargs)
    else:
        res = _match_mast3r(img_a, img_b, max_size, cache_pair=cache_pair, **kwargs)

    if cache_pair is not None and not res.get("error"):
        _record_agent_match_union(cache_pair[0], cache_pair[1], res, session=session)
    return res


def _match_loftr(
    img_a: np.ndarray, img_b: np.ndarray, max_size: int,
    K_a=None, K_b=None, cache_pair: tuple[str, str] | None = None,
    session: str = "",
) -> dict[str, Any]:
    """Match using LoFTR (kornia)."""
    try:
        matcher = _get_loftr()
    except ImportError:
        return {"error": "kornia not available", "matcher": "loftr"}
    except Exception as e:
        return {"error": f"LoFTR load failed: {e}", "matcher": "loftr"}

    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    scale_a = min(1.0, max_size / max(h_a, w_a))
    scale_b = min(1.0, max_size / max(h_b, w_b))

    img_a_r = cv2.resize(img_a, (int(w_a * scale_a), int(h_a * scale_a)))
    img_b_r = cv2.resize(img_b, (int(w_b * scale_b), int(h_b * scale_b)))

    # LoFTR expects grayscale (1-channel) input
    if img_a_r.ndim == 3:
        img_a_r = cv2.cvtColor(img_a_r, cv2.COLOR_RGB2GRAY)
    if img_b_r.ndim == 3:
        img_b_r = cv2.cvtColor(img_b_r, cv2.COLOR_RGB2GRAY)

    t_a = torch.from_numpy(img_a_r).float().unsqueeze(0).unsqueeze(0) / 255.0
    t_b = torch.from_numpy(img_b_r).float().unsqueeze(0).unsqueeze(0) / 255.0

    device = _torch_device()
    t_a, t_b = t_a.to(device), t_b.to(device)

    with torch.no_grad():
        input_dict = {"image0": t_a, "image1": t_b}
        correspondences = matcher(input_dict)

    mkpts_a = correspondences["keypoints0"].cpu().numpy()
    mkpts_b = correspondences["keypoints1"].cpu().numpy()
    confidence = correspondences["confidence"].cpu().numpy()

    # Scale back to original pixels
    mkpts_a = mkpts_a / scale_a
    mkpts_b = mkpts_b / scale_b

    pose_result = _estimate_pose_ransac(
        mkpts_a, mkpts_b, (w_a, h_a), (w_b, h_b), K_a=K_a, K_b=K_b
    )

    if cache_pair is not None:
        # Cache the FULL correspondence set for the agent-matches → COLMAP
        # importer (the API response keeps [:100] for the visualiser; the
        # reconstruction needs denser keypoints to triangulate).
        _record_agent_match(cache_pair[0], cache_pair[1], {
            "keypoints_a": mkpts_a.tolist(),
            "keypoints_b": mkpts_b.tolist(),
            "inlier_mask": pose_result.get("inlier_mask") or [],
            "matcher": "loftr",
        }, session=session)

    return {
        "matcher": "loftr",
        "num_matches": len(mkpts_a),
        "num_inliers": pose_result["num_inliers"],
        "pose": pose_result["pose"],
        "inlier_ratio": pose_result["inlier_ratio"],
        "mean_residual": pose_result.get("mean_inlier_residual"),
        "residual_units": pose_result.get("residual_units"),
        "keypoints_a": mkpts_a.tolist()[:100],
        "keypoints_b": mkpts_b.tolist()[:100],
        "inlier_mask": (pose_result.get("inlier_mask") or [])[:100],
    }


def _match_mast3r(
    img_a: np.ndarray, img_b: np.ndarray, max_size: int,
    K_a=None, K_b=None, cache_pair: tuple[str, str] | None = None,
    session: str = "",
) -> dict[str, Any]:
    """Match using MASt3R."""
    try:
        from mast3r.fast_nn import fast_reciprocal_NNs
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        model = _get_mast3r()
    except ImportError:
        logger.warning("MASt3R not available, falling back to LoFTR")
        res = _match_loftr(img_a, img_b, max_size, K_a=K_a, K_b=K_b,
                           cache_pair=cache_pair, session=session)
        res["matcher_fallback"] = "mast3r->loftr"
        return res
    except Exception:
        logger.warning("Cannot load MASt3R, falling back to LoFTR")
        res = _match_loftr(img_a, img_b, max_size, K_a=K_a, K_b=K_b,
                           cache_pair=cache_pair, session=session)
        res["matcher_fallback"] = "mast3r->loftr"
        return res

    device = _torch_device()

    # Save to temp files for dust3r loader
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_a, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_b:
        Image.fromarray(img_a).save(f_a.name)
        Image.fromarray(img_b).save(f_b.name)
        tmp_a, tmp_b = f_a.name, f_b.name

    try:
        images = load_images([tmp_a, tmp_b], size=max_size)
        output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

        view1, pred1 = output["view1"], output["pred1"]
        view2, pred2 = output["view2"], output["pred2"]

        desc1 = pred1["desc"].squeeze(0).detach()
        desc2 = pred2["desc"].squeeze(0).detach()

        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2, subsample_or_initxy1=8, device=device, dist="dot", block_size=2**13
        )

        # Filter border matches
        H0, W0 = view1["true_shape"][0]
        H1, W1 = view2["true_shape"][0]
        valid = (
            (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) &
            (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3) &
            (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) &
            (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)
        )
        matches_im0 = matches_im0[valid].astype(np.float64)
        matches_im1 = matches_im1[valid].astype(np.float64)
        # dust3r resizes to max_size; map keypoints back to original pixels for GT K.
        h_orig, w_orig = img_a.shape[:2]
        h_orig_b, w_orig_b = img_b.shape[:2]
        matches_im0[:, 0] *= w_orig / max(float(W0), 1.0)
        matches_im0[:, 1] *= h_orig / max(float(H0), 1.0)
        matches_im1[:, 0] *= w_orig_b / max(float(W1), 1.0)
        matches_im1[:, 1] *= h_orig_b / max(float(H1), 1.0)

        pose_result = _estimate_pose_ransac(
            matches_im0, matches_im1, (w_orig, h_orig), (w_orig_b, h_orig_b),
            K_a=K_a, K_b=K_b,
        )

        if cache_pair is not None:
            _record_agent_match(cache_pair[0], cache_pair[1], {
                "keypoints_a": matches_im0.tolist(),
                "keypoints_b": matches_im1.tolist(),
                "inlier_mask": pose_result.get("inlier_mask") or [],
                "matcher": "mast3r",
            }, session=session)

        return {
            "matcher": "mast3r",
            "num_matches": len(matches_im0),
            "num_inliers": pose_result["num_inliers"],
            "pose": pose_result["pose"],
            "inlier_ratio": pose_result["inlier_ratio"],
            "mean_residual": pose_result.get("mean_inlier_residual"),
            "residual_units": pose_result.get("residual_units"),
            "keypoints_a": matches_im0.tolist()[:100],
            "keypoints_b": matches_im1.tolist()[:100],
            "inlier_mask": (pose_result.get("inlier_mask") or [])[:100],
        }
    finally:
        os.unlink(tmp_a)
        os.unlink(tmp_b)


def _run_vggt_pair(img_a: np.ndarray, img_b: np.ndarray, max_size: int) -> dict[str, Any]:
    """Run VGGT on an image pair; returns point maps, confidence, extrinsics.

    The ONLY function that talks to the model — tests monkeypatch this so the
    matcher logic (NN correspondence, pose RANSAC, cache) is exercisable
    without model weights or torch.

    Both images are letterboxed (aspect-preserving resize + centered black
    padding) onto a shared square canvas and run through a SINGLE S=2 forward
    pass. This is required, not cosmetic: a single-view (S=1) forward emits
    its point map in that view's own canonical camera frame, so two separate
    forwards would produce points in unrelated coordinate frames and the
    NN correspondences / extrinsics would be meaningless. Do NOT "optimize"
    this into per-image forwards.

    The per-image letterbox parameters are returned as ``map_a``/``map_b``
    (``scale``, ``scale_y`` = post-/14-snap resize factors; ``pad_x``,
    ``pad_y`` = centered offsets) so callers can invert point-map pixels to
    original pixels exactly: ``x_orig = (x_pm - pad_x) / scale``. Note the
    dense head predicts a point for EVERY canvas pixel, including the black
    pad bars — callers must bound-check inverted coordinates against the
    original image dimensions.
    """
    import torchvision.transforms as TF

    model = _get_vggt()
    try:
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except ImportError:
        def pose_encoding_to_extri_intri(pose_enc: torch.Tensor, image_size_hw: tuple[int, int]):
            b, s, _ = pose_enc.shape
            ext = torch.eye(4, dtype=pose_enc.dtype, device=pose_enc.device)[:3, :4].unsqueeze(0).unsqueeze(0).expand(b, s, 3, 4).clone()
            intri = torch.eye(3, dtype=pose_enc.dtype, device=pose_enc.device).unsqueeze(0).unsqueeze(0).expand(b, s, 3, 3).clone()
            return ext, intri
    device = _torch_device()
    to_tensor = TF.ToTensor()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_a, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_b:
        Image.fromarray(img_a).save(f_a.name)
        Image.fromarray(img_b).save(f_b.name)
        tmp_a, tmp_b = f_a.name, f_b.name

    # Both views MUST go through ONE forward as an S=2 sequence: a single-view
    # (S=1) forward puts that view's point map in its own canonical camera
    # frame, so two separate forwards yield points in unrelated frames and the
    # NN correspondences/extrinsics would be garbage. Letterboxing to a common
    # square canvas keeps the mapping back to original pixels exact.
    # Square canvas: VGGT is trained at 518 (37*14); /14 snap required by the
    # patchify. max_size < 14 floors at 14 (never exceeds 518 either way).
    canvas = min(518, max_size - max_size % 14)
    canvas = max(canvas, 14)

    def _prep(path: str) -> tuple[torch.Tensor, dict[str, float]]:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        scale = canvas / max(w, h)
        new_w = min(canvas, max(14, round(w * scale / 14) * 14))
        new_h = min(canvas, max(14, round(h * scale / 14) * 14))
        scale = new_w / w  # after the /14 snap, x and y may differ
        scale_y = new_h / h
        im = im.resize((new_w, new_h), Image.Resampling.BICUBIC)
        pad_x = (canvas - new_w) // 2
        pad_y = (canvas - new_h) // 2
        sq = Image.new("RGB", (canvas, canvas), (0, 0, 0))
        sq.paste(im, (pad_x, pad_y))
        # Exact inverse: x_orig = (x_pm - pad_x) / scale
        return to_tensor(sq), {
            "scale": scale, "scale_y": scale_y,
            "pad_x": float(pad_x), "pad_y": float(pad_y),
        }

    def _forward(t: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            if device == "cuda":
                dtype = (torch.bfloat16
                         if torch.cuda.get_device_capability()[0] >= 8
                         else torch.float16)
                with torch.cuda.amp.autocast(dtype=dtype):
                    return model(t.to(device))
            return model(t)

    try:
        t_a, map_a = _prep(tmp_a)  # (3, canvas, canvas)
        t_b, map_b = _prep(tmp_b)
        # One forward, S=2 sequence — model adds the batch dim itself.
        preds = _forward(torch.stack([t_a, t_b]))

        def _n(x: torch.Tensor) -> np.ndarray:
            return x.detach().float().cpu().numpy()

        ext, _int = pose_encoding_to_extri_intri(
            preds["pose_enc"], (canvas, canvas)
        )
        ext = _n(ext[0])

        return {
            "points_a": _n(preds["world_points"][0, 0]).astype(np.float32),
            "points_b": _n(preds["world_points"][0, 1]).astype(np.float32),
            "conf_a": _n(preds["world_points_conf"][0, 0]).astype(np.float32),
            "conf_b": _n(preds["world_points_conf"][0, 1]).astype(np.float32),
            "ext_a": ext[0].astype(np.float64),
            "ext_b": ext[1].astype(np.float64),
            "map_a": map_a,
            "map_b": map_b,
        }
    finally:
        os.unlink(tmp_a)
        os.unlink(tmp_b)


def _json_sanitize(v):
    """Recursively replace NaN/inf floats so nothing non-JSON leaks out."""
    if isinstance(v, np.floating):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.ndarray):
        return _json_sanitize(v.tolist())
    if isinstance(v, float):
        return v if np.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _json_sanitize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_sanitize(x) for x in v]
    return v


def _match_vggt(
    img_a: np.ndarray, img_b: np.ndarray, max_size: int,
    K_a=None, K_b=None, cache_pair: tuple[str, str] | None = None,
    session: str = "",
) -> dict[str, Any]:
    """Match using VGGT dense point maps + extrinsics prior."""
    try:
        out = _run_vggt_pair(img_a, img_b, max_size)
        from agentic_sfm.pose_prior import (
            extrinsics_to_relative_pose,
            reciprocal_nn_matches,
        )
    except ImportError:
        logger.warning("VGGT not available, falling back to LoFTR")
        res = _match_loftr(img_a, img_b, max_size, K_a=K_a, K_b=K_b,
                           cache_pair=cache_pair, session=session)
        res["matcher_fallback"] = "vggt->loftr"
        return res
    except Exception:
        logger.warning("Cannot load/run VGGT, falling back to LoFTR", exc_info=True)
        res = _match_loftr(img_a, img_b, max_size, K_a=K_a, K_b=K_b,
                           cache_pair=cache_pair, session=session)
        res["matcher_fallback"] = "vggt->loftr"
        return res

    # world_points_conf uses conf_activation="expp1" — conf = 1 + exp(raw),
    # living in [1, inf), NOT a probability. The 5.0 default (VGGT's demo
    # value) left 862 of 1,170 benchmark pairs with no matches. On six ScanNet
    # pairs (outputs/diagnose_vggt_conf) 3.0 kept every pair's matches except
    # one that dropped from 6042 to 562, so 3.0 is the working default.
    conf_thr = float(os.environ.get("AGENTIC_SFM_VGGT_CONF_THRESHOLD", "3.0"))
    dist_thr_env = os.environ.get("AGENTIC_SFM_VGGT_DIST_THRESHOLD", "")
    dist_thr = float(dist_thr_env) if dist_thr_env else None
    max_pts = int(os.environ.get("AGENTIC_SFM_VGGT_MAX_POINTS", "20000"))

    pts_a_pm, pts_b_pm = reciprocal_nn_matches(
        out["points_a"], out["points_b"], out["conf_a"], out["conf_b"],
        conf_threshold=conf_thr, dist_threshold=dist_thr, max_points=max_pts,
    )

    # Map matched pixels from the point-map frame back to ORIGINAL pixels.
    # _run_vggt_pair letterboxes each image to a square canvas, so the exact
    # inverse is x_orig = (x_pm - pad_x) / scale. If a stub omits the maps,
    # fall back to plain proportional scaling (pm frame == full image).
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    Ha, Wa = out["points_a"].shape[:2]
    Hb, Wb = out["points_b"].shape[:2]
    matches_a = pts_a_pm.astype(np.float64)
    matches_b = pts_b_pm.astype(np.float64)
    if len(matches_a):
        map_a = out.get("map_a")
        map_b = out.get("map_b")
        if map_a and map_b:
            matches_a[:, 0] = (matches_a[:, 0] - map_a["pad_x"]) / map_a["scale"]
            matches_a[:, 1] = (matches_a[:, 1] - map_a["pad_y"]) / map_a.get(
                "scale_y", map_a["scale"])
            matches_b[:, 0] = (matches_b[:, 0] - map_b["pad_x"]) / map_b["scale"]
            matches_b[:, 1] = (matches_b[:, 1] - map_b["pad_y"]) / map_b.get(
                "scale_y", map_b["scale"])
        else:
            matches_a[:, 0] *= w_a / max(float(Wa), 1.0)
            matches_a[:, 1] *= h_a / max(float(Ha), 1.0)
            matches_b[:, 0] *= w_b / max(float(Wb), 1.0)
            matches_b[:, 1] *= h_b / max(float(Hb), 1.0)

    # Drop correspondences that land outside the original images — the dense
    # head predicts a point for every canvas pixel, including the letterbox
    # bars, and bar-to-bar matches can be mutually consistent (same black
    # bars in both views). Inverted pad pixels come out < 0 or >= w/h.
    # Mirrors the MASt3R border filter; the COLMAP cache must never see
    # out-of-bounds keypoints.
    num_matches_raw = len(matches_a)
    if num_matches_raw:
        in_bounds = (
            (matches_a[:, 0] >= 0) & (matches_a[:, 0] < w_a) &
            (matches_a[:, 1] >= 0) & (matches_a[:, 1] < h_a) &
            (matches_b[:, 0] >= 0) & (matches_b[:, 0] < w_b) &
            (matches_b[:, 1] >= 0) & (matches_b[:, 1] < h_b)
        )
        matches_a = matches_a[in_bounds]
        matches_b = matches_b[in_bounds]
        pts_a_pm = pts_a_pm[in_bounds]
        pts_b_pm = pts_b_pm[in_bounds]

    pose_result = _estimate_pose_ransac(
        matches_a, matches_b, (w_a, h_a), (w_b, h_b), K_a=K_a, K_b=K_b,
    )

    # Mean model confidence at the matched point-map pixels
    conf_vals = []
    if len(pts_a_pm):
        ia = np.clip(pts_a_pm.astype(int), [0, 0], [Wa - 1, Ha - 1])
        ib = np.clip(pts_b_pm.astype(int), [0, 0], [Wb - 1, Hb - 1])
        conf_vals = np.concatenate([
            out["conf_a"][ia[:, 1], ia[:, 0]],
            out["conf_b"][ib[:, 1], ib[:, 0]],
        ])
    prior_conf = float(np.mean(conf_vals)) if len(conf_vals) else None

    prior_pose = extrinsics_to_relative_pose(out["ext_a"], out["ext_b"])

    if cache_pair is not None:
        _record_agent_match(cache_pair[0], cache_pair[1], {
            "keypoints_a": matches_a.tolist(),
            "keypoints_b": matches_b.tolist(),
            "inlier_mask": pose_result.get("inlier_mask") or [],
            "matcher": "vggt",
        }, session=session)

    return _json_sanitize({
        "matcher": "vggt",
        "num_matches": len(matches_a),
        "num_matches_raw": num_matches_raw,
        "num_inliers": pose_result["num_inliers"],
        "pose": pose_result["pose"],
        "inlier_ratio": pose_result["inlier_ratio"],
        "mean_residual": pose_result.get("mean_inlier_residual"),
        "residual_units": pose_result.get("residual_units"),
        "keypoints_a": matches_a.tolist()[:100],
        "keypoints_b": matches_b.tolist()[:100],
        "inlier_mask": (pose_result.get("inlier_mask") or [])[:100],
        "prior_pose": prior_pose,
        "prior_confidence": prior_conf,
        "pointmap_size": [int(Wa), int(Ha)],
    })


def _run_mapanything_pair(img_a: np.ndarray, img_b: np.ndarray, max_size: int) -> dict[str, Any]:
    """Run MapAnything (facebook/map-anything-apache) on a two-view pair.

    Returns the same point-map + confidence + extrinsics dictionary contract as
    ``_run_vggt_pair`` so ``_match_mapanything`` shares the exact reciprocal-NN
    and ``extrinsics_to_relative_pose`` pipeline.
    """
    import torchvision.transforms as TF

    model = _get_mapanything()
    device = _torch_device()
    to_tensor = TF.ToTensor()

    canvas = min(518, max_size - max_size % 14)
    canvas = max(canvas, 14)

    def _prep_arr(arr: np.ndarray) -> tuple[torch.Tensor, dict[str, float]]:
        im = Image.fromarray(arr).convert("RGB")
        w, h = im.size
        scale = canvas / max(w, h)
        new_w = min(canvas, max(14, round(w * scale / 14) * 14))
        new_h = min(canvas, max(14, round(h * scale / 14) * 14))
        sx = new_w / w
        sy = new_h / h
        im = im.resize((new_w, new_h), Image.Resampling.BICUBIC)
        pad_x = (canvas - new_w) // 2
        pad_y = (canvas - new_h) // 2
        sq = Image.new("RGB", (canvas, canvas), (0, 0, 0))
        sq.paste(im, (pad_x, pad_y))
        return to_tensor(sq), {
            "scale": sx,
            "scale_y": sy,
            "pad_x": float(pad_x),
            "pad_y": float(pad_y),
        }

    t_a, map_a = _prep_arr(img_a)
    t_b, map_b = _prep_arr(img_b)
    views = [{"img": t_a.unsqueeze(0).to(device)}, {"img": t_b.unsqueeze(0).to(device)}]
    with torch.no_grad():
        preds = model(views)

    def _n(x: torch.Tensor) -> np.ndarray:
        return x.detach().float().cpu().numpy()

    p0, p1 = preds[0], preds[1]
    pts_a = _n(p0["pts3d"][0]).astype(np.float32)
    pts_b = _n(p1["pts3d"][0]).astype(np.float32)
    conf_a = _n(p0.get("conf", torch.ones_like(p0["pts3d"][..., 0]))[0]).astype(np.float32)
    conf_b = _n(p1.get("conf", torch.ones_like(p1["pts3d"][..., 0]))[0]).astype(np.float32)
    ext_a = _n(p0.get("camera_poses", torch.eye(4).unsqueeze(0))[0])[:3, :4].astype(np.float64)
    ext_b = _n(p1.get("camera_poses", torch.eye(4).unsqueeze(0))[0])[:3, :4].astype(np.float64)
    return {
        "points_a": pts_a,
        "points_b": pts_b,
        "conf_a": conf_a,
        "conf_b": conf_b,
        "ext_a": ext_a,
        "ext_b": ext_b,
        "map_a": map_a,
        "map_b": map_b,
    }


def _match_mapanything(
    img_a: np.ndarray, img_b: np.ndarray, max_size: int,
    K_a=None, K_b=None, cache_pair: tuple[str, str] | None = None,
    session: str = "",
) -> dict[str, Any]:
    """Match using MapAnything metric 3D point maps + camera pose prior."""
    try:
        out = _run_mapanything_pair(img_a, img_b, max_size)
        from agentic_sfm.pose_prior import (
            extrinsics_to_relative_pose,
            reciprocal_nn_matches,
        )
    except Exception:
        logger.warning("MapAnything not available, falling back to LoFTR")
        res = _match_loftr(img_a, img_b, max_size, K_a=K_a, K_b=K_b,
                           cache_pair=cache_pair, session=session)
        res["matcher_fallback"] = "mapanything->loftr"
        return res

    conf_thr = float(os.environ.get("AGENTIC_SFM_MAPANYTHING_CONF_THRESHOLD", "1.5"))
    max_pts = int(os.environ.get("AGENTIC_SFM_MAPANYTHING_MAX_POINTS", "20000"))
    pts_a_pm, pts_b_pm = reciprocal_nn_matches(
        out["points_a"], out["points_b"], out["conf_a"], out["conf_b"],
        conf_threshold=conf_thr, dist_threshold=None, max_points=max_pts,
    )

    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    Ha, Wa = out["points_a"].shape[:2]
    Hb, Wb = out["points_b"].shape[:2]
    matches_a = pts_a_pm.astype(np.float64)
    matches_b = pts_b_pm.astype(np.float64)
    if len(matches_a):
        map_a = out.get("map_a")
        map_b = out.get("map_b")
        if map_a and map_b:
            matches_a[:, 0] = (matches_a[:, 0] - map_a["pad_x"]) / map_a["scale"]
            matches_a[:, 1] = (matches_a[:, 1] - map_a["pad_y"]) / map_a.get("scale_y", map_a["scale"])
            matches_b[:, 0] = (matches_b[:, 0] - map_b["pad_x"]) / map_b["scale"]
            matches_b[:, 1] = (matches_b[:, 1] - map_b["pad_y"]) / map_b.get("scale_y", map_b["scale"])
        else:
            matches_a[:, 0] *= w_a / max(float(Wa), 1.0)
            matches_a[:, 1] *= h_a / max(float(Ha), 1.0)
            matches_b[:, 0] *= w_b / max(float(Wb), 1.0)
            matches_b[:, 1] *= h_b / max(float(Hb), 1.0)

    num_matches_raw = len(matches_a)
    if num_matches_raw:
        in_bounds = (
            (matches_a[:, 0] >= 0) & (matches_a[:, 0] < w_a) &
            (matches_a[:, 1] >= 0) & (matches_a[:, 1] < h_a) &
            (matches_b[:, 0] >= 0) & (matches_b[:, 0] < w_b) &
            (matches_b[:, 1] >= 0) & (matches_b[:, 1] < h_b)
        )
        matches_a = matches_a[in_bounds]
        matches_b = matches_b[in_bounds]
        pts_a_pm = pts_a_pm[in_bounds]
        pts_b_pm = pts_b_pm[in_bounds]

    pose_result = _estimate_pose_ransac(
        matches_a, matches_b, (w_a, h_a), (w_b, h_b), K_a=K_a, K_b=K_b,
    )
    conf_vals = []
    if len(pts_a_pm):
        ia = np.clip(pts_a_pm.astype(int), [0, 0], [Wa - 1, Ha - 1])
        ib = np.clip(pts_b_pm.astype(int), [0, 0], [Wb - 1, Hb - 1])
        conf_vals = np.concatenate([
            out["conf_a"][ia[:, 1], ia[:, 0]],
            out["conf_b"][ib[:, 1], ib[:, 0]],
        ])
    prior_conf = float(np.mean(conf_vals)) if len(conf_vals) else None
    prior_pose = extrinsics_to_relative_pose(out["ext_a"], out["ext_b"])

    if cache_pair is not None:
        _record_agent_match(cache_pair[0], cache_pair[1], {
            "keypoints_a": matches_a.tolist(),
            "keypoints_b": matches_b.tolist(),
            "inlier_mask": pose_result.get("inlier_mask") or [],
            "matcher": "mapanything",
        }, session=session)

    return _json_sanitize({
        "matcher": "mapanything",
        "num_matches": len(matches_a),
        "num_matches_raw": num_matches_raw,
        "num_inliers": pose_result["num_inliers"],
        "pose": pose_result["pose"],
        "inlier_ratio": pose_result["inlier_ratio"],
        "mean_residual": pose_result.get("mean_inlier_residual"),
        "residual_units": pose_result.get("residual_units"),
        "keypoints_a": matches_a.tolist()[:100],
        "keypoints_b": matches_b.tolist()[:100],
        "inlier_mask": (pose_result.get("inlier_mask") or [])[:100],
        "prior_pose": prior_pose,
        "prior_confidence": prior_conf,
        "pointmap_size": [int(Wa), int(Ha)],
    })


def _match_lightglue(
    img_a: np.ndarray, img_b: np.ndarray, max_size: int,
    K_a=None, K_b=None, cache_pair: tuple[str, str] | None = None,
    session: str = "",
) -> dict[str, Any]:
    """Match using LightGlue (kornia)."""
    try:
        lg, sp = _get_lightglue()
    except ImportError:
        return {"error": "kornia not available", "matcher": "lightglue"}
    except Exception as e:
        return {"error": f"LightGlue load failed: {e}", "matcher": "lightglue"}

    device = _torch_device()

    t_a = _to_tensor(img_a, max_size).to(device)
    t_b = _to_tensor(img_b, max_size).to(device)

    with torch.no_grad():
        la = sp(t_a)
        lb = sp(t_b)
        matches = lg(la["descriptors"], lb["descriptors"], la["keypoints"], lb["keypoints"])

    mkpts_a = la["keypoints"][0][matches[0]].cpu().numpy()
    mkpts_b = lb["keypoints"][0][matches[1]].cpu().numpy()

    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    scale_a = max(h_a, w_a) / max_size
    scale_b = max(h_b, w_b) / max_size
    mkpts_a = mkpts_a * scale_a
    mkpts_b = mkpts_b * scale_b

    pose_result = _estimate_pose_ransac(
        mkpts_a, mkpts_b, (w_a, h_a), (w_b, h_b), K_a=K_a, K_b=K_b
    )

    if cache_pair is not None:
        _record_agent_match(cache_pair[0], cache_pair[1], {
            "keypoints_a": mkpts_a.tolist(),
            "keypoints_b": mkpts_b.tolist(),
            "inlier_mask": pose_result.get("inlier_mask") or [],
            "matcher": "lightglue",
        }, session=session)

    return {
        "matcher": "lightglue",
        "num_matches": len(mkpts_a),
        "num_inliers": pose_result["num_inliers"],
        "pose": pose_result["pose"],
        "inlier_ratio": pose_result["inlier_ratio"],
        "mean_residual": pose_result.get("mean_inlier_residual"),
        "residual_units": pose_result.get("residual_units"),
        "keypoints_a": mkpts_a.tolist()[:100],
        "keypoints_b": mkpts_b.tolist()[:100],
        "inlier_mask": (pose_result.get("inlier_mask") or [])[:100],
    }


def _to_tensor(img: np.ndarray, max_size: int) -> torch.Tensor:
    h, w = img.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    img_r = cv2.resize(img, (int(w * scale), int(h * scale)))
    return torch.from_numpy(img_r).float().permute(2, 0, 1).unsqueeze(0) / 255.0


def _estimate_pose_ransac(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    size_a: tuple,
    size_b: tuple,
    K_a=None,
    K_b=None,
) -> dict[str, Any]:
    """Estimate relative pose via MAGSAC essential matrix + GT/guessed K."""
    from agentic_sfm.geometry import estimate_relative_pose

    return estimate_relative_pose(pts_a, pts_b, size_a, size_b, K_a=K_a, K_b=K_b)


_DOPPELGANGER_DETECTOR = None


def _get_doppelganger_detector():
    """Lazy-load the shared DINOv2+geometry doppelganger detector."""
    global _DOPPELGANGER_DETECTOR
    if _DOPPELGANGER_DETECTOR is not None:
        return _DOPPELGANGER_DETECTOR
    try:
        from agentic_sfm.tools.doppelganger import DoppelgangerDetector

        _DOPPELGANGER_DETECTOR = DoppelgangerDetector(device=_torch_device())
    except Exception as e:
        logger.warning("Could not create DoppelgangerDetector: %s", e)
        _DOPPELGANGER_DETECTOR = None
    return _DOPPELGANGER_DETECTOR


def tool_doppelganger_check(image_a: str, image_b: str) -> dict[str, Any]:
    """Check if image pair is a doppelganger (visually similar but distinct).

    Uses the DINOv2 + geometric-consistency detector from
    ``agentic_sfm.tools.doppelganger``: high appearance similarity with
    failed essential-matrix verification -> calibrated confidence.
    Falls back to the legacy match-count heuristic on failure.
    """
    match_result = tool_match(image_a, image_b, matcher="loftr")

    detector = _get_doppelganger_detector()
    if detector is not None:
        # Resolve registered ids to paths / arrays for the embedder.
        entry_a = _IMAGE_STORE.get(image_a)
        entry_b = _IMAGE_STORE.get(image_b)
        if isinstance(entry_a, str) and isinstance(entry_b, str):
            try:
                return detector.check(entry_a, entry_b, match_result=match_result)
            except Exception as e:
                logger.error("Doppelganger detector failed: %s", e)

    # Heuristic fallback — preserves the old contract on any failure.
    num_matches = match_result.get("num_matches", 0)
    inlier_ratio = match_result.get("inlier_ratio", 0.0)
    is_doppelganger = num_matches > 50 and inlier_ratio < 0.1
    confidence = (1.0 - inlier_ratio) * 0.6 if is_doppelganger else 0.0

    return {
        "is_doppelganger": is_doppelganger,
        "confidence": confidence,
        "score": confidence,
        "similarity_score": None,
        "num_matches": num_matches,
        "inlier_ratio": inlier_ratio,
        "num_inliers": int(match_result.get("num_inliers") or 0),
        "residual_units": match_result.get("residual_units"),
        "verdict": "doppelganger" if is_doppelganger else "uncertain",
        "method": "heuristic_fallback",
    }


def tool_retrieve(query_image: str, k: int = 5) -> dict[str, Any]:
    """Retrieve top-k candidate matching images."""
    # Placeholder: return all registered images
    candidates = [
        img_id for img_id in _IMAGE_STORE
        if img_id != query_image and "_crop_" not in img_id
    ]
    return {"candidates": candidates[:k]}


def tool_build_initial_graph(
    image_ids: list[str],
    k_neighbors: int = 5,
    matcher: str = "loftr",
    session: str = "",
) -> dict[str, Any]:
    """Initialize the view graph using learned visual retrieval and batch matching."""
    sess = session or _DEFAULT_SESSION
    if not image_ids:
        return {"error": "image_ids cannot be empty", "status": "error"}

    # Compute or retrieve DINOv2 embeddings
    valid_ids: list[str] = []
    embs: list[np.ndarray] = []
    for img_id in image_ids:
        path = _store_path(img_id)
        if path is None:
            continue
        valid_ids.append(img_id)
        if path in _EMBEDDING_CACHE:
            embs.append(_EMBEDDING_CACHE[path])
        else:
            try:
                emb = compute_image_embedding(path)
            except Exception as e:
                logger.warning(f"Embedding failed for {path}: {e}")
                emb = np.zeros(384, dtype=np.float32)
            _EMBEDDING_CACHE[path] = emb
            embs.append(emb)

    if len(valid_ids) < 2:
        return {
            "error": f"Fewer than 2 valid images resolved (found {len(valid_ids)})",
            "status": "error",
        }

    # Cosine similarity matrix
    embs_arr = np.stack(embs)
    norms = np.linalg.norm(embs_arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embs_norm = embs_arr / norms
    sim = embs_norm @ embs_norm.T

    # Select top-k nearest neighbors per image (canonical undirected pairs)
    k = max(1, min(k_neighbors, len(valid_ids) - 1))
    candidate_pairs: set[tuple[str, str]] = set()
    for i in range(len(valid_ids)):
        scores = sim[i].copy()
        scores[i] = -1e9  # exclude self
        top_idx = np.argsort(scores)[::-1][:k]
        for j in top_idx:
            u, v = valid_ids[i], valid_ids[j]
            key = (u, v) if u < v else (v, u)
            candidate_pairs.add(key)

    # Initialize graph
    graph = ViewGraph(valid_ids)

    # Batch match candidate pairs. Edges are labelled with the backend that
    # actually produced them: a fallback (e.g. vggt->loftr) must not be
    # recorded under the requested name.
    n_fallback = n_zero = 0
    inlier_counts: list[int] = []
    for u, v in sorted(candidate_pairs):
        match_res = tool_match(u, v, matcher=matcher, session=sess)
        if match_res.get("matcher_fallback"):
            n_fallback += 1
        num_inliers = int(match_res.get("num_inliers") or 0)
        inlier_counts.append(num_inliers)
        if int(match_res.get("num_matches") or 0) == 0:
            n_zero += 1
        num_matches = int(match_res.get("num_matches") or 0)
        inlier_ratio = float(match_res.get("inlier_ratio") or 0.0)
        pose = match_res.get("pose")
        R_rel = np.array(pose["R"], dtype=np.float64) if (pose and "R" in pose) else None
        t_rel = np.array(pose["t"], dtype=np.float64) if (pose and "t" in pose) else None
        status = "verified" if num_inliers >= 15 else "failed"

        edge = EdgeData(
            image_a=u,
            image_b=v,
            num_matches=num_matches,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            R_rel=R_rel,
            t_rel=t_rel,
            matcher=match_res.get("matcher", matcher),
            status=status,
        )
        graph.add_edge(u, v, edge)

    _SESSION_GRAPHS[sess] = graph

    summary = graph.get_summary()
    summary["num_candidates_selected"] = len(candidate_pairs)
    summary["matcher_requested"] = matcher
    summary["pairs_fallback"] = n_fallback
    summary["pairs_zero_matches"] = n_zero
    summary["median_inliers"] = float(np.median(inlier_counts)) if inlier_counts else 0.0
    summary["status"] = "ok"
    return summary


def tool_inspect_graph(session: str = "") -> dict[str, Any]:
    """Inspect the current session's view graph topology and diagnostics."""
    sess = session or _DEFAULT_SESSION
    graph = _SESSION_GRAPHS.get(sess)
    if graph is None:
        return {
            "error": f"No active view graph found for session '{sess}'. Call build_initial_graph first.",
            "status": "error",
        }
    summary = graph.get_summary()
    summary["components"] = graph.get_connected_components()
    # Active edge list so the episode runner knows which pairs were actually
    # in play (used to derive GT doppelganger labels at episode end).  Capped
    # so huge graphs don't blow the agent's context budget.
    active = list(graph.active_edges().keys())
    summary["edges"] = [[u, v] for (u, v) in active[:_INSPECT_GRAPH_MAX_EDGES]]
    summary["num_edges_listed"] = len(summary["edges"])
    summary["status"] = "ok"
    return summary


def tool_verify_doppelganger(
    image_a: str,
    image_b: str,
    session: str = "",
    matcher: str = "loftr",
) -> dict[str, Any]:
    """Verify if an image pair is a visual doppelganger using multi-modal evidence."""
    sess = session or _DEFAULT_SESSION
    graph = _SESSION_GRAPHS.get(sess)

    path_a = _store_path(image_a)
    path_b = _store_path(image_b)

    # 1. Visual similarity via embeddings
    vis_sim = 0.0
    if path_a and path_b:
        emb_a = _EMBEDDING_CACHE.get(path_a)
        if emb_a is None:
            emb_a = compute_image_embedding(path_a)
            _EMBEDDING_CACHE[path_a] = emb_a
        emb_b = _EMBEDDING_CACHE.get(path_b)
        if emb_b is None:
            emb_b = compute_image_embedding(path_b)
            _EMBEDDING_CACHE[path_b] = emb_b

        denom = max(float(np.linalg.norm(emb_a) * np.linalg.norm(emb_b)), 1e-8)
        vis_sim = float(np.dot(emb_a, emb_b) / denom)

    # 2. Local matching evidence
    u, v = (image_a, image_b) if image_a < image_b else (image_b, image_a)
    edge = graph.edges.get((u, v)) if graph else None

    if edge is not None and edge.status == "verified":
        inliers = edge.num_inliers
        ratio = edge.inlier_ratio
        R_uv = edge.R_rel
    else:
        match_res = tool_match(image_a, image_b, matcher=matcher, session=sess)
        if match_res.get("error"):
            # Surface matcher errors (e.g. an unknown matcher name) as an
            # error so the episode counts the call as invalid, instead of
            # reading missing fields as a genuine "no inliers" verdict.
            return {"error": match_res["error"], "pair": [image_a, image_b]}
        inliers = int(match_res.get("num_inliers") or 0)
        ratio = float(match_res.get("inlier_ratio") or 0.0)
        pose = match_res.get("pose")
        R_uv = np.array(pose["R"], dtype=np.float64) if (pose and "R" in pose) else None

    # 3. Cycle rotation residual check across common neighbors
    cycle_errors: list[float] = []
    if graph is not None and R_uv is not None:
        active = graph.active_edges()
        adj: dict[str, dict[str, np.ndarray]] = {n: {} for n in graph.nodes}
        for (i, j), e in active.items():
            if e.R_rel is not None and e.num_inliers >= 15:
                R = np.asarray(e.R_rel, dtype=np.float64).reshape(3, 3)
                adj[i][j] = R
                adj[j][i] = R.T

        if u in adj and v in adj:
            common = set(adj[u].keys()) & set(adj[v].keys())
            for w in common:
                # Loop: u -> v -> w -> u: R_wu @ R_vw @ R_uv
                R_loop = adj[w][u] @ adj[v][w] @ R_uv
                tr = np.clip((np.trace(R_loop) - 1.0) / 2.0, -1.0, 1.0)
                angle_deg = float(np.degrees(np.arccos(tr)))
                cycle_errors.append(angle_deg)

    max_cycle_err = max(cycle_errors) if cycle_errors else None

    # Multi-modal verdict
    is_dop = False
    evidence_parts: list[str] = []
    if max_cycle_err is not None and max_cycle_err > 15.0 and vis_sim > 0.70:
        is_dop = True
        evidence_parts.append(
            f"High cycle rotation residual ({max_cycle_err:.1f}° > 15°) with high appearance similarity ({vis_sim:.2f}) indicates symmetry fold/doppelganger."
        )
    elif vis_sim > 0.80 and inliers < 15:
        is_dop = True
        evidence_parts.append(
            f"High appearance similarity ({vis_sim:.2f}) but low inlier count ({inliers} < 15) indicates repetitive visual texture without valid epipolar geometry."
        )
    elif vis_sim > 0.80 and ratio < 0.15:
        is_dop = True
        evidence_parts.append(
            f"High appearance similarity ({vis_sim:.2f}) but poor inlier ratio ({ratio:.2f}) suggests spurious matches on repetitive structures."
        )
    else:
        evidence_parts.append(
            f"Consistent geometry (inliers={inliers}, ratio={ratio:.2f}, sim={vis_sim:.2f}"
            + (f", cycle_err={max_cycle_err:.1f}°" if max_cycle_err is not None else "")
            + ")."
        )

    evidence = " ".join(evidence_parts)

    return {
        "pair": [image_a, image_b],
        "visual_similarity": round(vis_sim, 3),
        "num_inliers": inliers,
        "inlier_ratio": round(ratio, 3),
        "cycle_rotation_error_deg": round(max_cycle_err, 2) if max_cycle_err is not None else None,
        "is_doppelganger": is_dop,
        "evidence": evidence,
        "status": "ok",
    }


def tool_prune_edges(pairs: list[list[str]], session: str = "") -> dict[str, Any]:
    """Prune specified edges from the session's active view graph."""
    sess = session or _DEFAULT_SESSION
    graph = _SESSION_GRAPHS.get(sess)
    if graph is None:
        return {"error": f"No active graph for session '{sess}'", "status": "error"}

    pruned_count = 0
    bucket = _match_store.get(sess, {})
    for p in pairs:
        if len(p) >= 2:
            u, v = str(p[0]), str(p[1])
            if graph.prune_edge(u, v):
                pruned_count += 1
            pa = _store_path(u)
            pb = _store_path(v)
            if pa and pb:
                bucket.pop(tuple(sorted((os.path.abspath(pa), os.path.abspath(pb)))), None)

    summary = graph.get_summary()
    comps = graph.get_connected_components()
    return {
        "num_pruned": pruned_count,
        "remaining_edges": len(graph.active_edges()),
        "num_components": len(comps),
        "components": comps,
        "summary": summary,
        "status": "ok",
    }


def tool_bridge_components(
    comp_a_id: int = 0,
    comp_b_id: int = 1,
    budget: int = 3,
    matcher: str = "loftr",
    session: str = "",
) -> dict[str, Any]:
    """Search for cross-component candidate pairs to connect disconnected components."""
    sess = session or _DEFAULT_SESSION
    graph = _SESSION_GRAPHS.get(sess)
    if graph is None:
        return {"error": f"No active graph for session '{sess}'", "status": "error"}

    comps = graph.get_connected_components()
    if comp_a_id < 0 or comp_a_id >= len(comps) or comp_b_id < 0 or comp_b_id >= len(comps):
        return {
            "error": f"Invalid component IDs {comp_a_id}, {comp_b_id}. Total components: {len(comps)}",
            "status": "error",
        }
    if comp_a_id == comp_b_id:
        return {"error": "comp_a_id and comp_b_id must be distinct", "status": "error"}

    nodes_a = comps[comp_a_id]
    nodes_b = comps[comp_b_id]

    b = max(1, min(budget, 5))

    candidates: list[tuple[str, str, float]] = []
    for u in nodes_a:
        path_u = _store_path(u)
        emb_u = _EMBEDDING_CACHE.get(path_u) if path_u else None
        for v in nodes_b:
            key = graph.canonical_key(u, v)
            if key in graph.edges and graph.edges[key].status == "verified":
                continue
            path_v = _store_path(v)
            emb_v = _EMBEDDING_CACHE.get(path_v) if path_v else None
            score = 0.0
            if emb_u is not None and emb_v is not None:
                denom = max(float(np.linalg.norm(emb_u) * np.linalg.norm(emb_v)), 1e-8)
                score = float(np.dot(emb_u, emb_v) / denom)
            candidates.append((u, v, score))

    candidates.sort(key=lambda x: x[2], reverse=True)
    tested_candidates = candidates[:b]

    bridges_added = 0
    new_edges: list[dict[str, Any]] = []
    for u, v, score in tested_candidates:
        match_res = tool_match(u, v, matcher=matcher, session=sess)
        num_inliers = int(match_res.get("num_inliers") or 0)
        num_matches = int(match_res.get("num_matches") or 0)
        inlier_ratio = float(match_res.get("inlier_ratio") or 0.0)
        pose = match_res.get("pose")
        R_rel = np.array(pose["R"], dtype=np.float64) if (pose and "R" in pose) else None
        t_rel = np.array(pose["t"], dtype=np.float64) if (pose and "t" in pose) else None
        status = "verified" if num_inliers >= 15 else "failed"

        edge = EdgeData(
            image_a=u,
            image_b=v,
            num_matches=num_matches,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            R_rel=R_rel,
            t_rel=t_rel,
            matcher=matcher,
            status=status,
        )
        graph.add_edge(u, v, edge)
        if status == "verified":
            bridges_added += 1
            new_edges.append({
                "pair": [u, v],
                "inliers": num_inliers,
                "inlier_ratio": round(inlier_ratio, 3),
                "status": "verified",
            })

    new_comps = graph.get_connected_components()
    return {
        "candidates_tested": len(tested_candidates),
        "bridges_added": bridges_added,
        "new_edges": new_edges,
        "num_components": len(new_comps),
        "components": new_comps,
        "summary": graph.get_summary(),
        "status": "ok",
    }


def _resolve_image_name(id_or_path: str, image_dir: str) -> str:
    """Map a registered image id or path to its COLMAP image name.

    COLMAP names are paths relative to ``image_dir``; for images directly
    inside it that is just the basename.  Falls back to basename when the
    path is not under ``image_dir``.
    """
    p = _store_path(id_or_path) or str(id_or_path)
    try:
        ap = os.path.abspath(p)
        ad = os.path.abspath(image_dir)
        if ap.startswith(ad + os.sep):
            return os.path.relpath(ap, ad)
    except Exception:
        pass
    return os.path.basename(p)


def _match_pair_list(
    database_path: str, pair_list: list[tuple[str, str]], image_dir: str
) -> int:
    """Match only the given image pairs via COLMAP's match-list importer.

    ``pair_list`` entries may be registered image ids or file paths; they
    are resolved to COLMAP image names.  Returns the number of pairs
    written to the match list.
    """
    import pycolmap

    lines = []
    for pair in pair_list:
        if len(pair) < 2:
            continue
        name_a = _resolve_image_name(str(pair[0]), image_dir)
        name_b = _resolve_image_name(str(pair[1]), image_dir)
        # COLMAP match lists are whitespace-separated; skip names with spaces.
        if not name_a or not name_b or " " in name_a or " " in name_b:
            continue
        # Skip entries that don't resolve to an actual file in image_dir —
        # otherwise garbage ids get written to the match list, match nothing,
        # and the run dies later with a misleading "no_convergence".
        if not (os.path.exists(os.path.join(image_dir, name_a))
                and os.path.exists(os.path.join(image_dir, name_b))):
            continue
        lines.append(f"{name_a} {name_b}\n")

    if not lines:
        return 0

    list_fd, list_path = tempfile.mkstemp(suffix=".txt", prefix="match_list_")
    try:
        with os.fdopen(list_fd, "w") as f:
            f.writelines(lines)
        pairing = pycolmap.ImportedPairingOptions(match_list_path=list_path)
        pycolmap.match_image_pairs(
            database_path=database_path,
            matching_options=colmap_matching_options(),
            pairing_options=pairing,
        )
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass
    return len(lines)


def _import_agent_matches(
    database_path: str,
    image_dir: str,
    pair_list: list[tuple[str, str]] | None = None,
    session: str = "",
) -> int:
    """Import the agent's own matcher correspondences into COLMAP's database.

    Instead of running SIFT ``match_exhaustive``/``match_image_pairs``, this
    writes the keypoints the agent's matcher produced (cached in
    ``_match_store`` under the caller's session) as the images' 2D keypoints,
    then writes the verified inlier correspondences as two-view geometries.
    ``incremental_mapping`` then triangulates the *agent's* matches directly —
    so reconstruction quality reflects the agent's matcher choices.

    Returns the number of verified pairs imported (0 → caller falls back to
    SIFT matching).
    """
    import pycolmap

    store = _match_store.get(session or _DEFAULT_SESSION, {})
    graph = _SESSION_GRAPHS.get(session or _DEFAULT_SESSION)

    # Active path pairs from the session's ViewGraph (if present)
    active_path_set: set[tuple[str, str]] | None = None
    if graph is not None:
        active_path_set = set()
        for (u, v), edge in graph.active_edges().items():
            pa = _store_path(u)
            pb = _store_path(v)
            if pa and pb:
                active_path_set.add(tuple(sorted((os.path.abspath(pa), os.path.abspath(pb)))))

    # Resolve the requested pairs (or all cached pairs when pair_list is None)
    # to their cached match records.
    records: list[dict[str, Any]] = []
    if pair_list:
        for pair in pair_list:
            if len(pair) < 2:
                continue
            a = _store_path(str(pair[0]))
            b = _store_path(str(pair[1]))
            if a is None or b is None:
                continue  # in-memory crops have no file path to key on
            key = tuple(sorted((os.path.abspath(a), os.path.abspath(b))))
            if active_path_set is not None and key not in active_path_set:
                continue
            rec = store.get(key)
            if rec is not None:
                records.append(rec)
    else:
        for key, r in store.items():
            if active_path_set is not None and key not in active_path_set:
                continue
            if os.path.exists(r["path_a"]) and os.path.exists(r["path_b"]):
                records.append(r)
    if not records:
        return 0

    db = pycolmap.Database.open(database_path)

    # Per-image deduplicated keypoint table (match endpoints → global indices).
    img_kpts: dict[str, list[list[float]]] = {}
    img_index: dict[str, dict[tuple, int]] = {}

    def kp_idx(name: str, xy) -> int:
        key = (round(float(xy[0]), 1), round(float(xy[1]), 1))
        m = img_index.setdefault(name, {})
        if key not in m:
            img_kpts.setdefault(name, []).append([float(xy[0]), float(xy[1])])
            m[key] = len(img_kpts[name]) - 1
        return m[key]

    pair_matches: list[tuple[str, str, list[tuple[int, int, bool]]]] = []
    for rec in records:
        name_a = _resolve_image_name(rec["path_a"], image_dir)
        name_b = _resolve_image_name(rec["path_b"], image_dir)
        img_kpts.setdefault(name_a, [])
        img_kpts.setdefault(name_b, [])
        mask = rec.get("inlier_mask") or []
        entries = []
        for i, (ka, kb) in enumerate(zip(rec["keypoints_a"], rec["keypoints_b"])):
            ia, ib = kp_idx(name_a, ka), kp_idx(name_b, kb)
            inl = bool(mask[i]) if i < len(mask) else True
            entries.append((ia, ib, inl))
        pair_matches.append((name_a, name_b, entries))

    # Register camera + image + keypoints for every referenced image.
    img_id: dict[str, int] = {}
    for name, kpts in img_kpts.items():
        try:
            w, h = Image.open(os.path.join(image_dir, name)).size
        except Exception:
            w, h = 1024, 1024
        cam = pycolmap.Camera(
            model="SIMPLE_PINHOLE", width=int(w), height=int(h),
            params=[1.2 * max(w, h), w / 2.0, h / 2.0],
        )
        cam_id = db.write_camera(cam)
        iid = db.write_image(pycolmap.Image(name=name, camera_id=cam_id))
        img_id[name] = iid
        db.write_keypoints(iid, np.asarray(kpts, dtype=np.float64))

    # Write the raw correspondences, then run COLMAP's geometric verification
    # so it estimates E / cam2_from_cam1 / inlier subsets itself (the mapper
    # needs the verified two-view geometry, not just index pairs).
    valid_pairs: list[tuple[str, str]] = []
    for name_a, name_b, entries in pair_matches:
        allpairs = np.asarray([[ia, ib] for ia, ib, _ in entries],
                              dtype=np.uint32)
        if len(allpairs) < 8:
            continue
        db.write_matches(img_id[name_a], img_id[name_b], allpairs)
        valid_pairs.append((name_a, name_b))
    db.close()

    if not valid_pairs:
        return 0

    # Geometric verification over the imported raw matches.
    list_fd, pairs_path = tempfile.mkstemp(suffix=".txt", prefix="verify_")
    try:
        with os.fdopen(list_fd, "w") as f:
            for a, b in valid_pairs:
                f.write(f"{a} {b}\n")
        pycolmap.verify_matches(database_path, pairs_path)
    finally:
        try:
            os.unlink(pairs_path)
        except OSError:
            pass
    return len(valid_pairs)


def _run_colmap_subprocess(
    db_path: str,
    image_dir: str,
    sparse_dir: str,
    used_agent_matches: bool,
    pair_list: list[list[str]] | None,
    num_pairs_requested: int,
    timeout_s: int = 1800,
) -> dict[str, Any]:
    """Run extract/match/mapping in a child process.

    In-process pycolmap SIGBUS (FAISS FeatureMatcherWorker) aborts uvicorn.
    Isolate native COLMAP so the tool server can keep serving /health.
    Set AGENTIC_SFM_SFM_IN_PROCESS=1 to skip the sandbox (tests / debug).
    """
    worker = _SFM_WORKER_PATH
    payload = {
        "db_path": db_path,
        "image_dir": image_dir,
        "sparse_dir": sparse_dir,
        "used_agent_matches": used_agent_matches,
        "pair_list": pair_list,
        "num_pairs_requested": num_pairs_requested,
    }

    def _in_process() -> dict[str, Any]:
        import importlib.util
        spec = importlib.util.spec_from_file_location("sfm_worker", worker)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod.run(payload)

    if os.environ.get("AGENTIC_SFM_SFM_IN_PROCESS", "").strip() in ("1", "true", "yes"):
        return _in_process()

    env = os.environ.copy()
    # COLMAP SIFT/FAISS must not share the matcher GPU with the parent.
    env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        proc = subprocess.Popen(
            [sys.executable, str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(
                json.dumps(payload), timeout=timeout_s
            )
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait(timeout=5)
            return {
                "error": f"COLMAP worker timed out after {timeout_s}s",
                "status": "timeout",
                "num_registered": 0,
                "num_points3d": 0,
            }
    except OSError as e:
        logger.error("Failed to spawn COLMAP worker: %s", e)
        return _in_process()

    if proc.returncode not in (0, 1) or not (stdout or "").strip():
        tail = (stderr or "")[-800:].strip()
        return {
            "error": (
                f"COLMAP worker crashed (exit {proc.returncode})"
                + (f": {tail}" if tail else "")
            ),
            "status": "crash",
            "num_registered": 0,
            "num_points3d": 0,
        }
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "error": f"COLMAP worker returned non-JSON (exit {proc.returncode}): "
                     f"{(stdout or stderr or '')[-400:]}",
            "status": "crash",
            "num_registered": 0,
            "num_points3d": 0,
        }


def default_sfm_output_dir() -> str:
    """COLMAP workdir: $SFM_OUTPUT_ROOT (or $TMPDIR), never ./outputs on scratch."""
    root = os.environ.get("SFM_OUTPUT_ROOT") or os.environ.get("TMPDIR") or tempfile.gettempdir()
    return str(Path(root) / f"sfm_run_{uuid.uuid4().hex[:8]}")


def tool_sfm_run(
    image_dir: str,
    pair_list: list[tuple[str, str]] | None = None,
    output_dir: str = "",
    use_agent_matches: bool = True,
    session: str = "",
) -> dict[str, Any]:
    """Run COLMAP SfM on a directory of images (pycolmap >= 4.x API)."""
    # Unique per-call dir unless the caller pinned one — a shared fixed dir
    # lets one sfm_run wipe another episode's reconstruction (database.db is
    # deleted and sparse/ is rewritten in place).
    if not output_dir:
        output_dir = default_sfm_output_dir()
    outdir = Path(output_dir)
    recon_id = f"recon_{uuid.uuid4().hex[:8]}"
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return {
            "error": f"could not create sfm output dir {output_dir}: {e}",
            "status": "io_error",
            "recon_id": recon_id,
            "num_registered": 0,
            "num_points3d": 0,
        }
    _RECON_STORE[recon_id] = {"image_dir": image_dir, "output_dir": output_dir}

    if not os.path.isdir(image_dir):
        return {
            "error": f"image_dir does not exist: {image_dir}",
            "recon_id": recon_id,
            "num_registered": 0,
            "num_points3d": 0,
        }

    if pair_list is not None and len(pair_list) == 0:
        # Agent filtered every pair as doppelganger — nothing to match. Check
        # before any feature extraction: an empty pair list is a decision,
        # not a request for the baseline.
        return {
            "error": "pair_list is empty (all pairs filtered)",
            "status": "no_pairs",
            "recon_id": recon_id,
            "num_registered": 0,
            "num_points3d": 0,
            "num_pairs_matched": 0,
        }

    try:
        import pycolmap

        db_path = str(outdir / "database.db")
        if os.path.exists(db_path):
            os.remove(db_path)

        # pycolmap>=4.x: the database file must exist before import.
        db = pycolmap.Database.open(db_path)
        db.close()

        # Agent-matches path: import the agent's own correspondences into the
        # DB (keypoints + verified two-view geometry) so incremental_mapping
        # triangulates them directly. This couples reconstruction quality to
        # the agent's matcher choices instead of an independent SIFT pass.
        used_agent_matches = False
        n_imported = 0
        if use_agent_matches:
            sess = session or _DEFAULT_SESSION
            graph = _SESSION_GRAPHS.get(sess)
            has_graph = graph is not None
            has_store = bool(_match_store.get(sess))

            if has_graph and len(graph.active_edges()) == 0:
                logger.warning(
                    "Active view graph has 0 verified edges; aborting (strict no-fallback contract)"
                )
                return {
                    "error": "No verified edges in agent graph. Reconstruction aborted.",
                    "status": "failed_empty_graph",
                    "recon_id": recon_id,
                    "num_registered": 0,
                    "num_points3d": 0,
                    "mean_reproj_error": 0.0,
                    "num_pairs_matched": 0,
                }

            n_imported = _import_agent_matches(db_path, image_dir, pair_list, session=session)
            if n_imported > 0:
                used_agent_matches = True
                logger.info(
                    "Imported %d agent-verified match pairs into COLMAP",
                    n_imported,
                )
            elif has_graph or has_store or session:
                logger.warning(
                    "use_agent_matches set but 0 verified matches imported; "
                    "aborting reconstruction (strict no-fallback contract)"
                )
                return {
                    "error": "No verified edges imported for COLMAP. Reconstruction aborted.",
                    "status": "failed_empty_graph",
                    "recon_id": recon_id,
                    "num_registered": 0,
                    "num_points3d": 0,
                    "mean_reproj_error": 0.0,
                    "num_pairs_matched": 0,
                }
            else:
                logger.warning(
                    "use_agent_matches set but no session graph/matches; "
                    "falling back to SIFT"
                )

        named_pairs: list[list[str]] | None = None
        num_pairs_requested = n_imported if used_agent_matches else 0
        if pair_list is not None:
            named_pairs = []
            for pair in pair_list:
                if len(pair) < 2:
                    continue
                name_a = _resolve_image_name(str(pair[0]), image_dir)
                name_b = _resolve_image_name(str(pair[1]), image_dir)
                if name_a and name_b:
                    named_pairs.append([name_a, name_b])
            if not used_agent_matches:
                num_pairs_requested = len(named_pairs)
            if not used_agent_matches and num_pairs_requested == 0:
                return {
                    "error": "pair_list entries did not resolve to known images; "
                             "use registered image ids (e.g. img_0000)",
                    "status": "no_pairs",
                    "recon_id": recon_id,
                    "num_registered": 0,
                    "num_points3d": 0,
                    "num_pairs_matched": 0,
                }

        sparse_dir = outdir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        native = _run_colmap_subprocess(
            db_path=db_path,
            image_dir=image_dir,
            sparse_dir=str(sparse_dir),
            used_agent_matches=used_agent_matches,
            pair_list=named_pairs,
            num_pairs_requested=num_pairs_requested,
        )
        if native.get("error"):
            native.setdefault("recon_id", recon_id)
            native.setdefault("num_registered", 0)
            native.setdefault("num_points3d", 0)
            return native

        num_registered = int(native.get("num_registered") or 0)
        num_points = int(native.get("num_points3d") or 0)
        model_dir = native.get("output_dir") or str(sparse_dir / "0")
        recon = None
        try:
            recon = pycolmap.Reconstruction(model_dir)
        except Exception:
            recon = None

        _RECON_STORE[recon_id].update({
            "num_images": num_registered,
            "num_points3d": num_points,
            "reconstruction": recon,
        })

        return {
            "recon_id": recon_id,
            "num_registered": num_registered,
            "num_points3d": num_points,
            "output_dir": model_dir,
            "num_pairs_matched": native.get("num_pairs_matched", num_pairs_requested),
            "mean_reproj_error": native.get("mean_reproj_error"),
            "num_components": native.get("num_components", 1),
        }
    except ImportError:
        return {
            "error": "pycolmap not available",
            "recon_id": recon_id,
            "num_registered": 0,
            "num_points3d": 0,
        }
    except Exception as e:
        logger.error("SfM run failed: %s", e)
        return {
            "error": str(e),
            "recon_id": recon_id,
            "num_registered": 0,
            "num_points3d": 0,
        }


def tool_inspect(recon_dir: str) -> dict[str, Any]:
    """Inspect a reconstruction."""
    try:
        import pycolmap

        recon = pycolmap.Reconstruction(recon_dir)
        return {
            "num_images": len(recon.images),
            "num_points3d": len(recon.points3D),
            "num_cameras": len(recon.cameras),
            "mean_reproj_error": float(np.mean([t.reprojection_error for t in recon.points3D.values()])) if recon.points3D else 0.0,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Agentic SfM Tool Server", version="0.1.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tools": [
            "crop", "match", "doppelganger_check", "retrieve", "sfm_run", "inspect",
            "build_initial_graph", "inspect_graph", "verify_doppelganger", "prune_edges", "bridge_components",
        ],
    }


@app.post("/crop")
def crop(req: CropRequest):
    return tool_crop(req.image_id, req.bbox)


@app.post("/match")
def match(req: MatchRequest):
    return tool_match(req.image_a, req.image_b, req.matcher, req.max_size,
                      req.K_a, req.K_b, session=req.session)


@app.post("/doppelganger_check")
def doppelganger_check(req: DoppelgangerRequest):
    return tool_doppelganger_check(req.image_a, req.image_b)


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    return tool_retrieve(req.query_image, req.k)


@app.post("/build_initial_graph")
def build_initial_graph_endpoint(req: BuildGraphRequest):
    return tool_build_initial_graph(
        req.image_ids, req.k_neighbors, req.matcher, session=req.session
    )


@app.post("/inspect_graph")
def inspect_graph_endpoint(req: InspectGraphRequest):
    return tool_inspect_graph(session=req.session)


@app.post("/verify_doppelganger")
def verify_doppelganger_endpoint(req: VerifyDoppelgangerRequest):
    return tool_verify_doppelganger(
        req.image_a, req.image_b, session=req.session, matcher=req.matcher
    )


@app.post("/prune_edges")
def prune_edges_endpoint(req: PruneEdgesRequest):
    return tool_prune_edges(req.pairs, session=req.session)


@app.post("/bridge_components")
def bridge_components_endpoint(req: BridgeComponentsRequest):
    return tool_bridge_components(
        req.comp_a_id, req.comp_b_id, req.budget, req.matcher, session=req.session
    )


@app.post("/sfm_run")
def sfm_run(req: SfMRequest):
    return tool_sfm_run(
        req.image_dir, req.pair_list, req.output_dir, req.use_agent_matches,
        session=req.session,
    )


@app.post("/inspect")
def inspect(req: InspectRequest):
    return tool_inspect(req.recon_dir)


@app.post("/register_image")
def register_image_endpoint(req: RegisterRequest):
    if not os.path.exists(req.path):
        return {"status": "error", "error": f"image path does not exist: {req.path}",
                "image_id": req.image_id}
    register_image(req.image_id, req.path)
    return {"status": "ok", "image_id": req.image_id}


def serve(host: str = "0.0.0.0", port: int = 8765):
    """Start the tool server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agentic SfM tool server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("TOOL_SERVER_PORT", "8765")),
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    serve(host=args.host, port=args.port)
