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
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sys

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

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


class DoppelgangerRequest(BaseModel):
    image_a: str
    image_b: str


class RetrieveRequest(BaseModel):
    query_image: str
    k: int = 5


class SfMRequest(BaseModel):
    image_dir: str
    pair_list: list[tuple[str, str]] | None = None
    output_dir: str = "./outputs/sfm_run"


class InspectRequest(BaseModel):
    recon_dir: str


# ---------------------------------------------------------------------------
# Image store — maps image_id → file path or numpy array
# ---------------------------------------------------------------------------

_IMAGE_STORE: dict[str, str | np.ndarray] = {}
_RECON_STORE: dict[str, dict[str, Any]] = {}
_CROP_META: dict[str, dict[str, Any]] = {}

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


def tool_crop(image_id: str, bbox: list[float]) -> dict[str, Any]:
    """Crop image to bbox [x1, y1, x2, y2] in normalized coords."""
    img = get_image(image_id)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    px1, py1 = int(x1 * w), int(y1 * h)
    px2, py2 = int(x2 * w), int(y2 * h)
    px1, px2 = max(0, px1), min(w, px2)
    py1, py2 = max(0, py1), min(h, py2)
    cropped = img[py1:py2, px1:px2]
    new_id = f"{image_id}_crop_{uuid.uuid4().hex[:8]}"
    _IMAGE_STORE[new_id] = cropped
    parent = _CROP_META.get(image_id, {})
    ox = int(parent.get("origin_xy", [0, 0])[0]) + px1
    oy = int(parent.get("origin_xy", [0, 0])[1]) + py1
    _CROP_META[new_id] = {"origin_xy": [ox, oy]}
    crop_path = str(Path(tempfile.gettempdir()) / f"{new_id}.jpg")
    try:
        Image.fromarray(cropped).save(crop_path)
    except Exception:
        crop_path = None
    return {
        "cropped_image_id": new_id,
        "crop_id": new_id,
        "origin_xy": [ox, oy],
        "crop_size": [px2 - px1, py2 - py1],
        "path": crop_path,
        "image_b64": image_to_base64(cropped, max_size=512),
    }


def tool_match(
    image_a: str,
    image_b: str,
    matcher: str = "loftr",
    max_size: int = 512,
    K_a: list | None = None,
    K_b: list | None = None,
) -> dict[str, Any]:
    """Match two images and estimate relative pose."""
    from agentic_sfm.geometry import k_for_image

    img_a = get_image(image_a)
    img_b = get_image(image_b)
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    Ka = k_for_image(K_a, _CROP_META.get(image_a, {}).get("origin_xy"), (w_a, h_a))
    Kb = k_for_image(K_b, _CROP_META.get(image_b, {}).get("origin_xy"), (w_b, h_b))
    kwargs = {"K_a": Ka, "K_b": Kb}

    if matcher == "loftr":
        return _match_loftr(img_a, img_b, max_size, **kwargs)
    elif matcher == "lightglue":
        return _match_lightglue(img_a, img_b, max_size, **kwargs)
    else:
        return _match_mast3r(img_a, img_b, max_size, **kwargs)


def _match_loftr(
    img_a: np.ndarray, img_b: np.ndarray, max_size: int,
    K_a=None, K_b=None,
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
    K_a=None, K_b=None,
) -> dict[str, Any]:
    """Match using MASt3R."""
    try:
        from mast3r.fast_nn import fast_reciprocal_NNs
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        model = _get_mast3r()
    except ImportError:
        logger.warning("MASt3R not available, falling back to LoFTR")
        return _match_loftr(img_a, img_b, max_size, K_a=K_a, K_b=K_b)
    except Exception:
        logger.warning("Cannot load MASt3R, falling back to LoFTR")
        return _match_loftr(img_a, img_b, max_size, K_a=K_a, K_b=K_b)

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


def _match_lightglue(
    img_a: np.ndarray, img_b: np.ndarray, max_size: int,
    K_a=None, K_b=None,
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


def tool_sfm_run(
    image_dir: str,
    pair_list: list[tuple[str, str]] | None = None,
    output_dir: str = "./outputs/sfm_run",
) -> dict[str, Any]:
    """Run COLMAP SfM on a directory of images."""
    os.makedirs(output_dir, exist_ok=True)
    recon_id = f"recon_{uuid.uuid4().hex[:8]}"
    _RECON_STORE[recon_id] = {"image_dir": image_dir, "output_dir": output_dir}

    try:
        import pycolmap

        database_path = os.path.join(output_dir, "database.db")
        if os.path.exists(database_path):
            os.unlink(database_path)

        pycolmap.import_images(image_dir, database_path, camera_model=pycolmap.CameraModelName.OPENCV)
        pycolmap.extract_features(database_path)
        pycolmap.match_exhaustive(database_path)

        if pair_list:
            # Filter pairs (doppelganger removal)
            pass

        reconstruction = pycolmap.incremental_mapping(database_path, image_dir, output_dir)

        num_images = len(reconstruction.images) if reconstruction else 0
        num_points3d = len(reconstruction.points3D) if reconstruction else 0

        _RECON_STORE[recon_id].update({
            "num_images": num_images,
            "num_points3d": num_points3d,
            "reconstruction": reconstruction,
        })

        return {
            "recon_id": recon_id,
            "num_registered": num_images,
            "num_points3d": num_points3d,
            "output_dir": output_dir,
        }
    except ImportError:
        return {"error": "pycolmap not available", "recon_id": recon_id}


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
    return {"status": "ok", "tools": ["crop", "match", "doppelganger_check", "retrieve", "sfm_run", "inspect"]}


@app.post("/crop")
def crop(req: CropRequest):
    return tool_crop(req.image_id, req.bbox)


@app.post("/match")
def match(req: MatchRequest):
    return tool_match(req.image_a, req.image_b, req.matcher, req.max_size, req.K_a, req.K_b)


@app.post("/doppelganger_check")
def doppelganger_check(req: DoppelgangerRequest):
    return tool_doppelganger_check(req.image_a, req.image_b)


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    return tool_retrieve(req.query_image, req.k)


@app.post("/sfm_run")
def sfm_run(req: SfMRequest):
    return tool_sfm_run(req.image_dir, req.pair_list, req.output_dir)


@app.post("/inspect")
def inspect(req: InspectRequest):
    return tool_inspect(req.recon_dir)


@app.post("/register_image")
def register_image_endpoint(req: RegisterRequest):
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
