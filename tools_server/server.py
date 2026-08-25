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
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    matcher: str = "mast3r"
    max_size: int = 512


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
    return {
        "cropped_image_id": new_id,
        "crop_size": [px2 - px1, py2 - py1],
        "image_b64": image_to_base64(cropped, max_size=512),
    }


def tool_match(
    image_a: str, image_b: str, matcher: str = "mast3r", max_size: int = 512
) -> dict[str, Any]:
    """Match two images and estimate relative pose."""
    img_a = get_image(image_a)
    img_b = get_image(image_b)

    if matcher == "loftr":
        return _match_loftr(img_a, img_b, max_size)
    elif matcher == "lightglue":
        return _match_lightglue(img_a, img_b, max_size)
    else:
        return _match_mast3r(img_a, img_b, max_size)


def _match_loftr(img_a: np.ndarray, img_b: np.ndarray, max_size: int) -> dict[str, Any]:
    """Match using LoFTR (kornia)."""
    try:
        import kornia.feature as KF
    except ImportError:
        return {"error": "kornia not available", "matcher": "loftr"}

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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_a, t_b = t_a.to(device), t_b.to(device)

    matcher = KF.LoFTR(pretrained="outdoor").to(device).eval()

    with torch.no_grad():
        input_dict = {"image0": t_a, "image1": t_b}
        correspondences = matcher(input_dict)

    mkpts_a = correspondences["keypoints0"].cpu().numpy()
    mkpts_b = correspondences["keypoints1"].cpu().numpy()
    confidence = correspondences["confidence"].cpu().numpy()

    # Scale back
    mkpts_a = mkpts_a / scale_a
    mkpts_b = mkpts_b / scale_b

    # Estimate pose with RANSAC
    pose_result = _estimate_pose_ransac(mkpts_a, mkpts_b, (w_a, h_a), (w_b, h_b))

    return {
        "matcher": "loftr",
        "num_matches": len(mkpts_a),
        "num_inliers": pose_result["num_inliers"],
        "pose": pose_result["pose"],
        "inlier_ratio": pose_result["inlier_ratio"],
        "keypoints_a": mkpts_a.tolist()[:100],
        "keypoints_b": mkpts_b.tolist()[:100],
    }


def _match_mast3r(img_a: np.ndarray, img_b: np.ndarray, max_size: int) -> dict[str, Any]:
    """Match using MASt3R."""
    try:
        from mast3r.model import AsymmetricMASt3R
        from mast3r.fast_nn import fast_reciprocal_NNs
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
    except ImportError:
        # Fallback to LoFTR
        logger.warning("MASt3R not available, falling back to LoFTR")
        return _match_loftr(img_a, img_b, max_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"

    try:
        model = AsymmetricMASt3R.from_pretrained(model_name).to(device).eval()
    except Exception:
        logger.warning("Cannot load MASt3R, falling back to LoFTR")
        return _match_loftr(img_a, img_b, max_size)

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
        matches_im0 = matches_im0[valid]
        matches_im1 = matches_im1[valid]

        pose_result = _estimate_pose_ransac(
            matches_im0.astype(np.float64), matches_im1.astype(np.float64),
            (W0, H0), (W1, H1),
        )

        return {
            "matcher": "mast3r",
            "num_matches": len(matches_im0),
            "num_inliers": pose_result["num_inliers"],
            "pose": pose_result["pose"],
            "inlier_ratio": pose_result["inlier_ratio"],
            "keypoints_a": matches_im0.tolist()[:100],
            "keypoints_b": matches_im1.tolist()[:100],
        }
    finally:
        os.unlink(tmp_a)
        os.unlink(tmp_b)


def _match_lightglue(img_a: np.ndarray, img_b: np.ndarray, max_size: int) -> dict[str, Any]:
    """Match using LightGlue (kornia)."""
    try:
        import kornia.feature as KF
    except ImportError:
        return {"error": "kornia not available", "matcher": "lightglue"}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Detect + describe with SuperPoint
    lg = KF.LightGlueMatcher("superpoint").to(device).eval()
    sp = KF.SuperPoint().to(device).eval()

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

    pose_result = _estimate_pose_ransac(mkpts_a, mkpts_b, (w_a, h_a), (w_b, h_b))

    return {
        "matcher": "lightglue",
        "num_matches": len(mkpts_a),
        "num_inliers": pose_result["num_inliers"],
        "pose": pose_result["pose"],
        "inlier_ratio": pose_result["inlier_ratio"],
        "keypoints_a": mkpts_a.tolist()[:100],
        "keypoints_b": mkpts_b.tolist()[:100],
    }


def _to_tensor(img: np.ndarray, max_size: int) -> torch.Tensor:
    h, w = img.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    img_r = cv2.resize(img, (int(w * scale), int(h * scale)))
    return torch.from_numpy(img_r).float().permute(2, 0, 1).unsqueeze(0) / 255.0


def _estimate_pose_ransac(
    pts_a: np.ndarray, pts_b: np.ndarray, size_a: tuple, size_b: tuple
) -> dict[str, Any]:
    """Estimate relative pose via essential matrix + RANSAC."""
    if len(pts_a) < 8:
        return {"num_inliers": 0, "inlier_ratio": 0.0, "pose": None}

    # Default intrinsics (normalized)
    K = np.array([
        [max(size_a) * 0.7, 0, size_a[0] / 2],
        [0, max(size_a) * 0.7, size_a[1] / 2],
        [0, 0, 1],
    ])

    E, mask = cv2.findEssentialMat(
        pts_a, pts_b, K, method=cv2.RANSAC, threshold=1.0, prob=0.999
    )
    if E is None:
        return {"num_inliers": 0, "inlier_ratio": 0.0, "pose": None}

    num_inliers = int(mask.sum())
    inlier_ratio = num_inliers / len(pts_a)

    _, R, t, _ = cv2.recoverPose(E, pts_a, pts_b, K, mask=mask)

    return {
        "num_inliers": num_inliers,
        "inlier_ratio": inlier_ratio,
        "pose": {
            "R": R.tolist(),
            "t": t.flatten().tolist(),
        },
    }


def tool_doppelganger_check(image_a: str, image_b: str) -> dict[str, Any]:
    """Check if image pair is a doppelganger (visually similar but distinct)."""
    # Placeholder: use MASt3R match count as proxy
    # TODO: load Doppelgangers++ checkpoint
    match_result = tool_match(image_a, image_b, matcher="loftr")
    num_matches = match_result.get("num_matches", 0)
    inlier_ratio = match_result.get("inlier_ratio", 0.0)

    # Heuristic: high matches but very low inlier ratio → likely doppelganger
    is_doppelganger = num_matches > 50 and inlier_ratio < 0.1
    score = 1.0 - inlier_ratio if num_matches > 50 else 0.0

    return {
        "is_doppelganger": is_doppelganger,
        "score": score,
        "num_matches": num_matches,
        "inlier_ratio": inlier_ratio,
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
    return tool_match(req.image_a, req.image_b, req.matcher, req.max_size)


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
    serve()
