"""FastAPI tool server for agentic SfM.

Endpoints:
  GET  /health
  POST /register_image
  POST /crop
  POST /match
  POST /doppelganger_check
  POST /sfm_run
  POST /inspect
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)
app = FastAPI(title="Agentic SfM Tool Server")

# In-memory image registry: image_id -> file path
_image_registry: dict[str, str] = {}
# Crop metadata: image_id -> {"origin_xy": [x1, y1]} in parent-image pixels
_crop_meta: dict[str, dict[str, Any]] = {}

# Lazy-loaded matcher
_matcher = None
_matcher_device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "0") == "0" else "cuda:0"

# Lazy-loaded doppelganger detector (DINOv2 + geometry fusion).
# Shares the tool-server GPU; DINOv2-small is ~90 MB so it does not
# conflict with the matcher models.
_doppelganger_detector = None


def _get_matcher():
    global _matcher
    if _matcher is not None:
        return _matcher
    try:
        import torch
        from kornia.feature import LoFTR

        device = torch.device(_matcher_device)
        _matcher = LoFTR(pretrained="outdoor").to(device).eval()
        logger.info(f"LoFTR loaded on {device}")
    except Exception as e:
        logger.warning(f"Could not load LoFTR: {e}")
        _matcher = None
    return _matcher


def _get_doppelganger_detector():
    """Lazy-load the DINOv2+geometry doppelganger detector.

    The detector itself never raises on init — the embedder degrades to a
    histogram fallback if DINOv2 cannot be loaded.
    """
    global _doppelganger_detector
    if _doppelganger_detector is not None:
        return _doppelganger_detector
    try:
        from agentic_sfm.tools.doppelganger import DoppelgangerDetector

        _doppelganger_detector = DoppelgangerDetector(device=_matcher_device)
    except Exception as e:
        logger.warning(f"Could not create DoppelgangerDetector: {e}")
        _doppelganger_detector = None
    return _doppelganger_detector


class CropRequest(BaseModel):
    image_id: str
    bbox: list[float]  # [x1, y1, x2, y2] normalized 0-1


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


class SfMRequest(BaseModel):
    image_dir: str
    pair_list: list[list[str]] | None = None
    output_dir: str = "./outputs/sfm_run"


class InspectRequest(BaseModel):
    recon_dir: str


class RegisterRequest(BaseModel):
    image_id: str
    path: str


@app.get("/health")
def health():
    return {"status": "ok", "tools": ["crop", "match", "doppelganger_check", "sfm_run", "inspect"]}


@app.post("/register_image")
def register_image(req: RegisterRequest):
    _image_registry[req.image_id] = req.path
    return {"status": "ok", "image_id": req.image_id}


@app.post("/crop")
def crop(req: CropRequest):
    from PIL import Image

    path = _image_registry.get(req.image_id)
    if path is None or not os.path.exists(path):
        return {"error": f"Image {req.image_id} not found"}

    img = Image.open(path).convert("RGB")
    w, h = img.size
    x1, y1, x2, y2 = req.bbox
    x1 = max(0, int(x1 * w))
    y1 = max(0, int(y1 * h))
    x2 = min(w, int(x2 * w))
    y2 = min(h, int(y2 * h))

    cropped = img.crop((x1, y1, x2, y2))
    crop_id = f"{req.image_id}_crop_{x1}_{y1}_{x2}_{y2}"
    crop_path = str(Path(tempfile.gettempdir()) / f"{crop_id}.jpg")
    cropped.save(crop_path)
    _image_registry[crop_id] = crop_path
    parent = _crop_meta.get(req.image_id, {})
    ox = int(parent.get("origin_xy", [0, 0])[0]) + x1
    oy = int(parent.get("origin_xy", [0, 0])[1]) + y1
    _crop_meta[crop_id] = {"origin_xy": [ox, oy]}

    import base64
    import io

    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "cropped_image_id": crop_id,
        "crop_id": crop_id,
        "path": crop_path,
        "origin_xy": [ox, oy],
        "size": [x2 - x1, y2 - y1],
        "crop_size": [x2 - x1, y2 - y1],
        "image_b64": image_b64,
    }


@app.post("/match")
def match(req: MatchRequest):
    path_a = _image_registry.get(req.image_a, req.image_a)
    path_b = _image_registry.get(req.image_b, req.image_b)

    if not os.path.exists(path_a) or not os.path.exists(path_b):
        return {"error": f"Image not found: {path_a} or {path_b}"}

    matcher = _get_matcher()
    if matcher is None:
        return {"error": "LoFTR not available", "num_inliers": 0, "inlier_ratio": 0.0}

    try:
        import cv2
        import torch
        from PIL import Image

        from agentic_sfm.geometry import estimate_relative_pose, k_for_image

        device = torch.device(_matcher_device)
        pil_a = Image.open(path_a).convert("RGB")
        pil_b = Image.open(path_b).convert("RGB")
        w_a, h_a = pil_a.size
        w_b, h_b = pil_b.size
        max_size = req.max_size
        scale_a = min(1.0, max_size / max(h_a, w_a))
        scale_b = min(1.0, max_size / max(h_b, w_b))
        img_a = pil_a.resize((int(w_a * scale_a), int(h_a * scale_a)))
        img_b = pil_b.resize((int(w_b * scale_b), int(h_b * scale_b)))
        gray_a = cv2.cvtColor(np.array(img_a), cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(np.array(img_b), cv2.COLOR_RGB2GRAY)

        inp = {
            "image0": torch.from_numpy(gray_a).float()[None, None] / 255.0,
            "image1": torch.from_numpy(gray_b).float()[None, None] / 255.0,
        }
        inp = {k: v.to(device) for k, v in inp.items()}

        with torch.no_grad():
            out = matcher(inp)

        conf = out["confidence"].cpu().numpy()
        pts0 = out["keypoints0"].cpu().numpy()
        pts1 = out["keypoints1"].cpu().numpy()
        if conf.ndim > 1:
            conf = conf[0]
            pts0 = pts0[0]
            pts1 = pts1[0]
        pts0 = pts0 / max(scale_a, 1e-8)
        pts1 = pts1 / max(scale_b, 1e-8)

        high_conf = conf > 0.5
        pts0_hc = pts0[high_conf]
        pts1_hc = pts1[high_conf]
        Ka = k_for_image(
            req.K_a, _crop_meta.get(req.image_a, {}).get("origin_xy"), (w_a, h_a)
        )
        Kb = k_for_image(
            req.K_b, _crop_meta.get(req.image_b, {}).get("origin_xy"), (w_b, h_b)
        )
        pose_result = estimate_relative_pose(
            pts0_hc, pts1_hc, (w_a, h_a), (w_b, h_b), K_a=Ka, K_b=Kb
        )
        return {
            "num_matches": int(len(pts0)),
            "num_inliers": pose_result["num_inliers"],
            "inlier_ratio": pose_result["inlier_ratio"],
            "mean_residual": pose_result.get("mean_inlier_residual"),
            "residual_units": pose_result.get("residual_units"),
            "pose": pose_result["pose"],
            "matches": [
                {"pt_a": pts0[i].tolist(), "pt_b": pts1[i].tolist(), "conf": float(conf[i])}
                for i in range(min(len(pts0), 20))
            ],
            # High-confidence correspondences that entered pose estimation,
            # aligned with inlier_mask — used by the match visualiser.
            "keypoints_a": pts0_hc.tolist()[:100],
            "keypoints_b": pts1_hc.tolist()[:100],
            "inlier_mask": (pose_result.get("inlier_mask") or [])[:100],
        }
    except Exception as e:
        logger.error(f"Match failed: {e}")
        return {"error": str(e), "num_inliers": 0, "inlier_ratio": 0.0}


@app.post("/doppelganger_check")
def doppelganger_check(req: DoppelgangerRequest):
    """DINOv2 + geometric-consistency doppelganger detection.

    Returns ``is_doppelganger`` plus a calibrated ``confidence`` in [0, 1],
    the DINOv2 ``similarity_score``, ``inlier_ratio`` from essential-matrix
    verification, and diagnostics (patch consistency, inlier residual).
    Falls back to the legacy count heuristic if anything goes wrong.
    """
    path_a = _image_registry.get(req.image_a, req.image_a)
    path_b = _image_registry.get(req.image_b, req.image_b)

    if not os.path.exists(path_a) or not os.path.exists(path_b):
        return {"error": f"Image not found: {path_a} or {path_b}"}

    match_result = match(
        MatchRequest(image_a=req.image_a, image_b=req.image_b, matcher="loftr")
    )

    detector = _get_doppelganger_detector()
    if detector is not None:
        try:
            return detector.check(path_a, path_b, match_result=match_result)
        except Exception as e:
            logger.error(f"Doppelganger detector failed: {e}")

    # Heuristic fallback — preserves the old contract on any failure.
    num_matches = int(match_result.get("num_matches", 0))
    inlier_ratio = float(match_result.get("inlier_ratio") or 0.0)
    is_doppelganger = num_matches > 50 and inlier_ratio < 0.1
    confidence = (1.0 - inlier_ratio) * 0.6 if is_doppelganger else 0.0
    return {
        "is_doppelganger": is_doppelganger,
        "confidence": confidence,
        "score": confidence,
        "similarity_score": None,
        "inlier_ratio": inlier_ratio,
        "num_matches": num_matches,
        "num_inliers": int(match_result.get("num_inliers") or 0),
        "residual_units": match_result.get("residual_units"),
        "verdict": "doppelganger" if is_doppelganger else "uncertain",
        "method": "heuristic_fallback",
    }


def _resolve_image_name(id_or_path: str, image_dir: str) -> str:
    """Map a registered image id or path to its COLMAP image name.

    COLMAP names are paths relative to ``image_dir``; for images directly
    inside it that is just the basename.  Falls back to basename when the
    path is not under ``image_dir``.
    """
    p = _image_registry.get(id_or_path, id_or_path)
    try:
        ap = os.path.abspath(p)
        ad = os.path.abspath(image_dir)
        if ap.startswith(ad + os.sep):
            return os.path.relpath(ap, ad)
    except Exception:
        pass
    return os.path.basename(p)


def _match_pair_list(
    database_path: str, pair_list: list[list[str]], image_dir: str
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
        lines.append(f"{name_a} {name_b}\n")

    if not lines:
        return 0

    list_fd, list_path = tempfile.mkstemp(suffix=".txt", prefix="match_list_")
    try:
        with os.fdopen(list_fd, "w") as f:
            f.writelines(lines)
        pairing = pycolmap.ImportedPairingOptions(match_list_path=list_path)
        pycolmap.match_image_pairs(
            database_path=database_path, pairing_options=pairing
        )
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass
    return len(lines)


@app.post("/sfm_run")
def sfm_run(req: SfMRequest):
    output_dir = Path(req.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.isdir(req.image_dir):
        return {
            "error": f"image_dir does not exist: {req.image_dir}",
            "num_registered": 0,
            "num_points3d": 0,
        }

    try:
        import pycolmap

        # Run incremental SfM
        db_path = str(output_dir / "database.db")
        if os.path.exists(db_path):
            os.remove(db_path)

        # pycolmap>=4.x: the database file must exist before import, and
        # extract_features performs image import + SIFT extraction in one step
        # (import_images alone leaves images with zero keypoints).
        db = pycolmap.Database.open(db_path)
        db.close()
        reader_opts = pycolmap.ImageReaderOptions(camera_model="OPENCV")
        pycolmap.extract_features(
            database_path=db_path,
            image_path=req.image_dir,
            reader_options=reader_opts,
        )

        num_pairs_requested = 0
        if req.pair_list is not None:
            if len(req.pair_list) == 0:
                # Agent filtered every pair as doppelganger — nothing to match.
                return {
                    "error": "pair_list is empty (all pairs filtered)",
                    "status": "no_pairs",
                    "num_registered": 0,
                    "num_points3d": 0,
                    "num_pairs_matched": 0,
                }
            # Restrict matching to the given pairs (doppelganger removal).
            num_pairs_requested = _match_pair_list(
                db_path, req.pair_list, req.image_dir
            )
            if num_pairs_requested == 0:
                logger.warning(
                    "pair_list given but no pairs resolved; "
                    "falling back to exhaustive matching"
                )
                pycolmap.match_exhaustive(database_path=db_path)
        else:
            pycolmap.match_exhaustive(database_path=db_path)

        recons = pycolmap.incremental_mapping(
            database_path=db_path,
            image_path=req.image_dir,
            output_path=str(output_dir / "sparse"),
        )

        # pycolmap>=4.x returns {model_idx: Reconstruction}; take the largest.
        recon = None
        if recons:
            recon = max(recons.values(), key=lambda r: len(r.images), default=None)

        if recon is None:
            return {
                "error": "incremental_mapping produced no reconstruction",
                "status": "no_convergence",
                "num_registered": 0,
                "num_points3d": 0,
                "num_pairs_matched": num_pairs_requested,
            }

        num_registered = len(recon.images)
        num_points = len(recon.points3D)

        # Persist the selected reconstruction so inspect/eval can read it.
        sparse_dir = output_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        # Clear stale contents to avoid confusing inspect with old subdirs.
        for old in sparse_dir.iterdir():
            if old.is_dir():
                import shutil
                shutil.rmtree(old)
            else:
                old.unlink()
        # pycolmap's Reconstruction.write requires the model dir to exist.
        model_dir = sparse_dir / "0"
        model_dir.mkdir(parents=True, exist_ok=True)
        recon.write(str(model_dir))

        mean_reproj_error = (
            float(recon.compute_mean_reprojection_error())
            if hasattr(recon, "compute_mean_reprojection_error")
            else None
        )

        return {
            "num_registered": num_registered,
            "num_points3d": num_points,
            "output_dir": str(sparse_dir / "0"),
            "num_pairs_matched": num_pairs_requested,
            "mean_reproj_error": mean_reproj_error,
            "num_components": len(recons) if recons else 1,
        }
    except Exception as e:
        logger.error(f"SfM run failed: {e}")
        return {"error": str(e), "num_registered": 0, "num_points3d": 0}


@app.post("/inspect")
def inspect(req: InspectRequest):
    try:
        import pycolmap

        # Try recon_dir directly, then the "0" model subdir produced by
        # incremental_mapping / our sfm_run persistence.
        recon = None
        for candidate in (req.recon_dir, os.path.join(req.recon_dir, "0")):
            try:
                r = pycolmap.Reconstruction(candidate)
                if len(r.images) > 0 or recon is None:
                    recon = r
                if len(r.images) > 0:
                    break
            except Exception:
                continue
        if recon is None:
            return {"error": f"no readable reconstruction under {req.recon_dir}"}

        mean_reproj_error = (
            float(recon.compute_mean_reprojection_error())
            if hasattr(recon, "compute_mean_reprojection_error")
            else None
        )
        num_reg = len(recon.images)

        return {
            "num_images": num_reg,
            "num_registered": num_reg,
            "num_points3d": len(recon.points3D),
            "num_cameras": len(recon.cameras),
            "mean_reproj_error": mean_reproj_error,
        }
    except Exception as e:
        return {"error": str(e)}
