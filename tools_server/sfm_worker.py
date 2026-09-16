#!/usr/bin/env python
"""COLMAP child process for ``sfm_run``.

pycolmap/FAISS can SIGBUS inside FeatureMatcherWorker (job 7882190, 16 Sep
2026 08:17) and that abort kills in-process uvicorn. Running extract / match /
incremental_mapping here means the FastAPI parent survives a native crash.

stdin: JSON
  db_path, image_dir, sparse_dir, used_agent_matches, pair_list (filenames)
stdout: JSON result
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


def _matching_options():
    """Matcher options that avoid COLMAP's FAISS descriptor index.

    The default SIFT matcher builds a per-image FAISS ``FeatureDescriptorIndex``
    (IVF clustering); that path SIGBUS'd in ``faiss::Clustering::train_encoded``
    and killed the tool server (job 7882190, 2026-09-16 08:17). This child
    process isolates such a crash, but every sfm_run would still fail — so use
    the brute-force matcher and never touch FAISS.
    """
    import pycolmap

    opts = pycolmap.FeatureMatchingOptions()
    try:
        opts.sift.cpu_brute_force_matcher = True
    except AttributeError:
        pass  # older pycolmap without the FAISS matcher
    return opts


def _match_named_pairs(database_path: str, pair_list: list[list[str]], image_dir: str) -> int:
    import pycolmap

    lines = []
    for pair in pair_list:
        if len(pair) < 2:
            continue
        name_a, name_b = str(pair[0]), str(pair[1])
        if " " in name_a or " " in name_b:
            continue
        if not (
            os.path.exists(os.path.join(image_dir, name_a))
            and os.path.exists(os.path.join(image_dir, name_b))
        ):
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
            matching_options=_matching_options(),
            pairing_options=pairing,
        )
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass
    return len(lines)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    import pycolmap

    db_path = cfg["db_path"]
    image_dir = cfg["image_dir"]
    sparse_dir = Path(cfg["sparse_dir"])
    used_agent_matches = bool(cfg.get("used_agent_matches"))
    pair_list = cfg.get("pair_list")
    num_pairs_requested = int(cfg.get("num_pairs_requested") or 0)

    sparse_dir.mkdir(parents=True, exist_ok=True)

    if not used_agent_matches:
        reader_opts = pycolmap.ImageReaderOptions(camera_model="OPENCV")
        pycolmap.extract_features(
            database_path=db_path,
            image_path=image_dir,
            reader_options=reader_opts,
        )
        if pair_list is not None:
            num_pairs_requested = _match_named_pairs(db_path, pair_list, image_dir)
            if num_pairs_requested == 0:
                return {
                    "error": "pair_list entries did not resolve to known images; "
                             "use registered image ids (e.g. img_0000)",
                    "status": "no_pairs",
                    "num_registered": 0,
                    "num_points3d": 0,
                    "num_pairs_matched": 0,
                }
        else:
            pycolmap.match_exhaustive(
                database_path=db_path,
                matching_options=_matching_options(),
            )

    recons = pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=image_dir,
        output_path=str(sparse_dir),
    )
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
    for old in sparse_dir.iterdir():
        if old.is_dir():
            import shutil
            shutil.rmtree(old)
        else:
            old.unlink()
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
        "output_dir": str(model_dir),
        "num_pairs_matched": num_pairs_requested,
        "mean_reproj_error": mean_reproj_error,
        "num_components": len(recons) if recons else 1,
    }


def main() -> int:
    cfg = json.loads(sys.stdin.read())
    try:
        result = run(cfg)
    except Exception as e:
        result = {
            "error": f"{type(e).__name__}: {e}",
            "num_registered": 0,
            "num_points3d": 0,
        }
        json.dump(result, sys.stdout)
        sys.stdout.flush()
        return 1
    json.dump(result, sys.stdout)
    sys.stdout.flush()
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
