#!/usr/bin/env python
"""Download additional MegaDepth scenes from the D2-Net preprocessed dataset.

The D2-Net preprocessed MegaDepth is hosted at dsmn.ml. Each scene is a
separate tar file containing the Undistorted_SfM images.

Usage:
    python scripts/download_megadepth_scenes.py --scenes 0080 0004 0331 --output-dir /scratch/kcwp264/data/megadepth/

Recommended scenes (by pair count from scene_info):
    0080: 6494 images, ~514k pairs (largest)
    0004: 4557 images, ~379k pairs
    0331: 3534 images, ~248k pairs
    0003: 5152 images, ~216k pairs
    0011: 2966 images, ~191k pairs
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# D2-Net MegaDepth download URL
# The full dataset is at: https://dsmn.ml/files/d2net/megadepth/
# Per-scene tars are at: https://dsmn.ml/files/d2net/megadepth/Undistorted_SfM/{scene}.tar.gz
# (This is inferred from the D2-Net download script; verify URL before running)
BASE_URL = "https://dsmn.ml/files/d2net/megadepth"


def download_scene(scene_id: str, output_dir: str, dry_run: bool = False) -> bool:
    """Download and extract a single MegaDepth scene."""
    output_path = Path(output_dir) / "Undistorted_SfM" / scene_id / "images"
    if output_path.exists() and len(list(output_path.iterdir())) > 0:
        logger.info(f"Scene {scene_id} already exists ({len(list(output_path.iterdir()))} images), skipping")
        return True

    url = f"{BASE_URL}/Undistorted_SfM/{scene_id}.tar.gz"
    tar_path = Path(output_dir) / f"{scene_id}.tar.gz"

    logger.info(f"Downloading scene {scene_id} from {url}")
    if dry_run:
        logger.info(f"  Would download to {tar_path} and extract to {output_path}")
        return True

    # Download with wget (supports resume)
    try:
        subprocess.run(
            ["wget", "-c", "-q", "--show-progress", url, "-O", str(tar_path)],
            check=True,
            timeout=3600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(f"Download failed for {scene_id}: {e}")
        logger.info(f"  Try manually: wget {url} -O {tar_path}")
        return False

    # Extract
    try:
        logger.info(f"Extracting {tar_path}...")
        subprocess.run(
            ["tar", "-xzf", str(tar_path), "-C", str(output_dir)],
            check=True,
            timeout=600,
        )
        # Remove tar after extraction
        tar_path.unlink()
        logger.info(f"Scene {scene_id} extracted successfully")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(f"Extraction failed for {scene_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", required=True,
                        help="Scene IDs to download (e.g., 0080 0004 0331)")
    parser.add_argument("--output-dir", default="/scratch/kcwp264/data/megadepth",
                        help="MegaDepth root directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without downloading")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check scene_info exists for requested scenes
    si_dir = output_dir / "scene_info_full" / "scene_info"
    for scene_id in args.scenes:
        si_path = si_dir / f"{scene_id}.npz"
        if not si_path.exists():
            logger.warning(f"No scene_info for {scene_id} at {si_path}")

    results = {}
    for scene_id in args.scenes:
        results[scene_id] = download_scene(scene_id, str(output_dir), args.dry_run)

    # Summary
    n_ok = sum(1 for v in results.values() if v)
    logger.info(f"Downloaded {n_ok}/{len(results)} scenes")
    for scene_id, ok in results.items():
        status = "OK" if ok else "FAILED"
        logger.info(f"  {scene_id}: {status}")

    if n_ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
