#!/usr/bin/env python3
"""Generate training pairs from MegaDepth D2-Net scene_info overlap matrices.

Uses the full overlap_matrix (not just pre-selected pair_infos) to generate
thousands of pairs from the 2 scenes with images on disk.

Usage:
  python scripts/generate_scaled_pairs.py \
      --scene-info-dir /scratch/kcwp264/data/megadepth/scene_info_full/scene_info \
      --image-root /scratch/kcwp264/data/megadepth/megadepth_test_1500 \
      --output-dir data \
      --max-per-bin 2000
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def difficulty_bin(overlap: float) -> str:
    if overlap > 0.7:
        return "easy"
    elif overlap > 0.3:
        return "medium"
    elif overlap > 0.1:
        return "hard"
    else:
        return "extreme"


def compute_relative_pose(pose_a: np.ndarray, pose_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute relative pose (R_rel, t_rel) from world-to-camera poses."""
    R_a = pose_a[:3, :3]
    t_a = pose_a[:3, 3]
    R_b = pose_b[:3, :3]
    t_b = pose_b[:3, 3]
    R_rel = R_b @ R_a.T
    t_rel = t_b - R_rel @ t_a
    return R_rel, t_rel


def load_scene_pairs_from_overlap_matrix(npz_path: str, image_root: str,
                                          scene: str) -> list[dict]:
    """Load pairs using the full overlap matrix from a D2-Net scene_info npz."""
    data = np.load(npz_path, allow_pickle=True)
    image_paths = data["image_paths"]
    intrinsics = data["intrinsics"]
    poses = data["poses"]
    overlap_matrix = data["overlap_matrix"]

    # Only use images that exist on disk
    existing_indices = []
    for i, p in enumerate(image_paths):
        if p is not None:
            p_str = p.decode() if isinstance(p, bytes) else str(p)
            full_path = os.path.join(image_root, p_str)
            if os.path.exists(full_path):
                existing_indices.append(i)

    logger.info(f"  {scene}: {len(existing_indices)}/{len(image_paths)} images on disk")
    if len(existing_indices) < 2:
        return []

    # Extract submatrix for existing images
    om = overlap_matrix[np.ix_(existing_indices, existing_indices)]

    pairs = []
    n = len(existing_indices)
    for i_local in range(n):
        i_global = existing_indices[i_local]
        for j_local in range(i_local + 1, n):
            j_global = existing_indices[j_local]
            overlap = float(om[i_local, j_local])
            if overlap <= 0:
                continue

            pa_str = image_paths[i_global]
            pb_str = image_paths[j_global]
            pa_str = pa_str.decode() if isinstance(pa_str, bytes) else str(pa_str)
            pb_str = pb_str.decode() if isinstance(pb_str, bytes) else str(pb_str)
            path_a = os.path.join(image_root, pa_str)
            path_b = os.path.join(image_root, pb_str)

            raw_a, raw_b = poses[i_global], poses[j_global]
            if raw_a is None or raw_b is None:
                continue
            pose_a = np.array(raw_a)
            pose_b = np.array(raw_b)
            if pose_a.ndim < 2 or pose_b.ndim < 2:
                continue

            R_rel, t_rel = compute_relative_pose(pose_a, pose_b)
            K_a = np.array(intrinsics[i_global]) if intrinsics[i_global] is not None else None
            K_b = np.array(intrinsics[j_global]) if intrinsics[j_global] is not None else None

            pair = {
                "pair_id": f"md_{scene}_{i_global:04d}_{j_global:04d}",
                "image_a": path_a,
                "image_b": path_b,
                "gt_R": R_rel.tolist(),
                "gt_t": t_rel.tolist(),
                "K_a": K_a.tolist() if K_a is not None else None,
                "K_b": K_b.tolist() if K_b is not None else None,
                "overlap_score": overlap,
                "difficulty": difficulty_bin(overlap),
                "dataset": "megadepth",
                "scene": scene,
                "extra": {"npz_file": Path(npz_path).stem},
            }
            pairs.append(pair)

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate scaled training pairs from overlap matrices")
    parser.add_argument("--scene-info-dir", type=str,
                        default="/scratch/kcwp264/data/megadepth/scene_info_full/scene_info")
    parser.add_argument("--image-root", type=str,
                        default="/scratch/kcwp264/data/megadepth/megadepth_test_1500")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-bin", type=int, default=2000,
                        help="Max pairs per difficulty bin (0 = no cap)")
    parser.add_argument("--scenes", type=str, nargs="*", default=None,
                        help="Specific scenes to use (default: all with images on disk)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find scenes with images on disk
    # Structure: image_root/Undistorted_SfM/{scene}/images/
    scene_info_dir = Path(args.scene_info_dir)
    image_root = Path(args.image_root)
    sfm_root = image_root / "Undistorted_SfM"
    if not sfm_root.exists():
        sfm_root = image_root  # fallback if no Undistorted_SfM subdirectory
    available_scenes = [d.name for d in sfm_root.iterdir()
                        if d.is_dir() and (d / "images").exists()]
    if args.scenes:
        available_scenes = [s for s in available_scenes if s in args.scenes]
    logger.info(f"Available scenes with images: {available_scenes}")

    all_pairs = []
    for scene in available_scenes:
        # Use the D2-Net style npz (no overlap range suffix)
        npz_path = scene_info_dir / f"{scene}.npz"
        if not npz_path.exists():
            logger.warning(f"  No scene_info for {scene}, skipping")
            continue

        logger.info(f"Loading {scene}.npz...")
        pairs = load_scene_pairs_from_overlap_matrix(str(npz_path), str(image_root), scene)
        logger.info(f"  Got {len(pairs)} valid pairs")
        all_pairs.extend(pairs)

    logger.info(f"Total valid pairs: {len(all_pairs)}")

    # Group by difficulty
    from collections import Counter, defaultdict
    by_diff = defaultdict(list)
    for p in all_pairs:
        by_diff[p["difficulty"]].append(p)

    for diff, pairs_list in by_diff.items():
        logger.info(f"  {diff}: {len(pairs_list)} pairs")

    # Cap per bin if needed
    rng = np.random.default_rng(args.seed)
    capped_pairs = []
    for diff in ["easy", "medium", "hard", "extreme"]:
        pairs_list = by_diff.get(diff, [])
        rng.shuffle(pairs_list)
        if args.max_per_bin > 0:
            capped_pairs.extend(pairs_list[:args.max_per_bin])
        else:
            capped_pairs.extend(pairs_list)

    logger.info(f"After capping: {len(capped_pairs)} pairs")

    # Stratified train/val split
    train_pairs = []
    val_pairs = []
    for diff in ["easy", "medium", "hard", "extreme"]:
        subset = [p for p in capped_pairs if p["difficulty"] == diff]
        rng.shuffle(subset)
        n_val = int(len(subset) * args.val_ratio)
        val_pairs.extend(subset[:n_val])
        train_pairs.extend(subset[n_val:])

    logger.info(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # Shuffle train/val so file order is not difficulty-ordered (subsets like
    # [:50] must sample all bins, not just the first difficulty).
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)

    # Save
    train_path = output_dir / "hard_pairs_train.json"
    val_path = output_dir / "hard_pairs_val.json"
    all_path = output_dir / "hard_pairs_all.json"

    with open(train_path, "w") as f:
        json.dump(train_pairs, f)
    logger.info(f"Saved {len(train_pairs)} train pairs to {train_path}")

    with open(val_path, "w") as f:
        json.dump(val_pairs, f)
    logger.info(f"Saved {len(val_pairs)} val pairs to {val_path}")

    with open(all_path, "w") as f:
        json.dump(capped_pairs, f)
    logger.info(f"Saved {len(capped_pairs)} total pairs to {all_path}")

    # Print stats
    train_diffs = Counter(p["difficulty"] for p in train_pairs)
    val_diffs = Counter(p["difficulty"] for p in val_pairs)
    logger.info(f"Train by difficulty: {dict(train_diffs)}")
    logger.info(f"Val by difficulty: {dict(val_diffs)}")


if __name__ == "__main__":
    main()
