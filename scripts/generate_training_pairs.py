#!/usr/bin/env python3
"""Generate training pairs from MegaDepth-1500 scene_info npz files.

Creates 500+ pairs with train/val split, stratified by difficulty bin.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MEGADEPTH_ROOT = "/scratch/kcwp264/data/megadepth"
SCENE_INFO_DIR = f"{MEGADEPTH_ROOT}/scene_info"
IMAGE_ROOT = f"{MEGADEPTH_ROOT}/megadepth_test_1500"
OUTPUT_DIR = Path(__file__).parent.parent / "data"


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
    """Compute relative pose (R_rel, t_rel) from world-to-camera poses.

    pose_a, pose_b are 4x4 world-to-camera transforms.
    Returns R_rel, t_rel such that pose_b ≈ [R_rel | t_rel] @ pose_a.
    """
    R_a = pose_a[:3, :3]
    t_a = pose_a[:3, 3]
    R_b = pose_b[:3, :3]
    t_b = pose_b[:3, 3]

    R_rel = R_b @ R_a.T
    t_rel = t_b - R_rel @ t_a
    return R_rel, t_rel


def load_scene_pairs(npz_path: str) -> list[dict]:
    """Load pairs from a single scene_info npz file."""
    data = np.load(npz_path, allow_pickle=True)
    image_paths = data["image_paths"]
    intrinsics = data["intrinsics"]
    poses = data["poses"]
    pair_infos = data["pair_infos"]

    # Map scene from filename: e.g. "0022_0.1_0.3.npz" -> "0022"
    scene = Path(npz_path).stem.split("_")[0]

    pairs = []
    for idx_pair, overlap, _ in pair_infos:
        i, j = int(idx_pair[0]), int(idx_pair[1])

        path_a = image_paths[i]
        path_b = image_paths[j]

        if path_a is None or path_b is None:
            continue

        full_path_a = os.path.join(IMAGE_ROOT, str(path_a))
        full_path_b = os.path.join(IMAGE_ROOT, str(path_b))

        if not os.path.exists(full_path_a) or not os.path.exists(full_path_b):
            continue

        pose_a = np.array(poses[i])
        pose_b = np.array(poses[j])

        if pose_a is None or pose_b is None:
            continue

        R_rel, t_rel = compute_relative_pose(pose_a, pose_b)

        K_a = np.array(intrinsics[i]) if intrinsics[i] is not None else None
        K_b = np.array(intrinsics[j]) if intrinsics[j] is not None else None

        overlap_val = float(overlap)
        pair = {
            "pair_id": f"md_{scene}_{i:04d}_{j:04d}",
            "image_a": full_path_a,
            "image_b": full_path_b,
            "gt_R": R_rel.tolist(),
            "gt_t": t_rel.tolist(),
            "K_a": K_a.tolist() if K_a is not None else None,
            "K_b": K_b.tolist() if K_b is not None else None,
            "overlap_score": overlap_val,
            "difficulty": difficulty_bin(overlap_val),
            "dataset": "megadepth",
            "scene": scene,
            "extra": {"npz_file": Path(npz_path).stem},
        }
        pairs.append(pair)

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate training pairs from MegaDepth-1500")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-bin", type=int, default=200,
                        help="Max pairs per difficulty bin (cap)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    npz_files = sorted(Path(SCENE_INFO_DIR).glob("*.npz"))
    logger.info(f"Found {len(npz_files)} scene_info files")

    for npz_path in npz_files:
        logger.info(f"Loading {npz_path.name}...")
        pairs = load_scene_pairs(str(npz_path))
        logger.info(f"  Got {len(pairs)} valid pairs (images exist on disk)")
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
        capped_pairs.extend(pairs_list[:args.max_per_bin])

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

    # Save
    train_path = output_dir / "hard_pairs_train.json"
    val_path = output_dir / "hard_pairs_val.json"
    all_path = output_dir / "hard_pairs_all.json"

    with open(train_path, "w") as f:
        json.dump(train_pairs, f, indent=2)
    logger.info(f"Saved {len(train_pairs)} train pairs to {train_path}")

    with open(val_path, "w") as f:
        json.dump(val_pairs, f, indent=2)
    logger.info(f"Saved {len(val_pairs)} val pairs to {val_path}")

    with open(all_path, "w") as f:
        json.dump(capped_pairs, f, indent=2)
    logger.info(f"Saved {len(capped_pairs)} total pairs to {all_path}")

    # Print stats
    train_diffs = Counter(p["difficulty"] for p in train_pairs)
    val_diffs = Counter(p["difficulty"] for p in val_pairs)
    logger.info(f"Train by difficulty: {dict(train_diffs)}")
    logger.info(f"Val by difficulty: {dict(val_diffs)}")


if __name__ == "__main__":
    main()
