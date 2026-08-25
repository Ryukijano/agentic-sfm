#!/usr/bin/env python
"""Build hard_pairs.json from MegaDepth-1500 LoFTR test split.

Uses the npz scene info files which contain image pairs with
known intrinsics, extrinsics, and overlap scores.
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image

MD_ROOT = Path("/scratch/kcwp264/data/megadepth/megadepth_test_1500")
SCENE_INFO = Path("/scratch/kcwp264/data/megadepth/scene_info")
OUTPUT = Path("/scratch/kcwp264/agentic-sfm/data/hard_pairs.json")

# Overlap ranges -> difficulty bins
# LoFTR uses: 0.1-0.3 (hard), 0.3-0.5 (medium), 0.5-0.7 (easy)
# We also want some "extreme" (<0.1) but LoFTR doesn't have that split,
# so we'll treat 0.1-0.3 as "hard" and call 0.1-0.2 "extreme"

def overlap_to_difficulty(overlap_range):
    """Map overlap range string to difficulty.
    
    LoFTR ranges: 0.1-0.3, 0.3-0.5, 0.5-0.7
    We split 0.1-0.3 into extreme (0.1-0.2) and hard (0.2-0.3).
    """
    lo, hi = map(float, overlap_range.split("_"))
    if lo >= 0.5:
        return "easy"
    elif lo >= 0.3:
        return "medium"
    elif lo >= 0.2:
        return "hard"
    else:
        # 0.1-0.3 range: assign half to extreme, half to hard
        # Use a hash of the pair indices for deterministic split
        return "extreme"  # default for 0.1-0.3


def rotation_error_deg(R1, R2):
    R_rel = R1 @ R2.T
    trace = np.clip(np.trace(R_rel), -1.0, 3.0)
    return float(np.degrees(np.arccos((trace - 1) / 2)))


def main():
    pairs = []
    pair_id_counter = 0

    npz_files = sorted(SCENE_INFO.glob("*.npz"))
    for npz_file in npz_files:
        data = dict(np.load(npz_file, allow_pickle=True))
        scene = npz_file.stem.split("_")[0]  # e.g. "0015" from "0015_0.1_0.3.npz"
        overlap_range = npz_file.stem.split("_", 1)[1]  # e.g. "0.1_0.3"

        image_paths = data["image_paths"]
        intrinsics = data["intrinsics"]
        poses = data["poses"]
        pair_infos = data["pair_infos"]

        for pi in pair_infos:
            idx1, idx2 = int(pi[0][0]), int(pi[0][1])
            # pi[1] is scale factor, pi[2] is crop bbox
            scale = float(pi[1]) if len(pi) > 1 else 1.0

            # Resolve image paths
            path_a = str(MD_ROOT / image_paths[idx1])
            path_b = str(MD_ROOT / image_paths[idx2])

            if not Path(path_a).exists() or not Path(path_b).exists():
                continue

            # Get poses (4x4 cam-from-world)
            pose_a = np.array(poses[idx1])  # 4x4
            pose_b = np.array(poses[idx2])
            R_a = pose_a[:3, :3]
            t_a = pose_a[:3, 3]
            R_b = pose_b[:3, :3]
            t_b = pose_b[:3, 3]

            # Relative pose: R_rel = R_b @ R_a^T, t_rel = t_b - R_rel @ t_a
            R_rel = R_b @ R_a.T
            t_rel = t_b - R_rel @ t_a

            # Overlap score from the range midpoint (actual per-pair overlap not stored)
            lo, hi = map(float, overlap_range.split("_"))
            overlap_score = (lo + hi) / 2.0

            K_a = np.array(intrinsics[idx1])
            K_b = np.array(intrinsics[idx2])

            pair = {
                "pair_id": f"md_{scene}_{pair_id_counter:04d}",
                "image_a": path_a,
                "image_b": path_b,
                "gt_R": R_rel.tolist(),
                "gt_t": t_rel.tolist(),
                "K_a": K_a.tolist(),
                "K_b": K_b.tolist(),
                "overlap_score": overlap_score,
                "difficulty": overlap_to_difficulty(overlap_range),
                "dataset": "megadepth",
                "scene": scene,
                "extra": {
                    "overlap_range": overlap_range,
                    "scale": scale,
                    "rot_error_deg": rotation_error_deg(R_a, R_b),
                },
            }
            pairs.append(pair)
            pair_id_counter += 1

    # Sample a manageable subset: ~10 per difficulty bin
    import random
    random.seed(42)

    # For 0.1-0.3 range, split into extreme and hard
    extreme_pairs = [p for p in pairs if p["difficulty"] == "extreme"]
    random.shuffle(extreme_pairs)
    subset = []
    subset.extend(extreme_pairs[:10])  # extreme
    # Relabel 5 more as "hard" from the same pool
    for p in extreme_pairs[10:20]:
        p["difficulty"] = "hard"
    subset.extend(extreme_pairs[10:20])  # hard from 0.1-0.3

    for diff in ["medium", "easy"]:
        diff_pairs = [p for p in pairs if p["difficulty"] == diff]
        random.shuffle(diff_pairs)
        n = min(10, len(diff_pairs))
        subset.extend(diff_pairs[:n])
        print(f"{diff}: {len(diff_pairs)} available, selected {n}")

    print(f"extreme (from 0.1-0.3): selected 10")
    print(f"hard (from 0.1-0.3): selected 10")

    # Save
    with open(OUTPUT, "w") as f:
        json.dump(subset, f, indent=2)

    print(f"\nTotal pairs saved: {len(subset)}")
    print(f"Saved to: {OUTPUT}")

    # Verify images load
    ok = 0
    for p in subset:
        try:
            img_a = Image.open(p["image_a"])
            img_b = Image.open(p["image_b"])
            ok += 1
        except Exception as e:
            print(f"ERROR loading {p['pair_id']}: {e}")
    print(f"Images verified: {ok}/{len(subset)}")


if __name__ == "__main__":
    main()
