"""Hard-pair dataset builder with overlap-binned difficulty.

Sources:
  - MegaDepth (GT poses + sparse reconstructions)
  - Map-free relocalization (extreme baselines)
  - ScanNet++ (RGB-D, dense GT)

Difficulty bins (overlap score ω ∈ [0,1]):
  - Easy:   ω > 0.7  (near-identical views)
  - Medium: 0.3 < ω ≤ 0.7
  - Hard:   0.1 < ω ≤ 0.3
  - Extreme: ω ≤ 0.1 (wide baseline, little overlap)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ImagePair:
    """A pair of images with ground truth pose."""

    pair_id: str
    image_a: str
    image_b: str
    gt_R: np.ndarray  # 3x3 rotation
    gt_t: np.ndarray  # 3x1 translation
    K_a: np.ndarray | None = None  # intrinsics
    K_b: np.ndarray | None = None
    overlap_score: float = 0.0  # ω ∈ [0,1]
    difficulty: str = "easy"  # easy/medium/hard/extreme
    dataset: str = "unknown"
    scene: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)


def compute_overlap_score_from_recon(
    shared_points: int, total_points_a: int, total_points_b: int
) -> float:
    """Overlap score = ratio of shared 3D landmarks."""
    if total_points_a == 0 or total_points_b == 0:
        return 0.0
    return 2.0 * shared_points / (total_points_a + total_points_b)


def difficulty_bin(overlap: float) -> str:
    """Map overlap score to difficulty bin."""
    if overlap > 0.7:
        return "easy"
    elif overlap > 0.3:
        return "medium"
    elif overlap > 0.1:
        return "hard"
    else:
        return "extreme"


def rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Rotation error in degrees."""
    R_rel = R1 @ R2.T
    trace = np.clip(np.trace(R_rel), -1.0, 3.0)
    return float(np.degrees(np.arccos((trace - 1) / 2)))


def translation_error_deg(t1: np.ndarray, t2: np.ndarray) -> float:
    """Translation direction error in degrees."""
    t1n = t1 / (np.linalg.norm(t1) + 1e-8)
    t2n = t2 / (np.linalg.norm(t2) + 1e-8)
    cos_angle = np.clip(np.dot(t1n, t2n), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def baseline_deg(R: np.ndarray, t: np.ndarray) -> float:
    """Approximate baseline angle between two views."""
    return rotation_error_deg(R, np.eye(3))


class HardPairDataset:
    """Dataset of hard image pairs with overlap-binned difficulty."""

    def __init__(self, pairs: list[ImagePair] | None = None):
        self.pairs = pairs or []

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> ImagePair:
        return self.pairs[idx]

    def filter_by_difficulty(self, difficulties: list[str]) -> "HardPairDataset":
        """Filter to specific difficulty bins."""
        return HardPairDataset([p for p in self.pairs if p.difficulty in difficulties])

    def split(
        self, val_ratio: float = 0.1, seed: int = 42
    ) -> tuple["HardPairDataset", "HardPairDataset"]:
        """Split into train/val, stratified by difficulty."""
        rng = np.random.default_rng(seed)
        train, val = [], []
        for diff in ["easy", "medium", "hard", "extreme"]:
            subset = [p for p in self.pairs if p.difficulty == diff]
            rng.shuffle(subset)
            n_val = int(len(subset) * val_ratio)
            val.extend(subset[:n_val])
            train.extend(subset[n_val:])
        return HardPairDataset(train), HardPairDataset(val)

    def stats(self) -> dict[str, Any]:
        """Return dataset statistics."""
        from collections import Counter

        diff_counts = Counter(p.difficulty for p in self.pairs)
        return {
            "total": len(self.pairs),
            "by_difficulty": dict(diff_counts),
            "mean_overlap": float(np.mean([p.overlap_score for p in self.pairs])) if self.pairs else 0.0,
        }

    def save(self, path: str) -> None:
        """Save dataset to JSON."""
        data = []
        for p in self.pairs:
            data.append({
                "pair_id": p.pair_id,
                "image_a": p.image_a,
                "image_b": p.image_b,
                "gt_R": p.gt_R.tolist(),
                "gt_t": p.gt_t.tolist(),
                "K_a": p.K_a.tolist() if p.K_a is not None else None,
                "K_b": p.K_b.tolist() if p.K_b is not None else None,
                "overlap_score": p.overlap_score,
                "difficulty": p.difficulty,
                "dataset": p.dataset,
                "scene": p.scene,
                "extra": p.extra,
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data)} pairs to {path}")

    @classmethod
    def load(cls, path: str) -> "HardPairDataset":
        """Load dataset from JSON."""
        with open(path) as f:
            data = json.load(f)
        pairs = []
        for d in data:
            pairs.append(ImagePair(
                pair_id=d["pair_id"],
                image_a=d["image_a"],
                image_b=d["image_b"],
                gt_R=np.array(d["gt_R"]),
                gt_t=np.array(d["gt_t"]),
                K_a=np.array(d["K_a"]) if d.get("K_a") else None,
                K_b=np.array(d["K_b"]) if d.get("K_b") else None,
                overlap_score=d["overlap_score"],
                difficulty=d["difficulty"],
                dataset=d["dataset"],
                scene=d["scene"],
                extra=d.get("extra", {}),
            ))
        return cls(pairs)


def build_megadepth_pairs(
    megadepth_dir: str,
    scenes: list[str] | None = None,
    max_pairs_per_scene: int = 100,
    min_overlap: float = 0.0,
) -> list[ImagePair]:
    """Build pairs from MegaDepth COLMAP reconstructions.

    Expects:
      megadepth_dir/
        scene_id/
          images/
          sparse/0/ (cameras.bin, images.bin, points3D.bin)
    """
    import pycolmap

    md_root = Path(megadepth_dir)
    if not md_root.exists():
        logger.warning(f"MegaDepth dir not found: {megadepth_dir}")
        return []

    scene_dirs = sorted(md_root.iterdir()) if scenes is None else [md_root / s for s in scenes]
    pairs = []

    for scene_dir in tqdm(scene_dirs, desc="MegaDepth scenes"):
        sparse_dir = scene_dir / "sparse" / "0"
        images_dir = scene_dir / "images"
        if not sparse_dir.exists() or not images_dir.exists():
            continue

        try:
            recon = pycolmap.Reconstruction(str(sparse_dir))
        except Exception as e:
            logger.warning(f"Cannot load {sparse_dir}: {e}")
            continue

        image_list = list(recon.images.values())
        if len(image_list) < 2:
            continue

        # Build pairs with overlap from shared 3D points
        for i in range(len(image_list)):
            for j in range(i + 1, min(i + 20, len(image_list))):
                img_a = image_list[i]
                img_b = image_list[j]

                # Compute overlap via shared observations
                pts_a = {p.point3D_id for p in img_a.points2D if p.has_point3D()}
                pts_b = {p.point3D_id for p in img_b.points2D if p.has_point3D()}
                shared = len(pts_a & pts_b)
                overlap = compute_overlap_score_from_recon(
                    shared, len(pts_a), len(pts_b)
                )

                if overlap < min_overlap:
                    continue

                # Relative pose
                R_a = img_a.cam_from_world.rotation().matrix if hasattr(img_a.cam_from_world, 'rotation') else np.eye(3)
                t_a = img_a.cam_from_world.translation if hasattr(img_a.cam_from_world, 'translation') else np.zeros(3)
                R_b = img_b.cam_from_world.rotation().matrix if hasattr(img_b.cam_from_world, 'rotation') else np.eye(3)
                t_b = img_b.cam_from_world.translation if hasattr(img_b.cam_from_world, 'translation') else np.zeros(3)

                R_rel = R_b @ R_a.T
                t_rel = t_b - R_rel @ t_a

                pair = ImagePair(
                    pair_id=f"md_{scene_dir.name}_{i}_{j}",
                    image_a=str(images_dir / img_a.name),
                    image_b=str(images_dir / img_b.name),
                    gt_R=R_rel,
                    gt_t=t_rel.reshape(3),
                    overlap_score=overlap,
                    difficulty=difficulty_bin(overlap),
                    dataset="megadepth",
                    scene=scene_dir.name,
                )
                pairs.append(pair)

                if len([p for p in pairs if p.scene == scene_dir.name]) >= max_pairs_per_scene:
                    break

    logger.info(f"Built {len(pairs)} MegaDepth pairs")
    return pairs


def build_dataset(
    megadepth_dir: str | None = None,
    output_path: str = "data/hard_pairs.json",
    max_pairs_per_scene: int = 100,
    min_overlap: float = 0.0,
) -> HardPairDataset:
    """Build the full hard-pair dataset."""
    all_pairs = []

    if megadepth_dir:
        all_pairs.extend(
            build_megadepth_pairs(megadepth_dir, max_pairs_per_scene=max_pairs_per_scene, min_overlap=min_overlap)
        )

    # TODO: add Map-free, ScanNet++ builders

    dataset = HardPairDataset(all_pairs)
    dataset.save(output_path)

    stats = dataset.stats()
    logger.info(f"Dataset stats: {stats}")

    return dataset
