#!/usr/bin/env python
"""Build hard-pair dataset from MegaDepth and other sources.

Usage:
  python scripts/build_pairs.py --megadepth-dir /scratch/kcwp264/data/megadepth --output data/hard_pairs.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.data.hard_pairs import build_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build hard-pair dataset")
    parser.add_argument("--megadepth-dir", type=str, default="/scratch/kcwp264/data/megadepth")
    parser.add_argument("--output", type=str, default="data/hard_pairs.json")
    parser.add_argument("--max-pairs-per-scene", type=int, default=100)
    parser.add_argument("--min-overlap", type=float, default=0.0)
    args = parser.parse_args()

    dataset = build_dataset(
        megadepth_dir=args.megadepth_dir,
        output_path=args.output,
        max_pairs_per_scene=args.max_pairs_per_scene,
        min_overlap=args.min_overlap,
    )

    stats = dataset.stats()
    logger.info(f"Final dataset: {stats}")


if __name__ == "__main__":
    main()
