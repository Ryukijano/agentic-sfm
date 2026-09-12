#!/usr/bin/env python
"""Build minimal format SFT data for the 2B policy.

LiteSearch-VL (arXiv 2608.29357) shows that 2B models need to learn the
"agent contract" (valid JSON tool-call format) before RL. Without it, the
2B model "almost never emits a usable answer."

This script generates ~200 valid crop_and_match examples using the heuristic
ORACLE_CROP_BOXES, WITHOUT requiring a running tool server. It creates the
JSONL format that the SFT trainer expects.

Usage:
  python scripts/build_format_sft.py \
      --pairs data/hard_pairs_train.json \
      --output data/sft_format.jsonl \
      --n-examples 200
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.agent.policy import SYSTEM_PROMPT
from agentic_sfm.data.hard_pairs import HardPairDataset
from agentic_sfm.geometry import ORACLE_CROP_BOXES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

USER_PROMPT = (
    'Match these two images. Call tools to achieve the best matching result, '
    'then output {"tool": "done"}.'
)


def build_example(pair, bbox: list[float], crop_img_id: str, other_img_id: str,
                  matcher: str = "loftr") -> dict:
    """Build a single SFT example with valid JSON tool-call format."""
    crop_call = json.dumps({
        "tool": "crop_and_match",
        "args": {
            "crop_image_id": crop_img_id,
            "bbox": bbox,
            "match_image_id": other_img_id,
            "matcher": matcher,
        },
    })
    done_call = json.dumps({"tool": "done", "args": {}})

    # Simulated observation (format only — actual match results come at RL time)
    obs_text = (
        f"Observation: crop_and_match result: cropped {crop_img_id} with bbox {bbox}, "
        f"matched against {other_img_id} using {matcher}. "
        f"num_inliers=45, inlier_ratio=0.3, pose estimated."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": pair.image_a},
            {"type": "image", "image": pair.image_b},
            {"type": "text", "text": USER_PROMPT},
        ]},
        {"role": "assistant", "content": crop_call},
        {"role": "user", "content": obs_text},
        {"role": "assistant", "content": done_call},
    ]

    return {
        "pair_id": pair.pair_id,
        "image_a": pair.image_a,
        "image_b": pair.image_b,
        "messages": messages,
        "difficulty": pair.difficulty,
    }


def build_match_only_example(pair, matcher: str = "loftr") -> dict:
    """Build a full-frame match example (no crop)."""
    match_call = json.dumps({
        "tool": "match",
        "args": {
            "image_a_id": "img_a",
            "image_b_id": "img_b",
            "matcher": matcher,
        },
    })
    done_call = json.dumps({"tool": "done", "args": {}})

    obs_text = (
        f"Observation: match result: matched img_a and img_b using {matcher}. "
        f"num_inliers=30, inlier_ratio=0.2, pose estimated."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": pair.image_a},
            {"type": "image", "image": pair.image_b},
            {"type": "text", "text": USER_PROMPT},
        ]},
        {"role": "assistant", "content": match_call},
        {"role": "user", "content": obs_text},
        {"role": "assistant", "content": done_call},
    ]

    return {
        "pair_id": pair.pair_id,
        "image_a": pair.image_a,
        "image_b": pair.image_b,
        "messages": messages,
        "difficulty": pair.difficulty,
    }


def main():
    parser = argparse.ArgumentParser(description="Build minimal format SFT data")
    parser.add_argument("--pairs", type=str, default="data/hard_pairs_train.json")
    parser.add_argument("--output", type=str, default="data/sft_format.jsonl")
    parser.add_argument("--n-examples", type=int, default=200)
    parser.add_argument("--matcher", type=str, default="loftr")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = HardPairDataset.load(args.pairs)
    logger.info(f"Loaded {len(dataset)} pairs from {args.pairs}")

    rng = random.Random(args.seed)
    pairs = list(dataset.pairs)
    rng.shuffle(pairs)

    examples = []
    n_crop = 0
    n_match = 0

    for pair in tqdm(pairs[:args.n_examples], desc="Building SFT examples"):
        # Alternate between crop_and_match and full-frame match
        if n_crop < args.n_examples * 0.7:
            # Pick a random oracle crop box
            bbox = rng.choice(ORACLE_CROP_BOXES)
            crop_img_id = rng.choice(["img_a", "img_b"])
            other_img_id = "img_b" if crop_img_id == "img_a" else "img_a"
            examples.append(build_example(pair, bbox, crop_img_id, other_img_id, args.matcher))
            n_crop += 1
        else:
            examples.append(build_match_only_example(pair, args.matcher))
            n_match += 1

    # Write JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    logger.info(f"Saved {len(examples)} SFT examples to {output_path}")
    logger.info(f"  crop_and_match: {n_crop}, full-frame match: {n_match}")


if __name__ == "__main__":
    main()
