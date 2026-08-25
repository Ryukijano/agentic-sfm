#!/usr/bin/env python
"""Convert the JSON hard-pair dataset into verl parquet format.

verl's ``RLHFDataset`` expects:
  - ``prompt``: list of chat messages (system + user).  Use ``<image>`` tokens
    for the two input images; the ``images`` column provides the actual paths.
  - ``images``: list of ``{"image": path}`` dicts.
  - ``extra_info``: dict with ``pair_id``, ``image_a``, ``image_b``,
    ``gt_R``, ``gt_t`` and any other per-sample metadata.
  - ``data_source``: a constant string tag.

Example:
  python scripts/prepare_verl_data.py \
      --input data/hard_pairs_train.json \
      --output data/verl/hard_pairs_train.parquet
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import datasets

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.agent.policy import SYSTEM_PROMPT


def _build_messages(user_text: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


USER_TEXT = (
    "<image>\n<image>\n"
    "Match these two images. Call tools to achieve the best matching result, "
    'then output {"tool": "done", "args": {}}.'
)


def build_record(sample: dict[str, Any]) -> dict[str, Any]:
    """Turn a raw hard-pair JSON record into a verl dataset row."""
    extra: dict[str, Any] = {
        "pair_id": sample["pair_id"],
        "image_a": sample["image_a"],
        "image_b": sample["image_b"],
    }
    if "gt_R" in sample and "gt_t" in sample:
        extra["gt_R"] = sample["gt_R"]
        extra["gt_t"] = sample["gt_t"]
    # Forward any other scalar metadata that may be useful for logging/debug.
    for key in ("K_a", "K_b", "overlap_score", "difficulty", "dataset", "scene"):
        if key in sample:
            extra[key] = sample[key]

    return {
        "prompt": _build_messages(USER_TEXT),
        "images": [{"image": sample["image_a"]}, {"image": sample["image_b"]}],
        "extra_info": extra,
        "data_source": "agentic_sfm",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert hard pairs JSON to verl parquet")
    parser.add_argument("--input", required=True, help="Path to hard pairs JSON file")
    parser.add_argument("--output", required=True, help="Output parquet path")
    args = parser.parse_args()

    with open(args.input) as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON list, got {type(raw).__name__}")

    records = [build_record(r) for r in raw]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # datasets handles nested dict/list schema transparently for parquet.
    ds = datasets.Dataset.from_list(records)
    ds.to_parquet(out_path)
    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
