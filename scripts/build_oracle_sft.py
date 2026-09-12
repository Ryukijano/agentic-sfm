#!/usr/bin/env python
"""Build crop SFT data without relying on the stock VLM.

Zero-shot Qwen rarely crops, so trajectories.jsonl is usually empty of useful
crop traces. This script matches each pair full-frame, then tries heuristic
overlap boxes (ORACLE_CROP_BOXES on img_a and img_b). Pairs where a crop beats
full-frame inliers become two-turn SFT chats: crop_and_match → observation → done.

Requires a running tool server (jobs/phase1_oracle_sft.slurm starts one).

Usage:
  python scripts/build_oracle_sft.py \
      --pairs data/hard_pairs_train.json \
      --output data/sft_train.jsonl \
      --tool-server-url http://localhost:8765
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.agent.policy import SYSTEM_PROMPT, ToolCall, execute_sfm_tool, format_observation
from agentic_sfm.data.hard_pairs import HardPairDataset
from agentic_sfm.geometry import crop_image_id, iter_oracle_crops, match_quality
from agentic_sfm.tools.client import ToolClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

USER_PROMPT = (
    'Match these two images. Call tools to achieve the best matching result, '
    'then output {"tool": "done"}.'
)


def _k_list(K):
    if K is None:
        return None
    return K.tolist() if hasattr(K, "tolist") else K


def _persist_crop(result: dict, dest: Path) -> str | None:
    crop = result.get("crop") if isinstance(result.get("crop"), dict) else result
    src = crop.get("path") if isinstance(crop, dict) else None
    if not src or not Path(src).exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return str(dest)


def _example_from_win(
    pair,
    crop_id: str,
    bbox: list[float],
    other_id: str,
    matcher: str,
    result: dict,
    crop_path: str | None,
) -> dict:
    args = {
        "image_id": crop_id,
        "bbox": bbox,
        "image_b": other_id,
        "matcher": matcher,
    }
    crop_call = json.dumps({"tool": "crop_and_match", "args": args})
    done_call = json.dumps({"tool": "done", "args": {}})
    obs_text = f"Observation: {format_observation(result)}"
    cid = crop_image_id(result)
    if cid and "cropped_image_id" not in obs_text:
        obs_text += f" Use id {cid} in match."

    obs_content: list[dict] = [{"type": "text", "text": obs_text}]
    crop_paths: list[str] = []
    if crop_path:
        obs_content.insert(0, {"type": "image"})
        crop_paths.append(crop_path)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
        {"role": "assistant", "content": crop_call},
        {"role": "user", "content": obs_content},
        {"role": "assistant", "content": done_call},
    ]
    return {
        "pair_id": pair.pair_id,
        "image_a": pair.image_a,
        "image_b": pair.image_b,
        "crop_image_paths": crop_paths,
        "messages": messages,
        "reward": 0.0,
        "reward_components": {
            "source": "oracle_crop",
            "num_inliers": result.get("num_inliers"),
            "inlier_ratio": result.get("inlier_ratio"),
        },
        "difficulty": pair.difficulty,
        "num_tool_calls": 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle crop SFT from heuristic boxes")
    parser.add_argument("--pairs", type=str, default="data/hard_pairs_train.json")
    parser.add_argument("--output", type=str, default="data/sft_train.jsonl")
    parser.add_argument("--tool-server-url", type=str, default="http://localhost:8765")
    parser.add_argument("--matcher", type=str, default="loftr")
    parser.add_argument("--min-inlier-gain", type=float, default=1.0,
                        help="Crop must beat full-frame inliers by at least this many")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--difficulties", nargs="*", default=None)
    parser.add_argument("--crop-dir", type=str, default="data/oracle_crops")
    args = parser.parse_args()

    tool_client = ToolClient(args.tool_server_url)
    health = tool_client.health()
    logger.info("Tool server: %s", health)

    dataset = HardPairDataset.load(args.pairs)
    pairs = dataset.pairs
    if args.difficulties:
        pairs = [p for p in pairs if p.difficulty in args.difficulties]
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    logger.info("Oracle search over %d pairs", len(pairs))

    crop_root = Path(args.crop_dir)
    crop_root.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_kept = 0
    n_full_better = 0
    n_errors = 0
    with open(out_path, "w") as fout:
        for pair in tqdm(pairs, desc="Oracle crops"):
            tool_client.register_image("img_a", pair.image_a)
            tool_client.register_image("img_b", pair.image_b)
            match_kwargs = {}
            ka, kb = _k_list(pair.K_a), _k_list(pair.K_b)
            if ka is not None:
                match_kwargs["K_a"] = ka
            if kb is not None:
                match_kwargs["K_b"] = kb

            full = execute_sfm_tool(
                tool_client,
                ToolCall(tool="match", args={"image_a": "img_a", "image_b": "img_b", "matcher": args.matcher}),
                match_kwargs,
            )
            if full.get("error"):
                n_errors += 1
                continue
            full_inliers = float(full.get("num_inliers") or 0)
            best = None
            best_meta = None
            for crop_id, bbox, other_id in iter_oracle_crops():
                result = execute_sfm_tool(
                    tool_client,
                    ToolCall(
                        tool="crop_and_match",
                        args={
                            "image_id": crop_id,
                            "bbox": bbox,
                            "image_b": other_id,
                            "matcher": args.matcher,
                        },
                    ),
                    match_kwargs,
                )
                if result.get("error"):
                    continue
                crop_inliers = float(result.get("num_inliers") or 0)
                if crop_inliers < full_inliers + args.min_inlier_gain:
                    continue
                if best is None or match_quality(result) > match_quality(best):
                    best = result
                    best_meta = (crop_id, bbox, other_id)

            if best is None or best_meta is None:
                n_full_better += 1
                continue

            crop_id, bbox, other_id = best_meta
            dest = crop_root / f"{pair.pair_id}_{crop_id}_{bbox[0]:.2f}_{bbox[1]:.2f}.jpg"
            crop_path = _persist_crop(best, dest)
            example = _example_from_win(
                pair, crop_id, bbox, other_id, args.matcher, best, crop_path
            )
            example["reward_components"]["fullframe_inliers"] = full_inliers
            fout.write(json.dumps(example) + "\n")
            fout.flush()
            n_kept += 1

    summary = {
        "pairs": len(pairs),
        "kept": n_kept,
        "fullframe_better_or_tie": n_full_better,
        "errors": n_errors,
        "output": str(out_path),
        "matcher": args.matcher,
        "min_inlier_gain": args.min_inlier_gain,
    }
    summary_path = out_path.parent / "oracle_sft_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Oracle SFT: kept %d / %d (full-frame won %d, errors %d) → %s",
                n_kept, len(pairs), n_full_better, n_errors, out_path)
    logger.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
