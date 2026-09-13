#!/usr/bin/env python
"""Baseline evaluation for agentic SfM — no policy model required.

Runs pure tool-server baselines on the hard-pairs dataset and scores each
against GT pose with ``compute_pair_reward``:

  1. direct_loftr        — full-frame match, matcher=loftr
  2. direct_mast3r       — full-frame match, matcher=mast3r
  3. direct_lightglue    — full-frame match, matcher=lightglue
  4. random_crop_loftr   — one seeded random crop (random side) + LoFTR match
  5. oracle_crop_loftr   — all ORACLE_CROP_BOXES on both images (+ full-frame
                           fallback), keep best via keep_best_match

Requires a running tool server (LoFTR/MASt3R/LightGlue on GPU):
  python tools_server/server.py          # or jobs/run_baselines.slurm

Usage:
  python scripts/run_baselines.py \
      --pairs data/hard_pairs_val.json \
      --tool-server-url http://localhost:8765 \
      --output outputs/baselines/baseline_results.json

Notes on correctness:
  * When cropping img_b we call match("img_a", crop_id) (NOT match(crop_id,
    "img_a")): the returned relative pose is the a→b transform, which is the
    convention of the GT (R_rel = R_b @ R_a.T). Putting the b-crop in slot a
    would estimate the inverse pose. Keeping the crop in its own side's slot
    also keeps the K arguments aligned — the server shifts the principal point
    for whichever image id carries crop metadata.
  * All baselines are scored with num_tool_calls=num_valid_calls=1 so that
    total_reward differences reflect geometric quality only (format and
    accumulative tool terms are identical). The true number of server calls is
    recorded per pair as ``num_server_calls``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.data.hard_pairs import HardPairDataset, ImagePair
from agentic_sfm.geometry import crop_image_id, iter_oracle_crops, keep_best_match
from agentic_sfm.rewards.pose_rewards import compute_pair_reward
from agentic_sfm.tools.client import ToolClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DIFFICULTY_BINS = ["easy", "medium", "hard", "extreme"]

ALL_BASELINES = [
    "direct_loftr",
    "direct_mast3r",
    "direct_lightglue",
    "random_crop_loftr",
    "oracle_crop_loftr",
]


def _k_list(K: Any) -> list | None:
    """ndarray → list for the wire; pass None through."""
    if K is None:
        return None
    return K.tolist() if hasattr(K, "tolist") else K


def _gt_pose(pair: ImagePair) -> dict[str, Any] | None:
    if pair.gt_R is None or pair.gt_t is None:
        return None
    return {
        "R": np.asarray(pair.gt_R).reshape(3, 3).tolist(),
        "t": np.asarray(pair.gt_t).reshape(3).tolist(),
    }


def _random_crop_box(rng: np.random.Generator, min_scale: float = 0.4,
                     max_scale: float = 0.9) -> list[float]:
    """Uniform random normalized box [x1, y1, x2, y2] with side fractions in
    [min_scale, max_scale]."""
    w = float(rng.uniform(min_scale, max_scale))
    h = float(rng.uniform(min_scale, max_scale))
    x1 = float(rng.uniform(0.0, 1.0 - w))
    y1 = float(rng.uniform(0.0, 1.0 - h))
    return [x1, y1, x1 + w, y1 + h]


def _crop_and_match(
    tool_client: ToolClient,
    crop_side: str,
    bbox: list[float],
    matcher: str,
    max_size: int,
    K_a: list | None,
    K_b: list | None,
) -> dict[str, Any]:
    """Crop ``crop_side`` ("img_a" or "img_b") then match against the other.

    The crop stays on its own side's match slot so the estimated pose remains
    the a→b transform and the full-frame K_a/K_b line up with the right image
    (the server adjusts K for the crop's origin_xy).
    """
    crop_res = tool_client.crop(crop_side, bbox)
    cid = crop_image_id(crop_res)
    if not cid:
        return {"error": f"crop returned no id for {crop_side}", "crop": crop_res}
    if crop_side == "img_a":
        match_res = tool_client.match(
            cid, "img_b", matcher=matcher, max_size=max_size, K_a=K_a, K_b=K_b
        )
    else:
        match_res = tool_client.match(
            "img_a", cid, matcher=matcher, max_size=max_size, K_a=K_a, K_b=K_b
        )
    match_res["crop"] = crop_res
    match_res["cropped_image_id"] = cid
    match_res["crop_side"] = crop_side
    match_res["crop_bbox"] = bbox
    return match_res


# ---------------------------------------------------------------------------
# Baselines — each returns (match_result, meta, num_server_calls)
# ---------------------------------------------------------------------------


def eval_direct(
    tool_client: ToolClient, pair: ImagePair, matcher: str, max_size: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Full-frame match, no crop."""
    res = tool_client.match(
        "img_a", "img_b", matcher=matcher, max_size=max_size,
        K_a=_k_list(pair.K_a), K_b=_k_list(pair.K_b),
    )
    return res, {}, 1


def eval_random_crop(
    tool_client: ToolClient, pair: ImagePair, matcher: str, max_size: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """One seeded random crop on a randomly chosen image, then match."""
    crop_side = "img_a" if rng.integers(0, 2) == 0 else "img_b"
    bbox = _random_crop_box(rng)
    res = _crop_and_match(
        tool_client, crop_side, bbox, matcher, max_size,
        _k_list(pair.K_a), _k_list(pair.K_b),
    )
    return res, {"crop_side": crop_side, "bbox": bbox}, 2


def eval_oracle_crop(
    tool_client: ToolClient, pair: ImagePair, matcher: str, max_size: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Try every ORACLE_CROP_BOXES box on both images plus the full frame;
    keep the best match by inlier quality (match_quality)."""
    best: dict[str, Any] | None = None
    best_meta: dict[str, Any] | None = None
    calls = 0
    for crop_side, bbox, _other in iter_oracle_crops():
        try:
            res = _crop_and_match(
                tool_client, crop_side, bbox, matcher, max_size,
                _k_list(pair.K_a), _k_list(pair.K_b),
            )
        except Exception as e:  # server hiccup on one box — keep going
            logger.debug("Oracle crop %s %s failed: %s", crop_side, bbox, e)
            calls += 2
            continue
        calls += 2
        if keep_best_match(best, res) is res:
            best, best_meta = res, {"crop_side": crop_side, "bbox": bbox}
    # Full-frame fallback — the oracle may always prefer no crop.
    try:
        full = tool_client.match(
            "img_a", "img_b", matcher=matcher, max_size=max_size,
            K_a=_k_list(pair.K_a), K_b=_k_list(pair.K_b),
        )
        calls += 1
        if keep_best_match(best, full) is full:
            best, best_meta = full, None  # None → full frame won
    except Exception as e:
        logger.debug("Oracle full-frame failed: %s", e)
        calls += 1
    return best or {"error": "all oracle crops failed"}, {"best_crop": best_meta}, calls


BASELINE_FNS = {
    "direct_loftr": eval_direct,
    "direct_mast3r": eval_direct,
    "direct_lightglue": eval_direct,
    "random_crop_loftr": eval_random_crop,
    "oracle_crop_loftr": eval_oracle_crop,
}


def _matcher_for(name: str, crop_matcher: str) -> str:
    if name in ("random_crop_loftr", "oracle_crop_loftr"):
        return crop_matcher
    return name.split("_", 1)[1]  # direct_<matcher>


# ---------------------------------------------------------------------------
# Aggregation + paired bootstrap
# ---------------------------------------------------------------------------


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    inliers = np.array([r["num_inliers"] for r in records], dtype=np.float64)
    aucs = np.array([r["pose_auc"] for r in records], dtype=np.float64)
    rewards = np.array([r["total_reward"] for r in records], dtype=np.float64)
    succ = np.array([r["success"] for r in records], dtype=np.float64)
    rot_errs = [r["rotation_error_deg"] for r in records
                if r.get("rotation_error_deg") is not None]
    tr_errs = [r["translation_error_deg"] for r in records
               if r.get("translation_error_deg") is not None]

    by_difficulty: dict[str, Any] = {}
    for d in DIFFICULTY_BINS:
        sub = [r for r in records if r["difficulty"] == d]
        if not sub:
            continue
        by_difficulty[d] = {
            "count": len(sub),
            "mean_inliers": float(np.mean([r["num_inliers"] for r in sub])),
            "mean_pose_auc": float(np.mean([r["pose_auc"] for r in sub])),
            "mean_total_reward": float(np.mean([r["total_reward"] for r in sub])),
            "success_rate": float(np.mean([r["success"] for r in sub])),
        }

    return {
        "num_pairs": n,
        "num_errors": int(sum(1 for r in records if r.get("error"))),
        "mean_inliers": float(inliers.mean()) if n else 0.0,
        "mean_inlier_ratio": float(np.mean([r["inlier_ratio"] for r in records])) if n else 0.0,
        "mean_pose_auc": float(aucs.mean()) if n else 0.0,
        "mean_total_reward": float(rewards.mean()) if n else 0.0,
        "success_rate": float(succ.mean()) if n else 0.0,
        "mean_rotation_error_deg": float(np.mean(rot_errs)) if rot_errs else None,
        "mean_translation_error_deg": float(np.mean(tr_errs)) if tr_errs else None,
        "mean_server_calls": float(np.mean([r["num_server_calls"] for r in records])) if n else 0.0,
        "by_difficulty": by_difficulty,
    }


def paired_bootstrap(
    x: np.ndarray, y: np.ndarray, n_boot: int = 10000, seed: int = 0
) -> dict[str, Any]:
    """Paired bootstrap on mean(x) - mean(y) over resampled pair indices.

    Reports the 95% percentile CI and a two-sided p-value
    2 * min(P(diff <= 0), P(diff >= 0)).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if n == 0:
        return {"n_pairs": 0}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        b = min(2000, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        diffs[done : done + b] = x[idx].mean(axis=1) - y[idx].mean(axis=1)
        done += b
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p_two_sided = 2.0 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    return {
        "n_pairs": int(n),
        "mean_x": float(x.mean()),
        "mean_y": float(y.mean()),
        "mean_diff": float(x.mean() - y.mean()),
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "p_value": float(min(p_two_sided, 1.0)),
        "significant_05": bool(min(p_two_sided, 1.0) < 0.05),
    }


def compare_baselines(
    per_pair: dict[str, list[dict[str, Any]]],
    baseline_names: list[str],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """All pairwise paired-bootstrap comparisons on pose_auc and total_reward."""
    comparisons: dict[str, Any] = {"pose_auc": {}, "total_reward": {}}
    for a, b in itertools.combinations(baseline_names, 2):
        ra = per_pair[a]
        rb = per_pair[b]
        m = min(len(ra), len(rb))
        for metric in ("pose_auc", "total_reward"):
            x = np.array([r[metric] for r in ra[:m]])
            y = np.array([r[metric] for r in rb[:m]])
            comparisons[metric][f"{a}__vs__{b}"] = paired_bootstrap(
                x, y, n_boot=n_boot, seed=seed
            )
    return comparisons


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool-server baseline evaluation")
    parser.add_argument("--pairs", type=str, default="data/hard_pairs_val.json",
                        help="JSON pair file (hard_pairs_train.json or hard_pairs_val.json)")
    parser.add_argument("--tool-server-url", type=str, default="http://localhost:8765")
    parser.add_argument("--output", type=str,
                        default="outputs/baselines/baseline_results.json")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--baselines", nargs="*", default=ALL_BASELINES,
                        choices=ALL_BASELINES)
    parser.add_argument("--crop-matcher", type=str, default="loftr",
                        help="Matcher for random/oracle crop baselines")
    parser.add_argument("--max-size", type=int, default=512,
                        help="Matcher input resize (server-side)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Tool client timeout per call (s); first call loads weights")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for random-crop boxes and bootstrap resampling")
    parser.add_argument("--n-boot", type=int, default=10000,
                        help="Bootstrap resamples per comparison")
    args = parser.parse_args()

    tool_client = ToolClient(args.tool_server_url, timeout=args.timeout)
    health = tool_client.health()
    logger.info("Tool server: %s", health)

    dataset = HardPairDataset.load(args.pairs)
    pairs = dataset.pairs
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    logger.info("Evaluating %d pairs from %s on baselines: %s",
                len(pairs), args.pairs, args.baselines)

    # One rng per baseline → reproducible random crops independent of order.
    rngs = {name: np.random.default_rng(args.seed + i)
            for i, name in enumerate(ALL_BASELINES)}
    per_pair: dict[str, list[dict[str, Any]]] = {name: [] for name in args.baselines}

    pbar = tqdm(pairs, desc="Baselines")
    for pair in pbar:
        gt_pose = _gt_pose(pair)
        try:
            tool_client.register_image("img_a", pair.image_a)
            tool_client.register_image("img_b", pair.image_b)
        except Exception as e:
            logger.warning("register_image failed for %s: %s", pair.pair_id, e)
            for name in args.baselines:
                per_pair[name].append({
                    "pair_id": pair.pair_id, "difficulty": pair.difficulty,
                    "overlap_score": pair.overlap_score, "dataset": pair.dataset,
                    "scene": pair.scene, "num_matches": 0, "num_inliers": 0,
                    "inlier_ratio": 0.0, "pose_auc": 0.0, "total_reward": 0.0,
                    "inlier_reward": 0.0, "rotation_error_deg": None,
                    "translation_error_deg": None, "success": False,
                    "num_server_calls": 0, "matcher_used": None,
                    "error": f"register_image: {e}", "meta": {},
                })
            continue

        for name in args.baselines:
            fn = BASELINE_FNS[name]
            matcher = _matcher_for(name, args.crop_matcher)
            try:
                match_result, meta, n_calls = fn(
                    tool_client, pair, matcher, args.max_size, rngs[name]
                )
            except Exception as e:
                match_result, meta, n_calls = {"error": str(e)}, {"exception": str(e)}, 0

            components = compute_pair_reward(
                match_result, gt_pose=gt_pose,
                num_tool_calls=1, num_valid_calls=1, num_invalid_calls=0,
            )
            rec = {
                "pair_id": pair.pair_id,
                "difficulty": pair.difficulty,
                "overlap_score": pair.overlap_score,
                "dataset": pair.dataset,
                "scene": pair.scene,
                "num_matches": match_result.get("num_matches", 0),
                "num_inliers": match_result.get("num_inliers", 0),
                "inlier_ratio": match_result.get("inlier_ratio", 0.0),
                "pose_auc": components.get("pose_reward", 0.0),
                "total_reward": components["total_reward"],
                "inlier_reward": components.get("inlier_reward", 0.0),
                "rotation_error_deg": components.get("rotation_error_deg"),
                "translation_error_deg": components.get("translation_error_deg"),
                "success": bool(components.get("pose_reward", 0.0) > 0),
                "num_server_calls": n_calls,
                "matcher_used": match_result.get("matcher", matcher),
                "error": match_result.get("error"),
                "meta": meta,
            }
            per_pair[name].append(rec)
            if rec["matcher_used"] != matcher:
                # e.g. server falls back to LoFTR when MASt3R isn't installed
                logger.debug("[%s] requested %s but server used %s",
                             pair.pair_id, matcher, rec["matcher_used"])
            pbar.set_postfix_str(
                f"{name}: inl={rec['num_inliers']} auc={rec['pose_auc']:.2f}"
            )

    # Aggregate + compare
    baselines_out: dict[str, Any] = {}
    for name in args.baselines:
        baselines_out[name] = {
            "summary": _summarize(per_pair[name]),
            "per_pair": per_pair[name],
        }

    logger.info("Running paired bootstrap (%d resamples) ...", args.n_boot)
    comparisons = compare_baselines(per_pair, args.baselines, args.n_boot, args.seed)

    output = {
        "config": {
            "pairs": args.pairs,
            "tool_server_url": args.tool_server_url,
            "baselines": args.baselines,
            "crop_matcher": args.crop_matcher,
            "max_size": args.max_size,
            "max_pairs": args.max_pairs,
            "seed": args.seed,
            "n_boot": args.n_boot,
        },
        "num_pairs": len(pairs),
        "baselines": baselines_out,
        "comparisons": comparisons,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results written to %s", out_path)

    # Summary table
    logger.info("\n=== Baseline summary (%d pairs) ===", len(pairs))
    logger.info("%-22s %8s %9s %9s %8s %8s",
                "baseline", "inliers", "pose_auc", "reward", "succ%", "calls")
    logger.info("-" * 70)
    for name in args.baselines:
        s = baselines_out[name]["summary"]
        logger.info("%-22s %8.1f %9.4f %9.4f %7.1f%% %8.1f",
                    name, s["mean_inliers"], s["mean_pose_auc"],
                    s["mean_total_reward"], 100 * s["success_rate"],
                    s["mean_server_calls"])
        for d in DIFFICULTY_BINS:
            dd = s["by_difficulty"].get(d)
            if dd:
                logger.info("  %-20s n=%-4d inl=%-7.1f auc=%-7.4f rew=%-7.4f succ=%.1f%%",
                            d, dd["count"], dd["mean_inliers"], dd["mean_pose_auc"],
                            dd["mean_total_reward"], 100 * dd["success_rate"])

    logger.info("\n=== Paired bootstrap on pose_auc (mean diff [95% CI], p) ===")
    for key, r in comparisons["pose_auc"].items():
        logger.info("%-46s %+.4f [%+.4f, %+.4f] p=%.4f",
                    key, r["mean_diff"], r["ci95_lo"], r["ci95_hi"], r["p_value"])

    tool_client.close()


if __name__ == "__main__":
    main()
