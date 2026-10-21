#!/usr/bin/env python
"""Summarise scripts/run_scene_oracle.py output: best single strategy vs per-scene oracle.

Pure stdlib. Primary metric is ``allpairs_succ_10`` (unregistered images count as failures).

  python scripts/analyze_scene_oracle.py <output-dir> [--metric allpairs_succ_10]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load(out_dir: Path, metric: str) -> dict[str, dict[str, float]]:
    """scene -> {"matcher/strategy": value}"""
    table: dict[str, dict[str, float]] = defaultdict(dict)
    for p in sorted(out_dir.glob("results_*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            table[r["scene_id"]][f"{r['matcher']}/{r['strategy']}"] = float(r.get(metric) or 0.0)
    return table


def fallback_rows(out_dir: Path) -> list[str]:
    bad = []
    for p in sorted(out_dir.glob("results_*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get("matcher") != "loftr" and (r.get("matcher_fallbacks") or 0) > 0:
                bad.append(f"{r['matcher']}/{r['strategy']}/{r['scene_id']}: {r['matcher_fallbacks']}")
    return bad


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--metric", default="allpairs_succ_10")
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--allow-fallback", action="store_true", help="Analyse even if mast3r/vggt rows fell back to LoFTR")
    args = ap.parse_args()

    bad = fallback_rows(Path(args.output_dir))
    if bad:
        print(f"WARNING: {len(bad)} non-LoFTR rows silently fell back to LoFTR, e.g. {bad[:3]}")
        if not args.allow_fallback:
            print("Refusing to analyse; those columns are not the requested matcher. Use --allow-fallback to override.")
            return 2

    table = load(Path(args.output_dir), args.metric)
    combos = sorted({c for v in table.values() for c in v})
    # Only scenes where every combo finished, so comparisons are paired.
    scenes = sorted(s for s, v in table.items() if all(c in v for c in combos))
    print(f"metric={args.metric}  scenes with all {len(combos)} combos: {len(scenes)}/{len(table)}")
    if not scenes:
        return 1

    means = {c: mean([table[s][c] for s in scenes]) for c in combos}
    print("\nPer-combo mean:")
    for c, m in sorted(means.items(), key=lambda kv: -kv[1]):
        print(f"  {c:24s} {m:.3f}")
    best = max(means, key=means.get)

    oracle = [max(table[s].values()) for s in scenes]
    best_vals = [table[s][best] for s in scenes]
    gap = [o - b for o, b in zip(oracle, best_vals)]
    print(f"\nBest single combo: {best} = {means[best]:.3f}")
    print(f"Per-scene oracle (best of {len(combos)} combos) = {mean(oracle):.3f}")
    print(f"Oracle gain = {mean(gap):+.3f}")

    rng = random.Random(0)
    boots = sorted(mean([gap[rng.randrange(len(gap))] for _ in gap]) for _ in range(args.bootstrap))
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
    print(f"Scene-bootstrap 95% CI of oracle gain: [{lo:+.3f}, {hi:+.3f}]  (oracle gain is upward-biased: max over noisy arms)")

    wins: dict[str, int] = defaultdict(int)
    for s in scenes:
        wins[max(table[s], key=table[s].get)] += 1
    print("\nScenes where each combo is best (ties -> first):")
    for c, k in sorted(wins.items(), key=lambda kv: -kv[1]):
        print(f"  {c:24s} {k}")

    # Gain from choosing the strategy per scene while the matcher is fixed.
    print("\nOracle over strategies with matcher fixed:")
    for m in sorted({c.split('/')[0] for c in combos}):
        cs = [c for c in combos if c.startswith(m + "/")]
        orc = mean([max(table[s][c] for c in cs) for s in scenes])
        bst = max(mean([table[s][c] for s in scenes]) for c in cs)
        print(f"  {m:8s} best single {bst:.3f}  oracle {orc:.3f}  gain {orc - bst:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
