#!/usr/bin/env python
"""Scene-level oracle study: fixed pair-selection strategies x matchers, no agent.

For every (scene, strategy) run the same cached-match ``sfm_run`` path and
record registration fraction plus relative-pose success over ALL image pairs
(unregistered images count as failures, so abstention is not rewarded).
``scripts/analyze_scene_oracle.py`` then compares the per-scene best-of-strategies
oracle with the best single strategy: a small gap means a learned scene-level
policy has little to gain.

Caveat: knn/exhaustive go through build_initial_graph, which only imports edges with
>=15 inliers into COLMAP; window strategies import every matched pair (no inlier floor).
Treat window-vs-others differences with that confound in mind.

Results are appended to ``<output-dir>/results_<matcher>.jsonl`` and the run is
resumable (finished (scene, strategy) keys are skipped).

Usage (tool server already running):
  python scripts/run_scene_oracle.py --matcher loftr --scannet-root <root> \
      --manifest data/frozen_eval_manifest.json --tool-server-url http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from agentic_sfm.agent.policy import ToolCall
from agentic_sfm.eval.scene_eval import compute_scene_metrics
from agentic_sfm.rl.scene_episode import (
    SceneRolloutEpisode,
    _inject_pose_error,
    _register_scene_images,
    cleanup_stage_dir,
    stage_episode_images,
)
from agentic_sfm.tools.client import ToolClient
from scripts.run_scene_baselines import (
    _score_baseline,
    run_exhaustive_baseline,
    run_knn_baseline,
)
from scripts.run_scene_grpo import SceneDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_scene_oracle")

THRESHOLDS = (5, 10, 20)


def run_window_baseline(
    scene: dict[str, Any],
    tool_client: ToolClient,
    window: int,
    matcher: str,
    output_dir_base: str,
) -> SceneRolloutEpisode:
    """Match each image with its next ``window`` frames (ScanNet frames are video order)."""
    image_paths = scene["image_paths"]
    n = len(image_paths)
    session_id = f"{scene['scene_id']}-win{window}-{uuid.uuid4().hex[:6]}"
    staged_dir = stage_episode_images(image_paths, image_root="", session_id=session_id)
    _register_scene_images(tool_client, image_paths, image_root="", staged_dir=staged_dir)
    ep = SceneRolloutEpisode(
        scene_id=scene["scene_id"],
        image_paths=image_paths,
        image_root="",
        num_images=n,
        gt_recon=scene.get("gt_recon"),
        session_id=session_id,
        staged_image_dir=staged_dir,
        done=True,
    )
    for u in range(n):
        for v in range(u + 1, min(n, u + 1 + window)):
            id_a, id_b = f"img_{u:04d}", f"img_{v:04d}"
            m_res = tool_client.match(id_a, id_b, matcher=matcher, session=session_id)
            ep.tool_calls.append(ToolCall(tool="match", args={"image_a": id_a, "image_b": id_b, "matcher": matcher}))
            ep.results.append(m_res)
    recon = tool_client.sfm_run(image_dir=staged_dir, output_dir=f"{output_dir_base}/{session_id}", session=session_id)
    ep.recon_result = recon
    ep.tool_calls.append(ToolCall(tool="sfm_run", args={}))
    ep.results.append(recon)
    ep.tool_calls.append(ToolCall(tool="done", args={}))
    _inject_pose_error(ep, scene.get("gt_recon"))
    _score_baseline(ep, scene)
    cleanup_stage_dir(staged_dir)
    return ep


def count_fallbacks(ep: SceneRolloutEpisode) -> int:
    """Pairs the server silently answered with LoFTR instead of the requested matcher."""
    n = 0
    for r in ep.results:
        if not isinstance(r, dict):
            continue
        if r.get("matcher_fallback"):
            n += 1
        n += int(r.get("pairs_fallback") or 0)  # build_initial_graph (knn / exhaustive)
    return n


def summarize(ep: SceneRolloutEpisode, scene: dict[str, Any], strategy: str, matcher: str, seconds: float) -> dict[str, Any]:
    m = compute_scene_metrics(ep, scene)
    n = int(m.get("registration/num_images") or len(scene["image_paths"]))
    total_pairs = n * (n - 1) / 2
    rel_pairs = float(m.get("pose/num_rel_pairs") or 0)
    row: dict[str, Any] = {
        "scene_id": scene["scene_id"],
        "strategy": strategy,
        "matcher": matcher,
        "num_images": n,
        "num_registered": m.get("registration/num_registered"),
        "registered_fraction": m.get("registration/registered_fraction"),
        "num_components": m.get("registration/num_components"),
        "num_pairs_matched": m.get("registration/num_pairs_matched"),
        "mean_pose_error_deg": m.get("pose/reported_mean_error_deg"),
        "seconds": round(seconds, 1),
        "tool_calls": len(ep.tool_calls),
        "matcher_fallbacks": count_fallbacks(ep),
        "match_errors": sum(1 for r in ep.results if isinstance(r, dict) and r.get("error")),
    }
    for t in THRESHOLDS:
        passed_of_registered = m.get(f"pose/rel_auc_{t}")
        # Success over ALL pairs: unregistered images contribute failures.
        row[f"allpairs_succ_{t}"] = (
            float(passed_of_registered) * rel_pairs / total_pairs
            if passed_of_registered is not None and total_pairs
            else 0.0
        )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matcher", required=True, choices=["loftr", "mast3r", "vggt"])
    ap.add_argument("--scannet-root", required=True)
    ap.add_argument("--manifest", default=str(_REPO / "data/frozen_eval_manifest.json"))
    ap.add_argument("--scene-ids", nargs="*", default=None, help="Override manifest scene list")
    ap.add_argument("--max-images", type=int, default=20)
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--strategies", nargs="+", default=["window1", "window2", "knn", "exhaustive"])
    ap.add_argument("--tool-server-url", default="http://127.0.0.1:8765")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sfm-output-base", default="sfm_runs", help="output_dir_base passed to sfm_run")
    args = ap.parse_args()

    scene_ids = args.scene_ids
    if not scene_ids:
        manifest = json.load(open(args.manifest))
        entries = manifest.get("scenes") or manifest.get("scene_ids")
        scene_ids = [e["scene_id"] if isinstance(e, dict) else e for e in entries]
    ds = SceneDataset.from_scannet(args.scannet_root, scene_ids=scene_ids, max_images_per_scene=args.max_images)
    scenes = ds.scenes
    logger.info("Loaded %d/%d scenes", len(scenes), len(scene_ids))
    if len(scenes) != len(scene_ids):
        missing = set(scene_ids) - {s["scene_id"] for s in scenes}
        logger.warning("Missing scenes: %s", sorted(missing))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_{args.matcher}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            if "error" in r:
                continue  # retry failed runs on resume
            done.add((r["scene_id"], r["strategy"]))

    client = ToolClient(base_url=args.tool_server_url)
    base = args.sfm_output_base

    for sc in scenes:
        for strat in args.strategies:
            key = (sc["scene_id"], strat)
            if key in done:
                continue
            t0 = time.time()
            try:
                if strat.startswith("window"):
                    ep = run_window_baseline(sc, client, int(strat[len("window"):]), args.matcher, f"{base}/{strat}")
                elif strat == "knn":
                    ep = run_knn_baseline(sc, client, k=args.knn_k, matcher=args.matcher, output_dir_base=f"{base}/knn")
                elif strat == "exhaustive":
                    ep = run_exhaustive_baseline(sc, client, matcher=args.matcher, output_dir_base=f"{base}/exhaustive")
                else:
                    raise ValueError(f"unknown strategy {strat}")
                row = summarize(ep, sc, strat, args.matcher, time.time() - t0)
            except (httpx.TransportError, ConnectionError):
                raise  # tool server down: abort so no zero rows are recorded
            except Exception as e:  # keep the sweep going; a failure is a data point
                logger.exception("[%s/%s] failed", sc["scene_id"], strat)
                row = {
                    "scene_id": sc["scene_id"], "strategy": strat, "matcher": args.matcher,
                    "error": str(e), "registered_fraction": 0.0,
                    **{f"allpairs_succ_{t}": 0.0 for t in THRESHOLDS},
                    "seconds": round(time.time() - t0, 1),
                }
            with open(out_path, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            if args.matcher != "loftr" and (row.get("matcher_fallbacks") or 0) > 0:
                raise SystemExit(
                    f"ABORT: {row['matcher_fallbacks']} {args.matcher} matches fell back to LoFTR on "
                    f"{sc['scene_id']}/{strat}; the {args.matcher} column would be invalid. Fix the matcher first."
                )
            logger.info("%s %s: reg=%.2f succ10=%.3f (%.0fs)", sc["scene_id"], strat,
                        row.get("registered_fraction") or 0.0, row["allpairs_succ_10"], row["seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
