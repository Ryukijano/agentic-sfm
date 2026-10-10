#!/usr/bin/env python
"""Three-baseline diagnostic benchmark for scene-level SfM (Phase 2).

Compares a scripted graph policy (not a trained agent) against three baselines:
  1. Fixed k-NN Graph Baseline (DINOv2 top-k nearest neighbors per image)
  2. Exhaustive COLMAP Baseline (All N(N-1)/2 pairs matched)
  3. Random Edge Sampling Baseline (E = k*N random pairs matched)
  4. Scripted graph policy (build -> inspect -> verify -> prune -> sfm_run)

Usage:
  # Synthetic offline smoke test (no GPU, no real server needed)
  python scripts/run_scene_baselines.py --synthetic --mock-server

  # Run on MegaDepth holdout scenes with real tool server
  python scripts/run_scene_baselines.py --scene-ids 0022 0024 --real --tool-server-url http://localhost:8765
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from agentic_sfm.agent.policy import ToolCall
from agentic_sfm.constants import DEFAULT_MATCHER
from agentic_sfm.eval.scene_eval import aggregate_scene_metrics, compute_scene_metrics
from agentic_sfm.rewards.pose_rewards import compute_scene_reward
from agentic_sfm.rl.scene_episode import (
    SceneRolloutEpisode,
    _inject_pose_error,
    _register_scene_images,
    cleanup_stage_dir,
    run_scene_episode,
    stage_episode_images,
)
from agentic_sfm.tools.client import ToolClient
from scripts.run_scene_grpo import SceneDataset
from scripts.scene_smoke_test import (
    MockToolClient,
    ScriptedGraphAgent,
    install_mock_retrieval,
    write_synthetic_scene,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_scene_baselines")

DEFAULT_SCENE_INFO_DIR = "/scratch/kcwp264/data/megadepth/scene_info_full/scene_info"
DEFAULT_IMAGE_ROOT = "/scratch/kcwp264/data/megadepth/megadepth_test_1500"


def _score_baseline(ep: SceneRolloutEpisode, scene: dict[str, Any]) -> None:
    ep.reward_components = compute_scene_reward(
        ep.recon_result,
        gt_recon=scene.get("gt_recon"),
        num_tool_calls=len(ep.tool_calls),
        num_valid_calls=sum(not r.get("error") for r in ep.results),
        tool_calls=ep.tool_calls,
        tool_results=ep.results,
        tool_cost=0.0,
        use_accumulative_tool_reward=False,
        doppelganger_weight=0.0,
        first_call_validity_weight=0.0,
    )
    ep.reward = ep.reward_components["total_reward"]


def run_exhaustive_baseline(
    scene: dict[str, Any],
    tool_client: ToolClient,
    image_root: str = "",
    matcher: str = DEFAULT_MATCHER,
    output_dir_base: str = "outputs/baselines/exhaustive",
) -> SceneRolloutEpisode:
    """Baseline 1: Exhaustive matching of all N*(N-1)/2 pairs."""
    image_paths = scene["image_paths"]
    n = len(image_paths)
    session_id = f"{scene['scene_id']}-exh-{uuid.uuid4().hex[:6]}"
    staged_dir = stage_episode_images(image_paths, image_root=image_root, session_id=session_id)
    registered_ids = _register_scene_images(
        tool_client, image_paths, image_root=image_root, staged_dir=staged_dir
    )

    ep = SceneRolloutEpisode(
        scene_id=scene["scene_id"],
        image_paths=image_paths,
        image_root=image_root,
        num_images=n,
        gt_recon=scene.get("gt_recon"),
        session_id=session_id,
        staged_image_dir=staged_dir,
        done=True,
    )

    # Build exhaustive graph (k_neighbors = n - 1)
    b_res = tool_client.build_initial_graph(image_ids=list(registered_ids.values()), k_neighbors=max(1, n - 1), matcher=matcher, session=session_id)
    ep.tool_calls.append(ToolCall(tool="build_initial_graph", args={"k_neighbors": n - 1, "matcher": matcher}))
    ep.results.append(b_res)

    # Run COLMAP reconstruction on all verified pairs
    recon = tool_client.sfm_run(
        image_dir=staged_dir,
        output_dir=f"{output_dir_base}/{session_id}",
        session=session_id,
    )
    ep.recon_result = recon
    ep.tool_calls.append(ToolCall(tool="sfm_run", args={}))
    ep.results.append(recon)
    ep.tool_calls.append(ToolCall(tool="done", args={}))

    _inject_pose_error(ep, scene.get("gt_recon"))
    _score_baseline(ep, scene)
    cleanup_stage_dir(staged_dir)
    return ep


def run_knn_baseline(
    scene: dict[str, Any],
    tool_client: ToolClient,
    k: int = 3,
    image_root: str = "",
    matcher: str = DEFAULT_MATCHER,
    output_dir_base: str = "outputs/baselines/knn",
) -> SceneRolloutEpisode:
    """Baseline 2: Fixed k-NN graph with no cycle inspection or verification."""
    image_paths = scene["image_paths"]
    n = len(image_paths)
    session_id = f"{scene['scene_id']}-knn{k}-{uuid.uuid4().hex[:6]}"
    staged_dir = stage_episode_images(image_paths, image_root=image_root, session_id=session_id)
    registered_ids = _register_scene_images(
        tool_client, image_paths, image_root=image_root, staged_dir=staged_dir
    )

    ep = SceneRolloutEpisode(
        scene_id=scene["scene_id"],
        image_paths=image_paths,
        image_root=image_root,
        num_images=n,
        gt_recon=scene.get("gt_recon"),
        session_id=session_id,
        staged_image_dir=staged_dir,
        done=True,
    )

    # Build k-NN graph
    b_res = tool_client.build_initial_graph(image_ids=list(registered_ids.values()), k_neighbors=k, matcher=matcher, session=session_id)
    ep.tool_calls.append(ToolCall(tool="build_initial_graph", args={"k_neighbors": k, "matcher": matcher}))
    ep.results.append(b_res)

    # Run COLMAP without inspection or pruning
    recon = tool_client.sfm_run(
        image_dir=staged_dir,
        output_dir=f"{output_dir_base}/{session_id}",
        session=session_id,
    )
    ep.recon_result = recon
    ep.tool_calls.append(ToolCall(tool="sfm_run", args={}))
    ep.results.append(recon)
    ep.tool_calls.append(ToolCall(tool="done", args={}))

    _inject_pose_error(ep, scene.get("gt_recon"))
    _score_baseline(ep, scene)
    cleanup_stage_dir(staged_dir)
    return ep


def run_random_baseline(
    scene: dict[str, Any],
    tool_client: ToolClient,
    edges_per_node: int = 3,
    image_root: str = "",
    matcher: str = DEFAULT_MATCHER,
    seed: int = 42,
    output_dir_base: str = "outputs/baselines/random",
) -> SceneRolloutEpisode:
    """Baseline 3: Random edge sampling with budget matching k-NN (E = edges_per_node * N)."""
    image_paths = scene["image_paths"]
    n = len(image_paths)
    session_id = f"{scene['scene_id']}-rand-{uuid.uuid4().hex[:6]}"
    staged_dir = stage_episode_images(image_paths, image_root=image_root, session_id=session_id)
    registered_ids = _register_scene_images(
        tool_client, image_paths, image_root=image_root, staged_dir=staged_dir
    )

    ep = SceneRolloutEpisode(
        scene_id=scene["scene_id"],
        image_paths=image_paths,
        image_root=image_root,
        num_images=n,
        gt_recon=scene.get("gt_recon"),
        session_id=session_id,
        staged_image_dir=staged_dir,
        done=True,
    )

    # Generate all possible pairs and randomly sample budget
    all_pairs = list(itertools.combinations(range(n), 2))
    budget = min(len(all_pairs), max(1, edges_per_node * n))
    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(all_pairs), size=budget, replace=False)
    chosen_pairs = [all_pairs[idx] for idx in chosen_indices]

    # Match sampled pairs
    matched_count = 0
    for u, v in chosen_pairs:
        id_a, id_b = f"img_{u:04d}", f"img_{v:04d}"
        m_res = tool_client.match(id_a, id_b, matcher=matcher, session=session_id)
        ep.tool_calls.append(ToolCall(tool="match", args={"image_a": id_a, "image_b": id_b, "matcher": matcher}))
        ep.results.append(m_res)
        if not m_res.get("error"):
            matched_count += 1

    # Run COLMAP on imported random matches
    recon = tool_client.sfm_run(
        image_dir=staged_dir,
        output_dir=f"{output_dir_base}/{session_id}",
        session=session_id,
    )
    ep.recon_result = recon
    ep.tool_calls.append(ToolCall(tool="sfm_run", args={}))
    ep.results.append(recon)
    ep.tool_calls.append(ToolCall(tool="done", args={}))

    _inject_pose_error(ep, scene.get("gt_recon"))
    _score_baseline(ep, scene)
    cleanup_stage_dir(staged_dir)
    return ep


def run_graph_agent_baseline(
    scene: dict[str, Any],
    tool_client: ToolClient,
    image_root: str = "",
    matcher: str = DEFAULT_MATCHER,
    k: int = 3,
) -> SceneRolloutEpisode:
    """Deterministic graph-action script, not a trained policy."""
    agent = ScriptedGraphAgent(matcher=matcher, k_neighbors=k)
    ep = run_scene_episode(
        agent=agent,
        scene_id=scene["scene_id"],
        image_paths=scene["image_paths"],
        tool_client=tool_client,
        gt_recon=scene.get("gt_recon"),
        overlap_matrix=scene.get("overlap_matrix"),
        image_indices=scene.get("image_indices"),
        max_tool_calls=20,
        max_turns=30,
        image_root=image_root,
    )
    _score_baseline(ep, scene)
    return ep


def print_comparison_table(results: dict[str, dict[str, Any]]) -> None:
    """Print formatted markdown comparison table."""
    keys = [
        ("registration/registered_fraction", "RegFrac", "{:.1%}"),
        ("registration/num_points3d", "3D Points", "{:.0f}"),
        ("registration/mean_reproj_error_px", "ReprojPx", "{:.2f}"),
        ("pose/rel_auc_10", "AUC@10°", "{:.3f}"),
        ("pose/rel_rot_err_mean_deg", "RotErr°", "{:.2f}"),
        ("doppelganger/f1", "DopF1", "{:.2f}"),
        ("efficiency/num_tool_calls", "Calls", "{:.0f}"),
        ("reward/total_reward", "GeometryScore", "{:.3f}"),
    ]
    header = f"| {'Baseline':<25} | " + " | ".join(f"{name:>9}" for _, name, _ in keys) + " |"
    sep = f"|:{'-'*25}-|-" + "-|-".join(f"{'-'*9}:" for _ in keys) + "-|"

    print("\n" + "=" * len(header))
    print("SCENE GRAPH ORCHESTRATION VS PHOTOGRAMMETRIC BASELINES")
    print("=" * len(header))
    print(header)
    print(sep)

    for method, metrics in results.items():
        row_vals = []
        for key, _, fmt in keys:
            norm_key = key.replace('/', '_')
            v = metrics.get(f"scene_eval/mean_{norm_key}")
            if v is None:
                v = metrics.get(f"scene_eval_mean_{norm_key}")
            if v is None:
                v = metrics.get(key)
            if v is not None:
                row_vals.append(f"{fmt.format(v):>9}")
            else:
                row_vals.append(f"{'—':>9}")
        print(f"| {method:<25} | " + " | ".join(row_vals) + " |")
    print("=" * len(header) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene-ids", nargs="+", default=["0015"])
    parser.add_argument("--scene-info-dir", default=DEFAULT_SCENE_INFO_DIR)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--scannet-root", default="", help="Evaluate held-out ScanNet scenes instead of MegaDepth")
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--k", type=int, default=3, help="k for k-NN graph baseline")
    parser.add_argument("--matcher", default=DEFAULT_MATCHER)
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic scene for offline test")
    parser.add_argument("--mock-server", action="store_true", help="Use MockToolClient instead of HTTP server")
    parser.add_argument("--real", action="store_true", help="Use the real tool server")
    parser.add_argument("--tool-server-url", default="http://localhost:8765")
    parser.add_argument("--output-dir", default="outputs/baselines")
    args = parser.parse_args()
    if args.real and args.mock_server:
        parser.error("--real and --mock-server are mutually exclusive")
    if args.real and args.synthetic:
        parser.error("--real requires a real scene dataset")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load scenes
    if args.synthetic:
        tmp_dir = Path(tempfile.mkdtemp(prefix="baseline_synth_"))
        scene_id = write_synthetic_scene(tmp_dir / "scene_info", tmp_dir / "images", n_images=args.max_images)
        ds = SceneDataset.from_megadepth(str(tmp_dir / "scene_info"), str(tmp_dir / "images"), scenes=[scene_id])
        scenes = ds.scenes
        image_root = str(tmp_dir / "images")
        logger.info(f"Loaded synthetic scene {scene_id} with {scenes[0]['num_images']} images")
    elif args.scannet_root:
        ds = SceneDataset.from_scannet(
            args.scannet_root, scene_ids=args.scene_ids, max_images_per_scene=args.max_images
        )
        scenes = ds.scenes
        image_root = ""
        logger.info(f"Loaded {len(scenes)} ScanNet scenes from {args.scannet_root}")
    else:
        ds = SceneDataset.from_megadepth(
            scene_info_dir=args.scene_info_dir,
            image_root=args.image_root,
            scenes=args.scene_ids,
            max_images=args.max_images,
        )
        scenes = ds.scenes
        image_root = args.image_root
        logger.info(f"Loaded {len(scenes)} scenes from {args.scene_info_dir}")
    if not scenes:
        parser.error("No usable scenes found for the requested IDs and dataset")

    # 2. Setup client
    if args.mock_server or args.synthetic:
        install_mock_retrieval()
        tool_client = MockToolClient()
    else:
        tool_client = ToolClient(base_url=args.tool_server_url)

    # 3. Evaluate each baseline
    baselines = {
        "1. Exhaustive COLMAP": lambda sc: run_exhaustive_baseline(
            sc, tool_client, image_root=image_root, matcher=args.matcher
        ),
        f"2. Fixed k-NN (k={args.k})": lambda sc: run_knn_baseline(
            sc, tool_client, k=args.k, image_root=image_root, matcher=args.matcher
        ),
        f"3. Random Sampling (E={args.k}N)": lambda sc: run_random_baseline(
            sc, tool_client, edges_per_node=args.k, image_root=image_root, matcher=args.matcher
        ),
        "4. Scripted graph policy": lambda sc: run_graph_agent_baseline(
            sc, tool_client, image_root=image_root, matcher=args.matcher, k=args.k
        ),
    }

    benchmark_summary: dict[str, dict[str, Any]] = {}

    for name, runner in baselines.items():
        logger.info(f"Running baseline: {name}")
        scene_metrics_list = []
        for sc in scenes:
            ep = runner(sc)
            m = compute_scene_metrics(ep, sc)
            m["reward/total_reward"] = ep.reward
            scene_metrics_list.append(m)
        agg = aggregate_scene_metrics(scene_metrics_list)
        agg["reward/total_reward"] = float(np.mean([m.get("reward/total_reward", 0.0) for m in scene_metrics_list]))
        benchmark_summary[name] = agg

    # 4. Display results table
    print_comparison_table(benchmark_summary)

    # 5. Save JSON
    save_path = out_dir / "baseline_benchmark_results.json"
    with open(save_path, "w") as f:
        json.dump(benchmark_summary, f, indent=2, default=str)
    logger.info(f"Benchmark results saved to {save_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
