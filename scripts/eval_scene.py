#!/usr/bin/env python
"""Phase 2: scene-level evaluation for agentic SfM.

Runs the scene-level pipeline on a set of MegaDepth scenes and reports
scene-level metrics (registration, pose accuracy, completeness,
doppelganger filtering, tool efficiency) via
``agentic_sfm.eval.scene_eval.compute_scene_metrics``.

Modes:
  agent   — full agent rollout via a running vLLM server (needs --vllm-url)
  oracle  — S-GRPO oracle trajectory (top-overlap pairs from GT), no vLLM
  direct  — no-agent baseline: register images + exhaustive COLMAP sfm_run

Usage:
  # No-agent COLMAP baseline (fast sanity check)
  python scripts/eval_scene.py --config configs/phase2_scene.yaml \
      --mode direct --max-scenes 5

  # Agent + oracle + direct comparison
  python scripts/eval_scene.py --config configs/phase2_scene.yaml \
      --mode agent oracle direct --vllm-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.agent.policy import ToolCall
from agentic_sfm.constants import DEFAULT_MATCHER
from agentic_sfm.eval.scene_eval import aggregate_scene_metrics, compute_scene_metrics
from agentic_sfm.rewards.pose_rewards import compute_scene_reward
from agentic_sfm.rl.scene_episode import (
    SceneRolloutEpisode,
    run_scene_episode,
    run_scene_oracle_episode,
)
from agentic_sfm.tools.client import ToolClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class _ShimAgent:
    """Minimal agent interface for oracle episodes (no vLLM needed).

    The oracle only uses ``_encode_image`` (thumbnails), ``matcher`` (match
    tool args), and ``reward_config`` (scene reward weights).
    """

    def __init__(self, matcher: str = DEFAULT_MATCHER, reward_config: dict | None = None):
        self.matcher = matcher
        self.reward_config = reward_config or {}

    def _encode_image(self, path: str) -> str:
        try:
            import base64

            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")
        except Exception:
            return ""


def run_direct_scene_episode(
    scene: dict[str, Any],
    tool_client: ToolClient,
    image_root: str = "",
    output_dir: str = "outputs/scene_eval/direct",
) -> SceneRolloutEpisode:
    """No-agent baseline: register all images, run exhaustive COLMAP."""
    image_paths = scene["image_paths"]
    ep = SceneRolloutEpisode(
        scene_id=scene["scene_id"],
        image_paths=image_paths,
        image_root=image_root,
        num_images=len(image_paths),
        gt_recon=scene.get("gt_recon"),
        done=True,
    )

    for i, path in enumerate(image_paths):
        try:
            full = str(Path(image_root) / path) if image_root else str(path)
            tool_client.register_image(f"img_{i:04d}", full)
        except Exception as e:
            logger.warning(f"[{scene['scene_id']}] register {path}: {e}")

    if image_paths:
        parent = Path(image_root) / Path(image_paths[0]).parent if image_root else Path(image_paths[0]).parent
        image_dir = str(parent)
    else:
        image_dir = ""

    try:
        recon = tool_client.sfm_run(
            image_dir=image_dir,
            output_dir=str(Path(output_dir) / scene["scene_id"]),
        )
    except Exception as e:
        recon = {"error": str(e), "num_registered": 0, "num_points3d": 0}

    ep.recon_result = recon
    ep.tool_calls = [ToolCall(tool="sfm_run", args={}), ToolCall(tool="done", args={})]
    ep.results = [recon]
    ep.reward_components = compute_scene_reward(
        recon, gt_recon=scene.get("gt_recon"), num_tool_calls=2, num_valid_calls=1
    )
    ep.reward = ep.reward_components["total_reward"]
    return ep


def evaluate_scenes(
    scenes: list[dict[str, Any]],
    mode: str,
    tool_client: ToolClient,
    agent: Any = None,
    image_root: str = "",
    max_tool_calls: int = 20,
    max_turns: int = 30,
    rollouts_per_scene: int = 1,
    output_dir: str = "outputs/scene_eval",
) -> list[dict[str, Any]]:
    """Run one mode over all scenes; return per-scene metric dicts."""
    per_scene: list[dict[str, Any]] = []

    for scene in scenes:
        scene_id = scene["scene_id"]
        episodes: list[SceneRolloutEpisode] = []

        for r in range(rollouts_per_scene):
            try:
                if mode == "direct":
                    ep = run_direct_scene_episode(
                        scene, tool_client, image_root=image_root,
                        output_dir=f"{output_dir}/direct",
                    )
                elif mode == "oracle":
                    ep = run_scene_oracle_episode(
                        agent=agent,
                        scene_id=scene_id,
                        image_paths=scene["image_paths"],
                        tool_client=tool_client,
                        gt_recon=scene.get("gt_recon"),
                        overlap_matrix=scene.get("overlap_matrix"),
                        image_indices=scene.get("image_indices"),
                        image_root=image_root,
                    )
                else:  # agent
                    ep = run_scene_episode(
                        agent=agent,
                        scene_id=scene_id,
                        image_paths=scene["image_paths"],
                        tool_client=tool_client,
                        gt_recon=scene.get("gt_recon"),
                        overlap_matrix=scene.get("overlap_matrix"),
                        image_indices=scene.get("image_indices"),
                        max_tool_calls=max_tool_calls,
                        max_turns=max_turns,
                        image_root=image_root,
                    )
                episodes.append(ep)
            except Exception as e:
                logger.error(f"[{scene_id}] {mode} rollout {r} failed: {e}")

        if not episodes:
            logger.warning(f"[{scene_id}] no successful {mode} episodes")
            continue

        # For multi-rollout modes keep the best episode by reward.
        ep = max(episodes, key=lambda e: e.reward)
        metrics = compute_scene_metrics(ep, scene)
        metrics["scene/mode"] = mode
        metrics["scene/num_rollouts"] = len(episodes)
        per_scene.append(metrics)

        logger.info(
            f"[{scene_id}] {mode}: registered "
            f"{metrics['registration/num_registered']}/"
            f"{metrics['registration/num_images']}, "
            f"points3d={metrics['registration/num_points3d']}, "
            f"reward={ep.reward:.3f}, "
            f"calls={metrics['efficiency/num_tool_calls']}, "
            f"rel_auc10={metrics.get('pose/rel_auc_10')}"
        )

    return per_scene


def _print_summary_table(results: dict[str, dict[str, Any]]) -> None:
    keys = [
        ("success_rate", "{:.2%}"),
        ("mean_registration_registered_fraction", "{:.3f}"),
        ("mean_registration_num_points3d", "{:.0f}"),
        ("mean_registration_mean_reproj_error_px", "{:.3f}"),
        ("mean_pose_rel_auc_10", "{:.3f}"),
        ("mean_pose_rel_rot_err_mean_deg", "{:.2f}"),
        ("mean_doppelganger_f1", "{:.3f}"),
        ("mean_efficiency_num_tool_calls", "{:.1f}"),
        ("reward_mean", "{:.3f}"),
    ]
    header = f"{'mode':10s} " + " ".join(f"{k[:18]:>18s}" for k, _ in keys)
    logger.info(header)
    logger.info("-" * len(header))
    for mode, agg in results.items():
        row = f"{mode:10s} "
        for key, fmt in keys:
            v = agg.get(f"scene_eval_{key}")
            row += f"{fmt.format(v):>18s} " if v is not None else f"{'—':>18s} "
        logger.info(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/phase2_scene.yaml")
    parser.add_argument("--tool-server-url", default="http://localhost:8765")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--mode", nargs="+", choices=["agent", "oracle", "direct"], default=["agent"])
    parser.add_argument("--scene-ids", nargs="+", default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--max-images-per-scene", type=int, default=None)
    parser.add_argument("--max-tool-calls", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--rollouts-per-scene", type=int, default=1)
    parser.add_argument("--lora-checkpoint", default=None,
                        help="LoRA adapter path to load into vLLM for agent mode")
    parser.add_argument("--output-dir", default="outputs/scene_eval")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to WandB")
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)
    data_cfg = config.get("data", {})
    reward_cfg = config.get("reward", {})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build scene dataset (reuses the Phase 2 training loader)
    from scripts.run_scene_grpo import SceneDataset

    scene_dataset = SceneDataset.from_megadepth(
        scene_info_dir=data_cfg.get("scene_info_dir", "data/megadepth/scene_info"),
        image_root=data_cfg.get("image_root", ""),
        scene_ids=args.scene_ids,
        max_images_per_scene=args.max_images_per_scene
        or data_cfg.get("max_images_per_scene", 20),
        min_images_per_scene=data_cfg.get("min_images_per_scene", 5),
    )
    scenes = scene_dataset.scenes[: args.max_scenes] if args.max_scenes else scene_dataset.scenes
    logger.info(f"Evaluating {len(scenes)} scenes: {[s['scene_id'] for s in scenes]}")
    if not scenes:
        logger.error("No scenes found — check data.scene_info_dir / image_root in config")
        sys.exit(1)

    image_root = data_cfg.get("image_root", "")

    tool_client = ToolClient(base_url=args.tool_server_url)
    try:
        logger.info(f"Tool server health: {tool_client.health()}")
    except Exception as e:
        logger.error(f"Tool server not reachable at {args.tool_server_url}: {e}")
        sys.exit(1)

    wandb_run = None
    if args.wandb:
        try:
            import wandb

            out_cfg = config.get("output", {})
            wandb_run = wandb.init(
                project=out_cfg.get("wandb_project", "agentic-sfm"),
                entity=out_cfg.get("wandb_entity"),
                name=out_cfg.get("wandb_run_name", "scene-eval") + "-eval",
                config={"modes": args.mode, "num_scenes": len(scenes)},
            )
        except Exception as e:
            logger.warning(f"WandB init failed ({e}); continuing without logging")

    results: dict[str, dict[str, Any]] = {}
    for mode in args.mode:
        agent = None
        if mode == "agent":
            from scripts.run_grpo import VLLMRolloutAgent

            agent = VLLMRolloutAgent(
                vllm_url=args.vllm_url,
                model_name=config["model"]["name"],
                max_new_tokens=config["model"].get("max_new_tokens", 512),
                max_tool_calls=args.max_tool_calls,
                temperature=0.1,  # low temperature for eval
                top_p=0.95,
                matcher=data_cfg.get("matcher", DEFAULT_MATCHER),
                accumulative_tool_coef=reward_cfg.get("accumulative_tool_coef", 0.1),
                ntep_intent_coef=reward_cfg.get("ntep_intent_coef", 0.05),
                ntep_redundancy_penalty=reward_cfg.get("ntep_redundancy_penalty", 0.05),
                use_ntep_rewards=reward_cfg.get("use_ntep_rewards", False),
            )
            # Scene-specific reward weights (registration/split/doppelganger)
            # are read via getattr(agent, "reward_config") in scene_episode.
            agent.reward_config = reward_cfg

            # Optionally load a LoRA checkpoint into the vLLM server.
            ckpt = args.lora_checkpoint or config["model"].get("checkpoint")
            if ckpt:
                try:
                    import httpx

                    httpx.post(
                        f"{args.vllm_url.rstrip('/')}/v1/load_lora_adapter",
                        json={"lora_name": "eval", "lora_path": str(Path(ckpt).resolve())},
                        headers={"Authorization": "Bearer EMPTY"},
                        timeout=60,
                    )
                    agent._lora_loaded = True
                    agent.vllm_model = "eval"
                    logger.info(f"Loaded LoRA adapter into vLLM: {ckpt}")
                except Exception as e:
                    logger.warning(f"Could not load LoRA adapter {ckpt}: {e}")
        elif mode == "oracle":
            agent = _ShimAgent(matcher=data_cfg.get("matcher", DEFAULT_MATCHER),
                             reward_config=reward_cfg)

        per_scene = evaluate_scenes(
            scenes,
            mode=mode,
            tool_client=tool_client,
            agent=agent,
            image_root=image_root,
            max_tool_calls=args.max_tool_calls,
            max_turns=args.max_turns,
            rollouts_per_scene=args.rollouts_per_scene if mode == "agent" else 1,
            output_dir=str(output_dir),
        )

        aggregate = aggregate_scene_metrics(per_scene)
        results[mode] = {"aggregate": aggregate, "per_scene": per_scene}

        if wandb_run is not None:
            wandb_run.log(
                {
                    f"{mode}/{k}": v
                    for k, v in aggregate.items()
                    if isinstance(v, (int, float))
                }
            )

    # Save + print
    results_path = output_dir / "scene_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    _print_summary_table({m: r["aggregate"] for m, r in results.items()})

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
