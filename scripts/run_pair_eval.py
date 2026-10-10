#!/usr/bin/env python
"""Honest pair-level eval of a LoRA policy on MegaDepth val pairs.

Loads the adapter into a running vLLM server (must succeed — base Qwen is
not an eval of this run) and scores episodes with pose-aligned metrics:
geometric_reward = pose_reward + inlier_reward, success = pose_reward > 0.

Does not instantiate GRPOTrainer (that would load the trainable PEFT on a
third GPU).

Usage:
  python scripts/run_pair_eval.py \\
      --config configs/phase1_grpo_hard.yaml \\
      --lora-checkpoint outputs/phase1_hard/checkpoints/best \\
      --vllm-url http://localhost:8000 \\
      --tool-server-url http://localhost:8765 \\
      --output outputs/eval_pair/phase1_hard_best.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_sfm.data.hard_pairs import HardPairDataset
from agentic_sfm.eval.evaluate import pair_episode_row, summarize_pair_eval
from agentic_sfm.tools.client import ToolClient
from scripts.run_grpo import VLLMRolloutAgent, load_vllm_lora

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LOFTR_HARD = {"geometric_reward": 0.9654661801388265, "success_rate": 0.88, "n": 50}
ORACLE_HARD = {"geometric_reward": 1.111316329060898, "success_rate": 0.90, "n": 50}


def _slice_from_baseline(baseline_json: Path | None, method: str, difficulty: str) -> dict:
    """Read a published baseline slice; fall back to the known hard numbers."""
    fallback = {
        "direct_loftr": LOFTR_HARD,
        "oracle_crop_loftr": ORACLE_HARD,
    }.get(method, {})
    if baseline_json is None or not baseline_json.is_file():
        return dict(fallback)
    try:
        blob = json.loads(baseline_json.read_text())
        summary = blob["baselines"][method]["summary"]
        if difficulty == "all":
            return {
                "geometric_reward": float(summary["mean_total_reward"]),
                "success_rate": float(summary["success_rate"]),
                "n": int(summary["num_pairs"]),
            }
        d = summary["by_difficulty"][difficulty]
        return {
            "geometric_reward": float(d["mean_total_reward"]),
            "success_rate": float(d["success_rate"]),
            "n": int(d["count"]),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        logger.warning(f"Could not read {method}/{difficulty} from {baseline_json}: {e}")
        return dict(fallback)


def _build_agent(config: dict, vllm_url: str) -> VLLMRolloutAgent:
    reward = config.get("reward", {})
    rl = config.get("rl", {})
    return VLLMRolloutAgent(
        vllm_url=vllm_url,
        model_name=config["model"]["name"],
        max_new_tokens=config["model"].get("max_new_tokens", 512),
        max_tool_calls=rl.get("max_tool_calls", 10),
        temperature=0.1,
        top_p=rl.get("top_p", 0.95),
        min_p=config.get("rollout", {}).get("min_p", 0.05),
        top_k=config.get("rollout", {}).get("top_k", 50),
        pose_weight=reward.get("pose_weight", 1.0),
        inlier_weight=reward.get("inlier_weight", 0.1),
        tool_cost=reward.get("tool_cost", 0.0),
        format_weight=reward.get("format_weight", 0.1),
        invalid_penalty=reward.get("invalid_penalty", 0.2),
        reward_schedule="static",
        reward_warmup_steps=0,
        matcher=config.get("data", {}).get("matcher", "loftr"),
        accumulative_tool_coef=reward.get("accumulative_tool_coef", 0.1),
        use_accumulative_tool_reward=reward.get("use_accumulative_tool_reward", True),
        ntep_intent_coef=reward.get("ntep_intent_coef", 0.05),
        ntep_redundancy_penalty=reward.get("ntep_redundancy_penalty", 0.05),
        use_ntep_rewards=reward.get("use_ntep_rewards", True),
        first_call_validity_weight=reward.get("first_call_validity_weight", 0.5),
        first_call_invalid_penalty=reward.get("first_call_invalid_penalty", 1.0),
        gate_geometry_on_validity=reward.get("gate_geometry_on_validity", True),
    )


def evaluate_pairs(pairs, rollout_agent, tool_client, pose_weight: float) -> list[dict]:
    rows = []
    for pair in tqdm(pairs, desc="pair eval"):
        gt_pose = (
            {"R": pair.gt_R.tolist(), "t": pair.gt_t.tolist()}
            if pair.gt_R is not None
            else None
        )
        ep = rollout_agent.run_episode(
            pair_id=pair.pair_id,
            image_a_path=pair.image_a,
            image_b_path=pair.image_b,
            tool_client=tool_client,
            gt_pose=gt_pose,
            K_a=pair.K_a,
            K_b=pair.K_b,
        )
        rows.append(pair_episode_row(pair, ep, pose_weight=pose_weight))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Pose-aligned LoRA pair eval")
    parser.add_argument("--config", type=str, default="configs/phase1_grpo_hard.yaml")
    parser.add_argument(
        "--lora-checkpoint",
        type=str,
        default="outputs/phase1_hard/checkpoints/best",
    )
    parser.add_argument("--pairs", type=str, default="data/hard_pairs_val.json")
    parser.add_argument("--tool-server-url", type=str, default="http://localhost:8765")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/eval_pair/phase1_hard_best.json",
    )
    parser.add_argument(
        "--baseline-json",
        type=str,
        default="outputs/baselines/baseline_results_latest.json",
    )
    parser.add_argument(
        "--difficulties",
        type=str,
        default="hard",
        help="Comma-separated difficulties to headline (still writes full table)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    project_root = Path(args.config).resolve().parent.parent
    os.chdir(project_root)

    lora_path = Path(args.lora_checkpoint)
    if not (lora_path / "adapter_model.safetensors").is_file():
        raise SystemExit(f"LoRA adapter missing: {lora_path}")

    dataset = HardPairDataset.load(args.pairs)
    if len(dataset.pairs) == 0:
        raise SystemExit(f"No pairs in {args.pairs}")
    logger.info(f"Loaded {len(dataset.pairs)} pairs from {args.pairs}")

    tool_client = ToolClient(args.tool_server_url)
    try:
        health = tool_client.health()
        logger.info(f"Tool server: {health}")
    except Exception as e:
        raise SystemExit(f"Cannot connect to tool server: {e}")

    rollout_agent = _build_agent(config, args.vllm_url)
    try:
        adapter = load_vllm_lora(
            args.vllm_url,
            lora_path,
            rollout_agent,
            adapter_name="phase1-hard-best",
        )
    except Exception as e:
        raise SystemExit(f"Failed to load LoRA into vLLM (refusing base-Qwen eval): {e}")
    if not getattr(rollout_agent, "_lora_loaded", False):
        raise SystemExit("vLLM LoRA flag not set after load — refusing base-Qwen eval")
    logger.info(f"Evaluating adapter {adapter} from {lora_path}")

    pose_w = float(config.get("reward", {}).get("pose_weight", 1.0)) or 1.0
    rows = evaluate_pairs(dataset.pairs, rollout_agent, tool_client, pose_w)
    full_stats = summarize_pair_eval(rows, method="phase1_hard_best")

    headline = [d.strip() for d in args.difficulties.split(",") if d.strip()]
    headline_pairs = [p for p in dataset.pairs if p.difficulty in headline] if headline else list(dataset.pairs)
    headline_ids = {p.pair_id for p in headline_pairs}
    headline_rows = [r for r in rows if r["pair_id"] in headline_ids]
    headline_stats = summarize_pair_eval(headline_rows, method=f"phase1_hard_best_{'+'.join(headline) or 'all'}")

    baseline_path = Path(args.baseline_json) if args.baseline_json else None
    loftr_hard = _slice_from_baseline(baseline_path, "direct_loftr", "hard")
    oracle_hard = _slice_from_baseline(baseline_path, "oracle_crop_loftr", "hard")
    hard_geo = float(headline_stats.get("mean_geometric_reward") or 0.0)
    hard_succ = float(headline_stats.get("success_rate") or 0.0)

    comparison = {
        "headline_difficulties": headline,
        "agent": {
            "geometric_reward": hard_geo,
            "success_rate": hard_succ,
            "n": int(headline_stats.get("total") or 0),
            "mean_reward": float(headline_stats.get("mean_reward") or 0.0),
            "mean_pose_auc": float(headline_stats.get("mean_pose_auc") or 0.0),
            "mean_tool_calls": float(headline_stats.get("mean_tool_calls") or 0.0),
        },
        "loftr_hard": loftr_hard,
        "oracle_hard": oracle_hard,
        "delta_vs_loftr_hard": {
            "geometric_reward": hard_geo - float(loftr_hard.get("geometric_reward") or 0.0),
            "success_rate": hard_succ - float(loftr_hard.get("success_rate") or 0.0),
        },
        "delta_vs_oracle_hard": {
            "geometric_reward": hard_geo - float(oracle_hard.get("geometric_reward") or 0.0),
            "success_rate": hard_succ - float(oracle_hard.get("success_rate") or 0.0),
        },
    }

    payload = {
        "checkpoint": str(lora_path.resolve()),
        "pairs": args.pairs,
        "vllm_adapter": adapter,
        "temperature": 0.1,
        "headline": headline_stats,
        "full_val": full_stats,
        "comparison": comparison,
        "per_pair": rows,
    }
    # Drop raw_results duplication inside the two summaries (kept in per_pair).
    for key in ("headline", "full_val"):
        payload[key] = {k: v for k, v in payload[key].items() if k != "raw_results"}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"Wrote {out}")

    logger.info("=== Hard-pair headline ===")
    logger.info(
        f"  agent  geo={hard_geo:.4f}  success={hard_succ:.1%}  "
        f"n={comparison['agent']['n']}"
    )
    logger.info(
        f"  LoFTR  geo={loftr_hard.get('geometric_reward', 0):.4f}  "
        f"success={float(loftr_hard.get('success_rate') or 0):.1%}"
    )
    logger.info(
        f"  oracle geo={oracle_hard.get('geometric_reward', 0):.4f}  "
        f"success={float(oracle_hard.get('success_rate') or 0):.1%}"
    )
    logger.info(
        f"  Δ LoFTR geo={comparison['delta_vs_loftr_hard']['geometric_reward']:+.4f}  "
        f"succ={comparison['delta_vs_loftr_hard']['success_rate']:+.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
