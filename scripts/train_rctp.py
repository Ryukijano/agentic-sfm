#!/usr/bin/env python
"""Phase 3c-i: Train Reward-Conditioned Trajectory Policy (RCTP).

SFT the model on mixed-quality trajectories with reward conditioning tokens
(<|high_reward|> or <|low_reward|>) injected into the prompt. This teaches
the model to produce different quality outputs conditioned on the reward token.

Usage:
  python scripts/train_rctp.py \
      --trajectories outputs/phase0/trajectories.jsonl \
      --config configs/phase1_sft.yaml \
      --output-dir outputs/rctp \
      --reward-threshold 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.rl.rc_grpo import prepare_rctp_training_data, HIGH_REWARD_TOKEN, LOW_REWARD_TOKEN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_trajectories(path: str) -> list[dict]:
    """Load trajectories from JSONL file."""
    trajectories = []
    with open(path) as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    return trajectories


def main():
    parser = argparse.ArgumentParser(description="Train RCTP for RC-GRPO")
    parser.add_argument("--trajectories", type=str, required=True,
                        help="Path to trajectories.jsonl")
    parser.add_argument("--config", type=str, default="configs/phase1_sft.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/rctp")
    parser.add_argument("--reward-threshold", type=float, default=0.3,
                        help="Reward above = high_reward, below = low_reward")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.chdir(Path(args.config).parent.parent)

    # Load and prepare RCTP training data
    trajectories = load_trajectories(args.trajectories)
    logger.info(f"Loaded {len(trajectories)} trajectories")

    rctp_examples = prepare_rctp_training_data(trajectories, args.reward_threshold)
    n_high = sum(1 for ex in rctp_examples if ex["reward_level"] == "high")
    n_low = len(rctp_examples) - n_high
    logger.info(f"RCTP examples: {len(rctp_examples)} (high: {n_high}, low: {n_low})")

    if not rctp_examples:
        logger.error("No RCTP examples — aborting.")
        return

    # Save RCTP data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rctp_data_path = output_dir / "rctp_train.jsonl"
    with open(rctp_data_path, "w") as f:
        for ex in rctp_examples:
            f.write(json.dumps(ex) + "\n")
    logger.info(f"Saved RCTP data to {rctp_data_path}")

    # Train using the same SFT trainer but with RCTP data
    from scripts.run_sft import SFTTrainer

    # Update config for RCTP
    config["data"]["sft_data"] = str(rctp_data_path)
    config["output"]["wandb_run_name"] = "rctp-training"

    trainer = SFTTrainer(
        config=config,
        output_dir=str(output_dir),
    )
    trainer.train(str(rctp_data_path))

    logger.info("RCTP training complete. Model ready for RC-GRPO.")


if __name__ == "__main__":
    main()
