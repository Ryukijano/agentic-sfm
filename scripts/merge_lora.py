#!/usr/bin/env python
"""Merge LoRA adapter weights into base model for vLLM serving.

After SFT (or GRPO), the LoRA adapter needs to be merged into the base
model weights so vLLM can serve the fine-tuned model without PEFT.

Usage:
  python scripts/merge_lora.py \
      --base-model "Qwen/Qwen3.5-2B" \
      --lora-path outputs/sft/checkpoints/epoch_5 \
      --output-dir outputs/sft/merged_model
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from peft import PeftModel

from agentic_sfm.agent.policy import load_policy_processor_and_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def merge_lora(base_model: str, lora_path: str, output_dir: str):
    """Load base model, attach LoRA, merge weights, save."""
    logger.info(f"Loading base model: {base_model}")
    processor, model = load_policy_processor_and_model(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    logger.info(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    logger.info(f"Saving merged model to: {output_dir}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    processor.save_pretrained(str(output_path))

    logger.info("Merge complete. Model ready for vLLM serving.")
    logger.info(f"Use with: --model {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--lora-path", type=str, required=True,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for merged model")
    args = parser.parse_args()

    merge_lora(args.base_model, args.lora_path, args.output_dir)


if __name__ == "__main__":
    main()
