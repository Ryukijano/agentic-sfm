#!/usr/bin/env python
"""Phase 1c: SFT (Supervised Fine-Tuning) warmup for agentic SfM.

LoRA fine-tuning of Qwen3-VL-8B on curated successful trajectories.
Trains with cross-entropy loss on assistant tokens only.

Usage:
  python scripts/run_sft.py --config configs/phase1_sft.yaml
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_sft_data(path: str) -> list[dict]:
    """Load SFT training data from JSONL."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


class SFTTrainer:
    """Supervised fine-tuning with LoRA on Qwen3-VL."""

    def __init__(self, config: dict, output_dir: str = "outputs/sft"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = config["model"]["name"]
        self.lora_config = config["model"]["lora"]
        self.lr = config["training"]["lr"]
        self.total_epochs = config["training"]["total_epochs"]
        self.batch_size = config["training"]["batch_size"]
        self.grad_accum = config["training"]["gradient_accumulation_steps"]
        self.max_grad_norm = config["training"]["max_grad_norm"]
        self.save_freq = config["training"].get("save_freq", 1)
        self.warmup_steps = config["training"].get("warmup_steps", 50)
        self.device = "cuda:0"

        self._model = None
        self._processor = None
        self._optimizer = None
        self._wandb = None

        wandb_cfg = config.get("output", {})
        if wandb_cfg.get("wandb_enabled", False):
            try:
                import wandb
                self._wandb = wandb.init(
                    project=wandb_cfg.get("wandb_project", "agentic-sfm"),
                    entity=wandb_cfg.get("wandb_entity"),
                    name=wandb_cfg.get("wandb_run_name", "phase1-sft"),
                    config=config,
                )
                logger.info("W&B logging enabled.")
            except Exception as e:
                logger.warning(f"W&B init failed: {e}")
                self._wandb = None

    def _load_model(self):
        """Load model + processor + LoRA."""
        if self._model is not None:
            return

        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info(f"Loading {self.model_name} on {self.device}...")
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )

        lora_cfg = LoraConfig(
            r=self.lora_config["rank"],
            lora_alpha=self.lora_config["alpha"],
            lora_dropout=self.lora_config["dropout"],
            target_modules=self.lora_config["target_modules"],
            task_type="CAUSAL_LM",
        )
        self._model = get_peft_model(self._model, lora_cfg)
        self._model.print_trainable_parameters()

        trainable = [p for p in self._model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(
            trainable, lr=self.lr,
            weight_decay=self.config["training"].get("weight_decay", 0.01),
        )
        logger.info("SFT model ready with LoRA.")

    def _compute_assistant_mask(self, messages: list[dict], images: list,
                                  processor) -> list[bool]:
        """Build boolean mask over tokenized conversation: True for assistant tokens."""
        full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        full_ids = processor(text=[full_text], images=images, return_tensors="pt")["input_ids"][0]
        n_tokens = len(full_ids)
        mask = [False] * n_tokens

        for k, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            try:
                prefix_text = processor.apply_chat_template(
                    messages[:k], tokenize=False, add_generation_prompt=True
                )
                prefix_ids = processor(text=[prefix_text], images=images, return_tensors="pt")["input_ids"][0]
                start = len(prefix_ids)

                through_text = processor.apply_chat_template(
                    messages[:k+1], tokenize=False, add_generation_prompt=False
                )
                through_ids = processor(text=[through_text], images=images, return_tensors="pt")["input_ids"][0]
                end = len(through_ids)

                for i in range(start, min(end, n_tokens)):
                    mask[i] = True
            except Exception as e:
                logger.warning(f"Mask computation failed for msg {k}: {e}")

        return mask

    def train_step(self, examples: list[dict], accum_step: int = 0,
                   is_last_accum: bool = True) -> dict:
        """One SFT step on a batch of examples."""
        self._load_model()
        self._model.train()

        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        valid_count = 0

        for ex in examples:
            messages = ex["messages"]
            try:
                # Load images
                images = [
                    Image.open(ex["image_a"]).convert("RGB"),
                    Image.open(ex["image_b"]).convert("RGB"),
                ]

                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                inputs = self._processor(
                    text=[text], images=images, return_tensors="pt", padding=True
                ).to(self.device)

                mask = self._compute_assistant_mask(messages, images, self._processor)
                mask_tensor = torch.tensor(mask, device=self.device, dtype=torch.bool)

                outputs = self._model(**inputs)
                logits = outputs.logits[:, :-1, :]
                target_ids = inputs["input_ids"][:, 1:]
                token_logps = F.log_softmax(logits, dim=-1)
                per_token_logps = token_logps.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

                # Shift mask: logps[i] predicts token i+1
                assistant_mask = mask_tensor[1:].to(per_token_logps.device)

                if assistant_mask.sum() == 0:
                    continue

                # Cross-entropy loss on assistant tokens only
                masked_logps = per_token_logps[0][assistant_mask]
                masked_targets = target_ids[0][assistant_mask]
                loss = F.nll_loss(
                    masked_logps.unsqueeze(0),
                    masked_targets.unsqueeze(0),
                )

                loss = loss / self.grad_accum
                loss.backward()

                total_loss = total_loss + loss.detach()
                valid_count += 1
            except Exception as e:
                logger.warning(f"SFT step failed for {ex.get('pair_id', '?')}: {e}")
                continue

        if valid_count > 0 and is_last_accum:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self._model.parameters() if p.requires_grad],
                self.max_grad_norm,
            )
            self._optimizer.step()
            self._optimizer.zero_grad()

        if is_last_accum:
            self._model.eval()

        return {
            "loss": (total_loss / max(valid_count, 1)).item(),
            "valid_count": valid_count,
        }

    def save_lora_checkpoint(self, epoch: int):
        ckpt_path = self.ckpt_dir / f"epoch_{epoch}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            self._model.save_pretrained(str(ckpt_path))
            logger.info(f"  Saved LoRA checkpoint: {ckpt_path}")
        else:
            logger.warning("  No model loaded — skipping checkpoint save.")

    def train(self, sft_data_path: str):
        """Main SFT training loop."""
        logger.info("Starting SFT training...")

        examples = load_sft_data(sft_data_path)
        logger.info(f"Loaded {len(examples)} SFT examples from {sft_data_path}")

        if not examples:
            logger.error("No SFT examples found — aborting.")
            return

        global_step = 0

        for epoch in range(self.total_epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.total_epochs} | Examples: {len(examples)}")

            import numpy as np
            rng = np.random.default_rng(epoch)
            indices = list(range(len(examples)))
            rng.shuffle(indices)

            epoch_losses = []
            micro_step = 0

            for batch_start in range(0, len(examples), self.batch_size):
                batch_indices = indices[batch_start:batch_start + self.batch_size]
                batch = [examples[i] for i in batch_indices]
                micro_step += 1
                is_last_accum = (micro_step % self.grad_accum == 0) or \
                                (batch_start + self.batch_size >= len(examples))

                stats = self.train_step(batch, accum_step=micro_step, is_last_accum=is_last_accum)
                epoch_losses.append(stats["loss"])

                if is_last_accum:
                    global_step += 1
                    if global_step % 10 == 0:
                        mean_loss = sum(epoch_losses[-10:]) / max(len(epoch_losses[-10:]), 1)
                        logger.info(f"  Step {global_step}: loss={mean_loss:.4f}")
                        if self._wandb:
                            self._wandb.log({
                                "sft/loss": mean_loss,
                                "sft/epoch": epoch + 1,
                                "sft/global_step": global_step,
                            })

            mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            logger.info(f"Epoch {epoch+1} mean loss: {mean_loss:.4f}")

            if (epoch + 1) % self.save_freq == 0:
                self.save_lora_checkpoint(epoch + 1)

        self.save_lora_checkpoint(self.total_epochs)
        if self._wandb:
            self._wandb.finish()
        logger.info("SFT training complete.")


def main():
    parser = argparse.ArgumentParser(description="Phase 1c: SFT warmup training")
    parser.add_argument("--config", type=str, default="configs/phase1_sft.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/sft")
    parser.add_argument("--sft-data", type=str, default=None,
                        help="Path to SFT JSONL (overrides config)")
    args = parser.parse_args()

    config = load_config(args.config)
    os.chdir(Path(args.config).parent.parent)

    sft_data_path = args.sft_data or config["data"].get("sft_data", "data/sft_train.jsonl")

    trainer = SFTTrainer(
        config=config,
        output_dir=args.output_dir,
    )
    trainer.train(sft_data_path)


if __name__ == "__main__":
    main()
