#!/usr/bin/env python
"""Phase 2b: OPSFT-style constrained SFT for agentic SfM (arXiv 2609.36659).

Like run_sft.py, but each optimizer step is constrained to a precomputed
update direction (mask + sign) extracted from a previous SFT->GRPO run:
gradients are masked before the step and parameters are projected back
onto the allowed direction after the step.

Usage:
  python scripts/run_sft_opsft.py --config configs/phase2_sft_opsft.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from run_sft import SFTTrainer, load_config  # noqa: E402

from agentic_sfm.opsft import (  # noqa: E402
    canonical_lora_name,
    fingerprint_state_dict,
    load_direction,
    mask_gradients_,
    project_parameters_,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class OPSFTTrainer(SFTTrainer):
    """SFT trainer constrained to a precomputed update direction."""

    def _load_model(self):
        if self._model is not None:
            return

        from peft import LoraConfig, PeftModel, get_peft_model

        from agentic_sfm.constants import DEFAULT_LORA_TARGET_MODULES

        logger.info(f"Loading {self.model_name} on {self.device}...")
        from agentic_sfm.agent.policy import load_policy_processor_and_model
        from run_sft import _set_max_image_pixels

        self._processor, self._model = load_policy_processor_and_model(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        max_image_pixels = self.config.get("data", {}).get("max_image_pixels", 1003520)
        _set_max_image_pixels(self._processor, max_image_pixels)

        init_adapter = self.config["model"].get("init_adapter")
        if init_adapter and Path(init_adapter).is_dir():
            logger.info(f"Initializing LoRA from adapter: {init_adapter}")
            self._model = PeftModel.from_pretrained(
                self._model, init_adapter, is_trainable=True
            )
        else:
            logger.warning(
                "OPSFT: no valid model.init_adapter configured (%r) — falling "
                "back to a FRESH random-LoRA get_peft_model. Direction "
                "semantics assume the constrained run starts from the same "
                "weights the direction was measured at (theta_base).",
                init_adapter,
            )
            lora_cfg = LoraConfig(
                r=self.lora_config["rank"],
                lora_alpha=self.lora_config["alpha"],
                lora_dropout=self.lora_config["dropout"],
                target_modules=self.lora_config.get("target_modules")
                or DEFAULT_LORA_TARGET_MODULES,
                task_type="CAUSAL_LM",
            )
            self._model = get_peft_model(self._model, lora_cfg)
        self._model.print_trainable_parameters()

        self._model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self._model.enable_input_require_grads()
        if getattr(self._model.config, "use_cache", None) is not None:
            self._model.config.use_cache = False

        trainable = [p for p in self._model.parameters() if p.requires_grad]
        self._optimizer = torch.optim.AdamW(
            trainable, lr=self.lr,
            weight_decay=self.config["training"].get("weight_decay", 0.01),
        )

        # --- OPSFT direction artifact + theta0 snapshot -------------------
        direction_path = self.config["opsft"]["direction_artifact"]
        direction = load_direction(direction_path)
        self._dir_mask = direction["mask"]
        self._dir_sign = direction["sign"]
        meta = direction.get("meta") or {}
        logger.info(
            f"Loaded OPSFT direction artifact {direction_path}: "
            f"{len(self._dir_mask)} mask entries, meta={meta}"
        )

        try:
            self._adapter_name = next(iter(self._model.peft_config.keys()))
        except Exception:
            self._adapter_name = "default"

        self._theta0 = {}
        missing = []
        for name, p in self._model.named_parameters():
            if not p.requires_grad:
                continue
            cname = canonical_lora_name(name, adapter_name=self._adapter_name)
            if cname not in self._dir_mask or cname not in self._dir_sign:
                missing.append(cname)
            self._theta0[cname] = p.detach().clone().cpu()

        if missing:
            raise RuntimeError(
                "OPSFT direction artifact does not cover all trainable "
                f"params ({len(missing)} missing): {sorted(missing)}"
            )

        base_fp = meta.get("base_fingerprint")
        if base_fp:
            theta0_fp = fingerprint_state_dict(self._theta0)
            if theta0_fp != base_fp:
                logger.warning(
                    "OPSFT init weights do NOT match the artifact's "
                    "extraction base (fingerprint mismatch) — displacement "
                    "semantics assume init == theta_base. Check "
                    "model.init_adapter vs the --base used by "
                    "extract_update_direction.py."
                )
            else:
                logger.info("OPSFT init weights verified against extraction base.")
        logger.info(
            f"OPSFT ready: adapter={self._adapter_name}, "
            f"{len(self._theta0)} constrained params, theta0 snapshot taken."
        )
        self._mask_step = 0
        logger.info("OPSFT model ready with LoRA.")

    def _pre_optim_step(self):
        stats = mask_gradients_(
            self._model.named_parameters(),
            self._dir_mask,
            self._dir_sign,
            adapter_name=self._adapter_name,
        )
        self._mask_step += 1
        if self._mask_step == 1 or self._mask_step % 10 == 0:
            logger.info(f"OPSFT mask_gradients_ step {self._mask_step}: {stats}")

    def _post_optim_step(self):
        with torch.no_grad():
            project_parameters_(
                self._model.named_parameters(),
                self._theta0,
                self._dir_mask,
                self._dir_sign,
                adapter_name=self._adapter_name,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2b: OPSFT constrained SFT training"
    )
    parser.add_argument("--config", type=str, default="configs/phase2_sft_opsft.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/opsft")
    parser.add_argument("--sft-data", type=str, default=None,
                        help="Path to SFT JSONL (overrides config)")
    parser.add_argument("--init-adapter", type=str, default=None,
                        help="Path to peft adapter dir to initialize LoRA from "
                             "(overrides config model.init_adapter)")
    parser.add_argument("--direction-artifact", type=str, default=None,
                        help="Path to OPSFT direction artifact (.pt) "
                             "(overrides config opsft.direction_artifact)")
    args = parser.parse_args()

    config = load_config(args.config)
    os.chdir(Path(args.config).parent.parent)

    from agentic_sfm.constants import assert_qwen35_runtime

    assert_qwen35_runtime()

    if args.init_adapter:
        config.setdefault("model", {})["init_adapter"] = args.init_adapter
    if args.direction_artifact:
        config.setdefault("opsft", {})["direction_artifact"] = args.direction_artifact

    sft_data_path = args.sft_data or config["data"].get("sft_data", "data/sft_train.jsonl")

    trainer = OPSFTTrainer(
        config=config,
        output_dir=args.output_dir,
    )
    trainer.train(sft_data_path)


if __name__ == "__main__":
    main()
