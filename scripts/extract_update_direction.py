#!/usr/bin/env python
"""Extract an OPSFT update-direction artifact between two LoRA checkpoints.

Compares a base adapter checkpoint (before on-policy GRPO) with a trained
one (after) and saves the exact support mask + int8 update sign used by
OPSFT constrained SFT (arXiv 2609.36659).

Usage:
  python scripts/extract_update_direction.py \
      --base checkpoints/lora_base \
      --trained checkpoints/lora_grpo \
      --output checkpoints/update_direction.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.opsft import extract_direction, save_direction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def resolve_path(path: str) -> Path:
    """Resolve a checkpoint path: dir -> adapter_model.safetensors inside."""
    p = Path(path)
    if p.is_dir():
        p = p / "adapter_model.safetensors"
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


def load_state_dict(path: str) -> dict:
    """Load a state dict from .safetensors or .pt/.bin."""
    p = resolve_path(path)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file

        sd = load_file(str(p))
    else:
        sd = torch.load(str(p), map_location="cpu", weights_only=True)
        # Unwrap common containers (e.g. {"state_dict": ...}).
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
    return {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="Base adapter dir or state-dict file")
    ap.add_argument("--trained", required=True, help="Trained adapter dir or state-dict file")
    ap.add_argument("--output", required=True, help="Output .pt artifact path")
    ap.add_argument(
        "--selection", choices=["exact_support", "topk"], default="exact_support",
        help="Support-mask selection: exact_support = all changed elements "
             "(paper default; near-dense under AdamW LoRA); topk = global "
             "top-|delta| fraction, approximating the paper's sparse regime.",
    )
    ap.add_argument(
        "--topk-frac", type=float, default=0.1,
        help="Fraction of elements kept when --selection topk (default 0.1).",
    )
    args = ap.parse_args()

    base_sd = load_state_dict(args.base)
    trained_sd = load_state_dict(args.trained)
    logger.info("Loaded %d base / %d trained tensors", len(base_sd), len(trained_sd))

    artifact = extract_direction(
        base_sd, trained_sd,
        selection=args.selection, topk_frac=args.topk_frac,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_direction(artifact, args.output)

    meta = artifact["meta"]
    print(f"n_tensors:        {meta['n_tensors']}")
    print(f"total elements:   {meta['n_elems']:,}")
    print(f"changed elements: {meta['n_changed_elems']:,} ({meta['frac_changed']:.4%})")
    print(f"selection:        {meta['selection']} "
          f"-> {meta['n_selected_elems']:,} in support ({meta['frac_selected']:.4%})")

    fracs = sorted(
        ((int(m.sum()) / m.numel(), name) for name, m in artifact["mask"].items()),
        reverse=True,
    )[:10]
    print("\nTop-10 most-changed tensors:")
    for frac, name in fracs:
        print(f"  {frac:8.4%}  {name}")


if __name__ == "__main__":
    main()
