"""OPSFT-style constrained SFT: update-direction extraction and enforcement.

Implements the LoRA adaptation of "On-Policy Parameter Update Direction
Underlies Generalization in LLM Post-Training" (arXiv 2609.36659).

A *direction artifact* is extracted between two LoRA adapter checkpoints
(before/after on-policy GRPO): for every shared tensor it stores the exact
support mask ``mask[name] = (base != trained)`` and the int8 sign of the
update ``sign[name] = sign(trained - base)``. During constrained SFT the
artifact enforces, each optimizer step:

* Stage 1 (``mask_gradients_``, after backward, before grad clipping):
  zero gradients outside the support, and zero gradients whose induced
  movement ``-grad`` opposes the reference direction ``D`` (``grad*D > 0``).
* Stage 2 (``project_parameters_``, after ``optimizer.step()``, under
  ``torch.no_grad()``): restore parameters outside the support to their
  init values ``theta0``, and snap back entries that moved against ``D``.

All artifact keys are canonical adapter-safetensors names (no adapter-name
segment); ``canonical_lora_name`` maps peft ``named_parameters()`` names to
them at train time.
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch

logger = logging.getLogger(__name__)

# peft named_parameters() inserts the adapter name (e.g. ".default.") between
# the lora_* attribute and ".weight"; safetensors keys omit it.
_LORA_SEGMENTS = (
    "lora_A",
    "lora_B",
    "lora_embedding_A",
    "lora_embedding_B",
    "lora_magnitude_vector",
)


def canonical_lora_name(name: str, adapter_name: str = "default") -> str:
    """Map a peft ``named_parameters()`` name to its adapter-safetensors key.

    ``x.lora_A.default.weight`` -> ``x.lora_A.weight``. Handles ``lora_A``,
    ``lora_B``, ``lora_embedding_A`` and ``lora_embedding_B``.

    DoRA is special: peft's ``renamed_dora_weights`` keeps the adapter
    segment and strips only the trailing ``.weight``, so
    ``x.lora_magnitude_vector.default.weight`` -> ``x.lora_magnitude_vector.default``.

    Names without a LoRA segment are returned unchanged.
    """
    if "lora_magnitude_vector" in name and name.endswith(".weight"):
        return name[: -len(".weight")]
    needle = f".{adapter_name}."
    if needle not in name:
        return name
    prefix, _, suffix = name.partition(needle)
    if any(seg in prefix for seg in _LORA_SEGMENTS):
        return prefix + "." + suffix
    return name


def extract_direction(
    base_sd: dict,
    trained_sd: dict,
    selection: str = "exact_support",
    topk_frac: float = 0.1,
) -> dict:
    """Extract the support mask and update sign between two state dicts.

    Both args map canonical names -> ``torch.Tensor`` (any dtype/device;
    moved to CPU internally). Returns::

        {"mask": {name: bool CPU tensor},
         "sign": {name: int8 CPU tensor},
         "meta": {...}}

    ``selection`` controls the support mask ``M``:

    * ``"exact_support"`` — ``M = (base != trained)`` elementwise, the
      paper's released default. Note: under AdamW-trained LoRA this is
      typically ~100% of elements, so only the sign constraint binds.
    * ``"topk"`` — ``M`` keeps only the elements whose |delta| is in the
      global top ``topk_frac`` fraction across ALL tensors. Approximates
      the paper's sparse-update regime when exact support is near-dense.

    ``sign`` is always ``sign(trained - base)`` computed in float32 (bf16
    diffs can cancel a real change to 0) on every element, regardless of
    ``selection``.

    Requires identical key sets and shapes; raises ``ValueError`` listing
    mismatches otherwise.
    """
    if selection not in ("exact_support", "topk"):
        raise ValueError(f"Unknown selection: {selection!r}")
    if not 0.0 < topk_frac <= 1.0:
        raise ValueError(f"topk_frac must be in (0, 1], got {topk_frac}")

    base_keys, trained_keys = set(base_sd), set(trained_sd)
    if base_keys != trained_keys:
        only_base = sorted(base_keys - trained_keys)
        only_trained = sorted(trained_keys - base_keys)
        raise ValueError(
            "Key mismatch between base and trained state dicts: "
            f"{len(only_base)} only in base {only_base[:5]}, "
            f"{len(only_trained)} only in trained {only_trained[:5]}"
        )

    deltas: dict[str, torch.Tensor] = {}
    shape_mismatch = []
    for name in sorted(base_keys):
        b = base_sd[name].detach().to("cpu")
        t = trained_sd[name].detach().to("cpu")
        if b.shape != t.shape:
            shape_mismatch.append((name, tuple(b.shape), tuple(t.shape)))
            continue
        deltas[name] = t.float() - b.float()
    if shape_mismatch:
        raise ValueError(
            f"Shape mismatch on {len(shape_mismatch)} tensors: {shape_mismatch[:5]}"
        )

    delta_thresh = 0.0
    if selection == "topk":
        all_abs = torch.cat([d.abs().flatten() for d in deltas.values()])
        n = all_abs.numel()
        # kthvalue has no element limit (torch.quantile caps at 2**24);
        # clamp k so tiny n * topk_frac cannot push k out of range.
        k = min(n, max(1, n - int(topk_frac * n) + 1))
        delta_thresh = float(torch.kthvalue(all_abs, k).values)

    mask: dict[str, torch.Tensor] = {}
    sign: dict[str, torch.Tensor] = {}
    n_elems = n_changed = n_selected = 0
    for name, d in deltas.items():
        s = torch.sign(d).to(torch.int8)
        if selection == "exact_support":
            m = d != 0
        else:
            # (d != 0) keeps the mask a subset of changed elements even
            # when delta_thresh == 0 (too few nonzero deltas to hit the
            # requested fraction); mask=True with sign=0 would be an
            # unconstrained coordinate wrongly counted as in-support.
            m = (d.abs() >= delta_thresh) & (d != 0)
        mask[name] = m
        sign[name] = s
        n_elems += m.numel()
        n_changed += int((d != 0).sum())
        n_selected += int(m.sum())

    frac_selected = (n_selected / n_elems) if n_elems else 0.0
    if selection == "topk" and frac_selected < 0.9 * topk_frac:
        logger.warning(
            "topk requested %.4f of elements but only %.4f changed and were "
            "selectable; support is sparser than requested",
            topk_frac, frac_selected,
        )

    meta = {
        "n_tensors": len(mask),
        "n_elems": n_elems,
        "n_changed_elems": n_changed,
        "frac_changed": (n_changed / n_elems) if n_elems else 0.0,
        "n_selected_elems": n_selected,
        "frac_selected": frac_selected,
        "selection": selection,
        "topk_frac": topk_frac if selection == "topk" else None,
        "delta_threshold": delta_thresh if selection == "topk" else None,
        "base_fingerprint": fingerprint_state_dict(base_sd),
    }
    logger.info(
        "extract_direction[%s]: %d tensors, %d/%d changed (%.4f), "
        "%d selected (%.4f)",
        selection, meta["n_tensors"], n_changed, n_elems,
        meta["frac_changed"], n_selected, frac_selected,
    )
    return {"mask": mask, "sign": sign, "meta": meta}


def fingerprint_state_dict(sd: dict) -> str:
    """Content fingerprint of a canonical-name state dict.

    SHA-256 over ``name\\0tensor_bytes`` for keys sorted lexically; used to
    verify that a constrained run's init weights match the extraction base.
    """
    import hashlib

    h = hashlib.sha256()
    for name in sorted(sd):
        # float32 upcast is exact for bf16 and numpy-safe.
        t = sd[name].detach().to("cpu").float().contiguous()
        h.update(name.encode())
        h.update(b"\x00")
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def save_direction(artifact: dict, path: str) -> None:
    """Persist a direction artifact via ``torch.save``."""
    torch.save(artifact, path)
    logger.info("Saved direction artifact to %s", path)


def load_direction(path: str) -> dict:
    """Load a direction artifact (``weights_only=True`` when supported)."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older torch without weights_only
        return torch.load(path, map_location="cpu")


def mask_gradients_(
    named_params: Iterable,
    mask: dict,
    sign: dict,
    adapter_name: str = "default",
) -> dict:
    """Stage 1: zero grads outside the support and grads opposing ``D``.

    ``named_params`` is an iterable of ``(name, Parameter)``; params whose
    canonical name is absent from ``mask`` (or with ``.grad is None``) are
    left untouched. Returns counts of zeroed entries.
    """
    from agentic_sfm.kernels import fused_opsft_mask_gradients_

    return fused_opsft_mask_gradients_(
        named_params,
        mask,
        sign,
        canonical_name_fn=canonical_lora_name,
        adapter_name=adapter_name,
    )


def project_parameters_(
    named_params: Iterable,
    theta0: dict,
    mask: dict,
    sign: dict,
    adapter_name: str = "default",
) -> dict:
    """Stage 2: restore params outside the support or moved against ``D``.

    ``theta0`` maps canonical names -> init-value tensors (never mutated).
    Caller must wrap in ``torch.no_grad()``. Params absent from ``mask``
    are untouched. Returns restore counts.
    """
    from agentic_sfm.kernels import fused_opsft_project_parameters_

    return fused_opsft_project_parameters_(
        named_params,
        theta0,
        mask,
        sign,
        canonical_name_fn=canonical_lora_name,
        adapter_name=adapter_name,
        logger=logger,
    )
