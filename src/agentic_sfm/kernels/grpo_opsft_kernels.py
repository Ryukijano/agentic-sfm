"""Fused Triton & PyTorch kernels for GRPO policy-gradient and OPSFT training.

Provides:
1. ``fused_grpo_kl_loss``:
   Single-pass fused forward + analytical backward kernel (Triton on CUDA
   ``sm_80``/``sm_89``/``sm_90``+, vectorized PyTorch fallback on CPU) for:
   - DAPO asymmetric clipped surrogate: ``-min(r * A, clip(r, 1-e_lo, 1+e_hi) * A)``
   - mu-GRPO off-support token veto: drops tokens with ``old_logp < thresh`` when ``A < 0``
   - K3 KL divergence leash against frozen SFT reference: ``exp(ref - new) - (ref - new) - 1``

2. ``fused_opsft_mask_gradients_`` & ``fused_opsft_project_parameters_``:
   Fused Triton kernels for OPSFT (arXiv:2609.36659) Stage-1 gradient masking
   and Stage-2 parameter projection across LoRA adapters, with automatic
   per-device caching of ``(mask, sign, theta0)`` tensors to eliminate
   per-step CPU->GPU PCIe transfers and per-tensor host synchronizations.
"""

from __future__ import annotations

from typing import Iterable

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


# ===========================================================================
# 1. Fused GRPO + mu-GRPO Veto + K3 KL Surrogate Kernel
# ===========================================================================

if HAS_TRITON:

    @triton.jit
    def _grpo_kl_fwd_bwd_kernel(
        new_ptr,
        old_ptr,
        ref_ptr,
        loss_out_ptr,
        grad_new_ptr,
        valid_out_ptr,
        n_elements,
        adv,
        clip_low,
        clip_high,
        kl_coef,
        off_support_thresh,
        apply_veto: tl.constexpr,
        has_ref: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        new_lp = tl.load(new_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        old_lp = tl.load(old_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        keep = mask
        if apply_veto:
            keep = mask & (old_lp >= off_support_thresh)

        log_ratio = new_lp - old_lp
        ratio = tl.exp(log_ratio)
        lo = 1.0 - clip_low
        hi = 1.0 + clip_high
        clipped_ratio = tl.minimum(tl.maximum(ratio, lo), hi)

        surr1 = ratio * adv
        surr2 = clipped_ratio * adv
        use_unclipped = surr1 <= surr2
        pg_loss = -tl.where(use_unclipped, surr1, surr2)

        # Analytical gradient d(pg_loss)/d(new_lp):
        # When surr1 < surr2, pg_loss = -adv * exp(new_lp - old_lp) -> grad = -adv * ratio
        # At the clipped flat branch, derivative wrt new_lp is 0.
        in_clip_band = (adv > 0.0 and ratio < hi) or (adv < 0.0 and ratio > lo)
        grad_pg = tl.where(use_unclipped & in_clip_band, -adv * ratio, 0.0)

        kl_loss = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        grad_kl = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        if has_ref:
            ref_lp = tl.load(ref_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            diff = ref_lp - new_lp
            exp_diff = tl.exp(diff)
            kl_loss = kl_coef * (exp_diff - diff - 1.0)
            # d/d(new_lp) [exp(ref - new) - (ref - new) - 1] = 1 - exp(ref - new)
            grad_kl = kl_coef * (1.0 - exp_diff)

        tok_loss = tl.where(keep, pg_loss + kl_loss, 0.0)
        tok_grad = tl.where(keep, grad_pg + grad_kl, 0.0)
        valid_f = tl.where(keep, 1.0, 0.0)

        tl.store(loss_out_ptr + offs, tok_loss, mask=mask)
        tl.store(grad_new_ptr + offs, tok_grad, mask=mask)
        tl.store(valid_out_ptr + offs, valid_f, mask=mask)


class _FusedGRPOKLAutograd(torch.autograd.Function):
    """Custom autograd function wrapping the fused GRPO + K3-KL kernel."""

    @staticmethod
    def forward(
        ctx,
        new_tok: torch.Tensor,
        old_tok: torch.Tensor,
        ref_tok: torch.Tensor | None,
        advantage: float,
        clip_low: float,
        clip_high: float,
        kl_coef: float,
        off_support_threshold: float | None,
        grad_accum: int,
    ) -> torch.Tensor:
        n = new_tok.numel()
        if n == 0:
            ctx.save_for_backward(torch.zeros_like(new_tok))
            return new_tok.sum() * 0.0

        apply_veto = bool(advantage < 0.0 and off_support_threshold is not None)
        thresh_val = float(off_support_threshold) if off_support_threshold is not None else -1e9
        has_ref = bool(ref_tok is not None and kl_coef > 0.0 and ref_tok.numel() > 0)

        if HAS_TRITON and new_tok.is_cuda:
            new_c = new_tok.detach().to(torch.float32).contiguous()
            old_c = old_tok.detach().to(torch.float32).contiguous()
            if has_ref:
                ref_c = ref_tok.detach().to(torch.float32).contiguous()
                if ref_c.numel() < n:
                    pad = new_c[ref_c.numel() :].clone()
                    ref_c = torch.cat([ref_c, pad], dim=0)
                else:
                    ref_c = ref_c[:n]
            else:
                ref_c = new_c

            loss_buf = torch.empty_like(new_c)
            grad_buf = torch.empty_like(new_c)
            valid_buf = torch.empty_like(new_c)

            BLOCK = 256
            grid = ((n + BLOCK - 1) // BLOCK,)
            _grpo_kl_fwd_bwd_kernel[grid](
                new_c,
                old_c,
                ref_c,
                loss_buf,
                grad_buf,
                valid_buf,
                n,
                float(advantage),
                float(clip_low),
                float(clip_high),
                float(kl_coef),
                thresh_val,
                apply_veto=apply_veto,
                has_ref=has_ref,
                BLOCK_SIZE=BLOCK,
                num_warps=4,
            )
            n_valid = valid_buf.sum()
            if float(n_valid.item()) == 0.0:
                ctx.save_for_backward(torch.zeros_like(new_tok))
                return new_tok.sum() * 0.0

            denom = n_valid * float(max(grad_accum, 1))
            grad_scaled = (grad_buf / denom).to(new_tok.dtype)
            ctx.save_for_backward(grad_scaled)
            return (loss_buf.sum() / denom).to(torch.float32)

        # Vectorized PyTorch path (CPU or non-Triton device) — matches exact semantics
        new_f = new_tok.detach().to(torch.float32)
        old_f = old_tok.detach().to(torch.float32)
        if apply_veto:
            keep = old_f >= thresh_val
        else:
            keep = torch.ones_like(old_f, dtype=torch.bool)

        n_keep = int(keep.sum().item())
        if n_keep == 0:
            ctx.save_for_backward(torch.zeros_like(new_tok))
            return new_tok.sum() * 0.0

        ratio = torch.exp(new_f - old_f)
        lo, hi = 1.0 - float(clip_low), 1.0 + float(clip_high)
        clipped = torch.clamp(ratio, lo, hi)
        adv_f = float(advantage)
        surr1 = ratio * adv_f
        surr2 = clipped * adv_f
        use_unclipped = surr1 <= surr2
        pg_loss = -torch.where(use_unclipped, surr1, surr2)

        in_band = (ratio < hi) if adv_f > 0.0 else ((ratio > lo) if adv_f < 0.0 else torch.zeros_like(keep))
        grad_pg = torch.where(use_unclipped & in_band, -adv_f * ratio, torch.zeros_like(ratio))

        kl_loss = torch.zeros_like(pg_loss)
        grad_kl = torch.zeros_like(grad_pg)
        if has_ref and ref_tok is not None:
            ref_f = ref_tok.detach().to(device=new_f.device, dtype=torch.float32)
            k = min(n, ref_f.numel())
            diff = ref_f[:k] - new_f[:k]
            exp_diff = torch.exp(diff)
            # Note: in run_grpo.py, pg_loss is averaged over n_keep and kl_loss
            # over k_keep; when k == n they share the same denominator n_keep.
            kl_Mask = keep[:k]
            n_kl = max(int(kl_Mask.sum().item()), 1)
            kl_term = float(kl_coef) * (exp_diff - diff - 1.0)
            kl_loss[:k] = torch.where(kl_Mask, kl_term * (n_keep / n_kl), torch.zeros_like(kl_term))
            grad_kl[:k] = torch.where(
                kl_Mask,
                float(kl_coef) * (1.0 - exp_diff) * (n_keep / n_kl),
                torch.zeros_like(exp_diff),
            )

        denom = float(n_keep * max(grad_accum, 1))
        total_tok_loss = torch.where(keep, pg_loss + kl_loss, torch.zeros_like(pg_loss))
        total_tok_grad = torch.where(keep, grad_pg + grad_kl, torch.zeros_like(grad_pg)) / denom
        ctx.save_for_backward(total_tok_grad.to(new_tok.dtype))
        return total_tok_loss.sum() / denom

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (grad_new,) = ctx.saved_tensors
        return grad_new * grad_output.to(grad_new.dtype), None, None, None, None, None, None, None, None


def fused_grpo_kl_loss(
    new_tok: torch.Tensor,
    old_tok: torch.Tensor,
    ref_tok: torch.Tensor | None,
    advantage: float,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    kl_coef: float = 0.03,
    off_support_threshold: float | None = -9.0,
    grad_accum: int = 1,
) -> tuple[torch.Tensor, bool]:
    """Compute the fused token-level GRPO + mu-GRPO off-support veto + K3 KL loss.

    Returns ``(loss_tensor, is_valid)`` where ``is_valid`` is ``False`` when
    the mu-GRPO off-support veto filtered out every token in the sequence.
    """
    if new_tok.numel() == 0:
        return new_tok.sum() * 0.0, False
    if advantage < 0.0 and off_support_threshold is not None:
        if not bool((old_tok >= off_support_threshold).any().item()):
            return new_tok.sum() * 0.0, False
    loss = _FusedGRPOKLAutograd.apply(
        new_tok,
        old_tok,
        ref_tok,
        float(advantage),
        float(clip_low),
        float(clip_high),
        float(kl_coef),
        off_support_threshold,
        int(max(grad_accum, 1)),
    )
    return loss, True


# ===========================================================================
# 2. Fused OPSFT Stage-1 Gradient Mask & Stage-2 Parameter Projection Kernels
# ===========================================================================

if HAS_TRITON:

    @triton.jit
    def _opsft_mask_grad_kernel(
        grad_ptr,
        mask_ptr,
        sign_ptr,
        counts_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        in_bounds = offs < n_elements

        g = tl.load(grad_ptr + offs, mask=in_bounds, other=0.0).to(tl.float32)
        m = tl.load(mask_ptr + offs, mask=in_bounds, other=1).to(tl.int1)
        d = tl.load(sign_ptr + offs, mask=in_bounds, other=0).to(tl.float32)

        outside = in_bounds & (~m)
        opposite = in_bounds & m & (g != 0.0) & (d != 0.0) & ((g * d) > 0.0)
        should_zero = outside | opposite

        g_out = tl.where(should_zero, 0.0, g)
        tl.store(grad_ptr + offs, g_out.to(grad_ptr.dtype.element_ty), mask=in_bounds)

        n_out = tl.sum(outside.to(tl.int32), axis=0)
        n_opp = tl.sum(opposite.to(tl.int32), axis=0)
        if n_out > 0:
            tl.atomic_add(counts_ptr + 0, n_out)
        if n_opp > 0:
            tl.atomic_add(counts_ptr + 1, n_opp)

    @triton.jit
    def _opsft_project_param_kernel(
        param_ptr,
        theta0_ptr,
        mask_ptr,
        sign_ptr,
        counts_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        in_bounds = offs < n_elements

        p = tl.load(param_ptr + offs, mask=in_bounds, other=0.0).to(tl.float32)
        t0 = tl.load(theta0_ptr + offs, mask=in_bounds, other=0.0).to(tl.float32)
        m = tl.load(mask_ptr + offs, mask=in_bounds, other=1).to(tl.int1)
        d = tl.load(sign_ptr + offs, mask=in_bounds, other=0).to(tl.float32)

        delta = p - t0
        outside = in_bounds & (~m)
        wrong = in_bounds & m & (d != 0.0) & ((delta * d) < 0.0)
        should_restore = outside | wrong

        p_out = tl.where(should_restore, t0, p)
        tl.store(param_ptr + offs, p_out.to(param_ptr.dtype.element_ty), mask=in_bounds)

        n_out = tl.sum(outside.to(tl.int32), axis=0)
        n_wrg = tl.sum(wrong.to(tl.int32), axis=0)
        if n_out > 0:
            tl.atomic_add(counts_ptr + 0, n_out)
        if n_wrg > 0:
            tl.atomic_add(counts_ptr + 1, n_wrg)


# Per-device cache keyed by (id(dict), cname, device) so CPU->GPU transfers of
# mask/sign/theta0 happen only once per training run instead of every step.
_DEVICE_TENSOR_CACHE: dict[tuple[int, str, torch.device, torch.dtype | None], torch.Tensor] = {}


def _cached_on_device(
    store: dict,
    key: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    cache_key = (id(store), key, device, dtype)
    cached = _DEVICE_TENSOR_CACHE.get(cache_key)
    src = store[key]
    if cached is not None and cached.shape == src.shape:
        return cached
    t = src.to(device=device, dtype=dtype if dtype is not None else src.dtype).contiguous()
    # Cap cache size to prevent unbounded growth in tests creating ephemeral dicts
    if len(_DEVICE_TENSOR_CACHE) > 2048:
        _DEVICE_TENSOR_CACHE.clear()
    _DEVICE_TENSOR_CACHE[cache_key] = t
    return t


def fused_opsft_mask_gradients_(
    named_params: Iterable,
    mask: dict,
    sign: dict,
    canonical_name_fn,
    adapter_name: str = "default",
) -> dict:
    """Fused Stage-1 OPSFT gradient masking across all LoRA parameters."""
    stats = {
        "zeroed_outside": 0,
        "zeroed_opposite": 0,
        "n_trainable": 0,
        "n_constrained": 0,
    }
    cuda_counts: dict[torch.device, torch.Tensor] = {}

    for name, p in named_params:
        if p.grad is None:
            continue
        stats["n_trainable"] += 1
        cname = canonical_name_fn(name, adapter_name)
        if cname not in mask:
            continue
        stats["n_constrained"] += 1
        g = p.grad
        dev = g.device

        if HAS_TRITON and g.is_cuda and g.is_contiguous():
            m_dev = _cached_on_device(mask, cname, dev, dtype=torch.bool)
            d_dev = _cached_on_device(sign, cname, dev, dtype=torch.int8)
            cnt = cuda_counts.get(dev)
            if cnt is None:
                cnt = torch.zeros(2, device=dev, dtype=torch.int32)
                cuda_counts[dev] = cnt
            n_el = g.numel()
            BLOCK = 512
            grid = ((n_el + BLOCK - 1) // BLOCK,)
            _opsft_mask_grad_kernel[grid](
                g, m_dev, d_dev, cnt, n_el, BLOCK_SIZE=BLOCK, num_warps=4
            )
        else:
            m_dev = _cached_on_device(mask, cname, dev, dtype=torch.bool)
            d_dev = _cached_on_device(sign, cname, dev, dtype=torch.float32)
            gf = g.float()
            outside = ~m_dev
            opposite = m_dev & (g != 0) & (d_dev != 0) & (gf * d_dev > 0)
            stats["zeroed_outside"] += int(outside.sum().item())
            stats["zeroed_opposite"] += int(opposite.sum().item())
            g.masked_fill_(outside | opposite, 0.0)

    for cnt in cuda_counts.values():
        c_cpu = cnt.cpu()
        stats["zeroed_outside"] += int(c_cpu[0].item())
        stats["zeroed_opposite"] += int(c_cpu[1].item())

    return stats


def fused_opsft_project_parameters_(
    named_params: Iterable,
    theta0: dict,
    mask: dict,
    sign: dict,
    canonical_name_fn,
    adapter_name: str = "default",
    logger=None,
) -> dict:
    """Fused Stage-2 OPSFT parameter projection across all LoRA parameters."""
    stats = {"restored_outside": 0, "restored_wrong": 0, "n_constrained": 0}
    cuda_counts: dict[torch.device, torch.Tensor] = {}

    for name, p in named_params:
        cname = canonical_name_fn(name, adapter_name)
        if cname not in mask:
            continue
        if cname not in theta0:
            if logger is not None:
                logger.warning("theta0 missing entry for %s; skipping", cname)
            continue
        stats["n_constrained"] += 1
        dev = p.device

        if HAS_TRITON and p.is_cuda and p.is_contiguous():
            m_dev = _cached_on_device(mask, cname, dev, dtype=torch.bool)
            d_dev = _cached_on_device(sign, cname, dev, dtype=torch.int8)
            t0_dev = _cached_on_device(theta0, cname, dev, dtype=p.dtype)
            cnt = cuda_counts.get(dev)
            if cnt is None:
                cnt = torch.zeros(2, device=dev, dtype=torch.int32)
                cuda_counts[dev] = cnt
            n_el = p.numel()
            BLOCK = 512
            grid = ((n_el + BLOCK - 1) // BLOCK,)
            _opsft_project_param_kernel[grid](
                p, t0_dev, m_dev, d_dev, cnt, n_el, BLOCK_SIZE=BLOCK, num_warps=4
            )
        else:
            m_dev = _cached_on_device(mask, cname, dev, dtype=torch.bool)
            d_dev = _cached_on_device(sign, cname, dev, dtype=torch.float32)
            t0_dev = _cached_on_device(theta0, cname, dev, dtype=p.dtype)
            delta = (p - t0_dev).float()
            outside = ~m_dev
            wrong = m_dev & (d_dev != 0) & (delta * d_dev < 0)
            stats["restored_outside"] += int(outside.sum().item())
            stats["restored_wrong"] += int(wrong.sum().item())
            restore = outside | wrong
            p[restore] = t0_dev[restore]

    for cnt in cuda_counts.values():
        c_cpu = cnt.cpu()
        stats["restored_outside"] += int(c_cpu[0].item())
        stats["restored_wrong"] += int(c_cpu[1].item())

    return stats
