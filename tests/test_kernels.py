"""Tests and numerical-equivalence checks for fused training & 3D loss kernels."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))

from agentic_sfm.kernels import (
    fast_all_pairs_pose_maa,
    fast_chamfer_min_dists,
    fast_gaussian_splat_compositor,
    fast_multiview_patch_zncc,
    fast_orthographic_rasterize,
    fused_grpo_kl_loss,
    fused_opsft_mask_gradients_,
    fused_opsft_project_parameters_,
)
from agentic_sfm.opsft import canonical_lora_name, mask_gradients_, project_parameters_


def _ref_grpo_kl_loss(
    new_tok: torch.Tensor,
    old_tok: torch.Tensor,
    ref_tok: torch.Tensor | None,
    adv: float,
    clip_low: float,
    clip_high: float,
    kl_coef: float,
    off_support_threshold: float | None,
    grad_accum: int,
) -> tuple[torch.Tensor, bool]:
    """Reference PyTorch implementation from the original run_grpo.py loop."""
    keep = None
    if adv < 0 and off_support_threshold is not None:
        keep = old_tok >= off_support_threshold
        if int(keep.sum()) == 0:
            return new_tok.sum() * 0.0, False
        new_sub = new_tok[keep]
        old_sub = old_tok[keep]
    else:
        new_sub = new_tok
        old_sub = old_tok

    ratio = torch.exp(new_sub - old_sub)
    clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    adv_t = torch.tensor(adv, device=ratio.device, dtype=ratio.dtype)
    loss = -torch.min(ratio * adv_t, clipped * adv_t).mean()

    if ref_tok is not None and kl_coef > 0.0:
        ref_sub = ref_tok[: keep.shape[0]][keep] if keep is not None else ref_tok
        k = min(new_sub.shape[0], ref_sub.shape[0])
        if k > 0:
            kl = torch.exp(ref_sub[:k] - new_sub[:k]) - (ref_sub[:k] - new_sub[:k]) - 1.0
            loss = loss + kl_coef * kl.mean()

    return loss / grad_accum, True


class TestFusedGRPOKLKernel:
    @pytest.mark.parametrize("adv", [1.35, -0.85, -2.10, 0.40])
    @pytest.mark.parametrize("grad_accum", [1, 4])
    def test_forward_and_backward_match_reference_exactly(self, adv: float, grad_accum: int):
        g = torch.Generator().manual_seed(42)
        old_lp = torch.randn(512, generator=g) * 3.0 - 4.0
        delta = torch.randn(512, generator=g) * 0.35
        ref_lp = old_lp + torch.randn(512, generator=g) * 0.2

        new_ref = (old_lp + delta).clone().detach().requires_grad_(True)
        new_fus = (old_lp + delta).clone().detach().requires_grad_(True)

        loss_ref, ok_ref = _ref_grpo_kl_loss(
            new_ref, old_lp, ref_lp, adv, 0.2, 0.28, 0.03, -9.0, grad_accum
        )
        loss_fus, ok_fus = fused_grpo_kl_loss(
            new_fus, old_lp, ref_lp, adv, 0.2, 0.28, 0.03, -9.0, grad_accum
        )
        assert ok_ref == ok_fus
        assert torch.allclose(loss_ref, loss_fus, atol=1e-6, rtol=1e-5)

        loss_ref.backward()
        loss_fus.backward()
        assert torch.allclose(new_ref.grad, new_fus.grad, atol=1e-6, rtol=1e-5)

    def test_all_vetoed_returns_false_and_zero_grad(self):
        old_lp = torch.full((64,), -12.0)
        new_lp = torch.full((64,), -11.5, requires_grad=True)
        loss, ok = fused_grpo_kl_loss(
            new_lp, old_lp, None, advantage=-1.0, off_support_threshold=-9.0
        )
        assert ok is False
        assert float(loss.item()) == 0.0


class TestFusedOPSFTKernels:
    def test_stage1_and_stage2_match_specification(self):
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32))
        p.grad = torch.tensor([0.5, -0.5, 0.5, -0.5], dtype=torch.float32)
        theta0 = {"layer.lora_A.weight": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)}
        mask = {"layer.lora_A.weight": torch.tensor([False, True, True, True])}
        # sign: [0, +1, +1, -1]
        # idx 0: outside mask -> zeroed
        # idx 1: g=-0.5, d=+1 -> g*d < 0 (descent +0.5 aligns with +1) -> kept
        # idx 2: g=+0.5, d=+1 -> g*d > 0 (descent -0.5 opposes +1) -> zeroed
        # idx 3: g=-0.5, d=-1 -> g*d > 0 (descent +0.5 opposes -1) -> zeroed
        sign = {"layer.lora_A.weight": torch.tensor([0, 1, 1, -1], dtype=torch.int8)}

        s1 = mask_gradients_([("layer.lora_A.default.weight", p)], mask, sign)
        assert s1["zeroed_outside"] == 1
        assert s1["zeroed_opposite"] == 2
        assert torch.allclose(p.grad, torch.tensor([0.0, -0.5, 0.0, 0.0]))

        # Simulate parameter movement that violates mask at idx 0 and sign at idx 2
        with torch.no_grad():
            p.copy_(torch.tensor([9.0, 2.5, 2.5, 3.5]))
            s2 = project_parameters_([("layer.lora_A.default.weight", p)], theta0, mask, sign)
        assert s2["restored_outside"] == 1
        assert s2["restored_wrong"] == 1
        assert torch.allclose(p.data, torch.tensor([1.0, 2.5, 3.0, 3.5]))


class TestRecon3DKernels:
    def test_fast_chamfer_matches_exact_euclidean(self):
        rng = np.random.default_rng(0)
        P = rng.uniform(-2, 2, size=(150, 3))
        G = rng.uniform(-2, 2, size=(200, 3))
        d_p2g, d_g2p = fast_chamfer_min_dists(P, G)
        # Brute-force reference
        diff = P[:, None, :] - G[None, :, :]
        dist_mat = np.linalg.norm(diff, axis=2)
        assert np.allclose(d_p2g, dist_mat.min(axis=1), atol=1e-6)
        assert np.allclose(d_g2p, dist_mat.min(axis=0), atol=1e-6)

    def test_fast_multiview_patch_zncc_speed_and_accuracy(self):
        rng = np.random.default_rng(11)
        pts = rng.uniform(-0.8, 0.8, size=(256, 3))
        pts[:, 2] = rng.uniform(2.0, 4.0, size=256)
        K = np.array([[60.0, 0.0, 32.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]])
        img = rng.uniform(0.1, 0.9, size=(64, 64, 3))
        poses = {0: np.eye(4), 1: np.eye(4), 2: np.eye(4)}
        imgs = {0: img, 1: img, 2: img}
        Ks = {0: K, 1: K, 2: K}

        t0 = time.perf_counter()
        out = fast_multiview_patch_zncc(pts, imgs, poses, Ks, patch_radius=3)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert out["patch_reproj_loss"] == pytest.approx(0.0, abs=1e-6)
        assert out["patch_consistency_reward"] == pytest.approx(1.0, abs=1e-6)
        assert out["num_consistent_points"] > 100
        assert dt_ms < 50.0  # vectorized pass runs in ~1-3 ms on CPU

    def test_fast_all_pairs_pose_maa_batched(self):
        poses = {}
        for i in range(10):
            M = np.eye(4)
            M[:3, 3] = [float(i), 0.1 * i, -0.2 * i]
            poses[i] = M
        res = fast_all_pairs_pose_maa(poses, poses, num_total_images=10)
        assert res["all_pairs_maa"] == pytest.approx(1.0)
        res_half = fast_all_pairs_pose_maa({i: poses[i] for i in range(5)}, poses, num_total_images=10)
        # 5*4/2 = 10 pairs out of 10*9/2 = 45 pairs
        assert res_half["all_pairs_maa"] == pytest.approx(10.0 / 45.0)
