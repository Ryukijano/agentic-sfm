"""Fused Triton & vectorized kernels for Agentic SfM training and 3D losses."""

from __future__ import annotations

from agentic_sfm.kernels.grpo_opsft_kernels import (
    HAS_TRITON,
    fused_grpo_kl_loss,
    fused_opsft_mask_gradients_,
    fused_opsft_project_parameters_,
)
from agentic_sfm.kernels.recon3d_kernels import (
    fast_all_pairs_pose_maa,
    fast_chamfer_min_dists,
    fast_gaussian_splat_compositor,
    fast_multiview_patch_zncc,
    fast_orthographic_rasterize,
)

__all__ = [
    "HAS_TRITON",
    "fast_all_pairs_pose_maa",
    "fast_chamfer_min_dists",
    "fast_gaussian_splat_compositor",
    "fast_multiview_patch_zncc",
    "fast_orthographic_rasterize",
    "fused_grpo_kl_loss",
    "fused_opsft_mask_gradients_",
    "fused_opsft_project_parameters_",
]
