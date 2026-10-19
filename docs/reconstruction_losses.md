# 3D Reconstruction Losses & Visual Model Verification for Agentic SfM

## 1. Motivation: Moving Beyond Tool-Call and Pose-Gauge Proxies

In Phase 1 and early Phase 2 (`configs/phase2_scene.yaml`, `configs/phase2_calder_8xh200.yaml`), episode rewards were dominated by tool-interaction shaping (`first_call_validity_reward = 0.5`, `accumulative_tool_reward`) and a single coarse pose scalar (`mean_pose_error_deg` over registered cameras only). Audits of job `8073061` / `8085187` (`docs/scene_eval_and_env_proposal.md`) exposed four failure modes:

1. **Tool-call reward dominance:** Syntax and validity bonuses accounted for ~40–50% of total episode reward, allowing policies to hill-climb tool formatting without improving 3D structure.
2. **Abstention and translation blindness:** Scoring geodesic rotation error only on *registered* images rewards registering an easy 3-camera subset, and ignores camera center translation errors ($\text{ATE}$) after $\mathrm{Sim}(3)$ alignment.
3. **Unused 3D geometry and RGB-D ground truth:** COLMAP's triangulated 3D point cloud $\mathcal{P} = \{(X_k, c_k, \mathcal{T}_k)\}$ (`sparse/0`) and ScanNet's dense RGB-D sensor depth maps (`depth/*.png`) were never used in the reward.
4. **No visual feedback from the 3D model:** When a visual doppelganger (symmetry fold) slips into the view graph, two walls or corridors fold on top of each other. Camera count stays high, yet the 3D model is visibly broken (ghosted walls, out-of-footprint rooms, scrambled reprojections).

This document specifies the **Mathematical + Visual 3D Reconstruction Loss** stack implemented in `src/agentic_sfm/rewards/reconstruction_loss.py` and wired into `pose_rewards.py` and `scene_episode.py`.

---

## 2. Pillar I: Mathematical 3D Reconstruction Losses

### 2.1 Intrinsic Bundle & View-Graph Rigidity (Zero-GT, Self-Supervised)

These terms require **no ground-truth poses or depth maps**. They can be evaluated both in the RL reward and at test time inside `inspect`.

1. **Track-Conditioned Robust Bundle Adjustment Loss ($\mathcal{L}_{\text{BA}}, \mathcal{R}_{\text{BA}}$)**
   A 2-view track has zero redundancy—any two non-parallel rays triangulate. Multi-view geometric rigidity only emerges when tracks span $|\mathcal{T}_k| \ge 3$ views with well-conditioned triangulation angles:
   $$\mathcal{L}_{\text{reproj}} = \frac{1}{|\mathcal{P}|} \sum_{X_k \in \mathcal{P}} \rho_{\text{Huber}}\!\left(\frac{e_k}{\sigma_{\text{px}}}\right), \qquad \mathcal{R}_{\text{track}} = \operatorname{clip}\!\left(\frac{\bar{\ell}_{\mathcal{T}} - 2}{4},\, 0,\, 1\right) \cdot \frac{|\{X_k \in \mathcal{P} : |\mathcal{T}_k| \ge 3\}|}{\max(|\mathcal{P}|, 1)}$$
   $$\mathcal{R}_{\text{BA}} = \exp\!\left(-\frac{\mathcal{L}_{\text{reproj}}}{2.0}\right) \cdot \mathcal{R}_{\text{track}}$$

2. **View-Graph Cycle Consistency & Algebraic Connectivity ($\mathcal{L}_{\text{cycle}}, \lambda_2$)**
   - **3-Cycle Rotation Residual:** Over all 3-cycles $(i,j,k) \in \mathcal{C}_3$ in the active view graph:
     $$\mathcal{L}_{\text{cycle}} = \frac{1}{|\mathcal{C}_3|} \sum_{(i,j,k) \in \mathcal{C}_3} \arccos\!\left(\frac{\operatorname{tr}(R_{ki} R_{jk} R_{ij}) - 1}{2}\right)$$
   - **Fiedler Value $\lambda_2(\mathcal{L}_{\text{norm}})$:** The second-smallest eigenvalue of the symmetrically normalized Laplacian $\mathcal{L}_{\text{norm}} = I - D^{-1/2} A D^{-1/2}$ bounds the Cramér–Rao variance of rotation averaging and detects articulated hinge bridges ($\lambda_2 \to 0$).

### 2.2 Extrinsic 3D Metric Losses (Against GT Poses, Depth Maps, & 3D Clouds)

Let $(s, R_s, t_s)$ be the gauge-fixed Umeyama $\mathrm{Sim}(3)$ alignment from predicted camera centers $\{\hat{C}_i\}$ to ground-truth camera centers $\{C_i^{\text{gt}}\}$, mapping reconstructed 3D points to metric world coordinates $X'_k = s R_s X_k + t_s$.

1. **All-Pairs Relative Pose mAA + Normalized Camera ATE ($\mathcal{R}_{\text{cam}}$)**
   - **All-pairs IMC mAA:** Evaluated across all $\binom{N}{2}$ image pairs in the episode. Any pair involving an unregistered image receives $\epsilon_{ij} = 180^\circ$, closing the abstention loophole:
     $$\mathcal{R}_{\text{mAA}} = \frac{1}{\binom{N}{2}} \sum_{i < j} \frac{1}{3} \sum_{\tau \in \{5^\circ, 10^\circ, 20^\circ\}} \mathbb{1}\!\left[\max(\epsilon^{(R)}_{ij}, \epsilon^{(t)}_{ij}) < \tau\right]$$
   - **Normalized Absolute Trajectory Error ($\mathcal{L}_{\text{ATE}}$):**
     $$\mathcal{L}_{\text{ATE}} = \frac{1}{|M|} \sum_{i \in M} \rho_{\text{Huber}}\!\left(\frac{\|s R_s \hat{C}_i + t_s - C_i^{\text{gt}}\|_2}{\max(\operatorname{diam}(\{C^{\text{gt}}\}), 10^{-3})}\right)$$

2. **Projected 3D-Point-to-Sensor-Depth + Gradient Loss ($\mathcal{L}_{\text{depth}}$)**
   For each triangulated 3D point $X'_k$ projected into registered camera $i$ with GT sensor depth $D^{\text{gt}}_i(u_{ik}) > 0$:
   $$\mathcal{L}_{\text{depth}} = \frac{1}{|\mathcal{O}|}\sum_{(i,k)\in\mathcal{O}} \left|\log \frac{\hat{z}_{ik}}{D^{\text{gt}}_i(u_{ik})}\right|_1 + \lambda_{\nabla} \frac{1}{|\mathcal{N}|}\sum_{(k,k')\in\mathcal{N}_i} \left|(\hat{z}_{ik} - \hat{z}_{ik'}) - (D^{\text{gt}}_i(u_{ik}) - D^{\text{gt}}_i(u_{ik'}))\right|$$
   Following VGGT (CVPR 2025, arXiv:2503.11651), the spatial gradient term $\lambda_{\nabla}$ penalizes depth-order inversions caused by wall-doubling.

3. **Bidirectional Truncated Chamfer & Voxel-Normalized 3D F-Score ($\mathcal{R}_{\text{3D-F1}}$)**
   Comparing $\mathcal{P}'_{\text{pred}} = \{X'_k\}$ against the voxel-downsampled dense GT surface cloud $\mathcal{V}_{\text{gt}}$ (unprojected from ScanNet `depth/*.png` + `pose/*.txt` via `viz.py:gt_pointcloud_arrays`):
   $$P@\tau = \frac{1}{|\mathcal{P}'_{\text{pred}}|}\sum_{x \in \mathcal{P}'_{\text{pred}}} \mathbb{1}\!\left[\min_{y \in \mathcal{V}_{\text{gt}}} \|x - y\|_2 < \tau\right], \qquad R@\tau = \frac{1}{|\mathcal{V}_{\text{gt}}|}\sum_{y \in \mathcal{V}_{\text{gt}}} \mathbb{1}\!\left[\min_{x \in \mathcal{P}'_{\text{pred}}} \|x - y\|_2 < \tau\right]$$
   $$\mathcal{R}_{\text{3D-F1}} = \frac{2 \, P@\tau \, R@\tau}{P@\tau + R@\tau + \epsilon} \qquad (\tau = 0.10\text{ m})$$

---

## 3. Pillar II: Visual Reconstruction Losses from the 3D Model

### 3.1 Multi-View Patch Photometric & Feature-Metric Reprojection Consistency ($\mathcal{R}_{\text{patch-cons}}$)
*(Self-supervised — requires zero GT poses or depth)*

Each triangulated 3D point $X_k \in \mathcal{P}$ projects to $\hat{u}_{ik} = \pi(K_i, \hat{R}_i X_k + \hat{t}_i)$ across its observing views $i \in \mathcal{T}_k$. Sampling a local $p \times p$ RGB patch $W_i(\hat{u}_{ik})$ around each projection, we measure zero-mean normalized cross-correlation (ZNCC) and L1 color consistency against the track centroid patch $\bar{W}_k$:
$$\mathcal{L}_{\text{patch-proj}} = \frac{1}{|\mathcal{P}|}\sum_{X_k \in \mathcal{P}} \frac{1}{|\mathcal{T}_k|} \sum_{i \in \mathcal{T}_k} \left[ 0.5\big(1 - \operatorname{ZNCC}(W_i(\hat{u}_{ik}), \bar{W}_k)\big) + 0.5\,\|W_i(\hat{u}_{ik}) - \bar{W}_k\|_1 \right]$$
When a symmetry-folded camera is registered at a wrong pose, 3D points project onto mismatched textures, spiking $\mathcal{L}_{\text{patch-proj}}$.

### 3.2 Differentiable Soft-Point / Gaussian Splatting Novel-View Synthesis ($\mathcal{R}_{\text{splat}}$)

Given colored 3D points $\mathcal{P} = \{(X_k, c_k)\}$ with k-NN adaptive splat radius $\sigma_k$, we render the 3D point cloud into registered or held-out camera views $(K_i, \hat{R}_i, \hat{t}_i)$ using front-to-back $\alpha$-compositing:
$$\hat{C}_i(u) = \sum_{m=1}^{K} c_m \alpha_m(u) \prod_{l < m} (1 - \alpha_l(u)), \qquad \hat{O}_i(u) = 1 - \prod_{m=1}^{K} (1 - \alpha_m(u))$$
Evaluating on the valid splat support mask $M_i = \mathbb{1}[\hat{O}_i > 0.1]$:
$$\mathcal{R}_{\text{splat}} = \frac{1}{|V|}\sum_{i \in V} \Big( 0.6\,\operatorname{SSIM}_{M_i}(\hat{C}_i, I_i) + 0.4\,\big(1 - \|\hat{C}_i - I_i\|_{1, M_i}\big) \Big)$$

### 3.3 Canonical 3-View Orthographic 3D Model Render Loss ($\mathcal{R}_{\text{ortho-3D}}$)

Both $\mathcal{P}'_{\text{pred}}$ and $\mathcal{P}_{\text{gt}}$ are rendered from three canonical orthographic cameras sharing the GT bounding box:
- **Top-Down BEV Floorplan ($X$-$Y$)**
- **Front Elevation ($X$-$Z$)**
- **Side Elevation ($Y$-$Z$)**

For each view $v \in \{\text{BEV}, \text{Front}, \text{Side}\}$, dilated occupancy $O_v$, normalized depth $Z_v$, and color $RGB_v$ grids are rasterized:
$$\mathcal{R}_{\text{ortho-3D}} = \frac{1}{3}\sum_{v} \Big[ 0.45\,\operatorname{IoU}\!\big(O^{\text{pred}}_v, O^{\text{gt}}_v\big) + 0.30\,\big(1 - \|RGB^{\text{pred}}_v - RGB^{\text{gt}}_v\|_{1, O^{\text{pred}}_v \wedge O^{\text{gt}}_v}\big) + 0.25\,\big(1 - |Z^{\text{pred}}_v - Z^{\text{gt}}_v|_{1, O^{\text{pred}}_v \wedge O^{\text{gt}}_v}\big) \Big]$$
Symmetry folding (doppelgangers) rotates a corridor or wall outside the true room footprint in the BEV floorplan, immediately dropping $\operatorname{IoU}(O^{\text{pred}}_{\text{BEV}}, O^{\text{gt}}_{\text{BEV}})$.

### 3.4 Closing the Visual Loop: 4-Panel Diagnostic Render (`render_diagnostic_4panel`)

To give `Qwen3.5-2B` direct visual perception of the 3D model it built, `render_diagnostic_4panel` renders a single $2 \times 2$ composite figure:
1. **Panel A (Top-Down BEV Floorplan):** Orthographic $X$-$Y$ projection of colored 3D points + camera frustums + trajectory polyline.
2. **Panel B (Isometric 3D View):** 3D point cloud + frustum cones.
3. **Panel C (3D Reprojection Residual Heatmap):** 3D points colored by reprojection error (green $\le 1\text{ px} \to$ red $\ge 3\text{ px}$) to highlight floaters and folded surfaces.
4. **Panel D (Splat-Rendered Camera View vs. Ground-Truth Frame):** Side-by-side / overlay of the splat-rendered 3D model from a registered camera against the observed RGB image.

---

## 4. Unified Reward Composition & Pre-Registered Ablation Protocol

### 4.1 Composite Scene Reward (`compute_scene_reward`)

When `use_reconstruction_loss: true` is enabled in the reward config (or when `recon_result` carries the injected reconstruction loss components), `compute_scene_reward` adds:
- `intrinsic_ba_reward`: $w_{\text{ba}} \cdot \mathcal{R}_{\text{BA}}$
- `all_pairs_maa_reward`: $w_{\text{maa}} \cdot \mathcal{R}_{\text{mAA}}$
- `ate_reward`: $w_{\text{ate}} \cdot \text{cov} \cdot \exp(-\mathcal{L}_{\text{ATE}})$
- `pointcloud_f1_reward`: $w_{\text{f1}} \cdot \mathcal{R}_{\text{3D-F1}}$
- `depth_consistency_reward`: $w_{\text{depth}} \cdot \exp(-\mathcal{L}_{\text{depth}})$
- `visual_3d_reward`: $w_{\text{vis3d}} \cdot \big(0.4\,\mathcal{R}_{\text{ortho-3D}} + 0.3\,\mathcal{R}_{\text{splat}} + 0.3\,\mathcal{R}_{\text{patch-cons}}\big)$

All new weights default to `0.0` when not specified so existing tests and configs remain 100% backward-compatible.

### 4.2 Pre-Registered Ablation Matrix (Calder 8×H200)

| Arm | Description | Active Reward Terms | Primary Metric (Frozen 30-Scene ScanNet) |
|---|---|---|---|
| **R0** | Legacy baseline | `registration_reward + pose_reward + first_call_validity (0.5)` | All-pairs mAA & 3D $\text{F}_1@10\text{cm}$ |
| **R1** | Intrinsic + Extrinsic Math 3D Loss | R0 (`first_call_validity=0`) + $\mathcal{R}_{\text{BA}} + \mathcal{R}_{\text{mAA}} + \mathcal{R}_{\text{ATE}} + \mathcal{R}_{\text{3D-F1}} + \mathcal{R}_{\text{depth}}$ | All-pairs mAA & 3D $\text{F}_1@10\text{cm}$ |
| **R2** | Math + Visual 3D Reconstruction Loss | R1 + $\mathcal{R}_{\text{ortho-3D}} + \mathcal{R}_{\text{splat}} + \mathcal{R}_{\text{patch-cons}}$ | All-pairs mAA, 3D $\text{F}_1@10\text{cm}$, BEV IoU |
| **R3** | Visual Loop Closure | R2 + `render_diagnostic_4panel` returned on `inspect` | All-pairs mAA, doppelganger F1, BEV IoU |

---

## 5. Key References

- **DUSt3R** (CVPR 2024): https://arxiv.org/abs/2312.14132
- **MASt3R** (ECCV 2024): https://arxiv.org/abs/2406.09756
- **VGGT** (CVPR 2025): https://arxiv.org/abs/2503.11651
- **VGGSfM** (CVPR 2024): https://arxiv.org/abs/2312.04563
- **FlowMap** (2024): https://arxiv.org/abs/2404.15259
- **InstantSplat** (2024/2025): https://arxiv.org/abs/2403.20309
- **ACE0** (ECCV 2024): https://arxiv.org/abs/2404.14351
- **Doppelgangers** (ICCV 2023): https://arxiv.org/abs/2309.02420
- **Spectral SLAM Sparsification / Fiedler $\lambda_2$**: https://arxiv.org/abs/2203.13897
- **Rotation Averaging Strong Duality** (CVPR 2018): https://arxiv.org/abs/1705.01362
- **GPTEval3D** (CVPR 2024): https://arxiv.org/abs/2401.04092
