# Scene-level evaluation and training environment: proposal and audit

Status: proposal, not yet implemented. Written after reviewing the Phase 2 scene GRPO run (job 8073061) and its 10-scene evaluation (job 8085187).

## 1. Where things stand

- Scene GRPO (job 8073061) finished 50 epochs. Checkpoints `best_eval` (epoch 25) and `epoch_50` load and contain finite, changed weights.
- The 10-scene evaluation (job 8085187) gives 3/10 scenes at success. Success means at least 50% registered images and mean pose error under 20 degrees. Mean pose error is 28.9 degrees, and mean coverage is 42.5%.
- Training reward rose from about 0.74 to 1.6, but reward is not evidence of reconstruction quality.

## 2. Problems with the current evaluation

These were checked against the repo and the eval JSON.

1. **Two of the ten eval scenes were training scenes.** MegaDepth 0015 and 0022 are in the curriculum (`configs/phase2_scene.yaml:79-81`). The held-out evidence is only the eight ScanNet scenes, and those are now development data.
2. **The direct baseline's success is 0 by construction.** `run_direct_scene_episode` never injects `mean_pose_error_deg` (`scripts/eval_scene.py:105-118`). Success then reads a missing value (`src/agentic_sfm/eval/scene_eval.py:818-829`). Relative AUC is still computed.
3. **The oracle is not an upper bound.** It matches only `top_pairs[:5]` (`src/agentic_sfm/rl/scene_episode.py:873`). Five pairs cannot register a 20-image scene. Eight of ten oracle episodes logged a tool error.
4. **The first-call validity bonus dominates reward.** It is 0.5 per episode (`configs/phase2_scene.yaml:66`). On scene 0015 it was 0.5 of a 1.17 total reward.
5. **Gating zeroes geometry after one bad first call.** `gate_geometry_on_validity: true` (`configs/phase2_scene.yaml:68`) still allows reward hacking through the first call.
6. **Agent and baseline pipelines differ.** Agent-side `sfm_run` imports only cached pairs and disables SIFT. A vanilla COLMAP baseline would measure a different backend, not a different policy.
7. **Registered-only pose AUC rewards abstention.** Registering three easy pairs can score perfect AUC.

## 3. Proposed evaluation

### 3.1 Held-out split

- Use ScanNet official test scenes `scene0715_00` through `scene0806_00`. Scenes 0707 to 0714 are excluded because job 8085187 has already used them. The official split file starts at `scene0707_00` and ends at `scene0806_00` (ScanNet repo `Tasks/Benchmark/scannetv2_test.txt`).
- Target at least 30 scenes, with 20 frames each.
- Write the manifest, with a content hash, before any new run.
- Exclude MegaDepth 0015 and 0022 from every held-out claim.
- ScanNet data requires signing the ScanNet Terms of Use before download.

### 3.2 Metrics

- Registration fraction, counting unregistered images as failures.
- Relative pose AUC@5/10/20, computed over all image pairs, not only registered ones.
- Component count and a failure taxonomy: no registration, split into components, and wrong pose.
- IMC-style relative-pose mAA, for comparability with published results. The IMC benchmark README defines mAA from relative-pose angular error (worst of rotation and translation-direction). It does not use camera centers after alignment.
- Keep the 50% and 20-degree gate only as a secondary summary.

### 3.3 Baselines, all through the same `sfm_run` path

1. Consecutive-frame pairs at equal call budget. This is the key falsifier. ScanNet frames are sequential video.
2. Random pairs at equal budget.
3. Top-k retrieval, with k = 5 and 10.
4. Direct baseline with pose error computed.
5. Oracle over GT-overlap pairs at equal budget. On ScanNet, overlap is a pose-similarity proxy, so this is heuristic there.
6. SFT-only checkpoint.
7. Optional external reference: GLUEMAP (`github.com/colmap/gluemap`, README verified).

### 3.4 Statistics

- Treat the scene as the unit. Use hierarchical or scene-level bootstrap intervals, not per-pair bootstraps, which underestimate between-scene variance.
- Report paired differences against each baseline.
- Claim superiority only if the interval excludes zero. Fifteen scenes is too few for that claim. Aim for at least 30.

### 3.5 Gates before any new RL run

- G1: the fair oracle reconstructs at least as well as exhaustive matching on the held-out set.
- G2: the direct baseline reports pose error.
- G3: at least 30 frozen held-out scenes.
- G4: the SFT-only checkpoint is evaluated.
- G5: the policy beats the consecutive-frame heuristic at equal budget, with a scene-level interval excluding zero.

## 4. Proposed training environment

- **Training pool:** MegaDepth scenes with covisibility from the npz overlap matrices. Only 0015 and 0022 currently have images on disk, so new scenes require downloads. Do not use ScanNet for the difficulty curriculum, because its overlap is a pose proxy (`scripts/run_scene_grpo.py:418-423`).
- **Match cache:** precompute exhaustive LoFTR matches for each sampled scene into one `.npz` per scene. Tool calls query the cache, and COLMAP still runs live. The SPAgent repo (`github.com/zhangzaibin/spagent`) uses an offline Pi3X cache for GRPO with multi-turn tool calling; its README was verified.
- **Reward:** pose AUC plus registered fraction, gated on a valid SfM run. Remove the first-call validity bonus. Penalise only calls above a budget and repeated identical calls.
- **Negative control:** shuffled-GT reward. It must not improve held-out metrics.
- **Trainer:** keep the custom GRPO trainer. Migrating to veRL adds risk without evident benefit at this stage. A veRL migration can be considered after the gates pass.
- **Files:** each episode creates one `sfm_run_*` directory per call. Use `$TMP_SHARED` for transient outputs. The file-count quota is 1.5M, and 1.19M are used.

### 4.1 Pose estimation model

Add a feed-forward pose model as a **tool the agent calls**, not as part of the policy input. Three distinct uses must not be conflated:

1. GT pose as policy input: never. The policy would see the target it is scored on.
2. Estimated pose as a tool output: yes. The agent gets relative pose plus confidence or inlier statistics and decides the next call.
3. Pose/intrinsics as conditioning on a reconstruction model: optional, only when those quantities are actually known (e.g. EXIF intrinsics).

Candidate models (paper numbers, not reproduced on our data):

| Model | What it gives | Verified | Concern |
|---|---|---|---|
| VGGT (CVPR 2025, arXiv 2503.11651) | Cameras, depth, point maps in one pass, ~0.2 s | Yes (paper Table 1: AUC@30 85.3 Re10K, 88.2 CO3Dv2; 91.8 with BA) | Exported poses lose 1–2 dB PSNR vs COLMAP without BA (repo issue #79); license unchecked |
| Pi3X (ICLR 2026, arXiv 2507.13347) | Permutation-equivariant poses; conditional injection of pose/intrinsics/depth; ~metric scale | Repo README only | Newer; benchmarks not cross-checked |
| MapAnything (3DV 2026, arXiv 2509.13414) | Any mix of images + intrinsics/poses/depth → metric geometry | Repo README only | Heavier; conditioning matters only with known inputs |
| MASt3R / MASt3R-SfM (arXiv 2409.19152) | Already in our tool server; sparse global alignment, own retrieval | Yes (in repo) | Small/medium scenes; it is the existing baseline |
| COLMAP / GLOMAP | Classical reference + BA refinement | Yes | Slow; failures are the motivation for this project |

Proposed wiring:

- **Pose-prior tool:** run VGGT (Pi3X as the alternative) on the candidate subset; return pairwise relative pose, confidence/inlier stats, and cross-pair consistency. Never return GT.
- **Decision use:** the agent reads the stats to pick the next action (re-match, bridge, stop).
- **Final refinement:** COLMAP/GLOMAP BA on the chosen pairs — feed-forward poses alone are not accurate enough to stand in for BA.
- **Baselines:** MASt3R-SfM and the same cached-match `sfm_run` path on the frozen split, equal budget.

Selection protocol: before wiring either model in, run VGGT and MASt3R on the frozen held-out pairs and compare AUC@5/10/20 against COLMAP. Pick whichever wins as the pose-prior tool; keep the other as an arm in the evaluation. Both go through the same held-out evaluation as the policy — otherwise we are comparing an unverified pose prior against our own RL pipeline.

## 5. Prior art

- The closest project is the author's own public write-up (`ryukijano.github.io/work/agentic-structure-from-motion`). Its claims need correction before publication. It states a training set of "1,535 hard pairs from 102 MegaDepth scenes" and a limit of 10 tool calls and 15 turns. The repo shows the pairs come from two scenes (0015: 402, 0022: 1,133), and the scene config uses 20 calls and 30 turns.
- ReasonMatch/DCRL (arXiv 2606.03577) is related, but it does matching only, with no tools or SfM. The paper's specific Base/SFT/DCRL numbers in the strategy doc were not verified and should not be cited until checked.
- GLUEMAP is a feed-forward global SfM baseline and a strong reference point.
- No published RL-trained SfM orchestrator was found. This is not proof of absence. The arXiv skill returned only weak matches for these queries.

## 6. Citation audit results

Verified against sources:
- DAPO (arXiv 2503.14476): NeurIPS 2025, all four techniques.
- S-GRPO (arXiv 2604.16557): CGI described as in the strategy doc.
- PyVision-RL (arXiv 2602.20739): accumulative reward formula matches.
- LiteSearch-VL (arXiv 2608.29357), NTEP (arXiv 2609.03493), XDG (arXiv 2608.29733), Glob3R (arXiv 2607.09225), Reasmory (arXiv 2606.00963): descriptions match abstracts.
- ScanNet official test split boundaries.

Not verified or incorrect:
- ReasonMatch/DCRL specific numbers: unsupported.
- Visual-ARFT: "ICCV 2025" and "20 annotated samples" appear to belong to Visual-RFT.
- Doppelgangers++ CVPR 2025 publication: arXiv ID is real, venue not confirmed.
- MegaDepth-1500 uses scenes 0015 and 0022: plausible, not confirmed here.
- "102 MegaDepth scenes" in the public project page: incorrect for the repo data.

## 7. Open decisions

1. Approve the ScanNet Terms of Use signature, if training scenes from ScanNet are wanted.
2. Approve the scale of MegaDepth image downloads for new train and eval scenes.
3. Approve a first step: write the frozen manifest and add the consecutive-frame and random-pair baselines before any GPU work.

## 8. Implementation status (pose prior)

Artifacts landed for section 4.1, none of them GPU-verified yet.

- **`src/agentic_sfm/pose_prior.py`** — point-map model outputs → pairwise correspondences and relative poses. Pure numpy so it tests without torch or weights. `reciprocal_nn_matches` (reciprocal nearest neighbours in 3D point-map space, confidence-gated, deterministic, explicit tie-break), `extrinsics_to_relative_pose` (world-to-camera → A→B relative pose, unit-norm `t`, the same convention as `estimate_relative_pose`), `relative_pose_angular_error` (IMC-style rotation + translation-direction error).
- **`tools_server/server.py`** — `matcher="vggt"` backend. One letterboxed forward per pair (both views on a common 518² canvas, so the two point maps share one frame), exact per-image inverse mapping for keypoints, pad-region correspondences filtered before they reach RANSAC or the match cache, confidence threshold 3.0 (VGGT's `world_points_conf` is `1 + exp(raw)` in `[1, ∞)`, not a probability — 5.0 mirrors VGGT's own demo default), and a LoFTR fallback tagged `matcher_fallback: "vggt->loftr"`. Returns the `_match_mast3r` contract plus `prior_pose` (the model's own extrinsics head) and `prior_confidence`.
- **`scripts/freeze_scene_manifest.py` + `data/frozen_eval_manifest.json`** — 30 scenes from `scene0715_00` onward (`scene0718_00` excluded: 19 frames, below the 20-frame criterion). sha256 `d62928353a2c3bff7ac0aa77b0f8b844059dc8035d23bd336228036a56bd0825`, recomputable with `--verify`.
- **`scripts/bench_pose_priors.py` + `jobs/bench_pose_priors.slurm`** — benchmarks `loftr` / `vggt` / `mast3r` on identical per-scene pair lists (19 consecutive + 20 seeded-random), GT intrinsics from `intrinsic_color.txt`, IMC-style worst-of-two error, AUC@5/10/20. Failures and fallbacks count as failures; the headline AUC excludes fallback pairs and a second "as-shipped" column includes them; a separate `prior_auc` scores VGGT's own pose head. `--dry-run` validated on `scene0715_00` (20 frames, 39 pairs). **Not submitted.**
- **Tests** — 17 for `pose_prior` (synthetic ground truth, independent projection path, exact tie-break, NaN masking, degenerate inputs), 13 for the VGGT backend (GT pose agreement within 1°, pad filtering, single-forward assertion, fallback paths), plus the existing suites. Full suite: 319 passed, 1 pre-existing failure (`test_scene_smoke.py::scene_images_encoded`, fails identically at HEAD without these changes).

### What is verified and what is not

Verified: the module-level math against synthetic ground truth; the A→B pose convention against `cv2.recoverPose` semantics; the manifest hash; the harness plumbing (dry run).

Not verified: any real VGGT output — no weights have been downloaded and no GPU job has run, so the accuracy claims in section 4.1 are still paper numbers. Also unverified: MASt3R is not installed in the environment, so the bench will record it as unavailable unless it is added first.

### D4RT

D4RT (CVPR 2026, arXiv 2512.08924) reports strong pose numbers (Re10K AUC@30 83.5, ScanNet ATE 0.014) and is a watch item, but it is not usable here yet: no official code or weights are released (the project page links only paper and blog; only unofficial reimplementations exist, unverified), it is video-oriented with no unordered sparse-view benchmark, and it does not accept intrinsics or poses as conditioning. It is excluded from the candidate table in 4.1.

### License constraint

VGGT's original weights and Pi3/Pi3X weights are non-commercial; MASt3R is CC BY-NC-SA. Fine for research publication. If commercial use is ever needed, MapAnything's Apache-2.0 weights are the alternative and the choice must be revisited.

### Next gate

Submitting `jobs/bench_pose_priors.slurm` requires approval: it downloads VGGT weights to `$SCRATCH` and runs on one L40S for up to 4 hours. Its result decides which prior is wired into the policy (Phase 2).
