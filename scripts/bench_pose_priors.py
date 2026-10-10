#!/usr/bin/env python
"""Benchmark pairwise relative-pose accuracy of matcher backends.

Runs every matcher in ``--matchers`` over an IDENTICAL, fixed pair list per
scene from the frozen evaluation manifest
(``data/frozen_eval_manifest.json``, written by
``scripts/freeze_scene_manifest.py``). That fixed-pair-list property is the
fairness requirement from docs/scene_eval_and_env_proposal.md section 3.

Per scene:
  - enumerate ``color/*.jpg`` deterministically (numeric sort), take at most
    ``--max-images`` frames evenly spread across the sequence (same
    even-spacing convention as
    ``scripts/run_scene_grpo.py::_load_scannet_scene``);
  - register them in the tool server's in-process image store;
  - build a FIXED pair list: consecutive pairs (i, i+1) plus
    ``--num-random-pairs`` pairs from a seeded RNG (default seed 0).

Ground truth: ScanNet ``pose/<frame>.txt`` files are camera-to-world; they
are inverted to world-to-camera and passed through
``agentic_sfm.pose_prior.extrinsics_to_relative_pose`` (R_rel = R_b @ R_a.T,
t_rel = t_b - R_rel @ t_a). Estimated poses are scored with
``relative_pose_angular_error``; the pair error is the worst of the
rotation / translation-direction angles (IMC convention).

Aggregation: per matcher, AUC@5/10/20 (fraction of pairs with error <=
threshold; failed/unregistered pairs count as failures and are never
excluded), median/mean error over scored pairs, pair counts, availability
fraction, median runtime.

Outputs ``outputs/bench_pose_priors/run_${SLURM_JOB_ID:-local}/`` with
``bench_pose_priors.json`` and ``summary.md``.

``--dry-run`` validates the manifest, pair construction and GT pose loading
for one scene WITHOUT importing the tool server or any model, and prints
what would run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))  # for `import scripts.freeze_scene_manifest`

from agentic_sfm.pose_prior import (
    extrinsics_to_relative_pose,
    relative_pose_angular_error,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MATCHERS = ["loftr", "vggt", "mast3r"]
AUC_THRESHOLDS_DEG = (5.0, 10.0, 20.0)


# ---------------------------------------------------------------------------
# Manifest + scene loading (conventions mirror scripts/run_scene_grpo.py)
# ---------------------------------------------------------------------------

def load_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    if "scene_ids" not in manifest or "sha256" not in manifest:
        raise ValueError(f"{path} is not a frozen eval manifest")
    from scripts.freeze_scene_manifest import verify_manifest

    if not verify_manifest(path):
        raise ValueError(
            f"{path}: manifest sha256 mismatch — refusing to benchmark a "
            "manifest that differs from what was frozen"
        )
    return manifest


def _read_mat(path: Path) -> np.ndarray:
    return np.loadtxt(str(path), dtype=np.float64)


def select_frames(
    scene_dir: Path, max_images: int
) -> list[tuple[int, Path, Path]]:
    """Evenly-spaced (fid, color_path, pose_path) frames, deterministically.

    Same selection as ``_load_scannet_scene``: numeric sort of color jpgs,
    keep only frames with a pose file, subsample with ``np.linspace`` so the
    subset spreads across the whole sequence.
    """
    color_dir = scene_dir / "color"
    pose_dir = scene_dir / "pose"
    frames: list[tuple[int, Path, Path]] = []
    for img in sorted(color_dir.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1):
        if not img.stem.isdigit():
            continue
        fid = int(img.stem)
        pose_f = pose_dir / f"{fid}.txt"
        if not pose_f.exists():
            continue
        frames.append((fid, img, pose_f))

    n = len(frames)
    k = min(n, max(1, int(max_images)))
    indices = sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))
    return [frames[i] for i in indices]


def build_pair_list(
    num_images: int, num_random_pairs: int, seed: int = 0
) -> list[tuple[int, int]]:
    """Fixed pair list: consecutive (i, i+1) + seeded random pairs."""
    pairs: list[tuple[int, int]] = [(i, i + 1) for i in range(num_images - 1)]
    seen = set(pairs)
    rng = np.random.default_rng(seed)
    attempts = 0
    while len(pairs) < num_images - 1 + num_random_pairs and attempts < 10000:
        attempts += 1
        i, j = sorted(rng.integers(0, num_images, size=2).tolist())
        if i == j or (i, j) in seen:
            continue
        seen.add((i, j))
        pairs.append((i, j))
    return pairs


def load_scene_intrinsics(scene_dir: Path) -> tuple[list | None, str]:
    """Colour-camera K (3x3 list) from ``intrinsic/intrinsic_color.txt``.

    Same convention as ``_load_scannet_scene``: the file is 4x4 and the
    top-left 3x3 is the colour intrinsics. Returns ``(None, "heuristic")``
    when unreadable so ``tool_match`` falls back to its f=1.2*max(w,h) guess.
    """
    path = scene_dir / "intrinsic" / "intrinsic_color.txt"
    try:
        k = _read_mat(path)
        if k.shape != (4, 4):
            raise ValueError(f"unexpected shape {k.shape}")
        return k[:3, :3].tolist(), "gt_intrinsic_color.txt"
    except Exception as e:
        logger.warning("[%s] unreadable intrinsics (%s); using heuristic K",
                       scene_dir.name, e)
        return None, "heuristic"


def gt_relative_poses(
    frames: list[tuple[int, Path, Path]], pairs: list[tuple[int, int]]
) -> dict[tuple[int, int], dict[str, list]]:
    """GT relative pose per pair.

    ScanNet pose files are camera-to-world; invert to world-to-camera before
    ``extrinsics_to_relative_pose``.
    """
    w2c = [np.linalg.inv(_read_mat(f[2]))[:3, :] for f in frames]
    return {
        (i, j): extrinsics_to_relative_pose(w2c[i], w2c[j]) for (i, j) in pairs
    }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import tools_server.server as server
    from tqdm import tqdm

    manifest = load_manifest(args.manifest)
    scannet_root = Path(manifest["scannet_root"])
    scene_ids = manifest["scene_ids"]

    run_id = os.environ.get("SLURM_JOB_ID", "local")
    out_dir = Path(args.output_dir) / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []

    for scene_id in tqdm(scene_ids, desc="scenes"):
        scene_dir = scannet_root / scene_id
        frames = select_frames(scene_dir, args.max_images)
        if len(frames) < 2:
            logger.warning("[%s] fewer than 2 usable frames; skipped", scene_id)
            continue

        image_ids: list[str] = []
        for idx, (_fid, img_path, _pose_path) in enumerate(frames):
            image_id = f"{scene_id}_img_{idx:04d}"
            server.register_image(image_id, str(img_path))
            image_ids.append(image_id)

        k_list, k_source = load_scene_intrinsics(scene_dir)

        pairs = build_pair_list(len(frames), args.num_random_pairs, seed=args.seed)
        gt_poses = gt_relative_poses(frames, pairs)

        for matcher in args.matchers:
            # Per-matcher session so _record_agent_match cache entries do
            # not overwrite each other across backends.
            session = f"bench_{run_id}_{scene_id}:{matcher}"
            for (i, j) in pairs:
                rec: dict[str, Any] = {
                    "scene_id": scene_id,
                    "matcher": matcher,
                    "pair": [i, j],
                    "frame_ids": [frames[i][0], frames[j][0]],
                    "k_source": k_source,
                    "error_deg": None,
                    "rot_err_deg": None,
                    "trans_err_deg": None,
                    "prior_error_deg": None,
                    "prior_rot_err_deg": None,
                    "prior_trans_err_deg": None,
                    "num_matches": None,
                    "num_inliers": None,
                    "runtime_s": None,
                    "failed": False,
                    "fallback": False,
                    "fallback_to": None,
                    "tool_error": None,
                }
                t0 = time.perf_counter()
                try:
                    res = server.tool_match(
                        image_ids[i], image_ids[j],
                        matcher=matcher, session=session,
                        K_a=k_list, K_b=k_list,
                    )
                    rec["runtime_s"] = time.perf_counter() - t0

                    if not isinstance(res, dict) or res.get("error"):
                        rec["failed"] = True
                        rec["tool_error"] = (
                            res.get("error") if isinstance(res, dict) else str(res)
                        )
                    else:
                        rec["num_matches"] = res.get("num_matches")
                        rec["num_inliers"] = res.get("num_inliers")
                        rec["fallback"] = bool(res.get("matcher_fallback"))
                        rec["fallback_to"] = res.get("matcher_fallback")
                        gt = gt_poses[(i, j)]
                        rec.update(
                            _score_pose(res.get("pose"), gt, prefix="")
                        )
                        # Feed-forward priors additionally return the model's
                        # own extrinsics head (`prior_pose`); score it too so
                        # the prior is what is actually being compared.
                        rec.update(
                            _score_pose(res.get("prior_pose"), gt, prefix="prior_")
                        )
                        if rec["error_deg"] is None:
                            rec["failed"] = True
                            rec["tool_error"] = "no pose returned"
                        elif not np.isfinite(rec["error_deg"]):
                            rec["failed"] = True
                            rec["tool_error"] = (
                                "undefined translation direction (NaN t error)"
                            )
                except Exception as e:  # model import/load/scoring failure
                    rec["runtime_s"] = time.perf_counter() - t0
                    rec["failed"] = True
                    rec["tool_error"] = f"{type(e).__name__}: {e}"
                all_records.append(rec)

    results = {
        "run_id": run_id,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": manifest.get("sha256"),
        "scannet_root": str(scannet_root),
        "matchers": args.matchers,
        "max_images": args.max_images,
        "num_random_pairs": args.num_random_pairs,
        "seed": args.seed,
        "aggregate": aggregate(all_records, args.matchers),
        "records": all_records,
    }

    json_path = out_dir / "bench_pose_priors.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    summary_path = out_dir / "summary.md"
    summary_path.write_text(render_summary(results))
    logger.info("Wrote %s and %s", json_path, summary_path)
    return results


def _score_pose(
    pose: Any, gt: dict[str, list], prefix: str = ""
) -> dict[str, Any]:
    """Score one pose dict against GT. Never raises on malformed input.

    A NaN/undefined translation-direction error means the pair FAILED — it
    must not silently pass on rotation alone.
    """
    out = {
        f"{prefix}error_deg": None,
        f"{prefix}rot_err_deg": None,
        f"{prefix}trans_err_deg": None,
    }
    if not isinstance(pose, dict):
        return out
    try:
        r = np.asarray(pose.get("R"), dtype=np.float64).reshape(3, 3)
        t = np.asarray(pose.get("t"), dtype=np.float64).reshape(3)
    except (ValueError, TypeError):
        return out  # malformed-but-truthy pose -> treat as no pose
    r_err, t_err = relative_pose_angular_error(gt["R"], gt["t"], r, t)
    if np.isnan(t_err):
        # undefined translation direction (e.g. zero baseline) -> failure
        out[f"{prefix}error_deg"] = float("inf")
        out[f"{prefix}rot_err_deg"] = r_err
        return out
    out[f"{prefix}rot_err_deg"] = r_err
    out[f"{prefix}trans_err_deg"] = t_err
    out[f"{prefix}error_deg"] = float(max(r_err, t_err))
    return out


def aggregate(
    records: list[dict[str, Any]], matchers: list[str]
) -> dict[str, Any]:
    """Per-matcher stats. Failed pairs count as failures for AUC — never dropped."""
    out: dict[str, Any] = {}
    for matcher in matchers:
        recs = [r for r in records if r["matcher"] == matcher]
        n_total = len(recs)
        n_failed = sum(1 for r in recs if r["failed"])
        n_fallback = sum(1 for r in recs if r["fallback"])
        real = [r for r in recs if not r["fallback"]]
        n_real = len(real)
        n_real_ok = sum(1 for r in real if not r["failed"])

        def _errs(rs):
            return np.array(
                [r["error_deg"] if not r["failed"] else np.inf for r in rs],
                dtype=np.float64,
            )

        def _auc(rs):
            errs = _errs(rs)
            return {
                f"auc@{int(t)}deg": float(np.mean(errs <= t)) if len(rs) else None
                for t in AUC_THRESHOLDS_DEG
            }

        # HEADLINE: non-fallback pairs only — fallback results are the other
        # matcher's numbers, not this one's.
        real_errs = _errs(real)
        real_scored = real_errs[np.isfinite(real_errs)]

        # Prior pose: only over pairs that actually returned one.
        prior_errs = np.array(
            [r["prior_error_deg"] for r in recs
             if r["prior_error_deg"] is not None],
            dtype=np.float64,
        )
        prior_scored = prior_errs[np.isfinite(prior_errs)]

        fallback_targets = sorted(
            {r["fallback_to"] for r in recs if r["fallback_to"]}
        )
        invalid = n_total > 0 and n_real == 0

        runtimes = [r["runtime_s"] for r in recs if r["runtime_s"] is not None]
        out[matcher] = {
            "num_pairs": n_total,
            "num_failed": n_failed,
            "num_fallback": n_fallback,
            "fallback_frac": n_fallback / n_total if n_total else 0.0,
            # availability = fraction served by the REQUESTED matcher without
            # error; fallbacks do not count as available.
            "availability_frac": n_real_ok / n_total if n_total else 0.0,
            "auc": _auc(real),
            "auc_as_shipped": _auc(recs),
            "prior_auc": (
                {
                    f"auc@{int(t)}deg": float(np.mean(prior_errs <= t))
                    for t in AUC_THRESHOLDS_DEG
                }
                if prior_errs.size else None
            ),
            "prior_median_error_deg": (
                float(np.median(prior_scored)) if prior_scored.size else None
            ),
            "prior_num_pairs": int(prior_errs.size),
            "median_error_deg": float(np.median(real_scored)) if real_scored.size else None,
            "mean_error_deg": float(np.mean(real_scored)) if real_scored.size else None,
            "median_runtime_s": float(np.median(runtimes)) if runtimes else None,
            "invalid": invalid,
            "invalid_reason": (
                f"all {n_total} pairs fell back to "
                f"{', '.join(fallback_targets) or 'unknown'}"
                if invalid else None
            ),
        }
    return out


def render_summary(results: dict[str, Any]) -> str:
    lines = [
        "# Pose-prior benchmark summary",
        "",
        f"- run_id: `{results['run_id']}`",
        f"- manifest: `{results['manifest']}` (sha256 `{results['manifest_sha256']}`)",
        f"- scenes evaluated via fixed pair lists "
        f"(consecutive + {results['num_random_pairs']} seeded random, "
        f"seed {results['seed']}; max {results['max_images']} images/scene)",
        "- pair error = worst of rotation / translation-direction angular "
        "error (deg); failed pairs count as failures in AUC",
        "- AUC columns are the HEADLINE metric over non-fallback pairs only; "
        "`ship*` columns include fallback pairs (as-shipped behaviour)",
        "- `prior*` columns score the model's own extrinsics-head pose "
        "(`prior_pose`), null for matchers that do not return one",
        "",
        "| matcher | avail | fallback | pairs | AUC@5 | AUC@10 | AUC@20 | ship@5 | ship@10 | ship@20 | prior@5 | prior@10 | prior@20 | prior med | med err | med s/call |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for matcher in results["matchers"]:
        a = results["aggregate"].get(matcher, {})
        if not a:
            continue

        def fmt(v, spec="{:.3f}"):
            return spec.format(v) if v is not None else "—"

        def auc(d, t):
            return fmt(d.get(f"auc@{t}deg")) if d else "—"

        label = matcher
        if a.get("invalid"):
            label = f"{matcher} (INVALID: {a['invalid_reason']})"

        lines.append(
            "| {m} | {av} | {fb} | {np} | {a5} | {a10} | {a20} | {s5} | {s10} | {s20} | {p5} | {p10} | {p20} | {pmed} | {med} | {rt} |".format(
                m=label,
                av=fmt(a.get("availability_frac")),
                fb=fmt(a.get("fallback_frac")),
                np=a.get("num_pairs", 0),
                a5=auc(a.get("auc"), 5),
                a10=auc(a.get("auc"), 10),
                a20=auc(a.get("auc"), 20),
                s5=auc(a.get("auc_as_shipped"), 5),
                s10=auc(a.get("auc_as_shipped"), 10),
                s20=auc(a.get("auc_as_shipped"), 20),
                p5=auc(a.get("prior_auc"), 5),
                p10=auc(a.get("prior_auc"), 10),
                p20=auc(a.get("prior_auc"), 20),
                pmed=fmt(a.get("prior_median_error_deg"), "{:.2f}"),
                med=fmt(a.get("median_error_deg"), "{:.2f}"),
                rt=fmt(a.get("median_runtime_s"), "{:.3f}"),
            )
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dry run — manifest + pair/GT plumbing only, no model imports
# ---------------------------------------------------------------------------

def dry_run(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    scannet_root = Path(manifest["scannet_root"])
    scene_id = manifest["scene_ids"][0]
    scene_dir = scannet_root / scene_id

    print(f"manifest: {Path(args.manifest).resolve()}")
    print(f"manifest sha256: {manifest.get('sha256')}")
    print(f"scenes: {len(manifest['scene_ids'])}")
    print(f"dry-run scene: {scene_id}")

    frames = select_frames(scene_dir, args.max_images)
    print(f"selected {len(frames)} frames "
          f"(fids: {[f[0] for f in frames]})")

    pairs = build_pair_list(len(frames), args.num_random_pairs, seed=args.seed)
    print(f"pair list: {len(pairs)} pairs "
          f"({len(frames) - 1} consecutive + {len(pairs) - (len(frames) - 1)} random, "
          f"seed {args.seed})")

    _k, k_source = load_scene_intrinsics(scene_dir)
    print(f"intrinsics: {k_source}")

    gt = gt_relative_poses(frames, pairs)
    sample = gt[pairs[0]]
    print(f"GT relative pose for pair {pairs[0]}: "
          f"R[0]={np.array(sample['R'])[0].round(4).tolist()}, "
          f"t={np.array(sample['t']).round(4).tolist()}")

    print(f"would run matchers: {args.matchers} "
          f"({len(manifest['scene_ids'])} scenes x {len(pairs)} pairs each)")
    print("dry-run OK: manifest, pair construction and GT poses all valid")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(_REPO / "data" / "frozen_eval_manifest.json"))
    ap.add_argument("--matchers", nargs="+", default=DEFAULT_MATCHERS)
    ap.add_argument("--max-images", type=int, default=20)
    ap.add_argument("--num-random-pairs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default=str(_REPO / "outputs" / "bench_pose_priors"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        return dry_run(args)
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
