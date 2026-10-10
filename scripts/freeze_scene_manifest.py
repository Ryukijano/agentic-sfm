#!/usr/bin/env python
"""Freeze the held-out ScanNet evaluation manifest.

Scans the extracted ``scannet_test_1500`` tree and writes a content-hashed
manifest of scene ids satisfying:

  - scene id >= ``--min-scene`` (lexicographic, default ``scene0715_00`` —
    scenes 0707-0714 were consumed by the dev eval, job 8085187);
  - at least ``--min-images`` color frames (``.jpg``) that each have a
    readable per-frame ``pose/<frame>.txt`` (ScanNet exports camera-to-world
    per frame, not a single ``pose/pose.txt``);
  - a readable ``intrinsic/intrinsic_color.txt``.

Scenes are sorted lexicographically and the first ``--num-scenes`` are kept.
The manifest records the selection criteria and a sha256 over the canonical
JSON of every field except the hash itself, so ``--verify`` can detect any
later edit.

Usage:
    python scripts/freeze_scene_manifest.py
    python scripts/freeze_scene_manifest.py --verify
    python scripts/freeze_scene_manifest.py --num-scenes 30 --out data/frozen_eval_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

DEFAULT_SCANNET_ROOT = (
    "/scratch/kcwp264/data/megadepth/scannet_test_1500/scannet_test_1500"
)
DEFAULT_OUT = _REPO / "data" / "frozen_eval_manifest.json"


def _resolve_root(root: Path) -> Path:
    """Tolerate being pointed one level above the nested scannet_test_1500 dir."""
    root = root.expanduser()
    if not list(root.glob("scene*_*")) and (root / "scannet_test_1500").is_dir():
        root = root / "scannet_test_1500"
    return root


def _readable(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            f.read(1)
        return True
    except OSError:
        return False


def scan_scenes(
    scannet_root: str | Path,
    min_scene: str = "scene0715_00",
    min_images: int = 20,
) -> list[str]:
    """Return sorted scene ids satisfying the freeze criteria."""
    root = _resolve_root(Path(scannet_root))
    if not root.is_dir():
        raise FileNotFoundError(f"ScanNet root not found: {root}")

    out: list[str] = []
    for scene_dir in sorted(root.glob("scene*_*")):
        if not scene_dir.is_dir():
            continue
        scene_id = scene_dir.name
        if scene_id < min_scene:
            continue

        color_dir = scene_dir / "color"
        pose_dir = scene_dir / "pose"
        intrinsic_file = scene_dir / "intrinsic" / "intrinsic_color.txt"
        if not color_dir.is_dir() or not pose_dir.is_dir():
            continue
        if not _readable(intrinsic_file):
            continue

        # Count color frames that also have a readable per-frame pose file,
        # matching the frame-validity convention in
        # scripts/run_scene_grpo.py::_load_scannet_scene.
        n_valid = 0
        for img in color_dir.glob("*.jpg"):
            if not img.stem.isdigit():
                continue
            pose_f = pose_dir / f"{img.stem}.txt"
            if pose_f.exists() and _readable(pose_f):
                n_valid += 1
        if n_valid < min_images:
            continue
        out.append(scene_id)
    return sorted(out)


def _canonical_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_manifest(
    scannet_root: str | Path,
    min_scene: str,
    min_images: int,
    num_scenes: int,
) -> dict:
    scene_ids = scan_scenes(scannet_root, min_scene, min_images)[:num_scenes]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scannet_root": str(_resolve_root(Path(scannet_root))),
        "criteria": {
            "min_scene": min_scene,
            "min_images": min_images,
            "num_scenes": num_scenes,
            "require": [
                "color/*.jpg frames with matching readable pose/<frame>.txt",
                "readable intrinsic/intrinsic_color.txt",
                "lexicographic sort, first num_scenes kept",
            ],
        },
        "scene_ids": scene_ids,
    }
    manifest["sha256"] = _canonical_hash(manifest)
    return manifest


def verify_manifest(path: str | Path) -> bool:
    manifest = json.loads(Path(path).read_text())
    claimed = manifest.get("sha256")
    body = {k: v for k, v in manifest.items() if k != "sha256"}
    actual = _canonical_hash(body)
    return claimed == actual


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scannet-root", default=DEFAULT_SCANNET_ROOT)
    ap.add_argument("--min-scene", default="scene0715_00")
    ap.add_argument("--min-images", type=int, default=20)
    ap.add_argument("--num-scenes", type=int, default=30)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Recompute the manifest sha256 and compare; do not rewrite.",
    )
    args = ap.parse_args()

    if args.verify:
        ok = verify_manifest(args.out)
        print(f"{args.out}: sha256 {'OK' if ok else 'MISMATCH'}")
        return 0 if ok else 1

    manifest = build_manifest(
        args.scannet_root, args.min_scene, args.min_images, args.num_scenes
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote {out}")
    print(f"scannet_root: {manifest['scannet_root']}")
    print(f"num scenes: {len(manifest['scene_ids'])}")
    for sid in manifest["scene_ids"]:
        print(f"  {sid}")
    print(f"sha256: {manifest['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
