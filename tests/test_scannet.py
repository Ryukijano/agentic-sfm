"""Tests for the ScanNet test-set scene source (Phase 2) + dense GT cloud.

Uses a tiny synthetic ``sceneNNNN_MM`` directory (color jpgs, uint16 depth
pngs, pose txts, intrinsic txts) so no external data is required.  A
real-data check against the extracted ``scannet_test_1500`` scene0772_00 is
skipped automatically when the data isn't present.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.eval.viz import (  # noqa: E402
    FIG_BG,
    gt_pointcloud_arrays,
    render_gt_pointcloud,
)
from scripts.run_scene_grpo import SceneDataset, _subsample_scene  # noqa: E402

REAL_ROOT = Path(
    "/scratch/kcwp264/data/megadepth/scannet_test_1500/scannet_test_1500"
)


def _write_mat(path: Path, m: np.ndarray) -> None:
    path.write_text("\n".join(" ".join(f"{v:.6f}" for v in row) for row in m))


def _make_scannet_scene(root: Path, name: str = "scene0001_00",
                        n_frames: int = 8, W: int = 80, H: int = 60,
                        drop_pose: int | None = None) -> Path:
    """Create a synthetic sceneNNNN_MM dir.  Returns the scene dir.

    Depth is 80x60 uint16 (mm), colour is 160x120 (2x), intrinsics follow the
    same 2x relationship as the real ScanNet export; extrinsic_color is
    identity.  Frame i's pose is c2w with a +0.1i X translation.
    """
    scene = root / name
    for sub in ("color", "depth", "pose", "intrinsic"):
        (scene / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    for i in range(n_frames):
        fid = i * 15  # sparse frame ids like the real export (0, 15, 30, ...)
        img = (rng.random((H * 2, W * 2, 3)) * 255).astype(np.uint8)
        Image.fromarray(img).save(scene / "color" / f"{fid}.jpg")
        depth = (2000 + 100 * i) * np.ones((H, W), dtype=np.uint16)
        depth[:5, :5] = 0  # a few invalid pixels
        Image.fromarray(depth).save(scene / "depth" / f"{fid}.png")
        if drop_pose is not None and i == drop_pose:
            continue  # pose missing -> frame must be skipped by the loader
        c2w = np.eye(4)
        c2w[0, 3] = 0.1 * i
        _write_mat(scene / "pose" / f"{fid}.txt", c2w)

    _write_mat(scene / "intrinsic" / "intrinsic_depth.txt",
               np.diag([70.0, 70.0, 1.0, 1.0])
               @ np.eye(4) + np.array([[0, 0, W / 2, 0],
                                       [0, 0, H / 2, 0],
                                       [0, 0, 0, 0], [0, 0, 0, 0]]))
    _write_mat(scene / "intrinsic" / "intrinsic_color.txt",
               np.array([[140.0, 0, W, 0], [0, 140.0, H, 0],
                         [0, 0, 1.0, 0], [0, 0, 0, 1.0]]))
    _write_mat(scene / "intrinsic" / "extrinsic_color.txt", np.eye(4))
    _write_mat(scene / "intrinsic" / "extrinsic_depth.txt", np.eye(4))
    return scene


def _make_root(tmp_path: Path, nested: bool = True) -> Path:
    root = tmp_path / ("scannet_test_1500" if nested else "")
    _make_scannet_scene(root, "scene0001_00", n_frames=8)
    _make_scannet_scene(root, "scene0002_00", n_frames=12)
    return tmp_path  # loader descends into scannet_test_1500/ automatically


# ---------------------------------------------------------------------------
# SceneDataset.from_scannet
# ---------------------------------------------------------------------------

def test_from_scannet_scene_dict_contract(tmp_path):
    root = _make_root(tmp_path)
    ds = SceneDataset.from_scannet(
        str(root), max_images_per_scene=5, min_images_per_scene=3)
    assert len(ds.scenes) == 2
    s = ds.scenes[0]
    assert s["scene_id"] == "scene0001_00"
    assert s["dataset"] == "scannet"
    assert s["num_images"] == 5
    assert len(s["image_paths"]) == 5
    for p in s["image_paths"]:
        assert Path(p).is_absolute() and Path(p).exists()
        assert p.endswith(".jpg")
    # gt poses keyed by str(position in image_paths), all 4x4
    poses = s["gt_recon"]["poses"]
    assert set(poses) == {str(i) for i in range(5)}
    for m in poses.values():
        assert np.asarray(m).shape == (4, 4)
    assert s["gt_recon"]["num_images"] == 5
    # image_indices index into the full valid frame list
    assert s["image_indices"] == sorted(s["image_indices"])
    assert max(s["image_indices"]) < 8
    # overlap proxy: square over the full frame list, unit diagonal, [0,1]
    om = s["overlap_matrix"]
    assert om.shape == (8, 8)
    assert np.allclose(np.diag(om), 1.0)
    assert om.min() >= 0.0 and om.max() <= 1.0
    # ScanNet extras
    assert len(s["depth_paths"]) == 5 and len(s["pose_paths"]) == 5
    assert len(s["frame_ids"]) == 5
    assert np.asarray(s["intrinsics"]).shape == (3, 3)
    assert s["intrinsics"][0][0] == pytest.approx(140.0)
    assert Path(s["scene_dir"]).is_dir()


def test_from_scannet_cam_from_world_conversion(tmp_path):
    """ScanNet pose files are camera-to-world; gt_recon must be inverted."""
    root = _make_root(tmp_path)
    ds = SceneDataset.from_scannet(
        str(root), max_images_per_scene=8, min_images_per_scene=3)
    s = ds.scenes[0]
    gt = np.asarray(s["gt_recon"]["poses"]["0"])
    # frame 0 has c2w = identity, so cam-from-world is identity too
    assert np.allclose(gt, np.eye(4))
    # frame index 1 -> c2w translation +0.1x; cam-from-world t = -R^T t = -0.1x
    gt1 = np.asarray(s["gt_recon"]["poses"]["1"])
    assert gt1[:3, 3] == pytest.approx([-0.1, 0.0, 0.0])
    # round-trip: inv(stored) == raw c2w file
    raw = np.loadtxt(str(Path(s["pose_paths"][1])))
    assert np.allclose(np.linalg.inv(gt1), raw)


def test_from_scannet_filters_and_missing_pose(tmp_path):
    root = tmp_path / "flat"
    _make_scannet_scene(root, "scene0003_00", n_frames=6, drop_pose=2)
    _make_scannet_scene(root, "scene0123_00", n_frames=6)
    ds = SceneDataset.from_scannet(
        str(root), scene_ids=["123"],  # numeric head matches scene0123_00
        max_images_per_scene=10, min_images_per_scene=3)
    assert [s["scene_id"] for s in ds.scenes] == ["scene0123_00"]
    ds2 = SceneDataset.from_scannet(
        str(root), scene_ids=["scene0003_00"],
        max_images_per_scene=10, min_images_per_scene=3)
    # frame with missing pose dropped: 5 valid frames
    assert ds2.scenes[0]["num_images"] == 5


def test_from_scannet_min_images(tmp_path):
    root = tmp_path / "flat"
    _make_scannet_scene(root, "scene0004_00", n_frames=3)
    ds = SceneDataset.from_scannet(str(root), min_images_per_scene=5)
    assert len(ds.scenes) == 0


def test_subsample_scene_keeps_scannet_lists_aligned(tmp_path):
    root = _make_root(tmp_path)
    ds = SceneDataset.from_scannet(
        str(root), max_images_per_scene=8, min_images_per_scene=3)
    s = ds.scenes[0]
    sub = _subsample_scene(s, 4)
    assert sub["num_images"] == 4
    assert len(sub["depth_paths"]) == 4
    assert len(sub["frame_ids"]) == 4
    assert set(sub["gt_recon"]["poses"]) == {"0", "1", "2", "3"}
    for p, d in zip(sub["image_paths"], sub["depth_paths"]):
        assert Path(p).stem == Path(d).stem


# ---------------------------------------------------------------------------
# render_gt_pointcloud / gt_pointcloud_arrays
# ---------------------------------------------------------------------------

def test_render_gt_pointcloud_synthetic(tmp_path):
    root = tmp_path / "flat"
    scene = _make_scannet_scene(root, "scene0005_00", n_frames=6)
    fig = render_gt_pointcloud(
        depth_dir=scene / "depth",
        color_dir=scene / "color",
        pose_dir=scene / "pose",
        intrinsic_path=scene / "intrinsic" / "intrinsic_depth.txt",
        scene_id="scene0005_00",
        max_points=5000,
    )
    import matplotlib.pyplot as plt
    assert fig._gt_point_count > 0
    out = tmp_path / "cloud.png"
    fig.savefig(out, facecolor=FIG_BG)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 10000


def test_gt_pointcloud_arrays_geometry(tmp_path):
    root = tmp_path / "flat"
    scene = _make_scannet_scene(root, "scene0006_00", n_frames=4)
    pts, cols, cams = gt_pointcloud_arrays(
        scene / "depth", scene / "color", scene / "pose",
        scene / "intrinsic" / "intrinsic_depth.txt", max_points=20000)
    assert pts.shape[1] == 3 and cols.shape == pts.shape
    assert np.isfinite(pts).all()
    # depth = 2.0m along camera +Z; with c2w = I + 0.1i*x the clouds sit in
    # front of each camera at z ~ +2.
    assert pts[:, 2].mean() == pytest.approx(2.0, abs=0.2)
    assert cams.shape == (4, 3)
    assert np.allclose(cams[:, 0], [0.0, 0.1, 0.2, 0.3])
    assert cols.min() >= 0.0 and cols.max() <= 1.0


@pytest.mark.skipif(
    not (REAL_ROOT / "scene0772_00" / "color").is_dir(),
    reason="extracted scannet_test_1500 not present",
)
def test_real_scene0772_loader_and_cloud():
    ds = SceneDataset.from_scannet(
        str(REAL_ROOT), scene_ids=["scene0772_00"],
        max_images_per_scene=20, min_images_per_scene=5)
    assert len(ds.scenes) == 1
    s = ds.scenes[0]
    assert s["num_images"] == 20
    assert all(Path(p).exists() for p in s["image_paths"])

    scene = REAL_ROOT / "scene0772_00"
    pts, cols, cams = gt_pointcloud_arrays(
        scene / "depth", scene / "color", scene / "pose",
        scene / "intrinsic" / "intrinsic_depth.txt", max_points=200000)
    assert len(pts) > 10_000
    assert np.isfinite(pts).all()
