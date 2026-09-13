"""Tests for scene-level curriculum in Phase 2 training.

Covers:
  - SceneDataset.from_megadepth ``scenes`` / ``max_images`` filtering
  - SceneDataset.filter (stage-boundary sub-sampling)
  - SceneCurriculum epoch→stage mapping
  - End-to-end stage progression over a simulated epoch loop
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.run_scene_grpo import (
    SceneCurriculum,
    SceneDataset,
    _normalize_scene_ids,
    _subsample_scene,
)


def _write_fake_scene(si_dir: Path, img_root: Path, scene_id: str,
                      n_images: int = 30, existing_frac: float = 1.0):
    """Write a minimal MegaDepth-style scene_info npz + dummy images."""
    si_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    n_existing = max(1, int(n_images * existing_frac))
    for i in range(n_images):
        rel = f"{scene_id}/img_{i:04d}.jpg"
        if i < n_existing:  # only some images exist on disk
            p = img_root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"fake-jpeg")
        paths.append(rel)

    poses = np.empty(n_images, dtype=object)
    intrinsics = np.empty(n_images, dtype=object)
    for i in range(n_images):
        M = np.eye(4)
        M[0, 3] = float(i)  # tag translation with image index for checks
        poses[i] = M
        intrinsics[i] = np.eye(3)
    overlap = np.eye(n_images)
    np.savez(
        si_dir / f"{scene_id}.npz",
        image_paths=np.array(paths),
        poses=poses,
        intrinsics=intrinsics,
        overlap_matrix=overlap,
    )


@pytest.fixture
def megadepth_tmp(tmp_path):
    """3 fake scenes: 0015 (30 imgs), 0022 (25 imgs), 0033 (40 imgs)."""
    si_dir = tmp_path / "scene_info"
    img_root = tmp_path / "images"
    _write_fake_scene(si_dir, img_root, "0015", n_images=30)
    _write_fake_scene(si_dir, img_root, "0022", n_images=25)
    _write_fake_scene(si_dir, img_root, "0033", n_images=40)
    # A dense-split file that must be skipped by the loader
    _write_fake_scene(si_dir, img_root, "0099_0.1_0.3", n_images=10)
    return si_dir, img_root


class TestFromMegadepth:
    def test_scenes_param_filters(self, megadepth_tmp):
        si_dir, img_root = megadepth_tmp
        ds = SceneDataset.from_megadepth(
            str(si_dir), str(img_root), scenes=["0015"], max_images=10)
        assert len(ds.scenes) == 1
        assert ds.scenes[0]["scene_id"] == "0015"

    def test_max_images_param_caps(self, megadepth_tmp):
        si_dir, img_root = megadepth_tmp
        ds = SceneDataset.from_megadepth(
            str(si_dir), str(img_root), scenes=["0015"], max_images=10)
        s = ds.scenes[0]
        assert s["num_images"] == 10
        assert len(s["image_paths"]) == 10
        assert len(s["image_indices"]) == 10
        assert s["gt_recon"]["num_images"] == 10
        assert sorted(s["gt_recon"]["poses"].keys(), key=int) == [
            str(i) for i in range(10)]

    def test_gt_poses_match_sampled_images(self, megadepth_tmp):
        """gt_recon poses are keyed by position in image_paths and must be
        the poses of the actually-sampled npz indices."""
        si_dir, img_root = megadepth_tmp
        ds = SceneDataset.from_megadepth(
            str(si_dir), str(img_root), scenes=["0015"], max_images=10)
        s = ds.scenes[0]
        for pos, orig_idx in enumerate(s["image_indices"]):
            pose = np.asarray(s["gt_recon"]["poses"][str(pos)])
            # _write_fake_scene tags translation x with the original index
            assert pose[0, 3] == pytest.approx(float(orig_idx))

    def test_scene_id_normalization(self, megadepth_tmp):
        si_dir, img_root = megadepth_tmp
        ds = SceneDataset.from_megadepth(
            str(si_dir), str(img_root), scenes=["15"], max_images=5)
        assert [s["scene_id"] for s in ds.scenes] == ["0015"]

    def test_backward_compat_params(self, megadepth_tmp):
        si_dir, img_root = megadepth_tmp
        ds = SceneDataset.from_megadepth(
            str(si_dir), str(img_root),
            scene_ids=["0022"], max_images_per_scene=7)
        assert len(ds.scenes) == 1
        assert ds.scenes[0]["num_images"] == 7

    def test_scenes_none_loads_all(self, megadepth_tmp):
        si_dir, img_root = megadepth_tmp
        ds = SceneDataset.from_megadepth(
            str(si_dir), str(img_root), max_images=5)
        ids = {s["scene_id"] for s in ds.scenes}
        assert ids == {"0015", "0022", "0033"}  # dense-split npz skipped


class TestSceneDatasetFilter:
    def _dataset(self, megadepth_tmp, max_images=20):
        si_dir, img_root = megadepth_tmp
        return SceneDataset.from_megadepth(
            str(si_dir), str(img_root), max_images=max_images)

    def test_filter_scenes(self, megadepth_tmp):
        ds = self._dataset(megadepth_tmp)
        out = ds.filter(scenes=["0015", "0022"])
        assert {s["scene_id"] for s in out.scenes} == {"0015", "0022"}

    def test_filter_scenes_normalized(self, megadepth_tmp):
        ds = self._dataset(megadepth_tmp)
        out = ds.filter(scenes=["15"])
        assert [s["scene_id"] for s in out.scenes] == ["0015"]

    def test_filter_max_images_subsamples(self, megadepth_tmp):
        ds = self._dataset(megadepth_tmp, max_images=20)
        out = ds.filter(scenes=["0015"], max_images=8)
        s = out.scenes[0]
        assert s["num_images"] == 8
        assert len(s["image_paths"]) == 8
        assert len(s["image_indices"]) == 8
        assert s["gt_recon"]["num_images"] == 8
        assert sorted(s["gt_recon"]["poses"].keys(), key=int) == [
            str(i) for i in range(8)]
        # image_indices still index into the original npz arrays
        for pos, idx in enumerate(s["image_indices"]):
            pose = np.asarray(s["gt_recon"]["poses"][str(pos)])
            assert pose[0, 3] == pytest.approx(float(idx))

    def test_filter_deterministic(self, megadepth_tmp):
        ds = self._dataset(megadepth_tmp, max_images=20)
        a = ds.filter(scenes=["0015"], max_images=8).scenes[0]
        b = ds.filter(scenes=["0015"], max_images=8).scenes[0]
        assert a["image_indices"] == b["image_indices"]
        assert a["image_paths"] == b["image_paths"]

    def test_filter_max_images_none_keeps_all(self, megadepth_tmp):
        ds = self._dataset(megadepth_tmp, max_images=12)
        out = ds.filter(max_images=None)
        assert all(s["num_images"] == 12 for s in out.scenes)

    def test_filter_max_images_above_size_is_noop(self, megadepth_tmp):
        ds = self._dataset(megadepth_tmp, max_images=12)
        out = ds.filter(scenes=["0015"], max_images=50)
        assert out.scenes[0]["num_images"] == 12

    def test_filter_no_match_returns_empty(self, megadepth_tmp):
        ds = self._dataset(megadepth_tmp)
        out = ds.filter(scenes=["9999"])
        assert out.scenes == []

    def test_filter_does_not_mutate_original(self, megadepth_tmp):
        ds = self._dataset(megadepth_tmp, max_images=20)
        before = ds.scenes[0]["num_images"]
        ds.filter(max_images=5)
        assert ds.scenes[0]["num_images"] == before


class TestSubsampleScene:
    def test_even_coverage(self):
        scene = {
            "scene_id": "x",
            "image_paths": [f"p{i}" for i in range(20)],
            "num_images": 20,
            "image_indices": list(range(20)),
            "gt_recon": {"num_images": 20,
                         "poses": {str(i): np.eye(4).tolist()
                                   for i in range(20)}},
        }
        out = _subsample_scene(scene, 5)
        assert out["num_images"] == 5
        # Evenly spaced: endpoints included
        assert out["image_indices"][0] == 0
        assert out["image_indices"][-1] == 19

    def test_padding_when_linspace_collapses(self):
        scene = {
            "scene_id": "x",
            "image_paths": [f"p{i}" for i in range(4)],
            "num_images": 4,
            "image_indices": list(range(4)),
            "gt_recon": {"num_images": 4,
                         "poses": {str(i): np.eye(4).tolist()
                                   for i in range(4)}},
        }
        out = _subsample_scene(scene, 3)
        assert out["num_images"] == 3
        assert len(set(out["image_indices"])) == 3


class TestSceneCurriculum:
    STAGES = [
        {"epochs": 10, "max_images": 10, "scenes": ["0015"]},
        {"epochs": 20, "max_images": 20, "scenes": ["0015", "0022"]},
        {"epochs": 20, "max_images": 50, "scenes": None},
    ]

    def test_stage_for_epoch_boundaries(self):
        cur = SceneCurriculum(self.STAGES)
        assert cur.total_epochs == 50
        for e in range(0, 10):
            assert cur.stage_for_epoch(e) == 0
        for e in range(10, 30):
            assert cur.stage_for_epoch(e) == 1
        for e in range(30, 50):
            assert cur.stage_for_epoch(e) == 2

    def test_last_stage_persists_past_end(self):
        cur = SceneCurriculum(self.STAGES)
        assert cur.stage_for_epoch(50) == 2
        assert cur.stage_for_epoch(999) == 2

    def test_describe(self):
        cur = SceneCurriculum(self.STAGES)
        d = cur.describe(0)
        assert "stage 1/3" in d
        assert "epochs 0-9" in d
        assert "0015" in d
        d2 = cur.describe(2)
        assert "scenes=all" in d2

    def test_from_real_config(self):
        cfg_path = Path(__file__).parent.parent / "configs" / "phase2_scene.yaml"
        if not cfg_path.exists():
            pytest.skip("phase2_scene.yaml not present")
        cfg = yaml.safe_load(cfg_path.read_text())
        cur = SceneCurriculum(cfg["curriculum"]["stages"])
        assert len(cur.stages) == 3
        assert cur.stages[0].scenes == ["0015"]
        assert cur.stages[2].scenes is None
        assert cur.total_epochs == 50

    def test_progression_over_epoch_loop(self, megadepth_tmp):
        """Simulate the training-loop filtering over all stage boundaries."""
        si_dir, img_root = megadepth_tmp
        full = SceneDataset.from_megadepth(str(si_dir), str(img_root),
                                           max_images=50)
        cur = SceneCurriculum(self.STAGES)

        active, current = full, -1
        seen = {}
        for epoch in range(50):
            idx = cur.stage_for_epoch(epoch)
            if idx != current:
                current = idx
                st = cur.stages[idx]
                active = full.filter(scenes=st.scenes, max_images=st.max_images)
            seen[epoch] = (
                {s["scene_id"] for s in active.scenes},
                {s["scene_id"]: s["num_images"] for s in active.scenes},
            )

        # Stage 1: only 0015 with 10 images
        ids, sizes = seen[0]
        assert ids == {"0015"} and sizes["0015"] == 10
        assert seen[9] == seen[0]
        # Stage 2: 0015 + 0022 with 20 images each
        ids, sizes = seen[10]
        assert ids == {"0015", "0022"}
        assert sizes["0015"] == 20 and sizes["0022"] == 20
        assert seen[29] == seen[10]
        # Stage 3: all three scenes at full loaded size
        ids, sizes = seen[30]
        assert ids == {"0015", "0022", "0033"}
        assert sizes["0015"] == 30  # 30 on disk < 50 cap
        assert sizes["0022"] == 25
        assert sizes["0033"] == 40
        assert seen[49] == seen[30]


class TestNormalizeSceneIds:
    def test_variants(self):
        out = _normalize_scene_ids(["15", "0022"])
        assert "15" in out and "0015" in out
        assert "0022" in out and "22" in out

    def test_none(self):
        assert _normalize_scene_ids(None) is None
