"""Tests for the agent-matches → COLMAP importer (_import_agent_matches).

Uses synthetic keypoints + a tiny on-disk image dir so no matcher or GPU is
needed — verifies the importer writes keypoints, raw matches, and verified
two-view geometries into the COLMAP database.
"""

import os
import sys
import tempfile

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "agentic_sfm", "tools"))
import server as S  # noqa: E402


def _make_image(path: str, size=(640, 480), seed=0):
    rng = np.random.default_rng(seed)
    Image.fromarray((rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)).save(path)


def _make_dir(tmp, n=3):
    names = []
    for i in range(n):
        p = os.path.join(tmp, f"im{i}.jpg")
        _make_image(p, seed=i)
        names.append(f"im{i}.jpg")
    return names


def _record(a, b, n=30, inliers=20):
    ka = np.random.default_rng(1).random((n, 2)) * 400
    kb = ka + np.random.default_rng(2).random((n, 2)) * 5
    mask = [True] * inliers + [False] * (n - inliers)
    return {
        "path_a": a, "path_b": b,
        "keypoints_a": ka.tolist(), "keypoints_b": kb.tolist(),
        "inlier_mask": mask, "matcher": "test",
    }


@pytest.fixture
def scene(tmp_path):
    d = str(tmp_path)
    names = _make_dir(d, 4)
    S._image_registry.clear()
    S._match_store.clear()
    for i, n in enumerate(names):
        S._image_registry[f"im{i}"] = os.path.join(d, n)
    yield d, names
    S._image_registry.clear()
    S._match_store.clear()


class TestAgentMatchesImport:
    def test_empty_store_returns_zero(self, scene):
        d, _ = scene
        assert S._import_agent_matches(os.path.join(d, "x.db"), d, None) == 0

    def test_imports_pairs_and_keypoints(self, scene):
        d, names = scene
        import pycolmap
        # populate match store for a chain
        for i in range(3):
            pa, pb = S._image_registry[f"im{i}"], S._image_registry[f"im{i+1}"]
            S._match_store[tuple(sorted((pa, pb)))] = _record(pa, pb)
        dbp = os.path.join(d, "m.db")
        n = S._import_agent_matches(dbp, d, None)
        assert n == 3
        db = pycolmap.Database.open(dbp)
        assert db.num_images() == 4
        assert db.num_keypoints() > 0
        assert db.num_matches() > 0
        db.close()

    def test_pair_list_filters_to_requested(self, scene):
        d, names = scene
        for i in range(3):
            pa, pb = S._image_registry[f"im{i}"], S._image_registry[f"im{i+1}"]
            S._match_store[tuple(sorted((pa, pb)))] = _record(pa, pb)
        dbp = os.path.join(d, "m2.db")
        # only request im0-im1 → only that pair imported
        n = S._import_agent_matches(dbp, d, [["im0", "im1"]])
        assert n == 1

    def test_sub8_pairs_not_written(self, scene):
        d, names = scene
        pa, pb = S._image_registry["im0"], S._image_registry["im1"]
        S._match_store[tuple(sorted((pa, pb)))] = _record(pa, pb, n=5, inliers=5)
        assert S._import_agent_matches(os.path.join(d, "m3.db"), d, None) == 0

    def test_record_agent_match_populates_store(self, scene):
        d, _ = scene
        a, b = "/abs/x.jpg", "/abs/y.jpg"
        S._record_agent_match(a, b, {"keypoints_a": [[1, 2]], "keypoints_b": [[3, 4]], "inlier_mask": [True]})
        assert tuple(sorted((a, b))) in S._match_store
        # no keypoints → not cached
        S._match_store.clear()
        S._record_agent_match(a, b, {"num_inliers": 3})
        assert len(S._match_store) == 0


class TestSfMRequestSchema:
    def test_use_agent_matches_field(self):
        req = S.SfMRequest(image_dir="/tmp/x")
        assert req.use_agent_matches is True or req.use_agent_matches is False  # field exists
        req2 = S.SfMRequest(image_dir="/tmp/x", use_agent_matches=True)
        assert req2.use_agent_matches is True
