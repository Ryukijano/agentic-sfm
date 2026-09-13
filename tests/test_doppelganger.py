"""Tests for the DINOv2 + geometry doppelganger detector.

Model-free paths are tested directly (scoring, patch stats, endpoint
plumbing); the heavy DINOv2 forward pass is stubbed or exercised only in
the optional GPU integration test at the bottom.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.tools.doppelganger import (
    DoppelgangerDetector,
    _patch_stats,
)


def _det() -> DoppelgangerDetector:
    # device=None -> CPU; embedder is lazy and never loaded by score().
    return DoppelgangerDetector(device=None)


class TestScoring:
    """Logistic scoring and veto behaviour (no model needed)."""

    def test_clear_doppelganger_flagged(self):
        det = _det()
        conf, terms, geom = det.score(
            similarity_score=0.9, patch_consistency=0.1, patch_coverage=0.6,
            inlier_ratio=0.05, num_matches=350, residual_units=1.2,
        )
        assert conf >= 0.9
        assert terms["geometry"] > 0 and terms["sim"] > 0
        assert geom is not None and geom < 0.2

    def test_true_match_not_flagged(self):
        det = _det()
        conf, _, geom = det.score(
            similarity_score=0.85, patch_consistency=0.7, patch_coverage=0.7,
            inlier_ratio=0.6, num_matches=300, residual_units=0.2,
            num_inliers=250,
        )
        assert geom == pytest.approx(1.0)
        assert conf <= det.weights.strong_verified_cap + 1e-9

    def test_unrelated_pair_low_confidence(self):
        """Low similarity + failed geometry = bad pair, not a doppelganger."""
        det = _det()
        conf, _, _ = det.score(
            similarity_score=0.3, patch_consistency=0.1, patch_coverage=0.2,
            inlier_ratio=0.0, num_matches=200, residual_units=None,
        )
        assert conf <= det.weights.sim_veto_cap + 1e-9

    def test_geometry_veto(self):
        """Strong geometric verification caps confidence regardless of sim."""
        det = _det()
        conf, _, geom = det.score(
            similarity_score=0.95, patch_consistency=0.0, patch_coverage=1.0,
            inlier_ratio=0.5, num_matches=500, residual_units=2.0,
            num_inliers=200,
        )
        assert geom == pytest.approx(1.0)
        assert conf <= det.weights.strong_verified_cap + 1e-9

    def test_moderate_verification_not_flagged(self):
        """80+ inliers at ~20% ratio is a verified pair (COLMAP semantics) —
        high visual similarity must not override it."""
        det = _det()
        conf, _, geom = det.score(
            similarity_score=0.9, patch_consistency=0.35, patch_coverage=0.8,
            inlier_ratio=0.21, num_matches=800, residual_units=0.4,
            num_inliers=146,
        )
        assert geom >= det.weights.verified_score
        assert conf <= det.weights.verified_cap + 1e-9

    def test_similarity_veto(self):
        """Low appearance similarity caps confidence."""
        det = _det()
        conf, _, _ = det.score(
            similarity_score=0.2, patch_consistency=0.0, patch_coverage=0.9,
            inlier_ratio=0.0, num_matches=500, residual_units=2.0,
        )
        assert conf <= det.weights.sim_veto_cap + 1e-9

    def test_inlier_support_scales_verification(self):
        """High inlier ratio on few points is weaker than on many."""
        det = _det()
        c_few, _, _ = det.score(0.85, None, None, 0.5, 200, None, num_inliers=5)
        c_many, _, _ = det.score(0.85, None, None, 0.5, 200, None, num_inliers=200)
        assert c_few > c_many
        # many inliers -> hard veto applies; few -> veto is gated off
        assert c_many <= det.weights.verified_cap + 1e-9
        assert c_few > det.weights.verified_cap

    def test_missing_geometry_capped(self):
        det = _det()
        conf, _, geom = det.score(
            similarity_score=0.95, patch_consistency=0.0, patch_coverage=0.9,
            inlier_ratio=None, num_matches=None, residual_units=None,
        )
        assert geom is None
        assert conf <= det.weights.no_geom_cap + 1e-9
        assert conf > 0.5  # still suspicious on appearance alone

    def test_missing_similarity_capped(self):
        det = _det()
        conf, _, _ = det.score(
            similarity_score=None, patch_consistency=None, patch_coverage=None,
            inlier_ratio=0.0, num_matches=500, residual_units=2.0,
        )
        assert conf <= det.weights.no_sim_cap + 1e-9

    def test_confidence_is_monotonic_in_evidence(self):
        det = _det()
        c_low = det.score(0.7, None, None, 0.30, 100, None)[0]
        c_mid = det.score(0.7, None, None, 0.15, 100, None)[0]
        c_high = det.score(0.7, None, None, 0.02, 100, None)[0]
        assert c_low < c_mid < c_high
        s_low = det.score(0.45, None, None, 0.10, 100, None)[0]
        s_high = det.score(0.85, None, None, 0.10, 100, None)[0]
        assert s_low < s_high


class TestPatchStats:
    def _grid(self, n=16, d=384, seed=0):
        rng = np.random.default_rng(seed)
        p = rng.normal(size=(n * n, d)).astype(np.float32)
        return p / np.linalg.norm(p, axis=1, keepdims=True)

    def test_identical_grids_full_consistency(self):
        pa = self._grid()
        cons, cov = _patch_stats(pa, pa)
        assert cov == pytest.approx(1.0)
        assert cons == pytest.approx(1.0)

    def test_shifted_grid_consistent(self):
        pa = self._grid()
        # Shift every patch by (+2, +1) cells — coherent displacement.
        n = 16
        pb = np.zeros_like(pa)
        for i in range(n * n):
            r, c = divmod(i, n)
            j = ((r + 2) % n) * n + ((c + 1) % n)
            pb[j] = pa[i]
        cons, cov = _patch_stats(pa, pb)
        assert cons > 0.7  # wrapped edge cells disagree with the mode

    def test_shuffled_grid_inconsistent(self):
        pa = self._grid(seed=0)
        rng = np.random.default_rng(1)
        pb = pa[rng.permutation(len(pa))]
        cons, cov = _patch_stats(pa, pb)
        assert cov == pytest.approx(1.0)  # all patches still match somewhere
        assert cons < 0.4                 # but layout is incoherent

    def test_unrelated_low_coverage(self):
        pa = self._grid(seed=0)
        pb = self._grid(seed=99)
        _, cov = _patch_stats(pa, pb)
        assert cov < 0.5

    def test_empty_patches(self):
        cons, cov = _patch_stats(None, self._grid())
        assert cons is None and cov is None


class TestCheck:
    """check() with a stubbed appearance model — verifies the response schema."""

    def _stub_check(self, det, sim, cons, cov, match_result):
        with patch.object(
            det, "_appearance_cached", return_value=(sim, cons, cov, "stub")
        ):
            return det.check("/nonexistent/a.png", "/nonexistent/b.png",
                             match_result=match_result)

    def test_response_schema(self):
        det = _det()
        r = self._stub_check(det, 0.9, 0.1, 0.6, {
            "num_matches": 400, "num_inliers": 15, "inlier_ratio": 0.04,
            "residual_units": 1.1,
        })
        for key in ("is_doppelganger", "confidence", "score",
                    "similarity_score", "inlier_ratio", "num_matches",
                    "num_inliers", "verdict", "method", "threshold"):
            assert key in r, f"missing key {key}"
        assert r["is_doppelganger"] is True
        assert r["confidence"] == r["score"]
        assert r["verdict"] == "doppelganger"
        assert r["similarity_score"] == pytest.approx(0.9)
        assert r["inlier_ratio"] == pytest.approx(0.04)

    def test_verdicts(self):
        det = _det()
        r = self._stub_check(det, 0.85, 0.7, 0.7,
                             {"num_matches": 300, "inlier_ratio": 0.5})
        assert r["verdict"] == "match" and r["is_doppelganger"] is False
        r = self._stub_check(det, 0.2, 0.0, 0.1,
                             {"num_matches": 100, "inlier_ratio": 0.0})
        assert r["verdict"] == "dissimilar" and r["is_doppelganger"] is False
        r = self._stub_check(det, 0.65, 0.3, 0.4,
                             {"num_matches": 100, "inlier_ratio": 0.15})
        assert r["verdict"] in ("uncertain", "doppelganger")

    def test_match_error_still_returns(self):
        det = _det()
        r = self._stub_check(det, 0.9, 0.1, 0.7, {"error": "matcher boom"})
        assert r["inlier_ratio"] is None
        assert r["confidence"] <= det.weights.no_geom_cap + 1e-9
        assert "error" not in r


class TestServerEndpoint:
    """FastAPI plumbing: /doppelganger_check wires detector + match()."""

    def _client(self):
        from fastapi.testclient import TestClient

        from agentic_sfm.tools import server

        return server, TestClient(server.app)

    def test_endpoint_with_stubbed_detector(self, tmp_path):
        from PIL import Image

        server, tc = self._client()
        pa = tmp_path / "a.png"
        pb = tmp_path / "b.png"
        Image.new("RGB", (32, 32), (10, 20, 30)).save(pa)
        Image.new("RGB", (32, 32), (200, 100, 50)).save(pb)
        tc.post("/register_image", json={"image_id": "a", "path": str(pa)})
        tc.post("/register_image", json={"image_id": "b", "path": str(pb)})

        fake_result = {
            "is_doppelganger": True, "confidence": 0.93, "score": 0.93,
            "similarity_score": 0.88, "inlier_ratio": 0.04,
            "num_matches": 300, "num_inliers": 12, "verdict": "doppelganger",
            "method": "dinov2+geometry", "threshold": 0.5,
        }
        fake_det = MagicMock()
        fake_det.check.return_value = fake_result
        fake_match = {"num_matches": 300, "num_inliers": 12,
                      "inlier_ratio": 0.04, "residual_units": 1.2}

        with patch.object(server, "match", return_value=fake_match), \
             patch.object(server, "_get_doppelganger_detector", return_value=fake_det):
            r = tc.post("/doppelganger_check", json={"image_a": "a", "image_b": "b"})
        assert r.status_code == 200
        data = r.json()
        assert data["is_doppelganger"] is True
        assert data["confidence"] == pytest.approx(0.93)
        assert data["similarity_score"] == pytest.approx(0.88)
        assert data["inlier_ratio"] == pytest.approx(0.04)
        fake_det.check.assert_called_once()

    def test_endpoint_missing_image(self):
        server, tc = self._client()
        r = tc.post("/doppelganger_check",
                    json={"image_a": "/no/such/a.png", "image_b": "/no/such/b.png"})
        assert "error" in r.json()

    def test_endpoint_heuristic_fallback(self, tmp_path):
        """Detector exception -> legacy heuristic still answers."""
        from PIL import Image

        server, tc = self._client()
        pa = tmp_path / "a.png"
        pb = tmp_path / "b.png"
        Image.new("RGB", (32, 32), (10, 20, 30)).save(pa)
        Image.new("RGB", (32, 32), (200, 100, 50)).save(pb)
        tc.post("/register_image", json={"image_id": "a", "path": str(pa)})
        tc.post("/register_image", json={"image_id": "b", "path": str(pb)})

        fake_det = MagicMock()
        fake_det.check.side_effect = RuntimeError("boom")
        fake_match = {"num_matches": 300, "num_inliers": 5, "inlier_ratio": 0.02}
        with patch.object(server, "match", return_value=fake_match), \
             patch.object(server, "_get_doppelganger_detector", return_value=fake_det):
            r = tc.post("/doppelganger_check", json={"image_a": "a", "image_b": "b"})
        data = r.json()
        assert data["method"] == "heuristic_fallback"
        assert data["is_doppelganger"] is True
        assert data["confidence"] > 0.5


class TestSceneEpisodeIntegration:
    """scene_episode records checks and filters the sfm pair list."""

    class _Client:
        def __init__(self, flagged_pair=None):
            self.sfm_calls = []
            self.flagged_pair = flagged_pair

        def register_image(self, img_id, path):
            return {"status": "ok"}

        def crop(self, image_id, bbox):
            return {"cropped_image_id": f"{image_id}_crop_1"}

        def match(self, image_a, image_b, matcher="loftr", **kw):
            return {"num_inliers": 80, "num_matches": 150,
                    "inlier_ratio": 0.55, "pose": None}

        def doppelganger_check(self, image_a, image_b):
            is_d = self.flagged_pair == (image_a, image_b) or \
                   self.flagged_pair == (image_b, image_a)
            return {
                "is_doppelganger": is_d,
                "confidence": 0.9 if is_d else 0.05,
                "similarity_score": 0.9,
                "inlier_ratio": 0.03 if is_d else 0.55,
            }

        def sfm_run(self, image_dir, pair_list=None, output_dir="./o"):
            self.sfm_calls.append({"image_dir": image_dir, "pair_list": pair_list})
            return {"num_registered": 3, "num_points3d": 100,
                    "output_dir": "/tmp/recon"}

        def inspect(self, recon_dir):
            return {"num_images": 3}

    class _Agent:
        matcher = "loftr"
        reward_config: dict = {}

        def __init__(self, responses):
            self.responses = list(responses)
            self._i = 0

        def _encode_image(self, path):
            return "b64"

        def _generate_turn(self, messages, images=None):
            if self._i < len(self.responses):
                r = self.responses[self._i]
                self._i += 1
                return r
            return json.dumps({"tool": "done", "args": {}})

    def test_flagged_pair_excluded_from_sfm(self):
        from agentic_sfm.rl.scene_episode import run_scene_episode

        client = self._Client(flagged_pair=("img_0000", "img_0002"))
        agent = self._Agent([
            json.dumps({"tool": "doppelganger_check",
                        "args": {"image_a": "img_0000", "image_b": "img_0002"}}),
            json.dumps({"tool": "sfm_run", "args": {}}),
            json.dumps({"tool": "done", "args": {}}),
        ])
        ep = run_scene_episode(
            agent=agent, scene_id="s", image_paths=["a.jpg", "b.jpg", "c.jpg"],
            tool_client=client, max_tool_calls=10,
        )
        assert ep.done
        # check was recorded under the canonical pair key
        assert "img_0000__img_0002" in ep.doppelganger_checks
        call = client.sfm_calls[0]
        pairs = {tuple(p) for p in call["pair_list"]}
        # flagged pair excluded, others retained
        assert ("img_0000", "img_0002") not in pairs
        assert ("img_0000", "img_0001") in pairs
        assert ("img_0001", "img_0002") in pairs
        # reward bookkeeping fields present
        assert ep.recon_result["num_doppelgangers_present"] == 1
        assert ep.recon_result["num_doppelgangers_filtered"] == 1

    def test_clean_check_keeps_pairs(self):
        from agentic_sfm.rl.scene_episode import run_scene_episode

        client = self._Client(flagged_pair=None)
        agent = self._Agent([
            json.dumps({"tool": "doppelganger_check",
                        "args": {"image_a": "img_0000", "image_b": "img_0001"}}),
            json.dumps({"tool": "sfm_run", "args": {}}),
            json.dumps({"tool": "done", "args": {}}),
        ])
        ep = run_scene_episode(
            agent=agent, scene_id="s", image_paths=["a.jpg", "b.jpg"],
            tool_client=client, max_tool_calls=10,
        )
        pairs = {tuple(p) for p in client.sfm_calls[0]["pair_list"]}
        assert ("img_0000", "img_0001") in pairs
        assert ep.recon_result["num_doppelgangers_present"] == 0

    def test_no_check_preserves_exhaustive(self):
        from agentic_sfm.rl.scene_episode import run_scene_episode

        client = self._Client()
        agent = self._Agent([
            json.dumps({"tool": "sfm_run", "args": {}}),
            json.dumps({"tool": "done", "args": {}}),
        ])
        ep = run_scene_episode(
            agent=agent, scene_id="s", image_paths=["a.jpg", "b.jpg"],
            tool_client=client, max_tool_calls=10,
        )
        assert client.sfm_calls[0]["pair_list"] is None  # old behaviour
        assert ep.recon_result["num_doppelgangers_present"] == 0

    def test_crop_ids_map_to_parent_pair(self):
        from agentic_sfm.rl.scene_episode import _pair_key

        key = _pair_key("img_0001_crop_0_0_10_10", "img_0002")
        assert key == "img_0001__img_0002"


class TestRealDino:
    """Optional GPU integration test — real DINOv2 forward pass."""

    def test_dinov2_embeds_and_scores(self, tmp_path):
        torch = pytest.importorskip("torch")
        pytest.importorskip("transformers")
        if not torch.cuda.is_available():
            pytest.skip("no GPU")
        from PIL import Image

        rng = np.random.default_rng(0)
        img_a = Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
        img_b = Image.fromarray(np.roll(np.asarray(img_a), 30, axis=1))
        pa, pb = str(tmp_path / "a.png"), str(tmp_path / "b.png")
        img_a.save(pa)
        img_b.save(pb)

        det = DoppelgangerDetector(device="cuda")
        r = det.check(pa, pb, match_result={
            "num_matches": 400, "num_inliers": 350, "inlier_ratio": 0.85,
            "residual_units": 0.1,
        })
        assert r["similarity_score"] is not None and r["similarity_score"] > 0.8
        assert r["is_doppelganger"] is False
        assert r["method"] == "dinov2+geometry"
