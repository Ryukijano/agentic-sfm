"""Regression tests for reward-integrity fixes.

The Phase 1 GRPO run reward-hacked a tolerant /crop endpoint (mixed-unit
bboxes were silently clamped into valid crops). The audit then found the same
bug class everywhere a rejection could still earn reward: silent argument
defaults, crop->full-frame fallback, cross-episode match caches, self-match,
shared SfM output dirs, and fabricated camera matrices.

Contract under test: a tool rejection must remain a rejection all the way
through dispatch and reward accounting.
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Load tools_server/server.py under a distinct module name — tests/
# already binds the name ``server`` to src/agentic_sfm/tools/server.py.
_spec = importlib.util.spec_from_file_location(
    "tools_server_main", ROOT / "tools_server" / "server.py"
)
TS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TS)

from agentic_sfm.agent.policy import execute_sfm_tool  # noqa: E402
from agentic_sfm.geometry import camera_matrix, k_for_image  # noqa: E402


def _img(path, size=(64, 48), seed=0):
    rng = np.random.default_rng(seed)
    arr = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return str(path)


@pytest.fixture
def two_images(tmp_path):
    a = _img(tmp_path / "a.png", seed=1)
    b = _img(tmp_path / "b.png", seed=2)
    TS._IMAGE_STORE.clear()
    TS._IMAGE_STORE["a"] = a
    TS._IMAGE_STORE["b"] = b
    yield a, b
    TS._IMAGE_STORE.clear()
    TS._match_store.clear()


class _RecordingClient:
    """Minimal ToolClient stand-in that records calls."""

    def __init__(self, crop_result=None):
        self.calls = []
        self.crop_result = crop_result or {
            "cropped_image_id": "a_crop", "crop_id": "a_crop"}

    def crop(self, image_id, bbox):
        self.calls.append(("crop", image_id, bbox))
        return self.crop_result

    def match(self, image_a, image_b, matcher="loftr", session="", **kw):
        self.calls.append(("match", image_a, image_b, session))
        return {"num_inliers": 42, "inlier_ratio": 0.5}

    def doppelganger_check(self, image_a, image_b):
        self.calls.append(("doppelganger", image_a, image_b))
        return {"is_doppelganger": False, "confidence": 0.1}

    def sfm_run(self, image_dir, pair_list=None, output_dir="", session=""):
        self.calls.append(("sfm_run", image_dir, session))
        return {"num_registered": 3}


class _TC:
    def __init__(self, tool, args):
        self.tool = tool
        self.args = args


# ---------------------------------------------------------------------------
# Crop validation (tools_server is what the Slurm jobs launch)
# ---------------------------------------------------------------------------

class TestCropValidation:
    def test_all_normalized_accepted(self, two_images):
        r = TS.tool_crop("a", [0.1, 0.1, 0.9, 0.9])
        assert "error" not in r
        assert r["bbox_normalized"] == pytest.approx([0.1, 0.1, 0.9, 0.9],
                                                   abs=0.02)

    def test_all_pixel_accepted(self, two_images):
        r = TS.tool_crop("a", [4, 4, 60, 40])
        assert "error" not in r
        assert r["crop_size"] == [56, 36]

    def test_mixed_units_rejected_with_teaching_error(self, two_images):
        # The exact reward-hacked pattern from the failed run.
        r = TS.tool_crop("a", [0.2, 0.3, 0.8, 937])
        assert "error" in r
        assert "all-normalized" in r["error"].lower()

    def test_degenerate_box_rejected(self, two_images):
        r = TS.tool_crop("a", [0.5, 0.5, 0.5, 0.9])
        assert "error" in r

    def test_wrong_length_rejected(self, two_images):
        assert "error" in TS.tool_crop("a", [0.1, 0.2, 0.9])


# ---------------------------------------------------------------------------
# Match validation
# ---------------------------------------------------------------------------

class TestMatchValidation:
    def test_same_id_rejected(self, two_images):
        r = TS.tool_match("a", "a")
        assert "error" in r

    def test_same_file_different_ids_rejected(self, two_images):
        a, _ = two_images
        TS._IMAGE_STORE["alias"] = a
        r = TS.tool_match("a", "alias")
        assert "error" in r

    def test_unknown_matcher_rejected(self, two_images):
        r = TS.tool_match("a", "b", matcher="magic_matchnet")
        assert "error" in r
        assert "magic_matchnet" in r["error"]

    def test_invalid_K_rejected_before_matching(self, two_images):
        # f <= 1 is physically meaningless for a pinhole camera.
        bad_K = [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]]
        r = TS.tool_match("a", "b", K_a=bad_K)
        assert "error" in r
        assert r["num_inliers"] == 0

    def test_malformed_K_rejected(self, two_images):
        r = TS.tool_match("a", "b", K_a=[[1, 2], [3, 4]])
        assert "error" in r


# ---------------------------------------------------------------------------
# Session-scoped match store
# ---------------------------------------------------------------------------

class TestSessionMatchStore:
    def test_record_goes_to_session_bucket(self):
        TS._match_store.clear()
        rec = {"keypoints_a": [[1, 2]], "keypoints_b": [[3, 4]],
               "inlier_mask": [True]}
        TS._record_agent_match("/x/a.jpg", "/x/b.jpg", rec, session="ep1")
        key = tuple(sorted((os.path.abspath("/x/a.jpg"),
                            os.path.abspath("/x/b.jpg"))))
        assert key in TS._match_store["ep1"]
        assert key not in TS._match_store.get("ep2", {})

    def test_import_only_sees_own_session(self, tmp_path):
        """A sibling rollout must not free-ride on another session's cache."""
        import pycolmap  # noqa: F401 — skip below handles absence

        d = str(tmp_path)
        names = []
        for i in range(2):
            p = os.path.join(d, f"im{i}.jpg")
            _img(p, seed=i)
            names.append(p)
            TS._IMAGE_STORE[f"im{i}"] = p

        n = 30
        ka = (np.random.default_rng(1).random((n, 2)) * 40).tolist()
        kb = (np.random.default_rng(2).random((n, 2)) * 40).tolist()
        rec = {"path_a": names[0], "path_b": names[1],
               "keypoints_a": ka, "keypoints_b": kb,
               "inlier_mask": [True] * n, "matcher": "test"}
        TS._record_agent_match(names[0], names[1], rec, session="other_ep")

        # Session "mine" sees nothing recorded by "other_ep".
        n_imp = TS._import_agent_matches(
            os.path.join(d, "x.db"), d, None, session="mine")
        assert n_imp == 0

        TS._IMAGE_STORE.clear()
        TS._match_store.clear()


# ---------------------------------------------------------------------------
# sfm_run pair_list semantics + unique output dirs
# ---------------------------------------------------------------------------

class TestSfMRun:
    def test_empty_pair_list_is_error_not_baseline(self, tmp_path):
        d = str(tmp_path)
        _img(os.path.join(d, "x.jpg"))
        r = TS.tool_sfm_run(d, pair_list=[], output_dir=str(tmp_path / "o"))
        assert "error" in r
        assert r["status"] == "no_pairs"

    def test_garbage_pair_list_errors_not_exhaustive(self, tmp_path):
        pytest.importorskip("pycolmap")
        d = str(tmp_path)
        for i in range(3):
            _img(os.path.join(d, f"im{i}.jpg"), seed=i)
        r = TS.tool_sfm_run(
            d, pair_list=[["nonexistent_a", "nonexistent_b"]],
            output_dir=str(tmp_path / "o2"), use_agent_matches=False)
        assert "error" in r
        assert r["status"] == "no_pairs"

    def test_empty_output_dir_generates_unique_dirs(self, tmp_path):
        d = str(tmp_path / "imgs")
        os.makedirs(d)
        r1 = TS.tool_sfm_run(d + "_missing", output_dir="")
        r2 = TS.tool_sfm_run(d + "_missing", output_dir="")
        # Both error on the missing dir, but each registered a distinct
        # output dir in _RECON_STORE — no shared ./outputs/sfm_run.
        o1 = TS._RECON_STORE[r1["recon_id"]]["output_dir"]
        o2 = TS._RECON_STORE[r2["recon_id"]]["output_dir"]
        assert o1 != o2


# ---------------------------------------------------------------------------
# Dispatcher: no silent defaults, rejection propagates
# ---------------------------------------------------------------------------

class TestDispatcherIntegrity:
    def test_match_missing_args_errors_no_call(self):
        c = _RecordingClient()
        r = execute_sfm_tool(c, _TC("match", {"image_a": "img_a"}))
        assert "error" in r
        assert c.calls == []

    def test_match_no_silent_defaults(self):
        c = _RecordingClient()
        r = execute_sfm_tool(c, _TC("match", {}))
        assert "error" in r
        assert "missing" in r["error"]

    def test_crop_and_match_propagates_crop_error(self):
        c = _RecordingClient(crop_result={"error": "mixed units"})
        r = execute_sfm_tool(c, _TC("crop_and_match", {
            "image_id": "a", "bbox": [0.2, 0.3, 0.8, 937], "image_b": "b"}))
        assert "error" in r
        # The rejected crop must NOT fall through to a full-frame match.
        assert not any(call[0] == "match" for call in c.calls)

    def test_session_threaded_to_match(self):
        c = _RecordingClient()
        execute_sfm_tool(c, _TC("match", {"image_a": "a", "image_b": "b"}),
                         session="ep-42")
        assert c.calls[0][3] == "ep-42"


# ---------------------------------------------------------------------------
# Geometry: provided-but-invalid K must not silently become a valid guess
# ---------------------------------------------------------------------------

class TestIntrinsics:
    def test_omitted_K_gets_default(self):
        K = camera_matrix((640, 480))
        assert K.shape == (3, 3) and K[0, 0] > 1.0

    def test_invalid_K_raises(self):
        with pytest.raises(ValueError):
            camera_matrix((640, 480), K=[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])

    def test_wrong_shape_K_raises(self):
        with pytest.raises(ValueError):
            camera_matrix((640, 480), K=[[1, 2], [3, 4]])

    def test_k_for_image_validates_eagerly(self):
        with pytest.raises(ValueError):
            k_for_image([[0, 0, 0], [0, 0, 0], [0, 0, 0]], None, (640, 480))


# ---------------------------------------------------------------------------
# Tiny-crop floor: the mode RL drifted into during Phase-1 epoch 2
# ---------------------------------------------------------------------------

class TestMinCropFloor:
    def test_tiny_box_rejected(self, two_images):
        # The exact mode that took over epoch 2 (10% of the image).
        r = TS.tool_crop("a", [0.2, 0.2, 0.3, 0.3])
        assert "error" in r
        assert "too small" in r["error"]

    def test_sft_style_box_still_accepted(self, two_images):
        # Smallest box the SFT data ever emits (min side 0.6).
        r = TS.tool_crop("a", [0.2, 0.3, 0.8, 0.9])
        assert "error" not in r

    def test_oracle_style_box_still_accepted(self, two_images):
        from agentic_sfm.geometry import ORACLE_CROP_BOXES

        for box in ORACLE_CROP_BOXES:
            r = TS.tool_crop("a", list(box))
            assert "error" not in r, (box, r)

    def test_floor_is_below_any_sft_box(self):
        # Guard: the floor must never reject a box the SFT data taught.
        assert TS.MIN_CROP_FRACTION < 0.6


# ---------------------------------------------------------------------------
# Identical-call loop guard
# ---------------------------------------------------------------------------

class TestRepeatBreaker:
    def test_signature_is_order_insensitive_for_args(self):
        from agentic_sfm.agent.policy import tool_call_signature

        a = _TC("match", {"image_a": "x", "image_b": "y"})
        b = _TC("match", {"image_b": "y", "image_a": "x"})
        assert tool_call_signature(a) == tool_call_signature(b)

    def test_signature_distinguishes_different_bboxes(self):
        from agentic_sfm.agent.policy import tool_call_signature

        a = _TC("crop_and_match", {"image_id": "i", "bbox": [0.2, 0.2, 0.3, 0.3]})
        b = _TC("crop_and_match", {"image_id": "i", "bbox": [0.2, 0.3, 0.8, 0.9]})
        assert tool_call_signature(a) != tool_call_signature(b)

    def test_breaker_message_mentions_alternatives(self):
        from agentic_sfm.agent.policy import repeat_breaker_message

        msg = repeat_breaker_message(("crop", "{}"), 4)
        assert "4 times" in msg and "bbox" in msg

    def test_limit_is_small(self):
        from agentic_sfm.agent.policy import MAX_IDENTICAL_TOOL_CALLS

        assert 1 <= MAX_IDENTICAL_TOOL_CALLS < 10
