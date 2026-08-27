"""Tests: heavy matcher weights load once per process, not per match call."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tools_server import server as tool_server


@pytest.fixture(autouse=True)
def _clear_cache():
    tool_server.clear_matcher_cache()
    yield
    tool_server.clear_matcher_cache()


def _rgb(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestMatcherCacheHelper:
    def test_loader_invoked_once_for_same_key(self):
        calls = []

        def loader():
            calls.append(1)
            return "model"

        a = tool_server._get_or_load_matcher("mast3r", "ckpt-a", "cpu", loader)
        b = tool_server._get_or_load_matcher("mast3r", "ckpt-a", "cpu", loader)
        assert a is b is "model"
        assert len(calls) == 1

    def test_different_checkpoint_or_device_reloads(self):
        calls = []

        def loader():
            calls.append(1)
            return object()

        tool_server._get_or_load_matcher("mast3r", "ckpt-a", "cpu", loader)
        tool_server._get_or_load_matcher("mast3r", "ckpt-b", "cpu", loader)
        tool_server._get_or_load_matcher("mast3r", "ckpt-a", "cuda", loader)
        assert len(calls) == 3


class TestMast3rLoadsOnce:
    def test_from_pretrained_not_called_on_second_match(self):
        mock_model = MagicMock(name="mast3r_model")
        load_calls = []

        def fake_from_pretrained(name):
            load_calls.append(name)
            return mock_model

        mock_AsymmetricMASt3R = MagicMock()
        mock_AsymmetricMASt3R.from_pretrained.side_effect = fake_from_pretrained
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        # Minimal inference graph so _match_mast3r can finish without real weights.
        desc = MagicMock()
        desc.squeeze.return_value.detach.return_value = MagicMock()
        fake_output = {
            "view1": {"true_shape": [np.array([64, 64])]},
            "view2": {"true_shape": [np.array([64, 64])]},
            "pred1": {"desc": desc},
            "pred2": {"desc": desc},
        }
        matches = np.array([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=np.float32)

        mast3r_mod = MagicMock()
        mast3r_mod.model.AsymmetricMASt3R = mock_AsymmetricMASt3R
        mast3r_mod.fast_nn.fast_reciprocal_NNs = MagicMock(
            return_value=(matches.copy(), matches.copy())
        )
        dust3r_inf = MagicMock()
        dust3r_inf.inference = MagicMock(return_value=fake_output)
        dust3r_img = MagicMock()
        dust3r_img.load_images = MagicMock(return_value=[{"img": 0}, {"img": 1}])

        with (
            patch.dict(
                sys.modules,
                {
                    "mast3r": mast3r_mod,
                    "mast3r.model": mast3r_mod.model,
                    "mast3r.fast_nn": mast3r_mod.fast_nn,
                    "dust3r": MagicMock(),
                    "dust3r.inference": dust3r_inf,
                    "dust3r.utils": MagicMock(),
                    "dust3r.utils.image": dust3r_img,
                },
            ),
            patch.object(
                tool_server,
                "_estimate_pose_ransac",
                return_value={"num_inliers": 2, "inlier_ratio": 0.5, "pose": None},
            ),
        ):
            r1 = tool_server._match_mast3r(_rgb(), _rgb(), max_size=64)
            r2 = tool_server._match_mast3r(_rgb(), _rgb(), max_size=64)

        assert r1["matcher"] == "mast3r"
        assert r2["matcher"] == "mast3r"
        assert len(load_calls) == 1, (
            f"from_pretrained must run once per process; got {len(load_calls)} calls"
        )
        mock_AsymmetricMASt3R.from_pretrained.assert_called_once()


class TestLoftrLoadsOnce:
    def test_loftr_pretrained_not_called_on_second_match(self):
        load_calls = []

        class FakeLoFTR:
            def __init__(self, pretrained="outdoor"):
                load_calls.append(pretrained)

            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, input_dict):
                n = 4
                return {
                    "keypoints0": __import__("torch").zeros(n, 2),
                    "keypoints1": __import__("torch").zeros(n, 2),
                    "confidence": __import__("torch").ones(n),
                }

        fake_kf = MagicMock()
        fake_kf.LoFTR = FakeLoFTR
        fake_kornia = MagicMock()
        fake_kornia.feature = fake_kf

        with (
            patch.dict(sys.modules, {"kornia": fake_kornia, "kornia.feature": fake_kf}),
            patch.object(
                tool_server,
                "_estimate_pose_ransac",
                return_value={"num_inliers": 1, "inlier_ratio": 0.25, "pose": None},
            ),
        ):
            # Gray images via RGB input path inside _match_loftr
            r1 = tool_server._match_loftr(_rgb(), _rgb(), max_size=64)
            r2 = tool_server._match_loftr(_rgb(), _rgb(), max_size=64)

        assert r1["matcher"] == "loftr"
        assert r2["matcher"] == "loftr"
        assert len(load_calls) == 1, (
            f"LoFTR(pretrained=...) must run once per process; got {len(load_calls)}"
        )


class TestLightglueLoadsOnce:
    def test_lightglue_not_rebuilt_on_second_match(self):
        lg_calls = []
        sp_calls = []

        class FakeLG:
            def __init__(self, *args, **kwargs):
                lg_calls.append(1)

            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, *args, **kwargs):
                import torch

                return (torch.tensor([0, 1]), torch.tensor([0, 1]))

        class FakeSP:
            def __init__(self, *args, **kwargs):
                sp_calls.append(1)

            def to(self, device):
                return self

            def eval(self):
                return self

            def __call__(self, t):
                import torch

                return {
                    "descriptors": torch.zeros(1, 2, 256),
                    "keypoints": torch.zeros(1, 2, 2),
                }

        fake_kf = MagicMock()
        fake_kf.LightGlueMatcher = FakeLG
        fake_kf.SuperPoint = FakeSP
        fake_kornia = MagicMock()
        fake_kornia.feature = fake_kf

        with (
            patch.dict(sys.modules, {"kornia": fake_kornia, "kornia.feature": fake_kf}),
            patch.object(
                tool_server,
                "_estimate_pose_ransac",
                return_value={"num_inliers": 1, "inlier_ratio": 0.5, "pose": None},
            ),
        ):
            r1 = tool_server._match_lightglue(_rgb(), _rgb(), max_size=64)
            r2 = tool_server._match_lightglue(_rgb(), _rgb(), max_size=64)

        assert r1["matcher"] == "lightglue"
        assert r2["matcher"] == "lightglue"
        assert len(lg_calls) == 1 and len(sp_calls) == 1
