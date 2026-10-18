"""Unit tests for Track B upgrades:
1. Strict full-frame tie-breaking floor in keep_best_match (geometry.py)
2. Monotonic correspondence union + crop-to-full-frame keypoint remapping (tools_server/server.py)
3. Monotonic baseline_delta_reward in compute_pair_reward and verl_agent_loop whitelist
4. Joint test-time budget & RL compute scaling law fitting (scripts/eval_budget_scaling.py)
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentic_sfm.constants import REWARD_TOTAL_KEYS  # noqa: E402
from agentic_sfm.geometry import keep_best_match  # noqa: E402
from agentic_sfm.rewards.pose_rewards import compute_pair_reward  # noqa: E402
from scripts.eval_budget_scaling import (  # noqa: E402
    DEFAULT_BUDGETS,
    fit_budget_compute_scaling_law,
    predict_scaling_error,
)

# Load tools_server/server.py under an isolated module name
_spec = importlib.util.spec_from_file_location(
    "tools_server_track_b", ROOT / "tools_server" / "server.py"
)
TS = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(TS)


def _make_img(path: Path, size: tuple[int, int] = (200, 160), seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    arr = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return str(path)


class TestKeepBestMatchFloor:
    def test_crop_tie_keeps_uncropped_full_frame_baseline(self):
        full_frame = {"num_inliers": 30, "inlier_ratio": 0.5, "pose": {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [1, 0, 0]}}
        crop_tie = {
            "num_inliers": 30,
            "inlier_ratio": 0.5,
            "cropped_image_id": "img_a_crop_1234",
            "pose": {"R": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], "t": [0, 1, 0]},
        }
        kept = keep_best_match(full_frame, crop_tie)
        assert kept is full_frame

    def test_strictly_better_crop_replaces_full_frame_baseline(self):
        full_frame = {"num_inliers": 30, "inlier_ratio": 0.5}
        crop_better = {
            "num_inliers": 35,
            "inlier_ratio": 0.5,
            "cropped_image_id": "img_a_crop_1234",
        }
        kept = keep_best_match(full_frame, crop_better)
        assert kept is crop_better

    def test_worse_crop_keeps_full_frame_baseline(self):
        full_frame = {"num_inliers": 30, "inlier_ratio": 0.5}
        crop_worse = {
            "num_inliers": 12,
            "inlier_ratio": 0.25,
            "cropped_image_id": "img_a_crop_1234",
        }
        kept = keep_best_match(full_frame, crop_worse)
        assert kept is full_frame


class TestMonotonicCorrespondenceUnion:
    def test_crop_records_parent_id_and_chained_origin_xy(self, tmp_path):
        TS._IMAGE_STORE.clear()
        TS._CROP_META.clear()
        TS._match_store.clear()

        pa = _make_img(tmp_path / "a.png", size=(200, 160), seed=1)
        TS.register_image("img_a", pa)

        c1 = TS.tool_crop("img_a", [0.2, 0.25, 0.8, 0.85])
        assert "error" not in c1
        cid1 = c1["cropped_image_id"]
        assert TS._CROP_META[cid1]["parent_id"] == "img_a"
        assert TS._CROP_META[cid1]["origin_xy"] == [40, 40]

        # Chained crop accumulates origin_xy and preserves root parent_id
        c2 = TS.tool_crop(cid1, [0.1, 0.1, 0.9, 0.9])
        assert "error" not in c2
        cid2 = c2["cropped_image_id"]
        assert TS._CROP_META[cid2]["parent_id"] == "img_a"
        ox2, oy2 = TS._CROP_META[cid2]["origin_xy"]
        assert ox2 > 40 and oy2 > 40
        root_path, offset = TS._root_path_and_offset(cid2)
        assert root_path == pa
        assert offset == (float(ox2), float(oy2))

    def test_crop_match_remaps_to_full_frame_and_unions_monotonically(self, tmp_path):
        TS._IMAGE_STORE.clear()
        TS._CROP_META.clear()
        TS._match_store.clear()

        pa = _make_img(tmp_path / "a.png", size=(200, 160), seed=10)
        pb = _make_img(tmp_path / "b.png", size=(200, 160), seed=20)
        TS.register_image("img_a", pa)
        TS.register_image("img_b", pb)

        sess = "ep_monotonic"
        # 1. Full-frame baseline match
        TS._record_agent_match(
            "img_a",
            "img_b",
            {
                "keypoints_a": [[10.0, 20.0], [50.0, 60.0]],
                "keypoints_b": [[12.0, 22.0], [52.0, 62.0]],
                "inlier_mask": [True, False],
                "matcher": "loftr",
            },
            session=sess,
        )
        key = tuple(sorted((os.path.abspath(pa), os.path.abspath(pb))))
        rec = TS._match_store[sess][key]
        assert len(rec["keypoints_a"]) == 2

        # 2. Crop img_a with origin (40, 40)
        crop_res = TS.tool_crop("img_a", [0.2, 0.25, 0.8, 0.85])
        crop_id = crop_res["cropped_image_id"]
        assert crop_res["origin_xy"] == [40, 40]

        # Match crop against img_b:
        #   - first point (10.0, 20.0) in crop -> (50.0, 60.0) in full frame (duplicate of point #2!)
        #   - second point (30.0, 35.0) in crop -> (70.0, 75.0) in full frame (new point!)
        TS._record_agent_match(
            crop_id,
            "img_b",
            {
                "keypoints_a": [[10.02, 20.01], [30.0, 35.0]],
                "keypoints_b": [[52.01, 61.98], [80.0, 85.0]],
                "inlier_mask": [True, True],
                "matcher": "mast3r",
            },
            session=sess,
        )

        rec_after = TS._match_store[sess][key]
        # Must NOT overwrite the first full-frame point [10.0, 20.0];
        # duplicate at 0.1px rounding is deduplicated and upgraded to inlier=True;
        # new shifted point [70.0, 75.0] is appended -> total 3 correspondences.
        assert len(rec_after["keypoints_a"]) == 3
        assert rec_after["keypoints_a"][0] == pytest.approx([10.0, 20.0])
        assert rec_after["keypoints_a"][2] == pytest.approx([70.0, 75.0])
        assert rec_after["keypoints_b"][2] == pytest.approx([80.0, 85.0])
        assert rec_after["inlier_mask"] == [True, True, True]


class TestBaselineDeltaRewardAndWhitelist:
    def test_baseline_delta_reward_in_total_keys_and_computed(self):
        assert "baseline_delta_reward" in REWARD_TOTAL_KEYS

        gt_pose = {
            "R": np.eye(3).tolist(),
            "t": [1.0, 0.0, 0.0],
        }
        # Baseline has 90 deg rotation & translation error -> 0 AUC
        r_90 = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        baseline_match = {
            "num_inliers": 10,
            "inlier_ratio": 0.2,
            "pose": {"R": r_90, "t": [0.0, 1.0, 0.0]},
        }
        # Improved crop match aligns with GT -> 1.0 AUC
        improved_match = {
            "num_inliers": 45,
            "inlier_ratio": 0.6,
            "pose": {"R": np.eye(3).tolist(), "t": [1.0, 0.0, 0.0]},
        }

        # Default baseline_delta_weight=0.0 preserves backwards-compatible total
        r_default = compute_pair_reward(
            improved_match,
            gt_pose=gt_pose,
            full_frame_baseline=baseline_match,
        )
        assert r_default["baseline_delta_reward"] == 0.0

        # With baseline_delta_weight=0.5, delta = 0.5 * (1.0 - 0.0) = 0.5
        r_delta = compute_pair_reward(
            improved_match,
            gt_pose=gt_pose,
            full_frame_baseline=baseline_match,
            baseline_delta_weight=0.5,
        )
        assert r_delta["baseline_delta_reward"] == pytest.approx(0.5)
        assert r_delta["total_reward"] == pytest.approx(r_default["total_reward"] + 0.5)

        # No negative penalty if candidate is worse than baseline (monotonic max(0, delta))
        r_worse = compute_pair_reward(
            baseline_match,
            gt_pose=gt_pose,
            full_frame_baseline=improved_match,
            baseline_delta_weight=0.5,
        )
        assert r_worse["baseline_delta_reward"] == 0.0

    def test_verl_agent_loop_whitelists_validity_and_baseline_delta_keys(self):
        src_text = (ROOT / "src" / "agentic_sfm" / "verl_agent_loop.py").read_text()
        for key in (
            "first_call_validity_weight",
            "first_call_invalid_penalty",
            "gate_geometry_on_validity",
            "format_weight",
            "baseline_delta_weight",
        ):
            assert f'"{key}"' in src_text, f"{key} missing from verl_agent_loop.py"


class TestBudgetComputeScalingLaw:
    def test_recovers_synthetic_2d_scaling_exponents(self):
        true_params = {
            "E_inf": 0.12,
            "a": 0.45,
            "alpha": 0.65,
            "c": 0.30,
            "beta": 0.40,
        }
        records = []
        for b in DEFAULT_BUDGETS:
            for c_rl in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
                err = (
                    true_params["E_inf"]
                    + true_params["a"] * (b ** (-true_params["alpha"]))
                    + true_params["c"] * (c_rl ** (-true_params["beta"]))
                )
                records.append({"budget": b, "compute": c_rl, "error": err})

        fit = fit_budget_compute_scaling_law(records)
        assert fit["r2"] > 0.9999
        assert fit["rmse"] < 1e-4
        assert fit["E_inf"] == pytest.approx(true_params["E_inf"], abs=1e-2)
        assert fit["a"] == pytest.approx(true_params["a"], abs=1e-2)
        assert fit["alpha"] == pytest.approx(true_params["alpha"], abs=1e-2)
        assert fit["c"] == pytest.approx(true_params["c"], abs=1e-2)
        assert fit["beta"] == pytest.approx(true_params["beta"], abs=1e-2)

        pred = predict_scaling_error(fit, budget=8, compute=16.0)
        expected = (
            true_params["E_inf"]
            + true_params["a"] * (8 ** (-true_params["alpha"]))
            + true_params["c"] * (16.0 ** (-true_params["beta"]))
        )
        assert pred == pytest.approx(expected, abs=1e-3)

    def test_isambard_config_and_slurm_exist(self):
        cfg_path = ROOT / "configs" / "phase2_isambard_4xgh200.yaml"
        slurm_path = ROOT / "jobs" / "isambard_4xgh200_hybrid.slurm"
        assert cfg_path.is_file()
        assert slurm_path.is_file()
        slurm_text = slurm_path.read_text()
        assert "#SBATCH --nodes=1" in slurm_text
        assert "#SBATCH --gpus=4" in slurm_text
        assert "#SBATCH --time=00:30:00" in slurm_text
        assert "apptainer exec --nv" in slurm_text
        assert "pytorch_26.09-py3.sif" in slurm_text
        assert "--max-epochs-this-run" in slurm_text
        cfg_text = cfg_path.read_text()
        assert "matcher_explore_eps: 0.25" in cfg_text
        assert "mapanything" in cfg_text


class TestDiagnosticSpecialistRoutingAndExploration:
    def test_select_diagnostic_specialist_matcher_prevents_loftr_collapse(self):
        from scripts.build_scene_sft import (
            HEAVY_3D_SPECIALISTS,
            build_oracle_episode,
            select_diagnostic_specialist_matcher,
        )

        rng = np.random.default_rng(7)
        # Explicit matcher preserved for backwards compatibility
        assert select_diagnostic_specialist_matcher("0015", 0.45, 0, rng, "loftr") == "loftr"
        assert select_diagnostic_specialist_matcher("0015", 0.45, 0, rng, "vggt") == "vggt"

        # Indoor ScanNet scene or low overlap routes to heavy 3D specialists
        m_indoor = select_diagnostic_specialist_matcher("scene0715_00", 0.40, 0, rng, "mixed")
        assert m_indoor in HEAVY_3D_SPECIALISTS
        m_low_ov = select_diagnostic_specialist_matcher("0015", 0.12, 1, rng, "mixed")
        assert m_low_ov in HEAVY_3D_SPECIALISTS

        # Bridge stage always escalates to a 3D specialist
        for idx in range(6):
            m_bridge = select_diagnostic_specialist_matcher("0015", 0.50, idx, rng, "mixed", stage="bridge")
            assert m_bridge in HEAVY_3D_SPECIALISTS

        # Across 10 outdoor mixed episodes, all 4 specialists appear
        chosen = {
            select_diagnostic_specialist_matcher("0015", 0.45, idx, rng, "mixed", stage="initial")
            for idx in range(10)
        }
        assert chosen == {"loftr", "vggt", "mast3r", "mapanything"}

        # End-to-end build_oracle_episode with matcher="mixed" emits non-LoFTR specialist calls
        fake_scene = {
            "scene_id": "scene0707_00",
            "num_images": 6,
            "image_paths": [f"img_{i}.jpg" for i in range(6)],
            "image_indices": list(range(6)),
            "overlap_matrix": np.array([
                [1.0, 0.35, 0.25, 0.0, 0.0, 0.0],
                [0.35, 1.0, 0.30, 0.0, 0.0, 0.0],
                [0.25, 0.30, 1.0, 0.02, 0.0, 0.0],
                [0.0, 0.0, 0.02, 1.0, 0.40, 0.30],
                [0.0, 0.0, 0.0, 0.40, 1.0, 0.35],
                [0.0, 0.0, 0.0, 0.30, 0.35, 1.0],
            ]),
            "gt_recon": {"num_images": 6, "poses": {str(i): np.eye(4).tolist() for i in range(6)}},
        }
        ep_dict = build_oracle_episode(fake_scene, image_root="", ep_idx=0, rng=rng, matcher="mixed")
        assert ep_dict["messages"]
        assistant_Payloads = [
            m["content"] for m in ep_dict["messages"] if m.get("role") == "assistant"
        ]
        assert any('"vggt"' in p or '"mast3r"' in p or '"mapanything"' in p for p in assistant_Payloads)

    def test_mapanything_backend_and_fallback(self, tmp_path, monkeypatch):
        TS._IMAGE_STORE.clear()
        TS._CROP_META.clear()
        TS._match_store.clear()

        pa = _make_img(tmp_path / "ma_a.png", size=(140, 140), seed=31)
        pb = _make_img(tmp_path / "ma_b.png", size=(140, 140), seed=32)
        TS.register_image("ma_a", pa)
        TS.register_image("ma_b", pb)

        assert "mapanything" in TS._VALID_MATCHERS

        # Test synthetic MapAnything pointmap forward
        H, W = 20, 20
        yy, xx = np.meshgrid(np.linspace(-0.5, 0.5, H), np.linspace(-0.5, 0.5, W), indexing="ij")
        pts = np.stack([xx, yy, np.full_like(xx, 2.0)], axis=-1).astype(np.float32)
        conf = np.full((H, W), 3.0, dtype=np.float32)
        ext_a = np.eye(3, 4, dtype=np.float64)
        ext_b = np.eye(3, 4, dtype=np.float64)
        ext_b[0, 3] = 0.1

        monkeypatch.setattr(
            TS,
            "_run_mapanything_pair",
            lambda a, b, ms: {
                "points_a": pts,
                "points_b": pts,
                "conf_a": conf,
                "conf_b": conf,
                "ext_a": ext_a,
                "ext_b": ext_b,
            },
        )
        res = TS.tool_match("ma_a", "ma_b", matcher="mapanything", session="sess_ma")
        assert res["matcher"] == "mapanything"
        assert res["num_matches"] > 0
        assert res["prior_pose"] is not None
        assert res["prior_confidence"] == pytest.approx(3.0)

    def test_run_scene_episode_forced_specialist_synchronizes_assistant_json(self, tmp_path):
        import json as _json
        from agentic_sfm.rl.scene_episode import run_scene_episode

        pa = _make_img(tmp_path / "s0.jpg", size=(64, 64), seed=1)
        pb = _make_img(tmp_path / "s1.jpg", size=(64, 64), seed=2)

        class _FakeAgent:
            reward_config = {"first_call_validity_weight": 0.0}

            def __init__(self):
                self.turn = 0

            def _encode_image(self, p):
                return "ZmFrZQ=="

            def _generate_turn(self, messages, images):
                self.turn += 1
                if self.turn == 1:
                    # Model tries to collapse to loftr
                    return '{"tool": "build_initial_graph", "args": {"k_neighbors": 2, "matcher": "loftr"}}'
                if self.turn == 2:
                    return '{"tool": "sfm_run", "args": {}}'
                return '{"tool": "done", "args": {}}'

        class _FakeToolClient:
            def __init__(self):
                self.recorded_matchers = []

            def register_image(self, img_id, path):
                return {"status": "ok"}

            def build_initial_graph(self, image_ids, k_neighbors=5, matcher="loftr", session=""):
                self.recorded_matchers.append(matcher)
                return {
                    "num_images": len(image_ids),
                    "num_candidate_edges": 1,
                    "num_verified_edges": 1,
                    "num_components": 1,
                    "matcher_requested": matcher,
                    "status": "ok",
                }

            def sfm_run(self, image_dir="", pair_list=None, session=""):
                return {"num_registered": 2, "num_points3d": 120}

        client = _FakeToolClient()
        ep = run_scene_episode(
            agent=_FakeAgent(),
            scene_id="scene0715_00",
            image_paths=[pa, pb],
            tool_client=client,
            max_tool_calls=5,
            max_turns=5,
            forced_specialist_matcher="vggt",
        )
        assert client.recorded_matchers == ["vggt"]
        assert ep.tool_calls[0].args["matcher"] == "vggt"
        first_assistant_json = _json.loads(ep.assistant_responses[0])
        assert first_assistant_json["args"]["matcher"] == "vggt"
        # Verify the message history seen by the GRPO logprob computation also has "vggt"
        assistant_msgs = [m["content"] for m in ep.messages if m.get("role") == "assistant"]
        assert _json.loads(assistant_msgs[0])["args"]["matcher"] == "vggt"
