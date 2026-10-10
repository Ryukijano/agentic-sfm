"""Unit and integration tests for the 3-Baseline Evaluation Benchmark."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.run_scene_baselines import (
    run_exhaustive_baseline,
    run_graph_agent_baseline,
    run_knn_baseline,
    run_random_baseline,
)
from scripts.run_scene_grpo import SceneDataset
from scripts.scene_smoke_test import MockToolClient, write_synthetic_scene


@pytest.fixture
def synthetic_scene(tmp_path):
    si_dir = tmp_path / "scene_info"
    img_root = tmp_path / "images"
    scene_id = write_synthetic_scene(si_dir, img_root, n_images=8, seed=42)
    ds = SceneDataset.from_megadepth(str(si_dir), str(img_root), scenes=[scene_id])
    assert len(ds.scenes) == 1
    return ds.scenes[0], str(img_root)


class TestBaselines:
    def test_exhaustive_baseline(self, synthetic_scene):
        scene, img_root = synthetic_scene
        client = MockToolClient()
        ep = run_exhaustive_baseline(scene, client, image_root=img_root)
        assert ep.done
        assert ep.recon_result is not None
        assert ep.recon_result.get("num_registered") == 8
        assert ep.reward > 0
        tools = [tc.tool for tc in ep.tool_calls]
        assert "build_initial_graph" in tools
        assert "sfm_run" in tools

    def test_knn_baseline(self, synthetic_scene):
        scene, img_root = synthetic_scene
        client = MockToolClient()
        ep = run_knn_baseline(scene, client, k=3, image_root=img_root)
        assert ep.done
        assert ep.recon_result is not None
        assert ep.recon_result.get("num_registered") == 8
        assert ep.reward > 0
        tools = [tc.tool for tc in ep.tool_calls]
        assert "build_initial_graph" in tools
        assert "sfm_run" in tools

    def test_random_baseline(self, synthetic_scene):
        scene, img_root = synthetic_scene
        client = MockToolClient()
        ep = run_random_baseline(scene, client, edges_per_node=2, image_root=img_root)
        assert ep.done
        assert ep.recon_result is not None
        assert ep.recon_result.get("num_registered") == 8
        assert ep.reward > 0
        tools = [tc.tool for tc in ep.tool_calls]
        assert "match" in tools
        assert "sfm_run" in tools

    def test_real_client_graph_signature_and_isolated_outputs(self, synthetic_scene):
        scene, img_root = synthetic_scene

        class StrictClient(MockToolClient):
            def build_initial_graph(self, image_ids, k_neighbors=5, matcher="loftr", session=""):
                assert set(image_ids) == set(self.registered)
                return super().build_initial_graph(image_ids, k_neighbors, matcher, session)

            def sfm_run(self, image_dir, pair_list=None, output_dir="", session=""):
                assert not output_dir or session in output_dir
                return super().sfm_run(image_dir, pair_list, output_dir, session)

        for runner in (run_exhaustive_baseline, run_knn_baseline):
            assert runner(scene, StrictClient(), image_root=img_root).done

    def test_mock_reward_does_not_favor_more_calls(self, synthetic_scene):
        scene, img_root = synthetic_scene
        knn = run_knn_baseline(scene, MockToolClient(), image_root=img_root)
        random = run_random_baseline(scene, MockToolClient(), image_root=img_root)
        assert random.recon_result["num_registered"] == knn.recon_result["num_registered"]
        assert random.reward == pytest.approx(knn.reward)

    def test_graph_agent_baseline(self, synthetic_scene):
        scene, img_root = synthetic_scene
        client = MockToolClient()
        ep = run_graph_agent_baseline(scene, client, image_root=img_root)
        assert ep.done
        assert ep.recon_result is not None
        assert ep.recon_result.get("num_registered") == 8
        assert ep.reward > 0
        tools = [tc.tool for tc in ep.tool_calls]
        assert tools == [
            "build_initial_graph",
            "inspect_graph",
            "verify_doppelganger",
            "prune_edges",
            "sfm_run",
            "done",
        ]
