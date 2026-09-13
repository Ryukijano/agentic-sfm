"""Tests for scene-level episode runner and retrieval."""

import json
import sys
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.agent.policy import ToolCall, parse_tool_call
from agentic_sfm.rl.scene_episode import (
    SceneRolloutEpisode,
    SCENE_SYSTEM_PROMPT,
    _execute_scene_tool,
    run_scene_episode,
    run_scene_oracle_episode,
)
from agentic_sfm.rl.retrieval import (
    compute_image_embedding,
    retrieve_pairs,
    retrieve_pairs_from_paths,
)
from agentic_sfm.rewards.pose_rewards import compute_scene_reward


class MockToolClient:
    """Mock tool client for testing scene episodes."""

    def __init__(self):
        self.registered = {}
        self.match_calls = []
        self.crop_calls = []
        self.sfm_calls = []
        self.inspect_calls = []

    def register_image(self, img_id: str, path: str):
        self.registered[img_id] = path
        return {"status": "ok", "image_id": img_id}

    def crop(self, image_id: str, bbox: list):
        self.crop_calls.append((image_id, bbox))
        return {"image_id": f"{image_id}_crop", "path": "/tmp/crop.png"}

    def match(self, image_a: str, image_b: str, matcher: str = "loftr", **kwargs):
        self.match_calls.append((image_a, image_b, matcher))
        return {
            "num_inliers": 100,
            "num_matches": 200,
            "inlier_ratio": 0.5,
            "pose": {"R": np.eye(3).tolist(), "t": [0, 0, 1]},
        }

    def doppelganger_check(self, image_a: str, image_b: str):
        return {"is_doppelganger": False, "num_matches": 200, "inlier_ratio": 0.5}

    def sfm_run(self, image_dir: str, **kwargs):
        self.sfm_calls.append(image_dir)
        return {
            "num_registered": 8,
            "num_points3d": 500,
            "output_dir": "/tmp/recon",
        }

    def inspect(self, recon_dir: str):
        self.inspect_calls.append(recon_dir)
        return {"num_images": 8, "num_points3d": 500, "num_cameras": 1}

    def health(self):
        return {"status": "ok"}


class MockAgent:
    """Mock agent for testing scene episodes."""

    def __init__(self, responses: Optional[list] = None):
        self.responses = responses or [
            json.dumps({"tool": "retrieve", "args": {}}),
            json.dumps({"tool": "match", "args": {"image_a": "img_0000", "image_b": "img_0001", "matcher": "loftr"}}),
            json.dumps({"tool": "sfm_run", "args": {}}),
            json.dumps({"tool": "done", "args": {}}),
        ]
        self._call_idx = 0
        self.matcher = "loftr"
        self.reward_config = {}

    def _encode_image(self, path: str) -> str:
        return "base64_fake_image"

    def _generate_turn(self, messages, images=None):
        if self._call_idx < len(self.responses):
            resp = self.responses[self._call_idx]
            self._call_idx += 1
            return resp
        return json.dumps({"tool": "done", "args": {}})


class TestSceneEpisode:
    def test_episode_dataclass(self):
        ep = SceneRolloutEpisode(scene_id="test", image_paths=["a.jpg"], num_images=1)
        assert ep.scene_id == "test"
        assert not ep.done

    def test_run_episode_completes(self):
        agent = MockAgent()
        client = MockToolClient()
        ep = run_scene_episode(
            agent=agent,
            scene_id="test",
            image_paths=["a.jpg", "b.jpg"],
            tool_client=client,
            max_tool_calls=10,
        )
        assert ep.done
        assert len(ep.tool_calls) == 4  # retrieve, match, sfm_run, done
        assert ep.reward > 0
        assert ep.recon_result.get("num_registered", 0) > 0

    def test_oracle_episode(self):
        agent = MockAgent()
        client = MockToolClient()
        # Create a simple overlap matrix (3 images, high overlap 0-1, low 0-2)
        om = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.1], [0.1, 0.1, 1.0]])
        ep = run_scene_oracle_episode(
            agent=agent,
            scene_id="test",
            image_paths=["a.jpg", "b.jpg", "c.jpg"],
            tool_client=client,
            overlap_matrix=om,
            image_indices=[0, 1, 2],
            failed_group_max_reward=0.0,
        )
        assert ep.done
        assert ep.reward > 0  # oracle should get positive reward
        assert len(ep.tool_calls) >= 3  # retrieve + match + sfm_run + done
        assert ep.recon_result.get("num_registered", 0) > 0

    def test_oracle_reward_floor(self):
        agent = MockAgent()
        client = MockToolClient()
        ep = run_scene_oracle_episode(
            agent=agent,
            scene_id="test",
            image_paths=["a.jpg"],
            tool_client=client,
            failed_group_max_reward=5.0,  # very high floor
        )
        # Oracle should get at least failed_group_max_reward + epsilon
        assert ep.reward > 5.0


class TestSceneToolExecution:
    def test_retrieve_uses_learned_model(self):
        """Verify retrieve uses retrieval.py, not just proximity."""
        ep = SceneRolloutEpisode(
            scene_id="test",
            image_paths=["a.jpg", "b.jpg"],
            image_root="/tmp",
            num_images=2,
        )
        client = MockToolClient()
        registered = {"a.jpg": "img_0000", "b.jpg": "img_0001"}

        with patch("agentic_sfm.rl.retrieval.retrieve_pairs_from_paths") as mock_ret:
            mock_ret.return_value = [{"image_a": "img_0000", "image_b": "img_0001", "score": 0.9}]
            tc = ToolCall(tool="retrieve", args={"top_k": 5})
            result = _execute_scene_tool(client, tc, registered, ep)
            mock_ret.assert_called_once()
            assert result["num_pairs"] == 1

    def test_sfm_run_tool(self):
        ep = SceneRolloutEpisode(
            scene_id="test",
            image_paths=["a.jpg"],
            image_root="/tmp",
            num_images=1,
        )
        client = MockToolClient()
        tc = ToolCall(tool="sfm_run", args={})
        result = _execute_scene_tool(client, tc, {}, ep)
        assert result["num_registered"] == 8
        assert client.sfm_calls

    def test_inspect_tool(self):
        ep = SceneRolloutEpisode(
            scene_id="test",
            image_paths=["a.jpg"],
            image_root="/tmp",
            num_images=1,
            recon_result={"output_dir": "/tmp/recon"},
        )
        client = MockToolClient()
        tc = ToolCall(tool="inspect", args={"recon_dir": "/tmp/recon"})
        result = _execute_scene_tool(client, tc, {}, ep)
        assert result["num_images"] == 8
        assert client.inspect_calls

    def test_done_tool(self):
        ep = SceneRolloutEpisode(scene_id="test", image_paths=["a.jpg"])
        client = MockToolClient()
        tc = ToolCall(tool="done", args={})
        result = _execute_scene_tool(client, tc, {}, ep)
        # done should not error, returns empty dict or similar
        assert isinstance(result, dict)


class TestSceneRewardIntegration:
    def test_scene_reward_with_ntep(self):
        rc = compute_scene_reward(
            {"num_registered": 8, "num_points3d": 500, "mean_pose_error_deg": 5.0,
             "num_doppelgangers_filtered": 1, "num_doppelgangers_present": 2},
            gt_recon={"num_images": 10},
            num_valid_calls=5,
            use_accumulative_tool_reward=True,
            use_ntep_rewards=True,
            tool_calls=[ToolCall(tool="match", args={}), ToolCall(tool="done", args={})],
            tool_results=[{"num_inliers": 100}, {}],
        )
        assert rc["total_reward"] > 0
        assert "accumulative_tool_reward" in rc
        assert "ntep_intent_reward" in rc
        assert "ntep_redundancy_penalty" in rc

    def test_scene_reward_all_components(self):
        rc = compute_scene_reward(
            {"num_registered": 8, "num_points3d": 500, "mean_pose_error_deg": 5.0,
             "num_doppelgangers_filtered": 2, "num_doppelgangers_present": 2},
            gt_recon={"num_images": 10},
        )
        assert "registration_reward" in rc
        assert "split_penalty" in rc
        assert "pose_reward" in rc
        assert "doppelganger_reward" in rc
        assert "tool_cost" in rc
        assert "accumulative_tool_reward" in rc
        assert "total_reward" in rc
