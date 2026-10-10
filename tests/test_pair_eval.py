"""Pose-aligned pair-eval rows: success is pose_reward > 0, not inliers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.data.hard_pairs import ImagePair
from agentic_sfm.eval.evaluate import pair_episode_row
from scripts.run_grpo import load_vllm_lora


def _pair() -> ImagePair:
    return ImagePair(
        pair_id="md_test_a_b",
        image_a="/tmp/a.jpg",
        image_b="/tmp/b.jpg",
        gt_R=np.eye(3),
        gt_t=np.array([0.0, 0.0, 1.0]),
        difficulty="hard",
    )


def test_pair_episode_row_success_is_pose_reward():
    ep = SimpleNamespace(
        reward=1.6,
        reward_components={"pose_reward": 0.8, "inlier_reward": 0.05},
        final_match={"num_inliers": 12, "inlier_ratio": 0.4},
        tool_calls=[object(), object()],
    )
    row = pair_episode_row(_pair(), ep, pose_weight=1.0)
    assert row["success"] is True
    assert row["geometric_reward"] == pytest.approx(0.85)
    assert row["num_inliers"] == 12
    assert row["num_tool_calls"] == 2


def test_pair_episode_row_inliers_alone_are_not_success():
    ep = SimpleNamespace(
        reward=0.6,
        reward_components={"pose_reward": 0.0, "inlier_reward": 0.04},
        final_match={"num_inliers": 7, "inlier_ratio": 0.2},
        tool_calls=[],
    )
    row = pair_episode_row(_pair(), ep)
    assert row["success"] is False
    assert row["geometric_reward"] == 0.04


def test_pair_episode_row_total_reward_bonus_is_not_success():
    ep = SimpleNamespace(
        reward=0.5,
        reward_components={"pose_reward": 0.0, "inlier_reward": 0.0},
        final_match={},
        tool_calls=[],
    )
    row = pair_episode_row(_pair(), ep)
    assert row["success"] is False
    assert row["geometric_reward"] == 0.0


def test_load_vllm_lora_requires_adapter_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        load_vllm_lora("http://127.0.0.1:9", tmp_path)
