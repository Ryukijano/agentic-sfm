"""Tests for shuffled full-val pair selection and per-difficulty summaries."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.eval.evaluate import select_eval_pairs, summarize_pair_eval


class _Pair:
    def __init__(self, difficulty: str, pair_id: str = ""):
        self.difficulty = difficulty
        self.pair_id = pair_id or difficulty


def test_select_eval_pairs_shuffles_bins():
    pairs = [_Pair("hard") for _ in range(50)]
    pairs += [_Pair("medium") for _ in range(50)]
    pairs += [_Pair("extreme") for _ in range(50)]
    first_50 = pairs[:50]
    assert {p.difficulty for p in first_50} == {"hard"}

    selected = select_eval_pairs(pairs, max_pairs=50, seed=0)
    assert len(selected) == 50
    assert len({p.difficulty for p in selected}) > 1


def test_select_eval_pairs_full_set_when_max_none():
    pairs = [_Pair("hard"), _Pair("medium"), _Pair("easy")]
    selected = select_eval_pairs(pairs, max_pairs=None, seed=0)
    assert len(selected) == 3
    assert {p.difficulty for p in selected} == {"hard", "medium", "easy"}


def test_summarize_pair_eval_by_difficulty_and_geometry():
    rows = [
        {
            "difficulty": "medium",
            "reward": 1.5,
            "geometric_reward": 1.0,
            "num_inliers": 100,
            "inlier_ratio": 0.5,
            "pose_auc": 0.9,
            "success": True,
            "num_tool_calls": 2,
            "rotation_error": 2.0,
            "translation_error": 3.0,
        },
        {
            "difficulty": "hard",
            "reward": 0.5,
            "geometric_reward": 0.0,
            "num_inliers": 0,
            "inlier_ratio": 0.0,
            "pose_auc": 0.0,
            "success": False,
            "num_tool_calls": 2,
            "rotation_error": -1,
            "translation_error": -1,
        },
    ]
    stats = summarize_pair_eval(rows)
    assert stats["num_pairs"] == 2
    assert stats["mean_reward"] == 1.0
    assert stats["mean_geometric_reward"] == 0.5
    assert stats["success_rate"] == 0.5
    assert stats["medium_mean_geometric_reward"] == 1.0
    assert stats["hard_success_rate"] == 0.0
    assert stats["medium_count"] == 1
