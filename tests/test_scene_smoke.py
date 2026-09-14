"""Integration test: scene episode end-to-end against MockToolClient.

No GPU / servers needed — exercises the full Phase-2 data flow:
SceneDataset (real loader on a synthetic scene) -> run_scene_episode with
the ScriptedSceneAgent from scripts/scene_smoke_test.py -> tool results,
reward components, and client-side call logging.
"""

import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_sfm.rl.scene_episode import _pair_key, run_scene_episode  # noqa: E402
from scripts.run_scene_grpo import SceneDataset  # noqa: E402
from scripts.scene_smoke_test import (  # noqa: E402
    MockToolClient,
    ScriptedSceneAgent,
    select_coherent_subset,
    verify_episode,
    write_synthetic_scene,
)

N_IMAGES = 12
N_SAMPLED = 10


@pytest.fixture
def scene(tmp_path):
    """Synthetic 12-image scene loaded through the real SceneDataset."""
    si_dir = tmp_path / "scene_info"
    img_root = tmp_path / "images"
    write_synthetic_scene(si_dir, img_root, scene_id="0015", n_images=N_IMAGES)
    ds = SceneDataset.from_megadepth(
        str(si_dir), str(img_root), scenes=["0015"], max_images=N_SAMPLED)
    assert len(ds.scenes) == 1
    return ds.scenes[0], str(img_root)


@pytest.fixture(autouse=True)
def mock_embeddings(monkeypatch):
    """Deterministic hash embeddings — keeps retrieval real but offline."""
    import agentic_sfm.rl.retrieval as retrieval

    def fake(image_path):
        seed = zlib.crc32(str(image_path).encode())
        return np.random.default_rng(seed).random(64).astype(np.float32)

    monkeypatch.setattr(retrieval, "compute_image_embedding", fake)


def _run(scene, client=None, num_matches=5, **agent_kwargs):
    s, img_root = scene
    client = client or MockToolClient()
    agent = ScriptedSceneAgent(num_matches=num_matches, **agent_kwargs)
    ep = run_scene_episode(
        agent=agent,
        scene_id=s["scene_id"],
        image_paths=s["image_paths"],
        tool_client=client,
        gt_recon=s.get("gt_recon"),
        overlap_matrix=s.get("overlap_matrix"),
        image_indices=s.get("image_indices"),
        image_root=img_root,
    )
    return ep, client, agent


class TestSceneEpisodeSmoke:
    def test_end_to_end_all_checks_pass(self, scene):
        ep, client, agent = _run(scene)
        checks = verify_episode(
            ep, client, scene[0], expected_tools=agent.expected_tool_sequence())
        failed = [c for c in checks if not c.ok]
        assert not failed, [(c.name, c.detail) for c in failed]

    def test_all_images_registered(self, scene):
        ep, client, _ = _run(scene)
        assert len(client.registered) == N_SAMPLED
        assert set(client.registered) == {
            f"img_{i:04d}" for i in range(N_SAMPLED)}
        # registration happens before the first assistant turn
        assert sum(e.startswith("register_image:") for e in client.call_log) \
            == N_SAMPLED

    def test_tool_call_sequence_logged(self, scene):
        ep, _, _ = _run(scene, num_matches=3)
        assert [tc.tool for tc in ep.tool_calls] == [
            "retrieve", "match", "match", "match",
            "doppelganger_check", "sfm_run", "inspect", "done",
        ]
        # one result per executed (non-done) call, responses per turn
        assert len(ep.results) == len(ep.tool_calls) - 1
        assert len(ep.assistant_responses) == len(ep.tool_calls)
        assert ep.done

    def test_retrieval_and_matches(self, scene):
        ep, _, _ = _run(scene)
        retr = next(r for tc, r in zip(
            [t for t in ep.tool_calls if t.tool != "done"], ep.results)
            if tc.tool == "retrieve")
        assert retr["num_pairs"] > 0 and len(retr["pairs"]) > 0
        for p in retr["pairs"]:
            assert {"image_a", "image_b", "score"} <= set(p)
        # every tracked pair match has inliers and a pose
        assert len(ep.pair_matches) == 5
        for m in ep.pair_matches.values():
            assert m["num_inliers"] > 0 and m.get("pose") is not None
        assert ep.final_match["num_inliers"] > 0

    def test_sfm_and_reward(self, scene):
        ep, _, _ = _run(scene)
        recon = ep.recon_result
        assert recon["num_registered"] == N_SAMPLED
        rc = ep.reward_components
        # deterministic expectations from the mock's canned numbers
        assert rc["registration_reward"] == pytest.approx(0.5)   # 10/10
        assert rc["pose_reward"] == pytest.approx(0.75)          # 1 - 5/20
        assert rc["accumulative_tool_reward"] == pytest.approx(0.9)  # 9 valid
        assert rc["split_penalty"] == 0.0
        assert rc["total_reward"] == pytest.approx(2.15)
        assert ep.reward == rc["total_reward"]


class TestDoppelgangerPath:
    def test_flagged_pair_filtered_from_sfm(self, scene):
        # confidence >= 0.5 flags the checked pair as a doppelganger
        client = MockToolClient(doppelganger_confidence=0.9)
        ep, client, _ = _run(scene, client=client)

        assert len(ep.doppelganger_checks) == 1
        checked_key = next(iter(ep.doppelganger_checks))
        pair_list = client.sfm_pair_lists[0]
        assert pair_list is not None
        flagged = [pair for pair in pair_list
                   if _pair_key(pair[0], pair[1]) == checked_key]
        assert flagged == []
        assert ep.recon_result["num_doppelgangers_present"] == 1
        assert ep.recon_result["num_doppelgangers_filtered"] == 1
        assert ep.reward_components["doppelganger_reward"] == pytest.approx(0.3)

    def test_unflagged_pair_kept_in_sfm(self, scene):
        client = MockToolClient(doppelganger_confidence=0.12)
        ep, client, _ = _run(scene, client=client)
        pair_list = client.sfm_pair_lists[0]
        # all C(10,2) pairs pass through when nothing is flagged
        assert len(pair_list) == N_SAMPLED * (N_SAMPLED - 1) // 2
        assert ep.recon_result["num_doppelgangers_present"] == 0

    def test_no_doppelganger_check_means_exhaustive(self, scene):
        client = MockToolClient()
        ep, client, _ = _run(scene, client=client, run_doppelganger=False)
        # without any check the pair list stays None -> server-side exhaustive
        assert client.sfm_pair_lists[0] is None
        assert ep.recon_result["num_doppelgangers_present"] == 0
        assert "doppelganger_check" not in [tc.tool for tc in ep.tool_calls]


class TestCoherentSubset:
    def _scene(self):
        # 6 images: a tight clique {0,1,2} and a straggler tail.
        om = np.array([
            [-1.0, 0.9, 0.8, 0.0, 0.0, 0.0],
            [0.9, -1.0, 0.9, 0.1, 0.0, 0.0],
            [0.8, 0.9, -1.0, 0.0, 0.1, 0.0],
            [0.0, 0.1, 0.0, -1.0, 0.9, 0.8],
            [0.0, 0.0, 0.1, 0.9, -1.0, 0.9],
            [0.0, 0.0, 0.0, 0.8, 0.9, -1.0],
        ])
        return {
            "scene_id": "x",
            "image_paths": [f"p{i}.jpg" for i in range(6)],
            "num_images": 6,
            "overlap_matrix": om,
            "image_indices": list(range(6)),
            "gt_recon": {
                "num_images": 6,
                "poses": {str(i): np.eye(4).tolist() for i in range(6)},
            },
        }

    def test_picks_densest_clique(self):
        out = select_coherent_subset(self._scene(), 3)
        # First clique wins (highest bottleneck overlaps)
        assert set(out["image_indices"]) == {0, 1, 2}
        assert out["num_images"] == 3
        assert out["image_paths"] == ["p0.jpg", "p1.jpg", "p2.jpg"]
        # poses remapped to positions within the subset
        assert sorted(out["gt_recon"]["poses"].keys(), key=int) == ["0", "1", "2"]
        assert out["gt_recon"]["num_images"] == 3

    def test_k_ge_n_is_noop(self):
        s = self._scene()
        out = select_coherent_subset(s, 10)
        assert out["num_images"] == 6
