"""Tests for graph orchestration tools on the tool server.

Covers:
  - build_initial_graph
  - inspect_graph
  - verify_doppelganger
  - prune_edges
  - bridge_components
  - strict no-fallback contract in sfm_run
"""

import importlib.util
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).parent.parent
_spec = importlib.util.spec_from_file_location(
    "tools_server_graph", ROOT / "tools_server" / "server.py"
)
TS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TS)


def _make_image(path: str, size=(128, 128), seed=0):
    rng = np.random.default_rng(seed)
    arr = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


@pytest.fixture
def test_scene(tmp_path, monkeypatch):
    """Fixture providing 4 registered synthetic images and mock embeddings."""
    d = str(tmp_path)
    img_ids = [f"img_{i:04d}" for i in range(4)]
    paths = []
    for i, img_id in enumerate(img_ids):
        p = os.path.join(d, f"{img_id}.jpg")
        _make_image(p, seed=i)
        paths.append(p)
        TS.register_image(img_id, p)

    # Deterministic mock embeddings
    def mock_emb(p):
        idx = int(Path(p).stem.split("_")[-1]) if "_" in Path(p).stem else 0
        v = np.zeros(384, dtype=np.float32)
        v[idx * 50:(idx + 1) * 50] = 1.0
        return v

    monkeypatch.setattr(TS, "compute_image_embedding", mock_emb)

    # Mock tool_match to return verified relative pose with MAGSAC-style R
    def mock_match(a, b, matcher="loftr", session="", **kw):
        # Register match record in match store
        pa = TS._store_path(a)
        pb = TS._store_path(b)
        res = {
            "num_matches": 100,
            "num_inliers": 80,
            "inlier_ratio": 0.8,
            "pose": {
                "R": np.eye(3).tolist(),
                "t": [0.0, 0.0, 1.0],
            },
            "keypoints_a": [[10.0, 10.0]] * 80,
            "keypoints_b": [[12.0, 10.0]] * 80,
            "inlier_mask": [True] * 80,
            "matcher": matcher,
        }
        if pa and pb:
            TS._record_agent_match(pa, pb, res, session=session)
        return res

    monkeypatch.setattr(TS, "tool_match", mock_match)

    session_id = f"test_session_{id(tmp_path)}"
    yield {
        "dir": d,
        "img_ids": img_ids,
        "paths": paths,
        "session": session_id,
    }

    # Clean up session
    TS._SESSION_GRAPHS.pop(session_id, None)
    TS._match_store.pop(session_id, None)


class TestGraphTools:
    def test_build_initial_graph(self, test_scene):
        sess = test_scene["session"]
        ids = test_scene["img_ids"]
        res = TS.tool_build_initial_graph(image_ids=ids, k_neighbors=2, session=sess)

        assert res["status"] == "ok"
        assert res["num_images"] == 4
        assert res["num_verified_edges"] > 0
        assert sess in TS._SESSION_GRAPHS
        graph = TS._SESSION_GRAPHS[sess]
        assert len(graph.active_edges()) == res["num_verified_edges"]

    def test_inspect_graph(self, test_scene):
        sess = test_scene["session"]
        ids = test_scene["img_ids"]

        # Before building: returns error
        err = TS.tool_inspect_graph(session=sess)
        assert err["status"] == "error"

        # After building: returns summary and components
        TS.tool_build_initial_graph(image_ids=ids, k_neighbors=2, session=sess)
        res = TS.tool_inspect_graph(session=sess)
        assert res["status"] == "ok"
        assert "components" in res
        assert "degrees" in res

        # Active edge list: exactly the session graph's active edges, as
        # [u, v] lists in the agent-facing id namespace.
        graph = TS._SESSION_GRAPHS[sess]
        expected = {tuple(e) for e in graph.active_edges()}
        assert expected  # graph actually has edges
        assert isinstance(res["edges"], list)
        assert {tuple(e) for e in res["edges"]} == expected
        assert res["num_edges_listed"] == len(res["edges"])

        # Pruning an edge is reflected in a subsequent inspect_graph call —
        # the episode runner relies on this to know which pairs are in play.
        pruned = list(graph.active_edges().keys())[0]
        TS.tool_prune_edges(pairs=[list(pruned)], session=sess)
        res2 = TS.tool_inspect_graph(session=sess)
        assert tuple(pruned) not in {tuple(e) for e in res2["edges"]}
        assert len(res2["edges"]) == len(res["edges"]) - 1

    def test_inspect_graph_edge_list_capped(self, test_scene):
        """Huge graphs return at most _INSPECT_GRAPH_MAX_EDGES edge entries."""
        from agentic_sfm.graph.view_graph import EdgeData, ViewGraph

        sess = test_scene["session"]
        ids = [f"cap_{i:04d}" for i in range(40)]  # 40*39/2 = 780 edges
        vg = ViewGraph(ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                vg.add_edge(ids[i], ids[j], EdgeData(ids[i], ids[j], num_inliers=50))
        TS._SESSION_GRAPHS[sess] = vg

        res = TS.tool_inspect_graph(session=sess)
        assert res["status"] == "ok"
        assert res["num_verified_edges"] == 780
        assert len(res["edges"]) == TS._INSPECT_GRAPH_MAX_EDGES
        assert res["num_edges_listed"] == TS._INSPECT_GRAPH_MAX_EDGES
        # Listed edges are real active edges (canonical u < v pairs).
        assert all(u < v for u, v in res["edges"])

    def test_verify_doppelganger(self, test_scene):
        sess = test_scene["session"]
        ids = test_scene["img_ids"]
        TS.tool_build_initial_graph(image_ids=ids, k_neighbors=2, session=sess)

        res = TS.tool_verify_doppelganger(ids[0], ids[1], session=sess)
        assert res["status"] == "ok"
        assert "visual_similarity" in res
        assert "num_inliers" in res
        assert "is_doppelganger" in res
        assert "evidence" in res
        assert isinstance(res["evidence"], str) and len(res["evidence"]) > 0

    def test_prune_edges(self, test_scene):
        sess = test_scene["session"]
        ids = test_scene["img_ids"]
        TS.tool_build_initial_graph(image_ids=ids, k_neighbors=3, session=sess)
        graph = TS._SESSION_GRAPHS[sess]
        n_before = len(graph.active_edges())

        pair_to_prune = list(graph.active_edges().keys())[0]
        res = TS.tool_prune_edges(pairs=[list(pair_to_prune)], session=sess)

        assert res["status"] == "ok"
        assert res["num_pruned"] == 1
        assert res["remaining_edges"] == n_before - 1
        assert len(graph.active_edges()) == n_before - 1

    def test_bridge_components(self, test_scene):
        sess = test_scene["session"]
        ids = test_scene["img_ids"]
        TS.tool_build_initial_graph(image_ids=ids, k_neighbors=1, session=sess)
        graph = TS._SESSION_GRAPHS[sess]

        # Artificially split into 2 components by creating disconnected nodes
        # e.g. prune edges between {0, 1} and {2, 3}
        for u in (ids[0], ids[1]):
            for v in (ids[2], ids[3]):
                graph.prune_edge(u, v)

        comps = graph.get_connected_components()
        if len(comps) >= 2:
            res = TS.tool_bridge_components(
                comp_a_id=0, comp_b_id=1, budget=2, session=sess
            )
            assert res["status"] == "ok"
            assert res["candidates_tested"] > 0
            assert "bridges_added" in res

    def test_sfm_reports_imported_graph_edges(self, test_scene, monkeypatch):
        sess = test_scene["session"]
        ids = test_scene["img_ids"]
        TS.tool_build_initial_graph(image_ids=ids, k_neighbors=2, session=sess)
        monkeypatch.setattr(TS, "_import_agent_matches", lambda *args, **kwargs: 3)
        monkeypatch.setattr(TS, "_run_colmap_subprocess", lambda **kwargs: {
            "num_registered": 3,
            "num_points3d": 20,
            "num_pairs_matched": kwargs["num_pairs_requested"],
            "output_dir": test_scene["dir"],
        })
        result = TS.tool_sfm_run(
            image_dir=test_scene["dir"],
            output_dir=os.path.join(test_scene["dir"], "recon_imported"),
            session=sess,
        )
        assert result["num_pairs_matched"] == 3

    def test_strict_no_fallback_aborts_on_empty_graph(self, test_scene):
        sess = test_scene["session"]
        ids = test_scene["img_ids"]
        # Build initial graph
        TS.tool_build_initial_graph(image_ids=ids, k_neighbors=2, session=sess)
        graph = TS._SESSION_GRAPHS[sess]

        # Prune ALL edges
        all_edges = list(graph.active_edges().keys())
        TS.tool_prune_edges(pairs=[list(e) for e in all_edges], session=sess)
        assert len(graph.active_edges()) == 0

        # sfm_run must abort with failed_empty_graph and 0 registered images
        out_dir = os.path.join(test_scene["dir"], "recon_empty")
        res = TS.tool_sfm_run(
            image_dir=test_scene["dir"],
            output_dir=out_dir,
            session=sess,
            use_agent_matches=True,
        )
        assert res["status"] == "failed_empty_graph"
        assert res["num_registered"] == 0
        assert "aborted" in res["error"].lower()


_REAL_TOOL_MATCH = TS.tool_match


class TestMatcherChoice:
    def test_fallback_edges_labelled_with_actual_backend(self, test_scene, monkeypatch):
        """A vggt request that fell back to loftr must be stored as loftr, not vggt."""
        real_mock = TS.tool_match

        def fallback_match(a, b, matcher="loftr", session="", **kw):
            res = real_mock(a, b, matcher="loftr", session=session)
            res["matcher"] = "loftr"
            res["matcher_fallback"] = "vggt->loftr"
            return res

        monkeypatch.setattr(TS, "tool_match", fallback_match)
        sess = test_scene["session"]
        res = TS.tool_build_initial_graph(
            image_ids=test_scene["img_ids"], k_neighbors=2, matcher="vggt", session=sess,
        )
        edges = TS._SESSION_GRAPHS[sess].active_edges().values()
        assert edges and all(e.matcher == "loftr" for e in edges)
        assert res["matcher_requested"] == "vggt"
        assert res["pairs_fallback"] == res["num_candidates_selected"]

    def test_summary_reports_zero_match_and_median_inliers(self, test_scene, monkeypatch):
        real_mock = TS.tool_match

        def sparse_match(a, b, matcher="loftr", session="", **kw):
            res = real_mock(a, b, matcher=matcher, session=session)
            res["num_matches"] = 0
            res["num_inliers"] = 0
            return res

        monkeypatch.setattr(TS, "tool_match", sparse_match)
        res = TS.tool_build_initial_graph(
            image_ids=test_scene["img_ids"], k_neighbors=2, session=test_scene["session"],
        )
        assert res["pairs_zero_matches"] == res["num_candidates_selected"]
        assert res["median_inliers"] == 0.0
        assert res["matcher_requested"] == "loftr"

    def test_verify_doppelganger_uses_requested_matcher(self, test_scene, monkeypatch):
        seen = []
        real_mock = TS.tool_match

        def spy(a, b, matcher="loftr", session="", **kw):
            seen.append(matcher)
            return real_mock(a, b, matcher=matcher, session=session)

        monkeypatch.setattr(TS, "tool_match", spy)
        ids = test_scene["img_ids"]
        TS.tool_verify_doppelganger(ids[0], ids[1], session=test_scene["session"], matcher="mast3r")
        assert seen == ["mast3r"]

    def test_verify_doppelganger_defaults_to_loftr(self, test_scene, monkeypatch):
        seen = []
        real_mock = TS.tool_match

        def spy(a, b, matcher="loftr", session="", **kw):
            seen.append(matcher)
            return real_mock(a, b, matcher=matcher, session=session)

        monkeypatch.setattr(TS, "tool_match", spy)
        ids = test_scene["img_ids"]
        TS.tool_verify_doppelganger(ids[0], ids[1], session=test_scene["session"])
        assert seen in ([], ["loftr"])

    def test_verify_doppelganger_rejects_unknown_matcher(self, test_scene, monkeypatch):
        # The fixture replaces tool_match with a permissive mock; restore the
        # real function so the matcher validation under test is the real one.
        monkeypatch.setattr(TS, "tool_match", _REAL_TOOL_MATCH)
        ids = test_scene["img_ids"]
        res = TS.tool_verify_doppelganger(
            ids[0], ids[1], session=test_scene["session"], matcher="not_a_matcher",
        )
        assert res.get("error"), "unknown matcher must be an error, not a valid verdict"
