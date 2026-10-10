"""Unit tests for ViewGraph data structure and geometric diagnostics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
from agentic_sfm.graph.view_graph import EdgeData, ViewGraph


def test_view_graph_components():
    nodes = ["img_0000", "img_0001", "img_0002", "img_0003", "img_0004"]
    vg = ViewGraph(nodes)

    # Component 1: 0-1-2
    vg.add_edge("img_0000", "img_0001", EdgeData("img_0000", "img_0001", num_inliers=50))
    vg.add_edge("img_0001", "img_0002", EdgeData("img_0001", "img_0002", num_inliers=40))

    # Component 2: 3-4
    vg.add_edge("img_0003", "img_0004", EdgeData("img_0003", "img_0004", num_inliers=30))

    comps = vg.get_connected_components(min_inliers=15)
    assert len(comps) == 2
    assert comps[0] == ["img_0000", "img_0001", "img_0002"]
    assert comps[1] == ["img_0003", "img_0004"]


def test_view_graph_pruning():
    vg = ViewGraph(["img_0", "img_1", "img_2"])
    vg.add_edge("img_0", "img_1", EdgeData("img_0", "img_1", num_inliers=50))
    vg.add_edge("img_1", "img_2", EdgeData("img_1", "img_2", num_inliers=40))

    assert len(vg.active_edges()) == 2
    assert vg.prune_edge("img_0", "img_1")
    assert len(vg.active_edges()) == 1

    # Pruned edge disconnects img_0
    comps = vg.get_connected_components()
    assert len(comps) == 2
    assert ["img_0"] in comps


def test_view_graph_cycle_inconsistency():
    vg = ViewGraph(["img_0", "img_1", "img_2"])

    # Consistent cycle: R01 = 90 deg around z, R12 = 90 deg around z, R20 = 180 deg around z
    # R01 @ R12 @ R20 = I
    def rot_z(deg):
        rad = np.radians(deg)
        return np.array([
            [np.cos(rad), -np.sin(rad), 0],
            [np.sin(rad), np.cos(rad), 0],
            [0, 0, 1]
        ])

    R01 = rot_z(90)
    R12 = rot_z(90)
    R20 = rot_z(180)  # R_20 so R_01 @ R_12 @ R_20 = rot_z(360) = I

    # Add edges: (0, 1), (1, 2), (0, 2) where edge (0,2) has R_02 = R_20.T = rot_z(-180)
    vg.add_edge("img_0", "img_1", EdgeData("img_0", "img_1", num_inliers=50, R_rel=R01))
    vg.add_edge("img_1", "img_2", EdgeData("img_1", "img_2", num_inliers=50, R_rel=R12))
    vg.add_edge("img_0", "img_2", EdgeData("img_0", "img_2", num_inliers=50, R_rel=R20.T))

    cycles = vg.find_cycle_inconsistencies(max_error_deg=10.0)
    assert len(cycles) == 0

    # Introduce a 45 degree doppelganger distortion on edge (0, 2)
    R02_bad = rot_z(-135)
    vg.add_edge("img_0", "img_2", EdgeData("img_0", "img_2", num_inliers=50, R_rel=R02_bad))

    bad_cycles = vg.find_cycle_inconsistencies(max_error_deg=10.0)
    assert len(bad_cycles) == 1
    assert pytest.approx(bad_cycles[0]["rotation_error_deg"], 0.5) == 45.0


def test_view_graph_summary():
    nodes = ["img_0", "img_1", "img_2", "img_3"]
    vg = ViewGraph(nodes)
    vg.add_edge("img_0", "img_1", EdgeData("img_0", "img_1", num_inliers=30))
    vg.add_edge("img_1", "img_2", EdgeData("img_1", "img_2", num_inliers=25))

    summary = vg.get_summary()
    assert summary["num_images"] == 4
    assert summary["num_verified_edges"] == 2
    assert summary["num_components"] == 2  # [0,1,2] and [3]
    assert "img_3" in summary["isolated_images"]
    assert "img_0" in summary["bottlenecks"]


def test_active_edge_keys_are_canonical():
    """Edge keys are always (min, max) pairs regardless of add order.

    The tool server serializes ``active_edges().keys()`` as ``[[u, v], ...]``
    for ``inspect_graph`` — canonical ordering keeps that list stable and
    deduplicated for the episode's GT doppelganger labelling.
    """
    vg = ViewGraph(["img_0", "img_1", "img_2"])
    # add_edge called with reversed argument order must not create a
    # duplicate or non-canonical key.
    vg.add_edge("img_1", "img_0", EdgeData("img_1", "img_0", num_inliers=10))
    vg.add_edge("img_0", "img_1", EdgeData("img_0", "img_1", num_inliers=20))

    keys = list(vg.active_edges())
    assert keys == [("img_0", "img_1")]
    assert all(u < v for u, v in keys)
    # round-trips through the [u, v] list form used by inspect_graph
    assert [list(k) for k in keys] == [["img_0", "img_1"]]
