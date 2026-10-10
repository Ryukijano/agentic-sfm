"""View graph data structure and geometric diagnostics for scene-level SfM.

Maintains scene view-graph topology and computes photogrammetric diagnostics:
connected components, cycle rotation consistency, degree distributions,
weak articulation bridges, and doppelganger candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EdgeData:
    """Attributes of a directed or undirected view graph edge."""

    image_a: str
    image_b: str
    num_matches: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    R_rel: np.ndarray | None = None
    t_rel: np.ndarray | None = None
    matcher: str = "loftr"
    status: str = "verified"  # "verified", "pruned", "failed"
    is_doppelganger: bool = False
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ViewGraph:
    """Maintains scene view-graph topology and computes photogrammetric diagnostics."""

    def __init__(self, image_ids: list[str]):
        self.nodes: list[str] = sorted(list(set(image_ids)))
        self.edges: dict[tuple[str, str], EdgeData] = {}

    @staticmethod
    def canonical_key(u: str, v: str) -> tuple[str, str]:
        """Canonical key for an undirected edge."""
        return (u, v) if u < v else (v, u)

    def add_edge(self, u: str, v: str, data: EdgeData) -> None:
        """Add or update an edge in the view graph."""
        key = self.canonical_key(u, v)
        self.edges[key] = data

    def prune_edge(self, u: str, v: str) -> bool:
        """Mark an edge as pruned so it is excluded from reconstruction."""
        key = self.canonical_key(u, v)
        if key in self.edges:
            self.edges[key].status = "pruned"
            return True
        return False

    def active_edges(self) -> dict[tuple[str, str], EdgeData]:
        """Return all verified, unpruned edges."""
        return {
            k: v for k, v in self.edges.items()
            if v.status == "verified"
        }

    def failed_edges(self) -> dict[tuple[str, str], EdgeData]:
        """Return all edges that failed matching or geometric verification."""
        return {
            k: v for k, v in self.edges.items()
            if v.status == "failed"
        }

    def get_connected_components(self, min_inliers: int = 15) -> list[list[str]]:
        """Return connected components of the active view graph."""
        adj: dict[str, set[str]] = {n: set() for n in self.nodes}
        for (u, v), edge in self.active_edges().items():
            if edge.num_inliers >= min_inliers:
                adj[u].add(v)
                adj[v].add(u)

        visited: set[str] = set()
        components: list[list[str]] = []
        for node in self.nodes:
            if node not in visited:
                comp: list[str] = []
                queue = [node]
                visited.add(node)
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for nbr in adj[curr]:
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append(nbr)
                components.append(sorted(comp))
        return sorted(components, key=len, reverse=True)

    def find_cycle_inconsistencies(
        self,
        max_error_deg: float = 15.0,
        min_inliers: int = 15,
    ) -> list[dict[str, Any]]:
        """Find 3-cycles with large rotation residual: R_uv @ R_vw @ R_wu != I.

        High cycle rotation error with high appearance similarity is a primary
        indicator of doppelgangers / visual symmetry folding in SfM.
        """
        adj: dict[str, dict[str, np.ndarray]] = {n: {} for n in self.nodes}
        for (u, v), edge in self.active_edges().items():
            if edge.R_rel is not None and edge.num_inliers >= min_inliers:
                R = np.asarray(edge.R_rel, dtype=np.float64).reshape(3, 3)
                adj[u][v] = R
                adj[v][u] = R.T

        inconsistent_cycles = []
        nodes = self.nodes
        n_len = len(nodes)
        for i in range(n_len):
            u = nodes[i]
            for j in range(i + 1, n_len):
                v = nodes[j]
                if v not in adj[u]:
                    continue
                for k in range(j + 1, n_len):
                    w = nodes[k]
                    if w in adj[v] and w in adj[u]:
                        R_uv = adj[u][v]
                        R_vw = adj[v][w]
                        R_wu = adj[w][u]
                        R_loop = R_uv @ R_vw @ R_wu
                        tr = np.clip((np.trace(R_loop) - 1.0) / 2.0, -1.0, 1.0)
                        angle_deg = float(np.degrees(np.arccos(tr)))
                        if angle_deg > max_error_deg:
                            inconsistent_cycles.append({
                                "cycle": [u, v, w],
                                "rotation_error_deg": round(angle_deg, 2),
                                "edges": [[u, v], [v, w], [w, u]],
                            })

        inconsistent_cycles.sort(key=lambda c: c["rotation_error_deg"], reverse=True)
        return inconsistent_cycles

    def find_weak_bridges(self) -> list[list[str]]:
        """Find articulation bridge edges whose removal would disconnect the graph."""
        active = list(self.active_edges().keys())
        if len(active) <= 1:
            return []

        base_components = len(self.get_connected_components())
        bridges: list[list[str]] = []

        for u, v in active:
            edge = self.edges[(u, v)]
            edge.status = "pruned"
            new_comps = len(self.get_connected_components())
            edge.status = "verified"
            if new_comps > base_components:
                bridges.append([u, v])

        return bridges

    def get_summary(self) -> dict[str, Any]:
        """Compute structured diagnostics for the MLLM agent."""
        comps = self.get_connected_components()
        active = self.active_edges()
        failed = self.failed_edges()

        degrees: dict[str, int] = {n: 0 for n in self.nodes}
        for (u, v) in active:
            degrees[u] += 1
            degrees[v] += 1

        deg_vals = list(degrees.values()) if degrees else [0]
        isolated = [n for n, deg in degrees.items() if deg == 0]
        bottlenecks = [n for n, deg in degrees.items() if deg == 1]
        cycles = self.find_cycle_inconsistencies()
        bridges = self.find_weak_bridges()

        return {
            "num_images": len(self.nodes),
            "num_verified_edges": len(active),
            "num_failed_edges": len(failed),
            "num_components": len(comps),
            "component_sizes": [len(c) for c in comps],
            "degrees": {
                "min": min(deg_vals),
                "mean": round(float(np.mean(deg_vals)), 2),
                "max": max(deg_vals),
            },
            "isolated_images": isolated,
            "bottlenecks": bottlenecks,
            "weak_bridges": bridges[:5],
            "suspicious_cycles": cycles[:5],
        }
