"""Eval subpackage."""

from agentic_sfm.eval.evaluate import (
    compare_methods,
    evaluate_agent,
    evaluate_direct_matching,
    save_results,
)
from agentic_sfm.eval.scene_eval import (
    aggregate_scene_metrics,
    compute_efficiency_frontier,
    compute_scene_metrics,
)

__all__ = [
    "aggregate_scene_metrics",
    "compare_methods",
    "compute_efficiency_frontier",
    "compute_scene_metrics",
    "evaluate_agent",
    "evaluate_direct_matching",
    "save_results",
]
