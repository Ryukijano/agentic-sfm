#!/usr/bin/env python
"""Budget-conditioned test-time & RL compute scaling law evaluator for Agentic-SfM.

Fits the joint scaling law:
    E(B, C_RL) = E_inf + a * B^(-alpha) + c * C_RL^(-beta)
where:
  - B is the test-time tool call budget (B in {1, 2, 4, 8, 16, 32}),
  - C_RL is the RL training compute (e.g. rollout episodes or gradient steps),
  - E(B, C_RL) is the geometric error metric (e.g. 1 - AUC@10 or pose error).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logger = logging.getLogger(__name__)

DEFAULT_BUDGETS = (1, 2, 4, 8, 16, 32)


def _solve_nonnegative_linear(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve min_{theta >= 0} ||X @ theta - y||_2^2 for small column count (<= 3)."""
    k = X.shape[1]
    best_theta = np.zeros(k, dtype=np.float64)
    best_rss = float(np.sum(y ** 2))

    # Enumerate all active subsets of columns (2^k <= 8 for k <= 3).
    for mask in range(1, 1 << k):
        cols = [j for j in range(k) if (mask >> j) & 1]
        X_sub = X[:, cols]
        theta_sub, _, _, _ = np.linalg.lstsq(X_sub, y, rcond=None)
        if np.all(theta_sub >= -1e-12):
            theta_sub = np.maximum(theta_sub, 0.0)
            theta = np.zeros(k, dtype=np.float64)
            theta[cols] = theta_sub
            resid = X @ theta - y
            rss = float(np.dot(resid, resid))
            if rss < best_rss:
                best_rss = rss
                best_theta = theta
    return best_theta


def fit_budget_compute_scaling_law(records: list[dict[str, Any]]) -> dict[str, float]:
    """Fit E(B, C_RL) = E_inf + a * B^(-alpha) + c * C_RL^(-beta).

    Each dict in ``records`` may provide:
      - ``budget`` or ``B``: tool call budget B > 0
      - ``compute`` or ``C_rl`` or ``C``: RL training compute C_RL > 0 (defaults to 1.0)
      - ``error`` or ``E``: error rate in [0, inf), or ``auc`` (converted via 1 - auc)

    Uses profile non-negative least squares over a coarse-to-fine grid in
    (alpha, beta) so linear parameters (E_inf, a, c) are globally optimal for
    every candidate exponent pair.
    """
    if not records:
        raise ValueError("records must not be empty")

    B_list: list[float] = []
    C_list: list[float] = []
    E_list: list[float] = []
    for r in records:
        b = r.get("budget", r.get("B"))
        if b is None or float(b) <= 0:
            raise ValueError(f"Record missing positive budget/B: {r}")
        c_val = r.get("compute", r.get("C_rl", r.get("C", 1.0)))
        if float(c_val) <= 0:
            raise ValueError(f"Record missing positive compute/C_rl: {r}")
        if "error" in r:
            e_val = float(r["error"])
        elif "E" in r:
            e_val = float(r["E"])
        elif "auc" in r:
            e_val = 1.0 - float(r["auc"])
        else:
            raise ValueError(f"Record missing error/E/auc field: {r}")
        B_list.append(float(b))
        C_list.append(float(c_val))
        E_list.append(e_val)

    B = np.asarray(B_list, dtype=np.float64)
    C = np.asarray(C_list, dtype=np.float64)
    y = np.asarray(E_list, dtype=np.float64)
    n = len(y)
    ones = np.ones(n, dtype=np.float64)
    vary_c = bool(np.ptp(C) > 1e-9)

    def _eval_exponents(alpha: float, beta: float) -> tuple[float, np.ndarray]:
        b_term = np.power(B, -alpha)
        if vary_c:
            c_term = np.power(C, -beta)
            X = np.column_stack([ones, b_term, c_term])
        else:
            X = np.column_stack([ones, b_term])
        theta = _solve_nonnegative_linear(X, y)
        resid = X @ theta - y
        rss = float(np.dot(resid, resid))
        if not vary_c:
            theta = np.array([theta[0], theta[1], 0.0], dtype=np.float64)
        return rss, theta

    # Stage 1: coarse grid search over exponents in [0.05, 2.5]
    alpha_grid = np.linspace(0.05, 2.5, 50)
    beta_grid = np.linspace(0.05, 2.5, 50) if vary_c else np.array([0.0])

    best_rss = float("inf")
    best_alpha = 0.5
    best_beta = 0.5 if vary_c else 0.0
    best_theta = np.zeros(3, dtype=np.float64)

    for a_cand in alpha_grid:
        for b_cand in beta_grid:
            rss, theta = _eval_exponents(float(a_cand), float(b_cand))
            if rss < best_rss:
                best_rss = rss
                best_alpha = float(a_cand)
                best_beta = float(b_cand)
                best_theta = theta

    # Stage 2: multi-scale local refinement around best (alpha, beta)
    radius_a = 0.1
    radius_b = 0.1 if vary_c else 0.0
    for _ in range(5):
        a_lo = max(1e-3, best_alpha - radius_a)
        a_hi = min(5.0, best_alpha + radius_a)
        a_sub = np.linspace(a_lo, a_hi, 21)
        if vary_c:
            b_lo = max(1e-3, best_beta - radius_b)
            b_hi = min(5.0, best_beta + radius_b)
            b_sub = np.linspace(b_lo, b_hi, 21)
        else:
            b_sub = np.array([0.0])
        for a_cand in a_sub:
            for b_cand in b_sub:
                rss, theta = _eval_exponents(float(a_cand), float(b_cand))
                if rss < best_rss:
                    best_rss = rss
                    best_alpha = float(a_cand)
                    best_beta = float(b_cand)
                    best_theta = theta
        radius_a *= 0.2
        radius_b *= 0.2

    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (best_rss / ss_tot) if ss_tot > 1e-15 else 1.0
    rmse = float(np.sqrt(best_rss / max(n, 1)))

    return {
        "E_inf": float(best_theta[0]),
        "a": float(best_theta[1]),
        "alpha": float(best_alpha),
        "c": float(best_theta[2]),
        "beta": float(best_beta),
        "r2": float(r2),
        "rmse": float(rmse),
    }


def predict_scaling_error(params: dict[str, float], budget: float, compute: float = 1.0) -> float:
    """Evaluate E(B, C_RL) under fitted scaling law parameters."""
    e_inf = float(params["E_inf"])
    a = float(params["a"])
    alpha = float(params["alpha"])
    c = float(params.get("c", 0.0))
    beta = float(params.get("beta", 0.0))
    term_c = c * (float(compute) ** (-beta)) if c > 0.0 and beta > 0.0 else 0.0
    return float(e_inf + a * (float(budget) ** (-alpha)) + term_c)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_BUDGETS),
        help="Tool call budgets B to evaluate (default: 1 2 4 8 16 32)",
    )
    parser.add_argument(
        "--records-json",
        type=str,
        default=None,
        help="Path to JSON file containing [{budget, compute, error}, ...] records to fit.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/budget_scaling/scaling_fit.json",
        help="Output JSON path for fitted scaling law parameters.",
    )
    args = parser.parse_args()

    if args.records_json:
        with open(args.records_json) as f:
            records = json.load(f)
    else:
        raise SystemExit(
            "Provide --records-json with evaluated (budget, compute, error) records."
        )

    fit = fit_budget_compute_scaling_law(records)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "budgets": args.budgets,
        "fit": fit,
        "num_records": len(records),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
