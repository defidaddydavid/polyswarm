"""
Weighted aggregation of agent estimates into a final probability.
Weights are based on: self-reported confidence + historical calibration score.
"""

from __future__ import annotations
from core.agent import AgentEstimate
import math


def aggregate(
    estimates: list[AgentEstimate],
    calibration_weights: dict[str, float] | None = None,
) -> dict:
    """
    Produce a final probability from a list of agent estimates.
    Uses confidence-weighted average, optionally adjusted by historical calibration.
    """
    if not estimates:
        raise ValueError("No estimates to aggregate")

    weights = []
    probs = []

    for est in estimates:
        base_weight = est.confidence
        cal_weight = calibration_weights.get(est.agent_id, 1.0) if calibration_weights else 1.0
        combined_weight = base_weight * cal_weight
        weights.append(combined_weight)
        probs.append(est.probability)

    total_weight = sum(weights)
    if total_weight == 0:
        weighted_prob = sum(probs) / len(probs)
    else:
        weighted_prob = sum(p * w for p, w in zip(probs, weights)) / total_weight

    # variance across estimates — high variance = less consensus
    mean = weighted_prob
    variance = sum(w * (p - mean) ** 2 for p, w in zip(probs, weights)) / total_weight if total_weight > 0 else 0
    std_dev = math.sqrt(variance)
    consensus_score = max(0.0, 1.0 - (std_dev * 2))  # 0=no consensus, 1=full consensus

    return {
        "probability": round(weighted_prob, 4),
        "probability_pct": f"{weighted_prob:.1%}",
        "consensus_score": round(consensus_score, 3),
        "std_dev": round(std_dev, 4),
        "n_agents": len(estimates),
        "individual_estimates": [
            {
                "agent_id": e.agent_id,
                "persona": e.persona,
                "probability": e.probability,
                "confidence": e.confidence,
                "reasoning": e.reasoning,
                "key_factors": e.key_factors,
                "round": e.round,
            }
            for e in estimates
        ],
    }


def ensemble_aggregate(
    weighted_result: dict,
    bayesian_result: dict,
    mc_result: dict,
    extremized_result: dict | None = None,
    logop_result: dict | None = None,
    sp_result: dict | None = None,
    cooke_result: dict | None = None,
) -> dict:
    """
    Ensemble of all aggregation methods using robust median.
    Combines: weighted, Bayesian, MC, extremized (IARPA),
    log opinion pool, surprisingly popular (Prelec), Cooke's classical.
    """
    w_prob = weighted_result["probability"]
    b_prob = bayesian_result["bayesian_probability"]
    mc_prob = mc_result["mean"]

    methods = {
        "weighted": round(w_prob, 4),
        "bayesian": round(b_prob, 4),
        "monte_carlo": round(mc_prob, 4),
    }
    all_probs = [w_prob, b_prob, mc_prob]

    if extremized_result:
        ext_p = extremized_result["extremized_probability"]
        methods["extremized"] = round(ext_p, 4)
        all_probs.append(ext_p)
    if logop_result:
        logop_p = logop_result["logop_probability"]
        methods["log_opinion_pool"] = round(logop_p, 4)
        all_probs.append(logop_p)
    if sp_result:
        sp_p = sp_result["sp_adjusted_probability"]
        methods["surprisingly_popular"] = round(sp_p, 4)
        all_probs.append(sp_p)
    if cooke_result:
        cooke_p = cooke_result["cooke_probability"]
        methods["cooke_classical"] = round(cooke_p, 4)
        all_probs.append(cooke_p)

    # robust median
    sorted_probs = sorted(all_probs)
    n = len(sorted_probs)
    if n % 2 == 0:
        median = (sorted_probs[n // 2 - 1] + sorted_probs[n // 2]) / 2
    else:
        median = sorted_probs[n // 2]

    mean = sum(all_probs) / n
    spread = max(all_probs) - min(all_probs)

    return {
        "ensemble_probability": round(median, 4),
        "ensemble_mean": round(mean, 4),
        "ensemble_pct": f"{median:.1%}",
        "n_methods": n,
        "method_spread": round(spread, 4),
        "methods": methods,
        "agreement": "high" if spread < 0.05 else "moderate" if spread < 0.10 else "low",
    }


# ---------------------------------------------------------------------------
# Stacking Aggregation (Ridge Regression Meta-Learner)
# ---------------------------------------------------------------------------

def stacking_aggregate(
    method_results: dict[str, float],
    history: list[dict] | None = None,
) -> dict:
    """
    Meta-learner that combines multiple aggregation methods via ridge regression.

    If historical data is provided (list of {methods: {name: prob}, outcome: float}),
    learns optimal weights. Otherwise uses equal weights.

    Args:
        method_results: dict of method_name -> probability for the current question.
        history: list of past records with method predictions and outcomes.

    Returns:
        dict with stacking_probability, method_weights, regularization,
        n_historical, method_used.
    """
    names = sorted(method_results.keys())
    current_probs = [method_results[n] for n in names]
    n_methods = len(names)

    if n_methods == 0:
        return {
            "stacking_probability": 0.5,
            "method_weights": {},
            "regularization": 0.0,
            "n_historical": 0,
            "method_used": "equal",
        }

    # Try ridge regression if we have enough history
    if history and len(history) >= max(n_methods + 2, 10):
        # Build X matrix and y vector from history
        X = []
        y = []
        for record in history:
            row = []
            methods_dict = record.get("methods", {})
            valid = True
            for name in names:
                if name in methods_dict:
                    row.append(methods_dict[name])
                else:
                    valid = False
                    break
            if valid:
                X.append(row)
                y.append(float(record["outcome"]))

        if len(X) >= max(n_methods + 2, 10):
            alpha = 1.0
            try:
                weights = _ridge_regression(X, y, alpha=alpha)

                # Compute stacking probability
                raw = sum(w * p for w, p in zip(weights, current_probs))
                stacking_prob = max(0.001, min(0.999, raw))

                method_weights = {n: round(w, 4) for n, w in zip(names, weights)}

                return {
                    "stacking_probability": round(stacking_prob, 4),
                    "method_weights": method_weights,
                    "regularization": alpha,
                    "n_historical": len(X),
                    "method_used": "ridge",
                }
            except Exception:
                pass  # fall through to equal weights

    # Equal weights fallback
    equal_prob = sum(current_probs) / n_methods
    equal_prob = max(0.001, min(0.999, equal_prob))
    method_weights = {n: round(1.0 / n_methods, 4) for n in names}

    return {
        "stacking_probability": round(equal_prob, 4),
        "method_weights": method_weights,
        "regularization": 0.0,
        "n_historical": 0,
        "method_used": "equal",
    }


def _ridge_regression(
    X: list[list[float]],
    y: list[float],
    alpha: float = 1.0,
) -> list[float]:
    """
    Ridge regression: w = (X^T X + alpha*I)^{-1} X^T y.
    Pure Python using Gauss-Jordan elimination for matrix inverse.
    """
    Xt = _matrix_transpose(X)
    XtX = _matrix_multiply(Xt, X)

    # Add regularization: XtX + alpha * I
    n = len(XtX)
    reg = [[alpha if i == j else 0.0 for j in range(n)] for i in range(n)]
    XtX_reg = _matrix_add(XtX, reg)

    # Invert
    XtX_inv = _matrix_inverse(XtX_reg)

    # X^T y (treat y as column vector)
    y_col = [[yi] for yi in y]
    Xty = _matrix_multiply(Xt, y_col)

    # w = XtX_inv * Xty
    w_col = _matrix_multiply(XtX_inv, Xty)
    return [row[0] for row in w_col]


def _matrix_multiply(
    A: list[list[float]],
    B: list[list[float]],
) -> list[list[float]]:
    """Multiply two matrices A (m x n) and B (n x p)."""
    m = len(A)
    n = len(B)
    p = len(B[0]) if n > 0 else 0
    result = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result


def _matrix_transpose(A: list[list[float]]) -> list[list[float]]:
    """Transpose a matrix."""
    if not A:
        return []
    m = len(A)
    n = len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def _matrix_add(
    A: list[list[float]],
    B: list[list[float]],
) -> list[list[float]]:
    """Element-wise addition of two matrices."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _matrix_inverse(A: list[list[float]]) -> list[list[float]]:
    """
    Matrix inverse via Gauss-Jordan elimination.
    Works on small matrices (typical: 3-10 aggregation methods).
    """
    n = len(A)
    # Augment with identity
    aug = [row[:] + [1.0 if j == i else 0.0 for j in range(n)] for i, row in enumerate(A)]

    for col in range(n):
        # Partial pivoting
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular or near-singular")

        # Scale pivot row
        for j in range(2 * n):
            aug[col][j] /= pivot

        # Eliminate column in other rows
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]

    # Extract inverse from augmented matrix
    return [row[n:] for row in aug]
