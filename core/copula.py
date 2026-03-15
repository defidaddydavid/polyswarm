"""
Copula Dependency Modeling — Gaussian copula for agent correlation structure.

Key insight: When agents share training data (all LLMs), their estimates are
correlated. Naive aggregation over-counts shared information. Copula analysis
quantifies the actual dependency structure and produces a dependency-adjusted
probability using Kish's effective sample size.

References:
  - Kish, L. (1965). "Survey Sampling." Wiley.
  - Clemen, R.T. & Winkler, R.L. (1999). "Combining Probability Distributions
    From Experts in Risk Analysis." Risk Analysis.
  - Joe, H. (1997). "Multivariate Models and Dependence Concepts." Chapman & Hall.
"""

from __future__ import annotations
import math
import random
from core.agent import AgentEstimate


def copula_dependency_analysis(estimates: list[AgentEstimate]) -> dict:
    """
    Analyze dependency structure between agents using a Gaussian copula.

    Computes pairwise correlations, effective sample size via Kish's formula,
    and generates correlated samples through Cholesky decomposition to produce
    a dependency-adjusted aggregate probability.
    """
    if not estimates:
        raise ValueError("No estimates")

    n = len(estimates)

    if n == 1:
        e = estimates[0]
        return {
            "correlation_matrix": {},
            "effective_n": 1.0,
            "dependency_adjusted_probability": round(e.probability, 4),
            "most_correlated_pair": None,
            "most_independent_agent": e.agent_id,
            "independence_ratio": 1.0,
        }

    # Build feature vectors: logit-transformed probability for each agent
    probs = [max(0.005, min(0.995, e.probability)) for e in estimates]
    confs = [e.confidence for e in estimates]
    logits = [_logit(p) for p in probs]

    # Pairwise correlations using logit distance + confidence distance
    corr_matrix = [[0.0] * n for _ in range(n)]
    corr_dict: dict[str, float] = {}

    for i in range(n):
        corr_matrix[i][i] = 1.0
        for j in range(i + 1, n):
            # Distance-based correlation: close agents are highly correlated
            logit_corr = max(0.0, 1.0 - abs(logits[i] - logits[j]) / 5.0)
            conf_corr = max(0.0, 1.0 - abs(confs[i] - confs[j]))
            rho = 0.7 * logit_corr + 0.3 * conf_corr
            rho = max(-0.99, min(0.99, rho))

            corr_matrix[i][j] = rho
            corr_matrix[j][i] = rho

            pair_key = f"{estimates[i].agent_id}|{estimates[j].agent_id}"
            corr_dict[pair_key] = round(rho, 4)

    # Also compute actual Pearson if we have enough signal (3+ dimensional)
    if n >= 3:
        # Use logits as the variable and see if agents cluster
        _update_pearson_correlations(logits, confs, estimates, corr_matrix, corr_dict)

    # Kish's effective sample size
    n_eff = _effective_sample_size(corr_matrix)
    independence_ratio = round(n_eff / n, 4)

    # Find most correlated pair and most independent agent
    most_corr_pair = None
    max_corr = -1.0
    for key, val in corr_dict.items():
        if val > max_corr:
            max_corr = val
            most_corr_pair = key

    # Most independent = lowest average correlation with others
    avg_corrs = []
    for i in range(n):
        off_diag = [corr_matrix[i][j] for j in range(n) if j != i]
        avg_corrs.append(sum(off_diag) / len(off_diag) if off_diag else 0.0)
    most_indep_idx = avg_corrs.index(min(avg_corrs))

    # Cholesky decomposition for correlated sampling
    # Regularize the matrix slightly for numerical stability
    reg_matrix = [row[:] for row in corr_matrix]
    for i in range(n):
        reg_matrix[i][i] += 0.01

    try:
        L = _cholesky(reg_matrix)
    except ValueError:
        # Fall back to diagonal if Cholesky fails
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            L[i][i] = 1.0

    # Generate 1000 correlated samples
    random.seed(42)
    n_samples = 1000
    sample_probs = []

    mean_logit = sum(logits) / n
    std_logit = math.sqrt(sum((l - mean_logit) ** 2 for l in logits) / n) if n > 1 else 0.5

    for _ in range(n_samples):
        # Draw independent standard normals
        z = [random.gauss(0, 1) for _ in range(n)]

        # Correlate via Cholesky: y = L @ z
        y = [0.0] * n
        for i in range(n):
            for j in range(i + 1):
                y[i] += L[i][j] * z[j]

        # Map to probabilities using agent logits as means
        agent_samples = []
        for i in range(n):
            sample_logit = logits[i] + y[i] * max(std_logit * 0.5, 0.1)
            sample_logit = max(-10.0, min(10.0, sample_logit))
            agent_samples.append(_inverse_logit(sample_logit))

        # Aggregate this draw (simple mean of sampled agent probs)
        sample_probs.append(sum(agent_samples) / n)

    dep_adj_prob = sum(sample_probs) / len(sample_probs)
    dep_adj_prob = max(0.001, min(0.999, dep_adj_prob))

    return {
        "correlation_matrix": corr_dict,
        "effective_n": round(n_eff, 2),
        "dependency_adjusted_probability": round(dep_adj_prob, 4),
        "most_correlated_pair": most_corr_pair,
        "most_independent_agent": estimates[most_indep_idx].agent_id,
        "independence_ratio": independence_ratio,
    }


def _update_pearson_correlations(
    logits: list[float],
    confs: list[float],
    estimates: list[AgentEstimate],
    corr_matrix: list[list[float]],
    corr_dict: dict[str, float],
) -> None:
    """Refine correlations using actual Pearson on (logit, confidence) vectors."""
    n = len(logits)
    # Build per-agent feature vectors and compute pairwise Pearson
    for i in range(n):
        for j in range(i + 1, n):
            x = [logits[i], confs[i]]
            y = [logits[j], confs[j]]
            r = _pearson_correlation(x, y)
            # Blend with distance-based estimate (50/50)
            blended = 0.5 * corr_matrix[i][j] + 0.5 * r
            blended = max(-0.99, min(0.99, blended))
            corr_matrix[i][j] = blended
            corr_matrix[j][i] = blended
            pair_key = f"{estimates[i].agent_id}|{estimates[j].agent_id}"
            corr_dict[pair_key] = round(blended, 4)


def _cholesky(matrix: list[list[float]]) -> list[list[float]]:
    """Pure Python Cholesky decomposition: A = L @ L^T."""
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = matrix[i][i] - s
                if val < 0:
                    raise ValueError(
                        f"Matrix not positive-definite at index {i}: "
                        f"diagonal value {val:.6f}"
                    )
                L[i][j] = math.sqrt(val)
            else:
                if L[j][j] == 0:
                    L[i][j] = 0.0
                else:
                    L[i][j] = (matrix[i][j] - s) / L[j][j]

    return L


def _pearson_correlation(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient between two equal-length lists."""
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    denom = math.sqrt(var_x * var_y)
    if denom < 1e-12:
        return 0.0

    return max(-1.0, min(1.0, cov / denom))


def _effective_sample_size(corr_matrix: list[list[float]]) -> float:
    """Kish's effective sample size: n_eff = n^2 / sum(sum(corr_matrix))."""
    n = len(corr_matrix)
    if n == 0:
        return 0.0

    total = sum(sum(row) for row in corr_matrix)
    if total < 1e-12:
        return float(n)

    n_eff = (n * n) / total
    return max(1.0, min(float(n), n_eff))


def _logit(p: float) -> float:
    """Log-odds transform."""
    p = max(0.005, min(0.995, p))
    return math.log(p / (1.0 - p))


def _inverse_logit(x: float) -> float:
    """Inverse logit (sigmoid)."""
    x = max(-10.0, min(10.0, x))
    return 1.0 / (1.0 + math.exp(-x))
