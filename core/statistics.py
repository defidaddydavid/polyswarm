"""
Statistical analysis module — distribution fitting, confidence intervals,
Monte Carlo simulation for probability estimates.
"""

from __future__ import annotations
import math
import random
from core.agent import AgentEstimate


def bootstrap_confidence_interval(
    estimates: list[AgentEstimate],
    n_samples: int = 1000,
    ci_level: float = 0.95,
) -> dict:
    """
    Bootstrap confidence interval for the swarm probability.
    Resamples agent estimates with replacement to estimate uncertainty.
    """
    probs = [e.probability for e in estimates]
    confs = [e.confidence for e in estimates]

    bootstrapped_means = []
    for _ in range(n_samples):
        sample_indices = [random.randint(0, len(probs) - 1) for _ in range(len(probs))]
        sample_probs = [probs[i] for i in sample_indices]
        sample_confs = [confs[i] for i in sample_indices]
        total_conf = sum(sample_confs)
        if total_conf > 0:
            weighted_mean = sum(p * c for p, c in zip(sample_probs, sample_confs)) / total_conf
        else:
            weighted_mean = sum(sample_probs) / len(sample_probs)
        bootstrapped_means.append(weighted_mean)

    bootstrapped_means.sort()
    alpha = (1 - ci_level) / 2
    lower_idx = int(alpha * n_samples)
    upper_idx = int((1 - alpha) * n_samples) - 1

    return {
        "mean": round(sum(bootstrapped_means) / n_samples, 4),
        "ci_lower": round(bootstrapped_means[lower_idx], 4),
        "ci_upper": round(bootstrapped_means[upper_idx], 4),
        "ci_level": ci_level,
        "ci_width": round(bootstrapped_means[upper_idx] - bootstrapped_means[lower_idx], 4),
        "std_error": round(_std(bootstrapped_means), 4),
    }


def monte_carlo_scenarios(
    estimates: list[AgentEstimate],
    n_simulations: int = 5000,
) -> dict:
    """
    Monte Carlo simulation treating each agent's estimate as a
    distribution (beta distribution parameterized by their probability and confidence).
    """
    results = []

    for _ in range(n_simulations):
        sim_probs = []
        for e in estimates:
            # use beta distribution: higher confidence = tighter distribution
            alpha_param = e.probability * e.confidence * 20 + 1
            beta_param = (1 - e.probability) * e.confidence * 20 + 1
            sample = _beta_sample(alpha_param, beta_param)
            sim_probs.append(sample)

        # weighted average for this simulation
        total_conf = sum(e.confidence for e in estimates)
        weighted = sum(p * e.confidence for p, e in zip(sim_probs, estimates)) / total_conf
        results.append(weighted)

    results.sort()

    # compute percentiles
    percentiles = {}
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        idx = int(pct / 100 * len(results))
        percentiles[f"p{pct}"] = round(results[idx], 4)

    # probability of being above/below key thresholds
    thresholds = {}
    for thresh in [0.25, 0.50, 0.75]:
        above = sum(1 for r in results if r > thresh) / len(results)
        thresholds[f"P(>{thresh:.0%})"] = round(above, 3)

    return {
        "percentiles": percentiles,
        "thresholds": thresholds,
        "mean": round(sum(results) / len(results), 4),
        "std": round(_std(results), 4),
        "skew": round(_skewness(results), 4),
        "n_simulations": n_simulations,
    }


def _beta_sample(alpha: float, beta: float) -> float:
    """Simple beta distribution sample using gamma sampling."""
    alpha = max(0.1, alpha)
    beta = max(0.1, beta)
    x = random.gammavariate(alpha, 1)
    y = random.gammavariate(beta, 1)
    return x / (x + y) if (x + y) > 0 else 0.5


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def _skewness(values: list[float]) -> float:
    n = len(values)
    if n < 3:
        return 0
    mean = sum(values) / n
    std = _std(values)
    if std == 0:
        return 0
    return sum(((v - mean) / std) ** 3 for v in values) * n / ((n - 1) * (n - 2))
