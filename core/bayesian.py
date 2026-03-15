"""
Bayesian belief updating engine.
Agents don't just average — they update beliefs using Bayes' theorem
with information-theoretic weighting.
"""

from __future__ import annotations
import math
from core.agent import AgentEstimate


def bayesian_aggregate(estimates: list[AgentEstimate], prior: float = 0.5) -> dict:
    """
    Bayesian aggregation of agent estimates.
    Treats each agent's estimate as evidence and updates from a prior.
    Weights by confidence and information content (Shannon entropy).
    """
    if not estimates:
        raise ValueError("No estimates")

    posterior = prior

    for est in estimates:
        p = max(0.001, min(0.999, est.probability))
        c = max(0.01, est.confidence)

        # information content: how surprising is this estimate relative to prior?
        kl_divergence = _kl_div(p, posterior)
        info_weight = min(c * (1 + kl_divergence), 1.0)

        # bayesian update: treat agent estimate as likelihood
        likelihood_yes = p ** info_weight
        likelihood_no = (1 - p) ** info_weight

        numerator = likelihood_yes * posterior
        denominator = numerator + likelihood_no * (1 - posterior)

        if denominator > 0:
            posterior = numerator / denominator

    posterior = max(0.001, min(0.999, posterior))

    # compute aggregate metrics
    probs = [e.probability for e in estimates]
    mean_prob = sum(probs) / len(probs)
    variance = sum((p - mean_prob) ** 2 for p in probs) / len(probs)

    return {
        "bayesian_probability": round(posterior, 4),
        "simple_mean": round(mean_prob, 4),
        "bayesian_shift": round(posterior - mean_prob, 4),
        "prior": prior,
        "variance": round(variance, 4),
        "entropy": round(_shannon_entropy(posterior), 4),
        "information_gain": round(_shannon_entropy(prior) - _shannon_entropy(posterior), 4),
    }


def _kl_div(p: float, q: float) -> float:
    """KL divergence D(p||q) for Bernoulli distributions."""
    p = max(0.001, min(0.999, p))
    q = max(0.001, min(0.999, q))
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def _shannon_entropy(p: float) -> float:
    """Shannon entropy of a Bernoulli distribution."""
    p = max(0.001, min(0.999, p))
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def compute_agent_agreement_matrix(estimates: list[AgentEstimate]) -> dict:
    """
    Compute pairwise agreement between agents.
    Returns a matrix of Jensen-Shannon divergences.
    """
    n = len(estimates)
    matrix = {}
    for i in range(n):
        for j in range(i + 1, n):
            pi = max(0.001, min(0.999, estimates[i].probability))
            pj = max(0.001, min(0.999, estimates[j].probability))
            m = (pi + pj) / 2
            jsd = 0.5 * _kl_div(pi, m) + 0.5 * _kl_div(pj, m)
            key = f"{estimates[i].agent_id}|{estimates[j].agent_id}"
            matrix[key] = round(jsd, 4)

    # find most agreeing and most disagreeing pairs
    if matrix:
        most_agree = min(matrix, key=matrix.get)
        most_disagree = max(matrix, key=matrix.get)
    else:
        most_agree = most_disagree = None

    return {
        "pairwise_jsd": matrix,
        "most_aligned": most_agree,
        "most_divergent": most_disagree,
        "mean_divergence": round(sum(matrix.values()) / len(matrix), 4) if matrix else 0,
    }
