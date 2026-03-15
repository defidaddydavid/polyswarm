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
