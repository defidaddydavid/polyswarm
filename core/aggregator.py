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
