"""
Game-theoretic analysis of agent positions.
Detects herding, contrarian signals, and information cascades.
"""

from __future__ import annotations
import math
from core.agent import AgentEstimate


def detect_herding(estimates: list[AgentEstimate]) -> dict:
    """
    Detect herding behavior — are agents clustering around a consensus
    more than expected from independent reasoning?

    Uses the Herfindahl-Hirschman Index (HHI) adapted for probability space.
    """
    if len(estimates) < 3:
        return {"herding_detected": False, "hhi": 0}

    probs = [e.probability for e in estimates]
    mean_p = sum(probs) / len(probs)

    # bucket probabilities into quintiles
    buckets = [0] * 5
    for p in probs:
        idx = min(int(p * 5), 4)
        buckets[idx] += 1

    # HHI: sum of squared shares
    n = len(probs)
    shares = [b / n for b in buckets]
    hhi = sum(s ** 2 for s in shares)

    # under uniform distribution, HHI = 0.2 (5 buckets)
    # herding when HHI > 0.5
    herding_score = max(0, (hhi - 0.2) / 0.8)

    # find the herd direction
    if herding_score > 0.3:
        herd_prob = sum(p * e.confidence for p, e in zip(probs, estimates)) / sum(e.confidence for e in estimates)
        herd_direction = "bullish" if herd_prob > 0.55 else "bearish" if herd_prob < 0.45 else "neutral"
    else:
        herd_direction = None

    # identify contrarians (agents >1.5 std devs from mean)
    std = math.sqrt(sum((p - mean_p) ** 2 for p in probs) / n)
    contrarians = [
        e.persona for e, p in zip(estimates, probs)
        if abs(p - mean_p) > 1.5 * std
    ] if std > 0.05 else []

    return {
        "herding_score": round(herding_score, 3),
        "herding_detected": herding_score > 0.3,
        "herd_direction": herd_direction,
        "hhi": round(hhi, 3),
        "contrarians": contrarians,
        "contrarian_signal": len(contrarians) > 0 and herding_score > 0.3,
    }


def compute_information_cascade(
    round1_estimates: list[AgentEstimate],
    round2_estimates: list[AgentEstimate],
) -> dict:
    """
    Detect information cascades between debate rounds.
    Did agents converge? Did contrarians flip? Who moved most?
    """
    if not round1_estimates or not round2_estimates:
        return {}

    r1_map = {e.agent_id: e for e in round1_estimates}
    r2_map = {e.agent_id: e for e in round2_estimates}

    movements = []
    flips = []
    total_convergence = 0

    r2_mean = sum(e.probability for e in round2_estimates) / len(round2_estimates)
    r1_mean = sum(e.probability for e in round1_estimates) / len(round1_estimates)

    for agent_id in r1_map:
        if agent_id in r2_map:
            r1_p = r1_map[agent_id].probability
            r2_p = r2_map[agent_id].probability
            delta = r2_p - r1_p

            # did they move toward consensus?
            dist_before = abs(r1_p - r1_mean)
            dist_after = abs(r2_p - r2_mean)
            converged = dist_after < dist_before

            movements.append({
                "agent": r1_map[agent_id].persona,
                "r1": round(r1_p, 3),
                "r2": round(r2_p, 3),
                "delta": round(delta, 3),
                "converged": converged,
            })

            if converged:
                total_convergence += 1

            # flip = changed side of 50%
            if (r1_p > 0.5 and r2_p < 0.5) or (r1_p < 0.5 and r2_p > 0.5):
                flips.append(r1_map[agent_id].persona)

    convergence_rate = total_convergence / len(movements) if movements else 0

    # sort by absolute movement
    movements.sort(key=lambda m: abs(m["delta"]), reverse=True)

    return {
        "convergence_rate": round(convergence_rate, 3),
        "cascade_detected": convergence_rate > 0.7,
        "mean_shift": round(r2_mean - r1_mean, 4),
        "flipped_agents": flips,
        "biggest_movers": movements[:3],
        "all_movements": movements,
    }


def nash_equilibrium_check(estimates: list[AgentEstimate]) -> dict:
    """
    Simplified Nash equilibrium analysis.
    Given the swarm output, would any individual agent benefit from
    deviating? Stable if no agent has incentive to move.
    """
    probs = [e.probability for e in estimates]
    confs = [e.confidence for e in estimates]
    mean_p = sum(p * c for p, c in zip(probs, confs)) / sum(confs)

    deviators = []
    for e in estimates:
        # agent benefits from deviating if their confidence-weighted
        # estimate is far from consensus AND their confidence is high
        distance = abs(e.probability - mean_p)
        if distance > 0.15 and e.confidence > 0.7:
            deviators.append({
                "agent": e.persona,
                "estimate": round(e.probability, 3),
                "consensus": round(mean_p, 3),
                "distance": round(distance, 3),
                "confidence": round(e.confidence, 3),
            })

    return {
        "stable": len(deviators) == 0,
        "potential_deviators": deviators,
        "stability_score": round(1.0 - len(deviators) / len(estimates), 3),
    }


# ---------------------------------------------------------------------------
# Scoring Rule Analysis
# ---------------------------------------------------------------------------

def scoring_rule_analysis(
    estimates: list[AgentEstimate],
    rule: str = "brier",
) -> dict:
    """
    For each agent, compute expected score under truthful reporting vs
    strategic shading (reporting closer to 0.5 to minimize worst-case loss).

    An agent is incentive-compatible if their truthful score is better
    (lower for Brier, higher for log) than their shaded score across
    both possible outcomes.

    Args:
        estimates: list of AgentEstimate objects.
        rule: "brier" or "log".

    Returns:
        dict with agent_scores, n_strategic, strategic_agents, mean_truthfulness.
    """
    score_fn = _brier_score if rule == "brier" else _log_score
    # For Brier, lower is better; for log, higher is better.
    brier_mode = rule == "brier"

    agent_scores = {}
    strategic_agents = []

    for e in estimates:
        p_true = e.probability
        p_shaded = _strategic_shade(p_true)

        # Expected score = p_true * score(forecast, 1) + (1-p_true) * score(forecast, 0)
        # assuming the agent's own probability is the true base rate
        truthful_ev = p_true * score_fn(p_true, 1.0) + (1 - p_true) * score_fn(p_true, 0.0)
        shaded_ev = p_true * score_fn(p_shaded, 1.0) + (1 - p_true) * score_fn(p_shaded, 0.0)

        # Incentive compatible if truthful is at least as good as shaded
        if brier_mode:
            # lower Brier is better
            incentive_compatible = truthful_ev <= shaded_ev + 1e-9
        else:
            # higher log score is better
            incentive_compatible = truthful_ev >= shaded_ev - 1e-9

        agent_scores[e.agent_id] = {
            "truthful_score": round(truthful_ev, 4),
            "shaded_score": round(shaded_ev, 4),
            "incentive_compatible": incentive_compatible,
        }

        if not incentive_compatible:
            strategic_agents.append(e.agent_id)

    n_strategic = len(strategic_agents)
    n_total = len(estimates)
    mean_truthfulness = round(1.0 - n_strategic / n_total, 4) if n_total > 0 else 1.0

    return {
        "agent_scores": agent_scores,
        "n_strategic": n_strategic,
        "strategic_agents": strategic_agents,
        "mean_truthfulness": mean_truthfulness,
    }


def _brier_score(forecast: float, outcome: float) -> float:
    """Brier score: (forecast - outcome)^2. Lower is better."""
    return (forecast - outcome) ** 2


def _log_score(forecast: float, outcome: float) -> float:
    """
    Logarithmic score: outcome*log(f) + (1-outcome)*log(1-f).
    Higher (less negative) is better.
    """
    f = max(1e-7, min(1 - 1e-7, forecast))
    return outcome * math.log(f) + (1 - outcome) * math.log(1 - f)


def _strategic_shade(probability: float, risk_aversion: float = 0.3) -> float:
    """
    Shade a probability toward 0.5 based on risk aversion.
    risk_aversion=0 means no shading, risk_aversion=1 means report 0.5.
    """
    return probability + risk_aversion * (0.5 - probability)
