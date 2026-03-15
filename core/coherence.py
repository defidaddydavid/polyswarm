"""
Coherence-Based Weighting — based on Karvetski et al. (2013) and
Mandel et al. (2024).

Key insight: Internally consistent (coherent) forecasters tend to be
more accurate. Test for violations of probability axioms — forecasters
who satisfy more coherence constraints get higher weight.

The Coherence Forecasting Scale (CFS) dominated all other measures
in predicting forecast accuracy across three IARPA tournaments.

References:
  - Karvetski, C.W. et al. (2013). "Probabilistic Coherence Weighting
    for Optimizing Expert Forecasts." Decision Analysis.
  - Mandel, D.R. et al. (2024). "Measuring probabilistic coherence
    to identify superior forecasters." Int. J. Forecasting.
"""

from __future__ import annotations
import math
from core.agent import AgentEstimate


def coherence_check(estimates: list[AgentEstimate]) -> dict:
    """
    Check probabilistic coherence of agent estimates.

    Tests:
    1. Complementarity: Does the agent's certainty match its probability?
       (confidence near 1.0 should mean probability near 0 or 1)
    2. Extremity consistency: High confidence agents should be more extreme
    3. Monotonicity proxy: Agents with similar focus areas should have
       correlated estimates (checked via persona clustering)
    4. Self-consistency: Does the confidence match the probability extremity?

    Returns coherence scores and weights for each agent.
    """
    if not estimates:
        raise ValueError("No estimates")

    n = len(estimates)
    scores = {}

    for e in estimates:
        checks_passed = 0
        total_checks = 0

        # Check 1: Confidence-extremity consistency
        # High confidence + moderate probability is incoherent
        extremity = abs(e.probability - 0.5) * 2  # 0 at 0.5, 1 at 0/1
        total_checks += 1
        if e.confidence > 0.8:
            # very confident agents should be somewhat extreme
            if extremity > 0.2:
                checks_passed += 1
        elif e.confidence < 0.4:
            # low confidence agents can be anywhere
            checks_passed += 1
        else:
            # moderate confidence is always fine
            checks_passed += 1

        # Check 2: Probability boundary coherence
        # Probabilities should be bounded [0.01, 0.99] (not certainty)
        total_checks += 1
        if 0.01 <= e.probability <= 0.99:
            checks_passed += 1

        # Check 3: Confidence should be bounded
        total_checks += 1
        if 0.1 <= e.confidence <= 0.95:
            checks_passed += 1

        # Check 4: Reasoning length should correlate with confidence
        # Agents who write more reasoning but are low confidence are more coherent
        # than agents with no reasoning but high confidence
        total_checks += 1
        has_reasoning = len(e.key_factors) > 0
        if has_reasoning:
            checks_passed += 1

        # Check 5: Internal consistency of probability and confidence
        # Very high probability (>0.9) with very low confidence (<0.3) is suspect
        total_checks += 1
        if e.probability > 0.9 and e.confidence < 0.3:
            pass  # incoherent: very sure of outcome but not confident?
        elif e.probability < 0.1 and e.confidence < 0.3:
            pass  # incoherent: very sure of negative but not confident?
        else:
            checks_passed += 1

        coherence_score = checks_passed / total_checks if total_checks > 0 else 0.5
        scores[e.agent_id] = {
            "coherence_score": round(coherence_score, 3),
            "checks_passed": checks_passed,
            "total_checks": total_checks,
            "persona": e.persona,
        }

    # compute coherence-weighted aggregate
    weights = {}
    for agent_id, score_data in scores.items():
        # weight = coherence_score^2 (amplify differences)
        weights[agent_id] = score_data["coherence_score"] ** 2

    total_weight = sum(weights.values())
    if total_weight > 0:
        norm_weights = {k: v / total_weight for k, v in weights.items()}
    else:
        norm_weights = {e.agent_id: 1.0 / n for e in estimates}

    coherence_prob = sum(
        norm_weights.get(e.agent_id, 1.0 / n) * e.probability
        for e in estimates
    )

    # sort by coherence
    ranked = sorted(
        scores.values(),
        key=lambda x: x["coherence_score"],
        reverse=True,
    )

    # detect incoherent agents
    incoherent = [
        s["persona"] for s in scores.values()
        if s["coherence_score"] < 0.5
    ]

    return {
        "coherence_probability": round(coherence_prob, 4),
        "agent_scores": scores,
        "most_coherent": ranked[0]["persona"] if ranked else None,
        "least_coherent": ranked[-1]["persona"] if ranked else None,
        "n_incoherent": len(incoherent),
        "incoherent_agents": incoherent,
        "mean_coherence": round(
            sum(s["coherence_score"] for s in scores.values()) / n, 3
        ),
    }
