"""
Surprisingly Popular Algorithm — based on Prelec et al. (2017).

Key insight: The correct answer is often the one that is "more popular
than people predict." If 60% of forecasters say YES but they predicted
that 75% would say YES, the "surprisingly popular" answer is actually NO
(since YES got fewer votes than expected).

This exploits private information that agents may not fully incorporate
into their first-order estimates but DO leak through their predictions
about what others will say.

Reference:
  - Prelec, D., Seung, H. S., & McCoy, J. (2017).
    "A solution to the single-question crowd wisdom problem."
    Nature, 541(7638), 532-535.
"""

from __future__ import annotations
import math
from core.agent import AgentEstimate


def surprisingly_popular(
    estimates: list[AgentEstimate],
    meta_predictions: list[float] | None = None,
) -> dict:
    """
    Apply the Surprisingly Popular algorithm.

    Each agent provides:
    1. Their own probability estimate (from AgentEstimate)
    2. Their prediction of what the AVERAGE estimate will be (meta_prediction)

    The SP algorithm then adjusts by:
    SP_score = actual_mean - predicted_mean

    If actual_mean > predicted_mean: the crowd is surprisingly bullish
    → boost the probability further toward YES
    If actual_mean < predicted_mean: the crowd is surprisingly bearish
    → adjust the probability further toward NO

    If meta_predictions are not available (agents don't provide them),
    we estimate them from the confidence-diversity pattern.
    """
    if not estimates:
        raise ValueError("No estimates")

    n = len(estimates)
    probs = [e.probability for e in estimates]
    confs = [e.confidence for e in estimates]

    actual_mean = sum(probs) / n

    # if no meta-predictions provided, estimate them
    if meta_predictions is None:
        meta_predictions = _estimate_meta_predictions(estimates)

    predicted_mean = sum(meta_predictions) / len(meta_predictions)

    # SP score: how much more popular is YES than expected?
    sp_score = actual_mean - predicted_mean

    # apply SP adjustment
    # the surprisingly popular answer gets boosted
    # scale the adjustment by the magnitude of surprise
    adjustment_strength = min(abs(sp_score) * 2.0, 0.15)  # cap at 15% shift
    sp_direction = 1.0 if sp_score > 0 else -1.0

    adjusted_probability = actual_mean + sp_direction * adjustment_strength
    adjusted_probability = max(0.001, min(0.999, adjusted_probability))

    # compute the vote share — fraction of agents above/below 50%
    votes_yes = sum(1 for p in probs if p > 0.5)
    votes_no = sum(1 for p in probs if p < 0.5)
    predicted_votes_yes = sum(1 for mp in meta_predictions if mp > 0.5)

    # "surprisingly popular" flag
    sp_direction_label = "yes" if sp_score > 0 else "no" if sp_score < 0 else "neutral"

    return {
        "sp_adjusted_probability": round(adjusted_probability, 4),
        "sp_score": round(sp_score, 4),
        "actual_mean": round(actual_mean, 4),
        "predicted_mean": round(predicted_mean, 4),
        "surprise_direction": sp_direction_label,
        "adjustment_strength": round(adjustment_strength, 4),
        "votes_yes": votes_yes,
        "votes_no": votes_no,
        "n_agents": n,
    }


def _estimate_meta_predictions(estimates: list[AgentEstimate]) -> list[float]:
    """
    Estimate what each agent thinks the crowd will say, when we don't
    have explicit meta-predictions.

    Heuristic: Agents with high confidence tend to think the crowd will
    agree with them (anchoring bias). Agents with low confidence tend to
    predict the crowd will be near 50%. We model this as a pull toward
    the agent's own estimate, weighted by confidence.

    Meta prediction ≈ confidence * own_estimate + (1 - confidence) * 0.5
    """
    meta = []
    for e in estimates:
        # higher confidence = agent thinks crowd agrees with them
        # lower confidence = agent expects crowd near 50%
        anchor = e.confidence * 0.6  # how much they anchor to themselves
        meta_pred = anchor * e.probability + (1 - anchor) * 0.5
        meta.append(meta_pred)
    return meta
