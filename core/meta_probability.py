"""
Meta-Probability Weighting (MPW) — based on Palley & Satopää (2023).

Key insight: Weight forecasters not just by their answer, but by the gap
between their answer and their meta-prediction of others' answers.
Forecasters whose estimates deviate most from what they expect others
to say are signaling that they have private information.

Also implements Neutral Pivoting (2024) which corrects for shared
information bias without needing to estimate crowd composition.

References:
  - Palley, A.B. & Satopää, V.A. (2023). "Boosting the Wisdom of
    Crowds Within a Single Judgment Problem." Working paper.
  - Decision Analysis (2024). Neutral Pivoting for shared-information correction.
"""

from __future__ import annotations
import math
from core.agent import AgentEstimate


def meta_probability_weight(
    estimates: list[AgentEstimate],
    meta_predictions: list[float] | None = None,
) -> dict:
    """
    Meta-Probability Weighting: weight agents by their information signal.

    w_i = |p_i - m_i| / sum(|p_j - m_j|)

    Where p_i is the forecast and m_i is agent i's prediction of the
    average forecast. Agents who diverge most from expected consensus
    are weighted higher — they're signaling private information.

    If meta_predictions not provided, estimates them from confidence.
    """
    if not estimates:
        raise ValueError("No estimates")

    n = len(estimates)
    probs = [e.probability for e in estimates]

    if meta_predictions is None:
        meta_predictions = _estimate_meta_predictions(estimates)

    # compute information signal for each agent
    signals = []
    for p, m in zip(probs, meta_predictions):
        signal = abs(p - m)
        signals.append(signal)

    total_signal = sum(signals)

    # compute weights (handle case where all signals are zero)
    if total_signal > 0:
        weights = [s / total_signal for s in signals]
    else:
        weights = [1.0 / n] * n

    # weighted aggregate
    mpw_prob = sum(w * p for w, p in zip(weights, probs))
    mpw_prob = max(0.001, min(0.999, mpw_prob))

    # identify which agents have strongest signals
    agent_signals = sorted(
        [
            {"agent": e.persona, "signal": round(s, 4), "weight": round(w, 4)}
            for e, s, w in zip(estimates, signals, weights)
        ],
        key=lambda x: x["signal"],
        reverse=True,
    )

    return {
        "mpw_probability": round(mpw_prob, 4),
        "top_signal_agents": agent_signals[:3],
        "mean_signal": round(sum(signals) / n, 4),
        "n_agents": n,
    }


def neutral_pivot(
    estimates: list[AgentEstimate],
    meta_predictions: list[float] | None = None,
    alpha: float = 1.0,
) -> dict:
    """
    Neutral Pivoting (Decision Analysis, 2024).

    Separates shared vs. private information to correct for
    shared-information bias (critical for LLM-based systems where
    all agents share training data).

    p_adjusted = p_mean + alpha * (p_mean - p_predicted_mean)

    This aggressively corrects for the shared-information component
    without needing to estimate crowd composition.
    """
    if not estimates:
        raise ValueError("No estimates")

    n = len(estimates)
    probs = [e.probability for e in estimates]
    actual_mean = sum(probs) / n

    if meta_predictions is None:
        meta_predictions = _estimate_meta_predictions(estimates)

    predicted_mean = sum(meta_predictions) / len(meta_predictions)

    # pivot: double-count the surprise
    pivot = actual_mean + alpha * (actual_mean - predicted_mean)
    pivot = max(0.001, min(0.999, pivot))

    return {
        "pivoted_probability": round(pivot, 4),
        "actual_mean": round(actual_mean, 4),
        "predicted_mean": round(predicted_mean, 4),
        "pivot_shift": round(pivot - actual_mean, 4),
        "alpha": alpha,
    }


def _estimate_meta_predictions(estimates: list[AgentEstimate]) -> list[float]:
    """
    Estimate what each agent thinks the crowd will say.
    Higher confidence agents anchor more to their own estimate.
    """
    meta = []
    for e in estimates:
        anchor = e.confidence * 0.6
        meta_pred = anchor * e.probability + (1 - anchor) * 0.5
        meta.append(meta_pred)
    return meta
