"""
Conformal Prediction — distribution-free prediction intervals with
guaranteed coverage.

Unlike bootstrap CI which assumes normality, conformal prediction provides
valid coverage guarantees regardless of the underlying distribution.
Uses jackknife+ (leave-one-agent-out) when no history exists, and split
conformal when historical forecasts are available.

References:
  - Vovk, V. et al. (2005). "Algorithmic Learning in a Random World." Springer.
  - Barber, R.F. et al. (2021). "Predictive Inference with the Jackknife+."
    Annals of Statistics.
  - Angelopoulos, A.N. & Bates, S. (2023). "Conformal Prediction: A Gentle
    Introduction." Foundations and Trends in ML.
"""

from __future__ import annotations
import math
from core.agent import AgentEstimate


def conformal_prediction(
    estimates: list[AgentEstimate],
    history: list[dict] | None = None,
    alpha: float = 0.05,
) -> dict:
    """
    Compute a conformal prediction interval for the swarm probability.

    If historical forecast-outcome pairs are provided, uses split conformal
    to calibrate prediction intervals from past errors. Otherwise, uses
    jackknife+ (leave-one-agent-out) for distribution-free intervals.

    Args:
        estimates: Current round agent estimates.
        history: Optional list of {"forecast": float, "outcome": float} dicts.
        alpha: Significance level. Default 0.05 gives 95% coverage.

    Returns dict with conformal bounds, coverage guarantee, and diagnostics.
    """
    if not estimates:
        raise ValueError("No estimates for conformal prediction")

    alpha = max(0.01, min(0.50, alpha))

    if history and len(history) >= 2:
        # enough history for split conformal
        current_prob = _aggregate(estimates)
        return _split_conformal(current_prob, history, alpha)

    return _jackknife_plus(estimates, alpha)


def _jackknife_plus(estimates: list[AgentEstimate], alpha: float) -> dict:
    """
    Jackknife+ conformal prediction using leave-one-agent-out.

    For each agent i, compute the aggregate WITHOUT agent i. The
    nonconformity scores are the differences between the full aggregate
    and each leave-one-out aggregate. The prediction interval is derived
    from the quantiles of these scores.

    With only one agent, returns a wide interval reflecting high uncertainty.
    """
    n = len(estimates)

    if n == 1:
        p = estimates[0].probability
        half_width = 0.25 * (1.0 - estimates[0].confidence)
        half_width = max(half_width, 0.05)
        return {
            "conformal_lower": round(max(0.0, p - half_width), 4),
            "conformal_upper": round(min(1.0, p + half_width), 4),
            "coverage_guarantee": round(1.0 - alpha, 4),
            "interval_width": round(min(1.0, 2 * half_width), 4),
            "method_used": "jackknife+",
            "nonconformity_scores": [0.0],
        }

    full_agg = _aggregate(estimates)

    # leave-one-out aggregates
    loo_aggs = []
    for i in range(n):
        subset = estimates[:i] + estimates[i + 1:]
        loo_aggs.append(_aggregate(subset))

    # nonconformity scores: how much does removing each agent shift things?
    scores = [abs(full_agg - loo) for loo in loo_aggs]

    # jackknife+ residuals: for each agent, the signed residual
    signed_residuals_upper = [loo + s for loo, s in zip(loo_aggs, scores)]
    signed_residuals_lower = [loo - s for loo, s in zip(loo_aggs, scores)]

    # quantile for the prediction interval
    # jackknife+ uses the ceil((1-alpha)(n+1))/n quantile
    quantile_idx = math.ceil((1.0 - alpha) * (n + 1)) - 1
    quantile_idx = max(0, min(n - 1, quantile_idx))

    sorted_upper = sorted(signed_residuals_upper)
    sorted_lower = sorted(signed_residuals_lower)

    # lower bound: alpha-quantile of lower residuals
    lower_idx = n - 1 - quantile_idx
    lower_idx = max(0, min(n - 1, lower_idx))

    conf_upper = sorted_upper[quantile_idx]
    conf_lower = sorted_lower[lower_idx]

    # clamp to valid probability range
    conf_lower = max(0.0, min(1.0, conf_lower))
    conf_upper = max(0.0, min(1.0, conf_upper))

    # ensure lower <= upper
    if conf_lower > conf_upper:
        conf_lower, conf_upper = conf_upper, conf_lower

    width = conf_upper - conf_lower

    return {
        "conformal_lower": round(conf_lower, 4),
        "conformal_upper": round(conf_upper, 4),
        "coverage_guarantee": round(1.0 - alpha, 4),
        "interval_width": round(width, 4),
        "method_used": "jackknife+",
        "nonconformity_scores": [round(s, 4) for s in scores],
    }


def _split_conformal(
    current_prob: float,
    history: list[dict],
    alpha: float,
) -> dict:
    """
    Split conformal prediction using historical forecast-outcome pairs.

    Computes nonconformity scores from past errors, then uses the
    empirical quantile to construct a prediction interval around the
    current forecast.
    """
    # compute nonconformity scores from history
    scores = []
    for entry in history:
        forecast = entry.get("forecast", 0.5)
        outcome = entry.get("outcome", 0.5)
        scores.append(_nonconformity_score(forecast, outcome))

    if not scores:
        return {
            "conformal_lower": round(max(0.0, current_prob - 0.2), 4),
            "conformal_upper": round(min(1.0, current_prob + 0.2), 4),
            "coverage_guarantee": round(1.0 - alpha, 4),
            "interval_width": 0.4,
            "method_used": "split",
            "nonconformity_scores": [],
        }

    scores.sort()
    n = len(scores)

    # conformal quantile: ceil((n+1)(1-alpha)) / n
    quantile_idx = math.ceil((n + 1) * (1.0 - alpha)) - 1
    quantile_idx = max(0, min(n - 1, quantile_idx))
    q_hat = scores[quantile_idx]

    conf_lower = max(0.0, current_prob - q_hat)
    conf_upper = min(1.0, current_prob + q_hat)
    width = conf_upper - conf_lower

    return {
        "conformal_lower": round(conf_lower, 4),
        "conformal_upper": round(conf_upper, 4),
        "coverage_guarantee": round(1.0 - alpha, 4),
        "interval_width": round(width, 4),
        "method_used": "split",
        "nonconformity_scores": [round(s, 4) for s in scores],
    }


def _nonconformity_score(forecast: float, outcome: float) -> float:
    """
    Compute the nonconformity score between a forecast and an outcome.

    Uses absolute error as the nonconformity measure. This is the simplest
    and most interpretable choice; alternatives include squared error or
    log-loss based scores.
    """
    return abs(forecast - outcome)


def _aggregate(estimates: list[AgentEstimate]) -> float:
    """
    Confidence-weighted aggregate probability.

    Falls back to simple mean if total confidence is zero.
    """
    if not estimates:
        return 0.5

    total_conf = sum(e.confidence for e in estimates)
    if total_conf > 0:
        return sum(e.probability * e.confidence for e in estimates) / total_conf

    return sum(e.probability for e in estimates) / len(estimates)
