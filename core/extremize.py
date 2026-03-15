"""
Extremized aggregation — based on Satopää et al. (2013) and Baron et al. (2014).

Key insight from IARPA ACE tournament research: simple averages of forecaster
probabilities are systematically under-confident. "Extremizing" pushes the
aggregate away from 50% toward 0 or 1, improving calibration.

Reference:
  - Satopää, V. A., et al. (2014). "Combining multiple probability predictions
    using a simple logit model." International Journal of Forecasting.
  - Baron, J., et al. (2014). "Two Reasons to Make Aggregated Probability
    Forecasts More Extreme." Decision Analysis.
  - Tetlock, P. E. (2015). "Superforecasting: The Art and Science of Prediction."
"""

from __future__ import annotations
import math
from core.agent import AgentEstimate


def extremize(
    estimates: list[AgentEstimate],
    d: float | None = None,
    method: str = "logit",
) -> dict:
    """
    Extremize the aggregated probability.

    Methods:
    - "logit" (default): Transform to log-odds space, average, then apply
      extremizing factor d > 1 to push away from 0.5.
      Formula: logit(p_ext) = d * mean(logit(p_i))
    - "power": p_ext = p^d / (p^d + (1-p)^d), which stretches toward extremes.

    If d is None, it's estimated from the number of agents using the
    Baron et al. heuristic: d ≈ 2.5 for independent forecasters,
    scaling down with correlation.
    """
    if not estimates:
        raise ValueError("No estimates to extremize")

    probs = [max(0.005, min(0.995, e.probability)) for e in estimates]
    confs = [e.confidence for e in estimates]
    n = len(probs)

    # confidence-weighted mean probability
    total_conf = sum(confs)
    if total_conf > 0:
        mean_p = sum(p * c for p, c in zip(probs, confs)) / total_conf
    else:
        mean_p = sum(probs) / n

    mean_p = max(0.005, min(0.995, mean_p))

    # estimate d if not provided
    if d is None:
        d = _estimate_extremizing_factor(probs)

    if method == "logit":
        # logit transform: log(p / (1 - p))
        mean_logit = _logit(mean_p)
        extremized_logit = d * mean_logit
        extremized_p = _inverse_logit(extremized_logit)
    elif method == "power":
        p = mean_p
        numerator = p ** d
        denominator = p ** d + (1 - p) ** d
        extremized_p = numerator / denominator if denominator > 0 else 0.5
    else:
        raise ValueError(f"Unknown method: {method}")

    extremized_p = max(0.001, min(0.999, extremized_p))

    # compute shift metrics
    shift = extremized_p - mean_p
    direction = "toward_yes" if extremized_p > 0.5 else "toward_no" if extremized_p < 0.5 else "neutral"

    return {
        "extremized_probability": round(extremized_p, 4),
        "raw_mean": round(mean_p, 4),
        "extremizing_factor": round(d, 3),
        "shift": round(shift, 4),
        "direction": direction,
        "method": method,
    }


def _estimate_extremizing_factor(probs: list[float]) -> float:
    """
    Estimate the optimal extremizing factor based on forecaster diversity.

    From Baron et al. (2014): more diverse (independent) forecasters
    warrant higher d. We estimate diversity via the coefficient of variation
    of the log-odds.

    Heuristic:
    - High diversity (CV > 1.5): d ≈ 2.5 (strong extremizing)
    - Moderate diversity (CV ≈ 1.0): d ≈ 1.8
    - Low diversity (CV < 0.5): d ≈ 1.2 (mild extremizing)
    - Very low diversity: d ≈ 1.0 (no extremizing — likely correlated)
    """
    n = len(probs)
    if n < 2:
        return 1.0

    logits = [_logit(p) for p in probs]
    mean_logit = sum(logits) / n
    std_logit = math.sqrt(sum((l - mean_logit) ** 2 for l in logits) / (n - 1))

    if abs(mean_logit) < 0.01:
        cv = std_logit  # avoid division by near-zero
    else:
        cv = std_logit / abs(mean_logit)

    # map CV to d
    if cv > 1.5:
        d = 2.5
    elif cv > 1.0:
        d = 1.8 + (cv - 1.0) * 1.4  # 1.8 to 2.5
    elif cv > 0.5:
        d = 1.2 + (cv - 0.5) * 1.2  # 1.2 to 1.8
    else:
        d = 1.0 + cv * 0.4  # 1.0 to 1.2

    return min(d, 3.0)


def _logit(p: float) -> float:
    """Log-odds transform."""
    p = max(0.005, min(0.995, p))
    return math.log(p / (1 - p))


def _inverse_logit(x: float) -> float:
    """Inverse logit (sigmoid)."""
    x = max(-10, min(10, x))  # prevent overflow
    return 1.0 / (1.0 + math.exp(-x))
