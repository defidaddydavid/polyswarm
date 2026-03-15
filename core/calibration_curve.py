"""
Calibration Curve Analysis — isotonic regression and Platt scaling for
forecast recalibration.

Builds calibration curves from historical data. Shows "when the swarm says
70%, it actually resolves YES X% of the time." Produces a calibrated
probability that corrects for systematic overconfidence or underconfidence.

Uses the Pool-Adjacent-Violators Algorithm (PAVA) for isotonic regression
— pure Python, no dependencies.

References:
  - Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines."
    Advances in Large Margin Classifiers.
  - Niculescu-Mizil, A. & Caruana, R. (2005). "Predicting Good Probabilities
    with Supervised Learning." ICML.
  - Brocker, J. (2009). "Reliability, Sufficiency, and the Decomposition
    of Proper Scores." Quarterly J. of the Royal Meteorological Society.
"""

from __future__ import annotations
import math


def calibrate_probability(
    raw_prob: float,
    history: list[dict] | None = None,
) -> dict:
    """
    Calibrate a raw probability using historical forecast-outcome data.

    Args:
        raw_prob: The raw probability to calibrate (0-1).
        history: List of {"forecast": float, "outcome": 0 or 1}.
                 If None or fewer than 10 entries, returns uncalibrated.

    Returns:
        dict with calibrated_probability, calibration_method, ECE,
        reliability_bins, overconfidence/underconfidence scores, etc.
    """
    raw_prob = max(0.0, min(1.0, raw_prob))

    if history is None or len(history) < 10:
        return {
            "calibrated_probability": round(raw_prob, 4),
            "calibration_method": "none",
            "expected_calibration_error": None,
            "reliability_bins": [],
            "overconfidence_score": None,
            "underconfidence_score": None,
            "n_historical": len(history) if history else 0,
        }

    forecasts = [h["forecast"] for h in history]
    outcomes = [h["outcome"] for h in history]

    # Build reliability diagram
    bins = _reliability_diagram(forecasts, outcomes)
    ece = _expected_calibration_error(forecasts, outcomes)

    # Compute overconfidence / underconfidence
    overconf = 0.0
    underconf = 0.0
    total_count = 0
    for b in bins:
        if b["count"] > 0:
            diff = b["predicted"] - b["actual"]
            if diff > 0:
                overconf += abs(diff) * b["count"]
            else:
                underconf += abs(diff) * b["count"]
            total_count += b["count"]
    if total_count > 0:
        overconf /= total_count
        underconf /= total_count

    # Try isotonic regression first (preferred with enough data)
    calibrated_prob = raw_prob
    method = "none"

    if len(history) >= 15:
        try:
            # Sort by forecast
            paired = sorted(zip(forecasts, outcomes), key=lambda x: x[0])
            sorted_f = [p[0] for p in paired]
            sorted_o = [p[1] for p in paired]

            iso_values = _isotonic_regression(sorted_f, sorted_o)

            # Apply isotonic calibration: find where raw_prob falls
            # and interpolate
            calibrated_prob = _apply_isotonic(raw_prob, sorted_f, iso_values)
            method = "isotonic"
        except Exception:
            method = "none"

    # Also try Platt scaling — use it if isotonic failed or as comparison
    if len(history) >= 10:
        try:
            A, B = _platt_scaling(forecasts, outcomes)
            platt_prob = _apply_platt(raw_prob, A, B)

            if method == "none":
                calibrated_prob = platt_prob
                method = "platt"
        except Exception:
            pass

    if method == "none":
        calibrated_prob = raw_prob

    calibrated_prob = max(0.001, min(0.999, calibrated_prob))

    return {
        "calibrated_probability": round(calibrated_prob, 4),
        "calibration_method": method,
        "expected_calibration_error": round(ece, 4),
        "reliability_bins": bins,
        "overconfidence_score": round(overconf, 4),
        "underconfidence_score": round(underconf, 4),
        "n_historical": len(history),
    }


def _isotonic_regression(x: list[float], y: list[float]) -> list[float]:
    """
    Pool-Adjacent-Violators Algorithm (PAVA) for isotonic regression.
    Assumes x is already sorted in ascending order.
    Returns isotonic-fitted y values (non-decreasing).
    """
    n = len(y)
    if n == 0:
        return []

    # Each block is [sum_of_values, count, start_index, end_index]
    blocks = [[float(y[i]), 1, i, i] for i in range(n)]

    # Merge adjacent blocks that violate monotonicity
    changed = True
    while changed:
        changed = False
        merged = []
        i = 0
        while i < len(blocks):
            if i + 1 < len(blocks):
                mean_curr = blocks[i][0] / blocks[i][1]
                mean_next = blocks[i + 1][0] / blocks[i + 1][1]
                if mean_curr > mean_next:
                    # Merge blocks
                    new_block = [
                        blocks[i][0] + blocks[i + 1][0],
                        blocks[i][1] + blocks[i + 1][1],
                        blocks[i][2],
                        blocks[i + 1][3],
                    ]
                    merged.append(new_block)
                    i += 2
                    changed = True
                    continue
            merged.append(blocks[i])
            i += 1
        blocks = merged

    # Expand blocks back to per-element values
    result = [0.0] * n
    for block in blocks:
        mean_val = block[0] / block[1]
        for j in range(block[2], block[3] + 1):
            result[j] = mean_val

    return result


def _apply_isotonic(
    raw_prob: float,
    sorted_forecasts: list[float],
    iso_values: list[float],
) -> float:
    """Apply isotonic calibration via linear interpolation."""
    if not sorted_forecasts:
        return raw_prob

    # Handle boundary cases
    if raw_prob <= sorted_forecasts[0]:
        return iso_values[0]
    if raw_prob >= sorted_forecasts[-1]:
        return iso_values[-1]

    # Find bracketing indices and interpolate
    for i in range(len(sorted_forecasts) - 1):
        if sorted_forecasts[i] <= raw_prob <= sorted_forecasts[i + 1]:
            span = sorted_forecasts[i + 1] - sorted_forecasts[i]
            if span == 0:
                return iso_values[i]
            frac = (raw_prob - sorted_forecasts[i]) / span
            return iso_values[i] + frac * (iso_values[i + 1] - iso_values[i])

    return raw_prob


def _platt_scaling(
    forecasts: list[float],
    outcomes: list[float],
    lr: float = 0.1,
    epochs: int = 200,
) -> tuple[float, float]:
    """
    Fit Platt scaling: P_cal = 1 / (1 + exp(A*f + B)).
    Uses simple gradient descent to find A, B that minimize log loss.

    Returns (A, B).
    """
    A = 0.0
    B = 0.0
    n = len(forecasts)

    for epoch in range(epochs):
        grad_A = 0.0
        grad_B = 0.0

        for f, o in zip(forecasts, outcomes):
            # clip forecast to avoid log(0)
            f_clip = max(0.001, min(0.999, f))
            z = A * f_clip + B
            # sigmoid
            p = 1.0 / (1.0 + math.exp(-z)) if z > -500 else 0.0
            if z > 500:
                p = 1.0
            p = max(1e-7, min(1 - 1e-7, p))

            # gradient of cross-entropy loss: d/dA = (p - o) * f, d/dB = (p - o)
            err = p - o
            grad_A += err * f_clip
            grad_B += err

        grad_A /= n
        grad_B /= n

        # Adaptive learning rate decay
        current_lr = lr / (1 + epoch * 0.01)
        A -= current_lr * grad_A
        B -= current_lr * grad_B

    return A, B


def _apply_platt(raw_prob: float, A: float, B: float) -> float:
    """Apply Platt scaling to a raw probability."""
    f_clip = max(0.001, min(0.999, raw_prob))
    z = A * f_clip + B
    if z > 500:
        return 1.0
    if z < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def _reliability_diagram(
    forecasts: list[float],
    outcomes: list[float],
    n_bins: int = 10,
) -> list[dict]:
    """
    Bin forecasts and compute mean predicted and mean actual per bin.
    Returns list of {bin_center, predicted, actual, count}.
    """
    bins = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        center = (lo + hi) / 2

        in_bin_f = []
        in_bin_o = []
        for f, o in zip(forecasts, outcomes):
            if lo <= f < hi or (i == n_bins - 1 and f == hi):
                in_bin_f.append(f)
                in_bin_o.append(o)

        count = len(in_bin_f)
        if count > 0:
            predicted = sum(in_bin_f) / count
            actual = sum(in_bin_o) / count
        else:
            predicted = center
            actual = center

        bins.append({
            "bin_center": round(center, 2),
            "predicted": round(predicted, 4),
            "actual": round(actual, 4),
            "count": count,
        })

    return bins


def _expected_calibration_error(
    forecasts: list[float],
    outcomes: list[float],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).
    Weighted average of |predicted - actual| per bin.
    """
    n = len(forecasts)
    if n == 0:
        return 0.0

    total_ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins

        in_bin_f = []
        in_bin_o = []
        for f, o in zip(forecasts, outcomes):
            if lo <= f < hi or (i == n_bins - 1 and f == hi):
                in_bin_f.append(f)
                in_bin_o.append(o)

        count = len(in_bin_f)
        if count > 0:
            predicted = sum(in_bin_f) / count
            actual = sum(in_bin_o) / count
            total_ece += abs(predicted - actual) * count

    return total_ece / n
