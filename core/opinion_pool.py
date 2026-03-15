"""
Logarithmic Opinion Pool (LogOP) — theory-optimal aggregation under
certain independence assumptions.

Unlike linear opinion pools (simple averages), logarithmic pools combine
probabilities multiplicatively in log space. This has the "external
Bayesianity" property: if all agents are Bayesian with different priors
but shared likelihood, LogOP recovers the correct posterior.

Also implements Cooke's Classical Model for performance-based weighting,
which weights forecasters by calibration AND informativeness.

References:
  - Genest, C., & Zidek, J. V. (1986). "Combining Probability
    Distributions: A Critique and Annotated Bibliography."
    Statistical Science.
  - Cooke, R. M. (1991). "Experts in Uncertainty: Opinion and
    Subjective Probability in Science." Oxford University Press.
  - Ranjan, R., & Gneiting, T. (2010). "Combining probability
    forecasts." Journal of the Royal Statistical Society.
"""

from __future__ import annotations
import math
from core.agent import AgentEstimate


def logarithmic_opinion_pool(
    estimates: list[AgentEstimate],
    weights: dict[str, float] | None = None,
) -> dict:
    """
    Logarithmic opinion pool: geometric mean in probability space.

    p_log = prod(p_i^w_i) / (prod(p_i^w_i) + prod((1-p_i)^w_i))

    Where w_i are normalized weights (sum to 1).

    Properties:
    - Extreme beliefs have strong influence (one agent at 0.01 pulls hard)
    - Satisfies external Bayesianity
    - More aggressive than linear pooling
    """
    if not estimates:
        raise ValueError("No estimates")

    n = len(estimates)

    # compute weights
    if weights:
        w = [weights.get(e.agent_id, 1.0 / n) for e in estimates]
    else:
        # weight by confidence
        total_conf = sum(e.confidence for e in estimates)
        w = [e.confidence / total_conf for e in estimates] if total_conf > 0 else [1/n] * n

    # normalize weights
    total_w = sum(w)
    w = [wi / total_w for wi in w]

    # compute in log space for numerical stability
    log_yes = sum(wi * math.log(max(0.001, min(0.999, e.probability))) for wi, e in zip(w, estimates))
    log_no = sum(wi * math.log(max(0.001, min(0.999, 1 - e.probability))) for wi, e in zip(w, estimates))

    # normalize
    log_max = max(log_yes, log_no)
    yes_term = math.exp(log_yes - log_max)
    no_term = math.exp(log_no - log_max)

    logop_prob = yes_term / (yes_term + no_term) if (yes_term + no_term) > 0 else 0.5
    logop_prob = max(0.001, min(0.999, logop_prob))

    # compare with linear pool
    linear_prob = sum(wi * e.probability for wi, e in zip(w, estimates))

    return {
        "logop_probability": round(logop_prob, 4),
        "linear_probability": round(linear_prob, 4),
        "logop_shift": round(logop_prob - linear_prob, 4),
        "more_extreme": abs(logop_prob - 0.5) > abs(linear_prob - 0.5),
        "n_agents": n,
    }


def cooke_classical_weights(
    estimates: list[AgentEstimate],
    calibration_scores: dict[str, float] | None = None,
    alpha: float = 0.05,
) -> dict:
    """
    Cooke's Classical Model: weight forecasters by both calibration
    AND informativeness.

    In Cooke's framework:
    - Calibration: statistical accuracy (p-value of chi-squared test
      on seed questions). Only forecasters above threshold alpha pass.
    - Informativeness: Shannon entropy relative to a reference
      (uniform) distribution. More informative = tighter distributions.

    Weight = calibration_score * informativeness

    Since we don't have seed questions, we approximate:
    - Calibration: from Brier scores (if available) or confidence consistency
    - Informativeness: inverse entropy of each agent's estimates
    """
    if not estimates:
        raise ValueError("No estimates")

    n = len(estimates)

    # compute informativeness for each agent
    # higher confidence + more extreme estimate = more informative
    informativeness = {}
    for e in estimates:
        p = max(0.001, min(0.999, e.probability))
        # Shannon information: -log2(1 - |p - 0.5| * 2)
        # More extreme = more informative
        extremity = abs(p - 0.5) * 2  # 0 at p=0.5, 1 at p=0/1
        info = e.confidence * (0.5 + extremity * 0.5)
        informativeness[e.agent_id] = info

    # compute calibration scores
    if calibration_scores:
        cal_scores = {e.agent_id: calibration_scores.get(e.agent_id, 0.5) for e in estimates}
    else:
        # without historical data, use confidence as a proxy
        # (consistent confidence suggests calibration awareness)
        cal_scores = {e.agent_id: min(e.confidence * 1.2, 1.0) for e in estimates}

    # apply threshold: only agents above alpha qualify
    qualified = {}
    disqualified = []
    for e in estimates:
        cal = cal_scores.get(e.agent_id, 0.0)
        if cal >= alpha:
            info = informativeness.get(e.agent_id, 0.5)
            qualified[e.agent_id] = cal * info
        else:
            disqualified.append(e.agent_id)

    # if no one qualifies, use equal weights
    if not qualified:
        qualified = {e.agent_id: 1.0 for e in estimates}
        disqualified = []

    # normalize
    total = sum(qualified.values())
    normalized = {k: v / total for k, v in qualified.items()}

    # compute weighted aggregate
    weighted_prob = 0.0
    for e in estimates:
        w = normalized.get(e.agent_id, 0.0)
        weighted_prob += w * e.probability

    return {
        "cooke_probability": round(weighted_prob, 4),
        "weights": {k: round(v, 4) for k, v in normalized.items()},
        "n_qualified": len(qualified),
        "n_disqualified": len(disqualified),
        "disqualified_agents": disqualified,
        "informativeness": {k: round(v, 4) for k, v in informativeness.items()},
    }
