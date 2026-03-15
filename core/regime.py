"""
Hidden Markov Model Regime Detection — adaptive strategy based on
market state.

Detects whether the forecast environment is in a "Consensus" (low
disagreement), "Debate" (moderate), or "Chaos" (high disagreement)
regime. Different regimes warrant different aggregation strategies
and confidence levels.

Uses Forward-Backward algorithm for state inference and Viterbi
decoding for most likely state path.

References:
  - Rabiner, L.R. (1989). "A Tutorial on Hidden Markov Models and
    Selected Applications in Speech Recognition." Proceedings IEEE.
  - Hamilton, J.D. (1989). "A New Approach to the Economic Analysis
    of Nonstationary Time Series." Econometrica.
  - Ang, A. & Bekaert, G. (2002). "Regime Switches in Interest Rates."
    Journal of Business & Economic Statistics.
"""

from __future__ import annotations
import math

# regime labels indexed by state number
_REGIME_NAMES = {0: "consensus", 1: "debate", 2: "chaos"}


def detect_regime(
    current_diagnostics: dict,
    history: list[dict] | None = None,
) -> dict:
    """
    Detect the current forecast regime from swarm diagnostics.

    Args:
        current_diagnostics: Dict with keys mean_prob, std_dev,
            herding_score, cascade_rate.
        history: Optional list of past diagnostics dicts (same keys).
            If provided, runs forward-backward HMM for posterior
            state probabilities. Otherwise, uses rule-based fallback.

    Returns dict with regime label, state probabilities, confidence,
    recommended strategy, and aggregation adjustments.
    """
    required = {"std_dev", "herding_score"}
    if not required.issubset(current_diagnostics.keys()):
        raise ValueError(
            f"current_diagnostics must contain: {required}. "
            f"Got: {set(current_diagnostics.keys())}"
        )

    model = _default_model()

    if history and len(history) >= 3:
        # convert history + current to observation sequences
        all_diags = list(history) + [current_diagnostics]
        observations = [
            [d.get("std_dev", 0.1), d.get("herding_score", 0.2)]
            for d in all_diags
        ]

        # forward-backward for posterior probabilities
        posteriors = _forward_backward(observations, model)
        current_posterior = posteriors[-1]

        # viterbi for most likely path
        state_path = _viterbi(observations, model)
        current_state = state_path[-1]

        # confidence: how peaked is the posterior?
        max_prob = max(current_posterior)
        confidence = max_prob
    else:
        # rule-based fallback
        rule_result = _classify_rule_based(current_diagnostics)
        current_state = rule_result["state"]
        current_posterior = rule_result["probabilities"]
        confidence = rule_result["confidence"]

    regime_name = _REGIME_NAMES[current_state]
    strategy = _regime_strategy(regime_name)

    return {
        "current_regime": regime_name,
        "regime_probabilities": {
            _REGIME_NAMES[i]: round(p, 4) for i, p in enumerate(current_posterior)
        },
        "confidence": round(confidence, 4),
        "recommended_strategy": strategy["strategy"],
        "aggregation_adjustments": strategy["adjustments"],
        "regime_description": strategy["description"],
    }


def _forward_backward(
    observations: list[list[float]],
    model: dict,
) -> list[list[float]]:
    """
    Forward-Backward algorithm for HMM posterior state probabilities.

    Args:
        observations: List of [std_dev, herding_score] pairs.
        model: HMM parameters (transition, emission, initial).

    Returns list of posterior probability vectors, one per time step.
    """
    n_states = len(model["initial_probs"])
    T = len(observations)

    if T == 0:
        return [model["initial_probs"][:]]

    trans = model["transition_matrix"]
    initial = model["initial_probs"]

    # --- forward pass ---
    alphas: list[list[float]] = []
    # t = 0
    alpha_0 = [
        initial[s] * _emission_prob(observations[0], s, model)
        for s in range(n_states)
    ]
    alpha_0 = _normalize(alpha_0)
    alphas.append(alpha_0)

    for t in range(1, T):
        alpha_t = []
        for j in range(n_states):
            val = sum(alphas[t - 1][i] * trans[i][j] for i in range(n_states))
            val *= _emission_prob(observations[t], j, model)
            alpha_t.append(val)
        alpha_t = _normalize(alpha_t)
        alphas.append(alpha_t)

    # --- backward pass ---
    betas: list[list[float]] = [None] * T  # type: ignore[list-item]
    betas[T - 1] = [1.0] * n_states

    for t in range(T - 2, -1, -1):
        beta_t = []
        for i in range(n_states):
            val = sum(
                trans[i][j]
                * _emission_prob(observations[t + 1], j, model)
                * betas[t + 1][j]
                for j in range(n_states)
            )
            beta_t.append(val)
        beta_t = _normalize(beta_t)
        betas[t] = beta_t

    # --- posterior = alpha * beta, normalized ---
    posteriors = []
    for t in range(T):
        gamma = [alphas[t][s] * betas[t][s] for s in range(n_states)]
        gamma = _normalize(gamma)
        posteriors.append(gamma)

    return posteriors


def _viterbi(
    observations: list[list[float]],
    model: dict,
) -> list[int]:
    """
    Viterbi algorithm — most likely state sequence.

    Works in log space to avoid underflow on long sequences.
    """
    n_states = len(model["initial_probs"])
    T = len(observations)

    if T == 0:
        return []

    trans = model["transition_matrix"]
    initial = model["initial_probs"]

    # log probabilities (guard against log(0))
    _log = lambda x: math.log(max(x, 1e-300))

    # t = 0
    V: list[list[float]] = []
    backptr: list[list[int]] = []

    v_0 = [
        _log(initial[s]) + _log(_emission_prob(observations[0], s, model))
        for s in range(n_states)
    ]
    V.append(v_0)
    backptr.append([0] * n_states)

    for t in range(1, T):
        v_t = []
        bp_t = []
        for j in range(n_states):
            candidates = [
                V[t - 1][i] + _log(trans[i][j]) for i in range(n_states)
            ]
            best_i = max(range(n_states), key=lambda i: candidates[i])
            best_val = candidates[best_i] + _log(
                _emission_prob(observations[t], j, model)
            )
            v_t.append(best_val)
            bp_t.append(best_i)
        V.append(v_t)
        backptr.append(bp_t)

    # backtrace
    path = [0] * T
    path[T - 1] = max(range(n_states), key=lambda s: V[T - 1][s])
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1][path[t + 1]]

    return path


def _default_model() -> dict:
    """
    Return sensible default HMM parameters.

    Transition tendencies:
      - Consensus tends to persist (0.7 self-transition)
      - Debate is transient (0.5 self-transition)
      - Chaos is moderately sticky (0.6 self-transition)

    Emission parameters (Gaussian) for [std_dev, herding_score]:
      - Consensus: low std_dev, low herding
      - Debate: moderate std_dev, moderate herding
      - Chaos: high std_dev, high herding
    """
    return {
        "transition_matrix": [
            # from consensus -> [consensus, debate, chaos]
            [0.70, 0.25, 0.05],
            # from debate -> [consensus, debate, chaos]
            [0.20, 0.50, 0.30],
            # from chaos -> [consensus, debate, chaos]
            [0.10, 0.30, 0.60],
        ],
        "emission_means": [
            # [std_dev_mean, herding_mean] per state
            [0.05, 0.10],  # consensus
            [0.12, 0.30],  # debate
            [0.25, 0.60],  # chaos
        ],
        "emission_stds": [
            [0.03, 0.08],  # consensus
            [0.05, 0.12],  # debate
            [0.10, 0.15],  # chaos
        ],
        "initial_probs": [0.5, 0.35, 0.15],
    }


def _emission_prob(
    obs: list[float],
    state: int,
    model: dict,
) -> float:
    """
    Gaussian emission probability for an observation given a state.

    Treats each feature dimension as independent (diagonal covariance).
    Returns the product of per-dimension Gaussian densities.
    """
    means = model["emission_means"][state]
    stds = model["emission_stds"][state]
    prob = 1.0

    for i in range(len(obs)):
        mu = means[i]
        sigma = max(stds[i], 1e-6)
        diff = obs[i] - mu
        exponent = -0.5 * (diff / sigma) ** 2
        # guard against extreme exponents
        exponent = max(exponent, -500.0)
        density = (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * math.exp(exponent)
        prob *= max(density, 1e-300)

    return prob


def _classify_rule_based(diagnostics: dict) -> dict:
    """
    Rule-based regime classification when no history is available.

    Uses simple thresholds on std_dev and herding_score to assign
    a regime with soft probability distribution.
    """
    std_dev = diagnostics.get("std_dev", 0.1)
    herding = diagnostics.get("herding_score", 0.2)

    # compute a "chaos score" from both signals
    chaos_signal = (std_dev / 0.25) * 0.5 + (herding / 0.6) * 0.5
    chaos_signal = max(0.0, min(2.0, chaos_signal))

    if chaos_signal < 0.4:
        state = 0  # consensus
        probs = [0.75, 0.20, 0.05]
        confidence = 0.7 + 0.3 * (0.4 - chaos_signal) / 0.4
    elif chaos_signal < 0.8:
        state = 1  # debate
        probs = [0.15, 0.70, 0.15]
        confidence = 0.5 + 0.2 * (1.0 - abs(chaos_signal - 0.6) / 0.4)
    else:
        state = 2  # chaos
        probs = [0.05, 0.20, 0.75]
        confidence = 0.6 + 0.3 * min(1.0, (chaos_signal - 0.8) / 0.5)

    confidence = max(0.3, min(0.95, confidence))

    return {
        "state": state,
        "probabilities": probs,
        "confidence": confidence,
    }


def _regime_strategy(regime: str) -> dict:
    """
    Return recommended aggregation strategy for each regime.

    Consensus: trust the aggregate, use extremizing.
    Debate: hedge with wider intervals, weight coherent agents.
    Chaos: be cautious, shrink toward base rate, widen intervals.
    """
    strategies = {
        "consensus": {
            "strategy": "Trust the aggregate. Agents largely agree, so "
                        "extremize the mean and use narrow confidence intervals.",
            "adjustments": {
                "extremize": 1.2,
                "confidence_weight": 1.0,
                "coherence_weight": 0.8,
                "interval_width": 0.8,
                "base_rate_shrinkage": 0.0,
            },
            "description": "Low disagreement regime. Agents converge on a "
                           "shared view. High confidence in aggregate.",
        },
        "debate": {
            "strategy": "Seek the informational edge. Weight coherent and "
                        "historically accurate agents more heavily. Use "
                        "moderate extremizing.",
            "adjustments": {
                "extremize": 1.0,
                "confidence_weight": 1.2,
                "coherence_weight": 1.3,
                "interval_width": 1.2,
                "base_rate_shrinkage": 0.1,
            },
            "description": "Moderate disagreement regime. Agents have "
                           "substantively different views. Careful weighting "
                           "can extract signal from the debate.",
        },
        "chaos": {
            "strategy": "Be humble. Shrink toward base rate, widen intervals "
                        "substantially, and reduce extremizing. The swarm is "
                        "confused — avoid false precision.",
            "adjustments": {
                "extremize": 0.7,
                "confidence_weight": 0.8,
                "coherence_weight": 1.5,
                "interval_width": 1.8,
                "base_rate_shrinkage": 0.3,
            },
            "description": "High disagreement regime. Agents strongly "
                           "diverge, possibly due to ambiguous evidence or "
                           "a genuinely uncertain situation.",
        },
    }

    return strategies.get(regime, strategies["debate"])


def _normalize(vec: list[float]) -> list[float]:
    """Normalize a vector to sum to 1, handling zero-sum gracefully."""
    total = sum(vec)
    if total > 0:
        return [v / total for v in vec]
    n = len(vec)
    return [1.0 / n] * n if n > 0 else []
