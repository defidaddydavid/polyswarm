"""
Information-Theoretic Analysis — mutual information, transfer entropy,
and information gain across debate rounds.

Measures how much new information each round produces, whether agents are
genuinely learning or just converging, and the causal information flow
between agents across rounds.

References:
  - Shannon, C.E. (1948). "A Mathematical Theory of Communication."
    Bell System Technical Journal.
  - Schreiber, T. (2000). "Measuring Information Transfer."
    Physical Review Letters.
  - Cover, T.M. & Thomas, J.A. (2006). "Elements of Information Theory." Wiley.
"""

from __future__ import annotations
import math
from core.agent import AgentEstimate


def information_analysis(
    estimates: list[AgentEstimate],
    round1_estimates: list[AgentEstimate] | None = None,
) -> dict:
    """
    Compute information-theoretic metrics across agents and optionally
    across debate rounds. Returns pairwise MI, redundancy, total/shared/unique
    information, and (if round1 provided) transfer entropy and info gain.
    """
    if not estimates:
        raise ValueError("No estimates")

    n = len(estimates)
    probs = [max(0.005, min(0.995, e.probability)) for e in estimates]
    confs = [e.confidence for e in estimates]
    individual_H = [_binary_entropy(p) for p in probs]
    total_information = sum(individual_H)

    # Pairwise mutual information
    pairwise_mi: dict[str, float] = {}
    mi_values = []
    if n >= 2:
        for i in range(n):
            for j in range(i + 1, n):
                mi = _mutual_information([probs[i], confs[i]], [probs[j], confs[j]])
                pair_key = f"{estimates[i].agent_id}|{estimates[j].agent_id}"
                pairwise_mi[pair_key] = round(mi, 4)
                mi_values.append(mi)

    shared_information = sum(mi_values) if mi_values else 0.0
    unique_information = max(0.0, total_information - shared_information)
    redundancy = _redundancy_ratio(estimates)
    diversity = _diversity_index(probs)

    # Cross-round analysis
    transfer_entropy: dict[str, float] = {}
    information_gain = 0.0
    most_influential = estimates[0].agent_id
    most_influenced = estimates[0].agent_id

    if round1_estimates and len(round1_estimates) > 0:
        r1_probs = [max(0.005, min(0.995, e.probability)) for e in round1_estimates]
        avg_H_r1 = sum(_binary_entropy(p) for p in r1_probs) / len(r1_probs)
        avg_H_r2 = sum(individual_H) / n if n else 0.0
        information_gain = max(0.0, avg_H_r1 - avg_H_r2)

        m = min(len(round1_estimates), n)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                te = _transfer_entropy(
                    [r1_probs[i]], [probs[i]], [probs[j]],
                )
                key = f"{round1_estimates[i].agent_id}->{estimates[j].agent_id}"
                transfer_entropy[key] = round(te, 4)

    if transfer_entropy:
        best_key = max(transfer_entropy, key=transfer_entropy.get)  # type: ignore
        parts = best_key.split("->")
        most_influential, most_influenced = parts[0], parts[1]
    elif n >= 2:
        mean_p = sum(probs) / n
        devs = [abs(p - mean_p) for p in probs]
        most_influential = estimates[devs.index(min(devs))].agent_id
        most_influenced = estimates[devs.index(max(devs))].agent_id

    return {
        "pairwise_mi": pairwise_mi,
        "total_information": round(total_information, 4),
        "shared_information": round(shared_information, 4),
        "unique_information": round(unique_information, 4),
        "redundancy_ratio": round(redundancy, 4),
        "information_gain_round2": round(information_gain, 4),
        "transfer_entropy": transfer_entropy,
        "most_influential_agent": most_influential,
        "most_influenced_agent": most_influenced,
        "diversity_index": round(diversity, 4),
    }


def _mutual_information(x: list[float], y: list[float], bins: int = 5) -> float:
    """Estimate MI between two continuous variables via histogram binning."""
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0
    x_d, y_d = _discretize(x, bins), _discretize(y, bins)
    joint: dict[tuple[int, int], int] = {}
    xc: dict[int, int] = {}
    yc: dict[int, int] = {}
    for xi, yi in zip(x_d, y_d):
        joint[(xi, yi)] = joint.get((xi, yi), 0) + 1
        xc[xi] = xc.get(xi, 0) + 1
        yc[yi] = yc.get(yi, 0) + 1
    mi = 0.0
    for (xi, yi), jc in joint.items():
        p_xy, p_x, p_y = jc / n, xc[xi] / n, yc[yi] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log(p_xy / (p_x * p_y))
    return max(0.0, mi)


def _transfer_entropy(
    source_r1: list[float], source_r2: list[float], target_r2: list[float],
    bins: int = 5,
) -> float:
    """
    Estimate transfer entropy from source to target across rounds.
    For single-point estimates, uses a distance-based heuristic.
    """
    n = min(len(source_r1), len(source_r2), len(target_r2))
    if n < 1:
        return 0.0
    if n == 1:
        source_change = abs(source_r2[0] - source_r1[0])
        target_toward = max(0.0, 1.0 - abs(target_r2[0] - source_r1[0]))
        return max(0.0, min(1.0, source_change * target_toward * 0.5))
    return max(0.0,
               _mutual_information(source_r1, target_r2, bins)
               - _mutual_information(source_r2, target_r2, bins))


def _shannon_entropy(probs: list[float]) -> float:
    """Shannon entropy of a discrete probability distribution."""
    return max(0.0, -sum(p * math.log2(p) for p in probs if p > 1e-12))


def _binary_entropy(p: float) -> float:
    """Binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)."""
    p = max(1e-12, min(1.0 - 1e-12, p))
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def _redundancy_ratio(estimates: list[AgentEstimate]) -> float:
    """
    Fraction of total information that is shared vs unique. Compares entropy
    of the mean probability to average individual entropy. Returns [0, 1].
    """
    n = len(estimates)
    if n < 2:
        return 0.0
    probs = [max(0.005, min(0.995, e.probability)) for e in estimates]
    mean_H = _binary_entropy(sum(probs) / n)
    avg_H = sum(_binary_entropy(p) for p in probs) / n
    if avg_H < 1e-12:
        return 1.0
    return max(0.0, min(1.0, mean_H / avg_H))


def _diversity_index(probs: list[float]) -> float:
    """Normalized diversity via coefficient of variation. Returns [0, 1]."""
    n = len(probs)
    if n < 2:
        return 0.0
    mean_p = sum(probs) / n
    std_p = math.sqrt(sum((p - mean_p) ** 2 for p in probs) / n)
    return max(0.0, min(1.0, std_p / 0.5))


def _discretize(values: list[float], bins: int) -> list[int]:
    """Bin continuous values into [0, bins-1] integer bins."""
    if not values:
        return []
    v_min, v_max = min(values), max(values)
    if v_max - v_min < 1e-12:
        return [0] * len(values)
    return [max(0, min(bins - 1, int((v - v_min) / (v_max - v_min) * (bins - 1) + 0.5)))
            for v in values]
