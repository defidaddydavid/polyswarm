"""
Shapley Value Attribution — fair contribution measurement for each agent.

Computes each agent's marginal contribution to forecast quality using
the Shapley value from cooperative game theory. Not just "Agent X gave
probability Y" but "if we removed Agent X, the aggregate would shift by Z."

Uses Monte Carlo sampling (random permutations) for efficiency since
exact computation requires 2^N coalitions.

References:
  - Shapley, L.S. (1953). "A Value for n-Person Games."
    Contributions to the Theory of Games.
  - Castro, J. et al. (2009). "Polynomial calculation of the Shapley value
    based on sampling." Computers & Operations Research.
  - Lundberg, S.M. & Lee, S.I. (2017). "A Unified Approach to Interpreting
    Model Predictions." NeurIPS.
"""

from __future__ import annotations
import math
import random
from core.agent import AgentEstimate


def shapley_values(
    estimates: list[AgentEstimate],
    n_permutations: int = 500,
) -> dict:
    """
    Compute Shapley values for each agent via Monte Carlo sampling.

    For each random permutation of agents, measure each agent's marginal
    contribution: the coalition value WITH the agent minus the value WITHOUT.
    Average these marginal contributions across all sampled permutations
    to get the Shapley value.

    Returns per-agent attribution, most/least valuable agents, redundancy
    detection, and a concentration index.
    """
    if not estimates:
        raise ValueError("No estimates to compute Shapley values")

    n = len(estimates)

    if n == 1:
        e = estimates[0]
        return {
            "per_agent_shapley": {
                e.agent_id: {
                    "shapley_value": round(e.probability, 4),
                    "rank": 1,
                    "persona": e.persona,
                },
            },
            "most_valuable_agent": {"agent_id": e.agent_id, "persona": e.persona},
            "least_valuable_agent": {"agent_id": e.agent_id, "persona": e.persona},
            "redundant_agents": [],
            "concentration_index": 1.0,
        }

    # accumulate marginal contributions per agent
    contribution_sums: dict[str, float] = {e.agent_id: 0.0 for e in estimates}
    agent_map: dict[str, AgentEstimate] = {e.agent_id: e for e in estimates}
    agent_ids = list(agent_map.keys())

    for _ in range(n_permutations):
        perm = agent_ids[:]
        random.shuffle(perm)
        coalition: list[AgentEstimate] = []

        for agent_id in perm:
            value_without = _coalition_value(coalition)
            coalition.append(agent_map[agent_id])
            value_with = _coalition_value(coalition)
            contribution_sums[agent_id] += value_with - value_without

    # average over permutations
    shapley: dict[str, float] = {
        aid: contribution_sums[aid] / n_permutations for aid in agent_ids
    }

    # rank agents by absolute Shapley value (higher = more influential)
    ranked_ids = sorted(agent_ids, key=lambda aid: abs(shapley[aid]), reverse=True)
    ranks = {aid: rank + 1 for rank, aid in enumerate(ranked_ids)}

    per_agent = {}
    for aid in agent_ids:
        per_agent[aid] = {
            "shapley_value": round(shapley[aid], 4),
            "rank": ranks[aid],
            "persona": agent_map[aid].persona,
        }

    # most and least valuable
    most_id = ranked_ids[0]
    least_id = ranked_ids[-1]

    # redundant agents: Shapley value near zero (< 1% of total range)
    abs_values = [abs(v) for v in shapley.values()]
    max_abs = max(abs_values) if abs_values else 1.0
    threshold = max(0.001, max_abs * 0.05)
    redundant = [
        {"agent_id": aid, "persona": agent_map[aid].persona}
        for aid in agent_ids
        if abs(shapley[aid]) < threshold
    ]

    # concentration index (Herfindahl-style on absolute Shapley values)
    total_abs = sum(abs_values)
    if total_abs > 0:
        shares = [v / total_abs for v in abs_values]
        hhi = sum(s * s for s in shares)
        # normalize: HHI ranges from 1/n (uniform) to 1 (single agent dominates)
        # map to 0..1 where 0 = perfectly equal, 1 = single agent
        if n > 1:
            concentration = (hhi - 1.0 / n) / (1.0 - 1.0 / n)
        else:
            concentration = 1.0
    else:
        concentration = 0.0

    concentration = max(0.0, min(1.0, concentration))

    return {
        "per_agent_shapley": per_agent,
        "most_valuable_agent": {
            "agent_id": most_id,
            "persona": agent_map[most_id].persona,
        },
        "least_valuable_agent": {
            "agent_id": least_id,
            "persona": agent_map[least_id].persona,
        },
        "redundant_agents": redundant,
        "concentration_index": round(concentration, 4),
    }


def _coalition_value(subset: list[AgentEstimate]) -> float:
    """
    Compute the aggregated probability of a coalition of agents.

    Uses confidence-weighted mean. An empty coalition has value 0.5
    (maximum uncertainty / uninformative prior).
    """
    if not subset:
        return 0.5

    total_conf = sum(e.confidence for e in subset)
    if total_conf > 0:
        return sum(e.probability * e.confidence for e in subset) / total_conf

    return sum(e.probability for e in subset) / len(subset)


def _marginal_contribution(
    agent: AgentEstimate,
    coalition: list[AgentEstimate],
) -> float:
    """
    Compute the marginal contribution of adding an agent to a coalition.

    Returns the signed difference: value(coalition + agent) - value(coalition).
    Positive means the agent pushes the aggregate away from the uninformative
    prior (0.5); negative means the agent dilutes the existing signal.
    """
    value_without = _coalition_value(coalition)
    value_with = _coalition_value(coalition + [agent])
    return value_with - value_without
