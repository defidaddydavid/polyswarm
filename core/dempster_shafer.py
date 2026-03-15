"""
Dempster-Shafer Evidence Theory — belief functions for uncertainty quantification.

Unlike Bayesian probability which forces a single number, Dempster-Shafer tracks
three quantities: belief (minimum evidence for), plausibility (maximum possible
support), and uncertainty (the gap). This explicitly represents "we don't know."

When agents disagree wildly, DS says "belief=0.3, plausibility=0.7, uncertainty=0.4"
— telling you the swarm genuinely doesn't know. Actionable: high uncertainty = don't
bet, or size down.

References:
  - Dempster, A.P. (1967). "Upper and Lower Probabilities Induced by a
    Multivalued Mapping." Annals of Mathematical Statistics.
  - Shafer, G. (1976). "A Mathematical Theory of Evidence." Princeton UP.
  - Sentz, K. & Ferson, S. (2002). "Combination of Evidence in Dempster-Shafer
    Theory." Sandia National Laboratories.
"""

from __future__ import annotations
from core.agent import AgentEstimate

# Focal elements
YES = "YES"
NO = "NO"
UNCERTAIN = "UNCERTAIN"  # the full frame of discernment {YES, NO}


def dempster_shafer_combine(estimates: list[AgentEstimate]) -> dict:
    """
    Combine agent estimates using Dempster-Shafer evidence theory.

    Each agent's probability and confidence are converted into a mass function,
    then iteratively combined via Dempster's rule. Returns belief, plausibility,
    uncertainty gap, and a pignistic probability for decision-making.
    """
    if not estimates:
        raise ValueError("No estimates")

    # Convert all agents to mass functions
    masses = [_agent_to_mass(e) for e in estimates]

    # Iteratively combine
    combined = masses[0]
    cumulative_conflict = 0.0

    for i in range(1, len(masses)):
        k = _conflict_coefficient(combined, masses[i])
        cumulative_conflict = 1.0 - (1.0 - cumulative_conflict) * (1.0 - k)
        combined = _combine_two_masses(combined, masses[i])

    # Extract belief and plausibility
    m_yes = combined.get(YES, 0.0)
    m_no = combined.get(NO, 0.0)
    m_unc = combined.get(UNCERTAIN, 0.0)

    # Belief = mass assigned directly to the hypothesis
    belief_yes = m_yes
    belief_no = m_no

    # Plausibility = 1 - belief in the negation
    # Pl(YES) = 1 - Bel(NO) = m(YES) + m(UNCERTAIN)
    plausibility_yes = m_yes + m_unc
    plausibility_no = m_no + m_unc

    # Uncertainty gap = plausibility - belief
    uncertainty_gap = plausibility_yes - belief_yes  # equals m_unc

    # Pignistic probability (Smets transform for decision-making)
    # BetP(YES) = m(YES) + m(UNCERTAIN) / 2
    pignistic = m_yes + m_unc / 2.0

    # Should abstain if uncertainty is too high
    should_abstain = uncertainty_gap > 0.4

    return {
        "belief_yes": round(belief_yes, 4),
        "belief_no": round(belief_no, 4),
        "plausibility_yes": round(plausibility_yes, 4),
        "plausibility_no": round(plausibility_no, 4),
        "uncertainty_gap": round(uncertainty_gap, 4),
        "conflict_coefficient": round(cumulative_conflict, 4),
        "combined_mass": {
            YES: round(m_yes, 4),
            NO: round(m_no, 4),
            UNCERTAIN: round(m_unc, 4),
        },
        "should_abstain": should_abstain,
        "pignistic_probability": round(pignistic, 4),
    }


def _agent_to_mass(estimate: AgentEstimate) -> dict:
    """
    Convert an agent's probability and confidence into a mass function.

    - m(YES) = probability * confidence
    - m(NO) = (1 - probability) * confidence
    - m(UNCERTAIN) = 1 - confidence  (mass assigned to ignorance)

    The confidence determines how much of the agent's belief is committed
    versus left as uncertainty. A low-confidence agent contributes mostly
    to the UNCERTAIN focal element.
    """
    prob = max(0.0, min(1.0, estimate.probability))
    conf = max(0.0, min(1.0, estimate.confidence))

    m_yes = prob * conf
    m_no = (1.0 - prob) * conf
    m_unc = 1.0 - conf

    # Ensure non-negative and normalized
    total = m_yes + m_no + m_unc
    if total < 1e-12:
        return {YES: 0.0, NO: 0.0, UNCERTAIN: 1.0}

    return {
        YES: m_yes / total,
        NO: m_no / total,
        UNCERTAIN: m_unc / total,
    }


def _combine_two_masses(m1: dict, m2: dict) -> dict:
    """
    Combine two mass functions using Dempster's rule of combination.

    For each pair of focal elements, compute intersection and accumulate mass.
    Conflicting mass (empty intersection) is tracked as K and used for
    normalization: combined_m(A) = sum_{B∩C=A} m1(B)*m2(C) / (1 - K).
    """
    # Define intersection rules for our three focal elements
    # YES ∩ YES = YES, NO ∩ NO = NO, UNCERTAIN ∩ X = X
    # YES ∩ NO = EMPTY (conflict)

    intersection_map = {
        (YES, YES): YES,
        (NO, NO): NO,
        (UNCERTAIN, UNCERTAIN): UNCERTAIN,
        (YES, UNCERTAIN): YES,
        (UNCERTAIN, YES): YES,
        (NO, UNCERTAIN): NO,
        (UNCERTAIN, NO): NO,
        (YES, NO): None,  # conflict
        (NO, YES): None,  # conflict
    }

    combined = {YES: 0.0, NO: 0.0, UNCERTAIN: 0.0}
    conflict = 0.0

    for a, ma in m1.items():
        for b, mb in m2.items():
            product = ma * mb
            if product < 1e-15:
                continue

            result = intersection_map.get((a, b))
            if result is None:
                conflict += product
            else:
                combined[result] += product

    # Normalize by 1 / (1 - K)
    if conflict >= 1.0 - 1e-12:
        # Total conflict: sources completely contradict each other
        # Return maximum uncertainty
        return {YES: 0.0, NO: 0.0, UNCERTAIN: 1.0}

    normalizer = 1.0 / (1.0 - conflict)
    return {
        YES: combined[YES] * normalizer,
        NO: combined[NO] * normalizer,
        UNCERTAIN: combined[UNCERTAIN] * normalizer,
    }


def _conflict_coefficient(m1: dict, m2: dict) -> float:
    """
    Compute the conflict coefficient K between two mass functions.

    K = sum of mass products where intersections are empty.
    K = 0 means no conflict; K approaching 1 means near-total contradiction.
    """
    intersection_map = {
        (YES, YES): YES,
        (NO, NO): NO,
        (UNCERTAIN, UNCERTAIN): UNCERTAIN,
        (YES, UNCERTAIN): YES,
        (UNCERTAIN, YES): YES,
        (NO, UNCERTAIN): NO,
        (UNCERTAIN, NO): NO,
        (YES, NO): None,
        (NO, YES): None,
    }

    conflict = 0.0
    for a, ma in m1.items():
        for b, mb in m2.items():
            if intersection_map.get((a, b)) is None:
                conflict += ma * mb

    return max(0.0, min(1.0, conflict))
