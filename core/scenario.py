"""
Scenario simulation engine.
Feed a scenario, get simulated crowd reactions, sentiment shift, and market impact.
This is the MiroFish-style mode: not just "will X happen" but "if X happens, what follows?"
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from core.agent import _get_llm_client, _call_llm, _parse_json

console = Console()


@dataclass
class ScenarioReaction:
    agent_id: str
    persona: str
    immediate_reaction: str       # what they do in first 5 minutes
    sentiment_shift: float        # -1.0 to +1.0
    price_impact_estimate: float  # % price change expected
    confidence: float
    reasoning: str
    actions: list[str]            # specific actions this persona takes


@dataclass
class ScenarioResult:
    scenario: str
    reactions: list[ScenarioReaction]
    aggregate_sentiment: float
    aggregate_price_impact: float
    consensus: float
    narrative: str                # what the crowd story becomes
    secondary_effects: list[str]  # second-order consequences


class ScenarioEngine:
    def __init__(self):
        self._provider, self._client = _get_llm_client()

    def simulate(self, scenario: str, context: str = "") -> ScenarioResult:
        """Simulate crowd reactions to a scenario across all market participant archetypes."""

        from agents.personas import PERSONA_DEFINITIONS

        console.print(Panel(f"[bold magenta]Scenario:[/bold magenta] {scenario}", title="PolySwarm Scenario Simulation"))

        reactions = []

        for persona_def in PERSONA_DEFINITIONS:
            console.print(f"  [dim]{persona_def['persona']}[/dim] reacting...", end=" ")

            system = f"""You are a {persona_def['persona']} reacting to a breaking news scenario in real-time.

Your profile: {persona_def['description']}
Your focus: {persona_def['information_focus']}
Your biases: {persona_def['bias_profile']}

Simulate your IMMEDIATE, AUTHENTIC reaction to the scenario — not a detached analysis, but how you personally would react and what you would actually DO.

Output ONLY valid JSON with these fields:
- immediate_reaction: string (what you think/say in the first 5 minutes)
- sentiment_shift: float from -1.0 (very bearish/negative) to +1.0 (very bullish/positive)
- price_impact_estimate: float (% price change you expect, e.g. 0.15 for +15%, -0.08 for -8%)
- confidence: float 0.0-1.0
- reasoning: string (2-3 sentences)
- actions: list of 3-4 strings (specific things you do: "buy spot BTC", "tweet bearish take", "close longs", etc.)"""

            user = f"""Scenario: {scenario}

{"Additional context: " + context if context else ""}

React now."""

            try:
                raw = _call_llm(self._provider, self._client, system, user)
                data = _parse_json(raw)

                reaction = ScenarioReaction(
                    agent_id=persona_def["agent_id"],
                    persona=persona_def["persona"],
                    immediate_reaction=data["immediate_reaction"],
                    sentiment_shift=float(data["sentiment_shift"]),
                    price_impact_estimate=float(data["price_impact_estimate"]),
                    confidence=float(data["confidence"]),
                    reasoning=data["reasoning"],
                    actions=data.get("actions", []),
                )
                reactions.append(reaction)
                sentiment_emoji = "🟢" if reaction.sentiment_shift > 0.2 else "🔴" if reaction.sentiment_shift < -0.2 else "🟡"
                console.print(f"{sentiment_emoji} sentiment={reaction.sentiment_shift:+.2f} impact={reaction.price_impact_estimate:+.1%}")
            except Exception as e:
                console.print(f"[red]error: {e}[/red]")

        # aggregate
        if reactions:
            weights = [r.confidence for r in reactions]
            total_w = sum(weights)
            agg_sentiment = sum(r.sentiment_shift * w for r, w in zip(reactions, weights)) / total_w
            agg_impact = sum(r.price_impact_estimate * w for r, w in zip(reactions, weights)) / total_w

            # consensus: how much agents agree on direction
            same_direction = sum(1 for r in reactions if (r.sentiment_shift > 0) == (agg_sentiment > 0))
            consensus = same_direction / len(reactions)
        else:
            agg_sentiment = 0.0
            agg_impact = 0.0
            consensus = 0.0

        # generate narrative summary
        narrative, secondary = self._generate_narrative(scenario, reactions, agg_sentiment, agg_impact)

        result = ScenarioResult(
            scenario=scenario,
            reactions=reactions,
            aggregate_sentiment=round(agg_sentiment, 3),
            aggregate_price_impact=round(agg_impact, 4),
            consensus=round(consensus, 3),
            narrative=narrative,
            secondary_effects=secondary,
        )

        self._print_results(result)
        return result

    def _generate_narrative(self, scenario: str, reactions: list[ScenarioReaction], sentiment: float, impact: float) -> tuple[str, list[str]]:
        """Generate a crowd narrative and secondary effects from the reactions."""
        reactions_summary = "\n".join([
            f"- {r.persona}: sentiment={r.sentiment_shift:+.2f}, actions={', '.join(r.actions[:2])}"
            for r in reactions
        ])

        system = "You are a market analyst synthesizing crowd reactions into a narrative."
        user = f"""Given this scenario and these market participant reactions, write:
1. A 2-sentence "crowd narrative" — the dominant story that emerges from these reactions combined
2. 3-4 second-order effects (what happens NEXT, after the initial reaction)

Scenario: {scenario}
Aggregate sentiment: {sentiment:+.2f}
Aggregate price impact: {impact:+.1%}

Reactions:
{reactions_summary}

Output JSON with:
- narrative: string
- secondary_effects: list of strings"""

        try:
            raw = _call_llm(self._provider, self._client, system, user, max_tokens=400)
            data = _parse_json(raw)
            return data["narrative"], data.get("secondary_effects", [])
        except Exception:
            return "Mixed reactions across market participants.", []

    def _print_results(self, result: ScenarioResult):
        table = Table(title="Scenario Reactions", box=box.ROUNDED, show_lines=True)
        table.add_column("Persona", style="cyan", min_width=20)
        table.add_column("Sentiment", justify="center")
        table.add_column("Price Impact", justify="right", style="green")
        table.add_column("Immediate Actions")

        for r in result.reactions:
            bar = "█" * int(abs(r.sentiment_shift) * 10)
            sentiment_str = f"[green]+{r.sentiment_shift:.2f} {bar}[/green]" if r.sentiment_shift > 0 else f"[red]{r.sentiment_shift:.2f} {bar}[/red]"
            table.add_row(
                r.persona,
                sentiment_str,
                f"{r.price_impact_estimate:+.1%}",
                " · ".join(r.actions[:2]),
            )
        console.print(table)

        direction = "BULLISH" if result.aggregate_sentiment > 0.1 else "BEARISH" if result.aggregate_sentiment < -0.1 else "NEUTRAL"
        color = "green" if result.aggregate_sentiment > 0.1 else "red" if result.aggregate_sentiment < -0.1 else "yellow"

        console.print(Panel(
            f"[bold {color}]{direction}[/bold {color}]  Sentiment: {result.aggregate_sentiment:+.2f}  |  Price Impact: {result.aggregate_price_impact:+.1%}  |  Consensus: {result.consensus:.0%}\n\n"
            f"[italic]{result.narrative}[/italic]\n\n"
            + "\n".join(f"  -> {e}" for e in result.secondary_effects),
            title="Crowd Simulation Result",
            border_style=color,
        ))
