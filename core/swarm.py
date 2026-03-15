"""
Swarm orchestrator. Runs N debate rounds across all agents,
then aggregates into a final probability estimate.
"""

from __future__ import annotations
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from core.agent import Agent, AgentEstimate
from core.aggregator import aggregate
from core.calibration import (
    init_db,
    save_forecast,
    save_swarm_forecast,
    get_calibration_weights,
)
from agents.personas import build_swarm
from data.context import build_context

console = Console()


class Swarm:
    def __init__(self, agents: list[Agent] | None = None):
        init_db()
        self.agents = agents or build_swarm()
        self.debate_rounds = int(os.getenv("DEBATE_ROUNDS", "2"))

    def forecast(self, question: str, market_odds: float | None = None) -> dict:
        console.print(Panel(f"[bold cyan]Question:[/bold cyan] {question}", title="PolySwarm Forecast"))

        # fetch context
        console.print("[dim]Fetching context...[/dim]")
        context = build_context(question)

        calibration_weights = get_calibration_weights()

        all_estimates: list[AgentEstimate] = []

        for round_num in range(1, self.debate_rounds + 1):
            console.print(f"\n[bold yellow]── Round {round_num} ──[/bold yellow]")
            round_estimates = []

            for agent in self.agents:
                console.print(f"  [dim]{agent.persona}[/dim] thinking...", end=" ")
                est = agent.estimate(
                    question=question,
                    context=context,
                    debate_round=round_num,
                    other_estimates=all_estimates if round_num > 1 else None,
                )
                round_estimates.append(est)
                console.print(f"[green]{est.probability:.0%}[/green] (confidence: {est.confidence:.0%})")

            all_estimates = round_estimates  # replace with latest round

        # save to calibration DB
        for est in all_estimates:
            save_forecast(question, est.agent_id, est.probability)

        # aggregate final result
        result = aggregate(all_estimates, calibration_weights)
        save_swarm_forecast(question, result["probability"], result["consensus_score"], market_odds)

        # print debate table
        self._print_results(result, market_odds)

        if market_odds is not None:
            edge = result["probability"] - market_odds
            result["market_odds"] = market_odds
            result["edge"] = round(edge, 4)
            result["edge_pct"] = f"{edge:+.1%}"

        return result

    def _print_results(self, result: dict, market_odds: float | None):
        table = Table(title="Agent Estimates", box=box.ROUNDED, show_lines=True)
        table.add_column("Persona", style="cyan")
        table.add_column("Probability", justify="right", style="green")
        table.add_column("Confidence", justify="right")
        table.add_column("Key Factors")

        for est in result["individual_estimates"]:
            table.add_row(
                est["persona"],
                f"{est['probability']:.0%}",
                f"{est['confidence']:.0%}",
                " | ".join(est["key_factors"][:2]),
            )
        console.print(table)

        odds_str = f"  Market odds: [yellow]{market_odds:.0%}[/yellow]  Edge: [bold]{'+'if result['probability']>market_odds else ''}{result['probability']-market_odds:.1%}[/bold]" if market_odds else ""
        console.print(Panel(
            f"[bold green]Swarm Probability: {result['probability_pct']}[/bold green]\n"
            f"Consensus: {result['consensus_score']:.0%}  |  Agents: {result['n_agents']}{odds_str}",
            title="Final Result",
            border_style="green",
        ))
