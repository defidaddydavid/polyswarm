"""
Swarm orchestrator — runs debate rounds, applies Bayesian updating,
game theory analysis, and statistical modeling.
"""

from __future__ import annotations
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from core.agent import Agent, AgentEstimate
from core.aggregator import aggregate
from core.bayesian import bayesian_aggregate, compute_agent_agreement_matrix
from core.game_theory import detect_herding, compute_information_cascade, nash_equilibrium_check
from core.statistics import bootstrap_confidence_interval, monte_carlo_scenarios
from core.extremize import extremize
from core.surprisingly_popular import surprisingly_popular
from core.opinion_pool import logarithmic_opinion_pool, cooke_classical_weights
from core.meta_probability import meta_probability_weight, neutral_pivot
from core.coherence import coherence_check
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
        console.print(Panel(f"[bold cyan]Question:[/bold cyan] {question}", title="⟐ PolySwarm Forecast", border_style="cyan"))

        console.print("[dim]Fetching context from 15+ live sources...[/dim]")
        context = build_context(question)

        calibration_weights = get_calibration_weights()

        round1_estimates: list[AgentEstimate] = []
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

            if round_num == 1:
                round1_estimates = round_estimates.copy()
            all_estimates = round_estimates

        # save to calibration DB
        for est in all_estimates:
            save_forecast(question, est.agent_id, est.probability)

        # ── ANALYSIS PIPELINE ──
        console.print(f"\n[bold magenta]── Analysis ──[/bold magenta]")

        # 1. Standard weighted aggregation
        result = aggregate(all_estimates, calibration_weights)

        # 2. Bayesian aggregation
        bayesian = bayesian_aggregate(all_estimates, prior=market_odds or 0.5)
        result["bayesian"] = bayesian
        console.print(f"  [dim]Bayesian posterior:[/dim] [bold]{bayesian['bayesian_probability']:.1%}[/bold] (info gain: {bayesian['information_gain']:.3f} bits)")

        # 3. Agent agreement matrix
        agreement = compute_agent_agreement_matrix(all_estimates)
        result["agreement"] = agreement

        # 4. Herding detection
        herding = detect_herding(all_estimates)
        result["herding"] = herding
        if herding["herding_detected"]:
            console.print(f"  [yellow]⚠ Herding detected[/yellow] (score: {herding['herding_score']:.2f}, direction: {herding['herd_direction']})")
            if herding["contrarians"]:
                console.print(f"    Contrarians: {', '.join(herding['contrarians'])}")
        else:
            console.print(f"  [green]✓ No herding[/green] (score: {herding['herding_score']:.2f})")

        # 5. Information cascade (if multi-round)
        if self.debate_rounds > 1 and round1_estimates:
            cascade = compute_information_cascade(round1_estimates, all_estimates)
            result["cascade"] = cascade
            if cascade.get("cascade_detected"):
                console.print(f"  [yellow]⚠ Information cascade detected[/yellow] (convergence: {cascade['convergence_rate']:.0%})")
            if cascade.get("flipped_agents"):
                console.print(f"    Flipped agents: {', '.join(cascade['flipped_agents'])}")

        # 6. Nash equilibrium
        nash = nash_equilibrium_check(all_estimates)
        result["nash_equilibrium"] = nash
        if nash["stable"]:
            console.print(f"  [green]✓ Nash stable[/green] — no agent has incentive to deviate")
        else:
            deviators = [d["agent"] for d in nash["potential_deviators"]]
            console.print(f"  [yellow]⚠ Unstable[/yellow] — potential deviators: {', '.join(deviators)}")

        # 7. Bootstrap confidence interval
        bootstrap = bootstrap_confidence_interval(all_estimates)
        result["confidence_interval"] = bootstrap
        console.print(f"  [dim]95% CI:[/dim] [{bootstrap['ci_lower']:.1%}, {bootstrap['ci_upper']:.1%}] (width: {bootstrap['ci_width']:.1%})")

        # 8. Monte Carlo
        mc = monte_carlo_scenarios(all_estimates)
        result["monte_carlo"] = mc
        console.print(f"  [dim]Monte Carlo (5K sims):[/dim] median={mc['percentiles']['p50']:.1%}, P(>50%)={mc['thresholds']['P(>50%)']:.0%}")

        # 9. Extremized aggregation (Satopää/Baron/Tetlock IARPA)
        ext = extremize(all_estimates)
        result["extremized"] = ext
        console.print(f"  [dim]Extremized (d={ext['extremizing_factor']:.2f}):[/dim] [bold]{ext['extremized_probability']:.1%}[/bold] (shift: {ext['shift']:+.1%})")

        # 10. Surprisingly Popular algorithm (Prelec 2017)
        sp = surprisingly_popular(all_estimates)
        result["surprisingly_popular"] = sp
        console.print(f"  [dim]Surprisingly Popular:[/dim] [bold]{sp['sp_adjusted_probability']:.1%}[/bold] (SP score: {sp['sp_score']:+.3f}, direction: {sp['surprise_direction']})")

        # 11. Logarithmic opinion pool
        logop = logarithmic_opinion_pool(all_estimates)
        result["log_opinion_pool"] = logop
        console.print(f"  [dim]Log Opinion Pool:[/dim] [bold]{logop['logop_probability']:.1%}[/bold] (vs linear: {logop['linear_probability']:.1%})")

        # 12. Cooke's Classical Model (performance-based weighting)
        cooke = cooke_classical_weights(all_estimates, calibration_weights)
        result["cooke_classical"] = cooke
        console.print(f"  [dim]Cooke's Classical:[/dim] [bold]{cooke['cooke_probability']:.1%}[/bold] ({cooke['n_qualified']}/{len(all_estimates)} agents qualified)")

        # 13. Meta-Probability Weighting (Palley & Satopää 2023)
        mpw = meta_probability_weight(all_estimates)
        result["meta_probability"] = mpw
        console.print(f"  [dim]Meta-Prob Weight:[/dim] [bold]{mpw['mpw_probability']:.1%}[/bold] (top signal: {mpw['top_signal_agents'][0]['agent']})")

        # 14. Neutral Pivoting (shared-information correction)
        pivot = neutral_pivot(all_estimates)
        result["neutral_pivot"] = pivot
        console.print(f"  [dim]Neutral Pivot:[/dim] [bold]{pivot['pivoted_probability']:.1%}[/bold] (shift: {pivot['pivot_shift']:+.3f})")

        # 15. Coherence check
        coherence = coherence_check(all_estimates)
        result["coherence"] = coherence
        if coherence["n_incoherent"] > 0:
            console.print(f"  [yellow]⚠ {coherence['n_incoherent']} incoherent agents[/yellow] (mean coherence: {coherence['mean_coherence']:.2f})")
        else:
            console.print(f"  [green]✓ All agents coherent[/green] (mean: {coherence['mean_coherence']:.2f})")

        save_swarm_forecast(question, result["probability"], result["consensus_score"], market_odds)

        # print results
        self._print_results(result, market_odds, bayesian, bootstrap, mc)

        if market_odds is not None:
            edge = result["probability"] - market_odds
            bayesian_edge = bayesian["bayesian_probability"] - market_odds
            result["market_odds"] = market_odds
            result["edge"] = round(edge, 4)
            result["edge_pct"] = f"{edge:+.1%}"
            result["bayesian_edge"] = round(bayesian_edge, 4)
            result["bayesian_edge_pct"] = f"{bayesian_edge:+.1%}"

        return result

    def _print_results(self, result: dict, market_odds, bayesian, bootstrap, mc):
        # agent table
        table = Table(title="Agent Estimates", box=box.ROUNDED, show_lines=True)
        table.add_column("Agent", style="cyan")
        table.add_column("P", justify="right", style="green")
        table.add_column("Conf", justify="right")
        table.add_column("Key Factors")

        for est in result["individual_estimates"]:
            table.add_row(
                est["persona"],
                f"{est['probability']:.0%}",
                f"{est['confidence']:.0%}",
                " | ".join(est["key_factors"][:2]),
            )
        console.print(table)

        # summary panel
        odds_line = ""
        if market_odds is not None:
            edge = result['probability'] - market_odds
            b_edge = bayesian['bayesian_probability'] - market_odds
            odds_line = (
                f"\n  Market: [yellow]{market_odds:.0%}[/yellow]  "
                f"Edge (weighted): [bold]{'+'if edge>0 else ''}{edge:.1%}[/bold]  "
                f"Edge (Bayesian): [bold]{'+'if b_edge>0 else ''}{b_edge:.1%}[/bold]"
            )

        # get new analysis results if available
        ext_line = ""
        if "extremized" in result:
            ext = result["extremized"]
            ext_line = f"\n  Extremized: {ext['extremized_probability']:.1%} (d={ext['extremizing_factor']:.2f})"
        sp_line = ""
        if "surprisingly_popular" in result:
            sp = result["surprisingly_popular"]
            sp_line = f"  |  SP: {sp['sp_adjusted_probability']:.1%}"
        logop_line = ""
        if "log_opinion_pool" in result:
            logop = result["log_opinion_pool"]
            logop_line = f"  |  LogOP: {logop['logop_probability']:.1%}"

        console.print(Panel(
            f"  [bold green]Weighted:  {result['probability_pct']}[/bold green]  |  "
            f"[bold cyan]Bayesian:  {bayesian['bayesian_probability']:.1%}[/bold cyan]  |  "
            f"Consensus: {result['consensus_score']:.0%}\n"
            f"  95% CI: [{bootstrap['ci_lower']:.1%}, {bootstrap['ci_upper']:.1%}]  |  "
            f"MC median: {mc['percentiles']['p50']:.1%}  |  "
            f"Entropy: {bayesian['entropy']:.3f} bits"
            f"{ext_line}{sp_line}{logop_line}"
            f"{odds_line}",
            title="⟐ Final Result",
            border_style="green",
        ))
