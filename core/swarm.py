"""
Swarm orchestrator -- runs debate rounds, applies Bayesian updating,
game theory analysis, and statistical modeling.
"""

from __future__ import annotations
import os
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box

from core.theme import (
    console, header, section, stat_row, stat_card,
    progress_bar, probability_color, edge_color, sentiment_bar,
    COLORS, LOGO_SMALL,
)
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


class Swarm:
    def __init__(self, agents: list[Agent] | None = None):
        init_db()
        self.agents = agents or build_swarm()
        self.debate_rounds = int(os.getenv("DEBATE_ROUNDS", "2"))

    def forecast(self, question: str, market_odds: float | None = None) -> dict:
        # ── Question Header ──
        console.print()
        console.print(Panel(
            f"  [bold white]{question}[/]",
            border_style=COLORS["brand"],
            title=f"[bold {COLORS['brand']}]<<<>>>  FORECAST[/]",
            title_align="left",
            subtitle=f"[{COLORS['dim']}]{self.debate_rounds} rounds  |  {len(self.agents)} agents[/]",
            subtitle_align="right",
            padding=(1, 2),
        ))

        # ── Data Fetch ──
        console.print()
        console.print(f"  [{COLORS['dim']}]Fetching live data from 23 sources...[/]")
        context = build_context(question)
        console.print(f"  [{COLORS['positive']}]Context ready[/]")

        calibration_weights = get_calibration_weights()

        round1_estimates: list[AgentEstimate] = []
        all_estimates: list[AgentEstimate] = []

        # ── Debate Rounds ──
        for round_num in range(1, self.debate_rounds + 1):
            section_title = f"ROUND {round_num}"
            console.print()
            console.print(f"  [bold {COLORS['brand']}]{'━' * 3} {section_title} {'━' * (50 - len(section_title))}[/]")
            console.print()

            round_estimates = []

            for agent in self.agents:
                persona_short = agent.persona[:22].ljust(22)
                console.print(f"  [{COLORS['dim']}]{persona_short}[/]", end="")
                est = agent.estimate(
                    question=question,
                    context=context,
                    debate_round=round_num,
                    other_estimates=all_estimates if round_num > 1 else None,
                )
                round_estimates.append(est)

                # Mini inline visualization
                p = est.probability
                p_color = probability_color(p)
                conf_bar = progress_bar(est.confidence, width=8, filled_color=COLORS["dim"])
                console.print(f" [bold {p_color}]{p:5.1%}[/]  {conf_bar}  [{COLORS['dim']}]conf {est.confidence:.0%}[/]")

            if round_num == 1:
                round1_estimates = round_estimates.copy()
            all_estimates = round_estimates

        # save to calibration DB
        for est in all_estimates:
            save_forecast(question, est.agent_id, est.probability)

        # ── ANALYSIS PIPELINE ──
        console.print()
        console.print(f"  [bold {COLORS['accent2']}]{'━' * 3} ANALYSIS {'━' * 44}[/]")
        console.print()

        # 1. Standard weighted aggregation
        result = aggregate(all_estimates, calibration_weights)

        # 2. Bayesian aggregation
        bayesian = bayesian_aggregate(all_estimates, prior=market_odds or 0.5)
        result["bayesian"] = bayesian

        # 3. Agent agreement matrix
        agreement = compute_agent_agreement_matrix(all_estimates)
        result["agreement"] = agreement

        # 4. Herding detection
        herding = detect_herding(all_estimates)
        result["herding"] = herding

        # 5. Information cascade
        if self.debate_rounds > 1 and round1_estimates:
            cascade = compute_information_cascade(round1_estimates, all_estimates)
            result["cascade"] = cascade

        # 6. Nash equilibrium
        nash = nash_equilibrium_check(all_estimates)
        result["nash_equilibrium"] = nash

        # 7. Bootstrap confidence interval
        bootstrap = bootstrap_confidence_interval(all_estimates)
        result["confidence_interval"] = bootstrap

        # 8. Monte Carlo
        mc = monte_carlo_scenarios(all_estimates)
        result["monte_carlo"] = mc

        # 9. Extremized aggregation
        ext = extremize(all_estimates)
        result["extremized"] = ext

        # 10. Surprisingly Popular
        sp = surprisingly_popular(all_estimates)
        result["surprisingly_popular"] = sp

        # 11. Log Opinion Pool
        logop = logarithmic_opinion_pool(all_estimates)
        result["log_opinion_pool"] = logop

        # 12. Cooke's Classical Model
        cooke = cooke_classical_weights(all_estimates, calibration_weights)
        result["cooke_classical"] = cooke

        # 13. Meta-Probability Weighting
        mpw = meta_probability_weight(all_estimates)
        result["meta_probability"] = mpw

        # 14. Neutral Pivoting
        pivot = neutral_pivot(all_estimates)
        result["neutral_pivot"] = pivot

        # 15. Coherence check
        coherence = coherence_check(all_estimates)
        result["coherence"] = coherence

        save_swarm_forecast(question, result["probability"], result["consensus_score"], market_odds)

        # ── Print Everything ──
        self._print_agent_table(result)
        self._print_methods(result, bayesian, bootstrap, mc, ext, sp, logop, cooke, mpw, pivot)
        self._print_diagnostics(herding, nash, coherence, result.get("cascade"))
        self._print_final(result, market_odds, bayesian, bootstrap)

        if market_odds is not None:
            edge = result["probability"] - market_odds
            bayesian_edge = bayesian["bayesian_probability"] - market_odds
            result["market_odds"] = market_odds
            result["edge"] = round(edge, 4)
            result["edge_pct"] = f"{edge:+.1%}"
            result["bayesian_edge"] = round(bayesian_edge, 4)
            result["bayesian_edge_pct"] = f"{bayesian_edge:+.1%}"

        return result

    def _print_agent_table(self, result: dict):
        """Print the agent estimates table."""
        console.print()
        table = Table(
            box=box.SIMPLE_HEAVY,
            border_style=COLORS["brand"],
            show_header=True,
            header_style=f"bold {COLORS['brand']}",
            padding=(0, 1),
            title=f"[bold {COLORS['brand']}]Agent Estimates[/]",
        )
        table.add_column("#", style=COLORS["dim"], justify="right", width=3)
        table.add_column("Agent", style="bold", min_width=22)
        table.add_column("Prob", justify="right", width=6)
        table.add_column("", width=14)  # bar
        table.add_column("Conf", justify="right", width=5)
        table.add_column("Key Factors", style=COLORS["dim"], max_width=40)

        for i, est in enumerate(result["individual_estimates"], 1):
            p = est["probability"]
            p_color = probability_color(p)
            bar = progress_bar(p, width=12, filled_color=p_color)
            table.add_row(
                str(i),
                est["persona"],
                f"[{p_color}]{p:.0%}[/]",
                bar,
                f"{est['confidence']:.0%}",
                " | ".join(est["key_factors"][:2]),
            )
        console.print(table)

    def _print_methods(self, result, bayesian, bootstrap, mc, ext, sp, logop, cooke, mpw, pivot):
        """Print aggregation methods results."""
        console.print()
        console.print(f"  [bold {COLORS['accent2']}]{'━' * 3} AGGREGATION METHODS {'━' * 33}[/]")
        console.print()

        # Build a nice two-column layout
        methods = [
            ("Weighted Mean",      result["probability"],       f"consensus {result['consensus_score']:.0%}"),
            ("Bayesian",           bayesian["bayesian_probability"], f"info gain {bayesian['information_gain']:.3f} bits"),
            ("Extremized",         ext["extremized_probability"],   f"d={ext['extremizing_factor']:.2f}, shift {ext['shift']:+.1%}"),
            ("Surprisingly Pop.",  sp["sp_adjusted_probability"],   f"SP score {sp['sp_score']:+.3f}"),
            ("Log Opinion Pool",   logop["logop_probability"],     f"vs linear {logop['linear_probability']:.1%}"),
            ("Cooke's Classical",  cooke["cooke_probability"],     f"{cooke['n_qualified']}/{len(result['individual_estimates'])} qualified"),
            ("Meta-Prob Weight",   mpw["mpw_probability"],         f"top: {mpw['top_signal_agents'][0]['agent'][:16]}"),
            ("Neutral Pivot",      pivot["pivoted_probability"],   f"shift {pivot['pivot_shift']:+.3f}"),
            ("Monte Carlo",        mc["percentiles"]["p50"],       f"P(>50%)={mc['thresholds']['P(>50%)']:.0%}"),
            ("Bootstrap CI",       (bootstrap["ci_lower"] + bootstrap["ci_upper"]) / 2,
                                                                   f"[{bootstrap['ci_lower']:.1%}, {bootstrap['ci_upper']:.1%}]"),
        ]

        table = Table(
            box=box.SIMPLE,
            show_header=False,
            padding=(0, 1),
            show_edge=False,
        )
        table.add_column("Method", style="bold", min_width=20)
        table.add_column("Prob", justify="right", width=6)
        table.add_column("", width=16)  # bar
        table.add_column("Detail", style=COLORS["dim"])

        for name, prob, detail in methods:
            p_color = probability_color(prob)
            bar = progress_bar(prob, width=14, filled_color=p_color)
            table.add_row(
                f"  {name}",
                f"[bold {p_color}]{prob:.1%}[/]",
                bar,
                detail,
            )

        console.print(table)

    def _print_diagnostics(self, herding, nash, coherence, cascade):
        """Print diagnostic checks."""
        console.print()
        console.print(f"  [bold {COLORS['accent']}]{'━' * 3} DIAGNOSTICS {'━' * 40}[/]")
        console.print()

        # Herding
        if herding["herding_detected"]:
            h_icon = f"[{COLORS['warning']}]![/]"
            h_text = f"[{COLORS['warning']}]Herding detected[/]  score={herding['herding_score']:.2f}  direction={herding['herd_direction']}"
            if herding["contrarians"]:
                h_text += f"\n                         [{COLORS['dim']}]Contrarians: {', '.join(herding['contrarians'])}[/]"
        else:
            h_icon = f"[{COLORS['positive']}]OK[/]"
            h_text = f"[{COLORS['dim']}]No herding[/]  [{COLORS['dim']}]score={herding['herding_score']:.2f}[/]"
        console.print(f"  {h_icon}  Herding     {h_text}")

        # Nash
        if nash["stable"]:
            console.print(f"  [{COLORS['positive']}]OK[/]  Nash        [{COLORS['dim']}]Equilibrium stable -- no agent has incentive to deviate[/]")
        else:
            deviators = [d["agent"] for d in nash["potential_deviators"]]
            console.print(f"  [{COLORS['warning']}]![/]   Nash        [{COLORS['warning']}]Unstable[/]  [{COLORS['dim']}]potential deviators: {', '.join(deviators)}[/]")

        # Coherence
        if coherence["n_incoherent"] > 0:
            console.print(f"  [{COLORS['warning']}]![/]   Coherence   [{COLORS['warning']}]{coherence['n_incoherent']} incoherent[/]  [{COLORS['dim']}]mean={coherence['mean_coherence']:.2f}[/]")
        else:
            console.print(f"  [{COLORS['positive']}]OK[/]  Coherence   [{COLORS['dim']}]All coherent  mean={coherence['mean_coherence']:.2f}[/]")

        # Cascade
        if cascade:
            if cascade.get("cascade_detected"):
                console.print(f"  [{COLORS['warning']}]![/]   Cascade     [{COLORS['warning']}]Information cascade[/]  [{COLORS['dim']}]convergence={cascade['convergence_rate']:.0%}[/]")
                if cascade.get("flipped_agents"):
                    console.print(f"                         [{COLORS['dim']}]Flipped: {', '.join(cascade['flipped_agents'])}[/]")
            else:
                console.print(f"  [{COLORS['positive']}]OK[/]  Cascade     [{COLORS['dim']}]No cascade detected  convergence={cascade.get('convergence_rate', 0):.0%}[/]")

    def _print_final(self, result, market_odds, bayesian, bootstrap):
        """Print the final result panel."""
        prob = result["probability"]
        p_color = probability_color(prob)
        b_prob = bayesian["bayesian_probability"]
        b_color = probability_color(b_prob)

        # Build the big final display
        big_bar = progress_bar(prob, width=30, filled_color=p_color)

        lines = []
        lines.append(f"  [bold {p_color}]{prob:.1%}[/]  {big_bar}")
        lines.append("")
        lines.append(f"  [{COLORS['dim']}]Weighted[/]    [bold {p_color}]{prob:.1%}[/]     [{COLORS['dim']}]Bayesian[/]   [bold {b_color}]{b_prob:.1%}[/]     [{COLORS['dim']}]Consensus[/]  [bold]{result['consensus_score']:.0%}[/]")
        lines.append(f"  [{COLORS['dim']}]95% CI[/]     [{COLORS['dim']}][{bootstrap['ci_lower']:.1%}, {bootstrap['ci_upper']:.1%}][/]   [{COLORS['dim']}]Entropy[/]    [{COLORS['dim']}]{bayesian['entropy']:.3f} bits[/]")

        if market_odds is not None:
            edge = prob - market_odds
            b_edge = b_prob - market_odds
            e_color = edge_color(edge)
            be_color = edge_color(b_edge)
            lines.append("")
            lines.append(f"  [{COLORS['dim']}]Market[/]     [bold]{market_odds:.0%}[/]")
            lines.append(f"  [{COLORS['dim']}]Edge[/]       [bold {e_color}]{edge:+.1%}[/]       [{COLORS['dim']}]Bayesian Edge[/]  [bold {be_color}]{b_edge:+.1%}[/]")

            if abs(edge) >= 0.05:
                direction = "LONG" if edge > 0 else "SHORT"
                dir_color = COLORS["positive"] if edge > 0 else COLORS["negative"]
                lines.append(f"  [{COLORS['dim']}]Signal[/]     [bold {dir_color}]{direction}[/] [{COLORS['dim']}]-- swarm sees {abs(edge):.0%} edge vs market[/]")

        console.print()
        console.print(Panel(
            "\n".join(lines),
            border_style=COLORS["brand"],
            title=f"[bold {COLORS['brand']}]<<<>>>  RESULT[/]",
            title_align="left",
            padding=(1, 2),
        ))
