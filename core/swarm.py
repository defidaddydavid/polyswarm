"""
Swarm orchestrator -- runs debate rounds, applies 26 mathematical analysis
methods including Bayesian updating, game theory, information theory,
Dempster-Shafer evidence theory, copula dependency modeling, and more.
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
from core.aggregator import aggregate, stacking_aggregate
from core.bayesian import bayesian_aggregate, compute_agent_agreement_matrix
from core.game_theory import (
    detect_herding, compute_information_cascade,
    nash_equilibrium_check, scoring_rule_analysis,
)
from core.statistics import (
    bootstrap_confidence_interval, monte_carlo_scenarios,
    kernel_density_estimate, mcmc_posterior,
)
from core.extremize import extremize
from core.surprisingly_popular import surprisingly_popular
from core.opinion_pool import logarithmic_opinion_pool, cooke_classical_weights
from core.meta_probability import meta_probability_weight, neutral_pivot
from core.coherence import coherence_check
from core.copula import copula_dependency_analysis
from core.dempster_shafer import dempster_shafer_combine
from core.information_theory import information_analysis
from core.shapley import shapley_values
from core.conformal import conformal_prediction
from core.regime import detect_regime
from core.optimal_transport import method_distance_analysis
from core.calibration_curve import calibrate_probability
from core.calibration import (
    init_db,
    save_forecast,
    save_swarm_forecast,
    get_calibration_weights,
    get_forecast_history,
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
            subtitle=f"[{COLORS['dim']}]{self.debate_rounds} rounds  |  {len(self.agents)} agents  |  26 methods[/]",
            subtitle_align="right",
            padding=(1, 2),
        ))

        # ── Data Fetch ──
        console.print()
        console.print(f"  [{COLORS['dim']}]Fetching live data from 23 sources...[/]")
        context = build_context(question)
        console.print(f"  [{COLORS['positive']}]Context ready[/]")

        calibration_weights = get_calibration_weights()
        history = get_forecast_history(limit=100)

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

        # ══════════════════════════════════════════════════════════
        #  ANALYSIS PIPELINE — 26 methods
        # ══════════════════════════════════════════════════════════
        console.print()
        console.print(f"  [bold {COLORS['accent2']}]{'━' * 3} ANALYSIS PIPELINE {'━' * 35}[/]")
        console.print(f"  [{COLORS['dim']}]Running 26 mathematical models...[/]")
        console.print()

        # ── Classical Aggregation ──
        # 1. Standard weighted aggregation
        result = aggregate(all_estimates, calibration_weights)

        # 2. Bayesian aggregation
        bayesian = bayesian_aggregate(all_estimates, prior=market_odds or 0.5)
        result["bayesian"] = bayesian

        # 3. Agent agreement matrix (Jensen-Shannon divergence)
        agreement = compute_agent_agreement_matrix(all_estimates)
        result["agreement"] = agreement

        # 4. Extremized aggregation (IARPA/Tetlock)
        ext = extremize(all_estimates)
        result["extremized"] = ext

        # 5. Surprisingly Popular (Prelec, Nature 2017)
        sp = surprisingly_popular(all_estimates)
        result["surprisingly_popular"] = sp

        # 6. Logarithmic Opinion Pool (Genest & Zidek)
        logop = logarithmic_opinion_pool(all_estimates)
        result["log_opinion_pool"] = logop

        # 7. Cooke's Classical Model
        cooke = cooke_classical_weights(all_estimates, calibration_weights)
        result["cooke_classical"] = cooke

        # 8. Meta-Probability Weighting (Palley & Satopää 2023)
        mpw = meta_probability_weight(all_estimates)
        result["meta_probability"] = mpw

        # 9. Neutral Pivoting (2024)
        pivot = neutral_pivot(all_estimates)
        result["neutral_pivot"] = pivot

        # ── Statistical Analysis ──
        # 10. Bootstrap confidence interval
        bootstrap = bootstrap_confidence_interval(all_estimates)
        result["confidence_interval"] = bootstrap

        # 11. Monte Carlo simulation (Beta distributions)
        mc = monte_carlo_scenarios(all_estimates)
        result["monte_carlo"] = mc

        # 12. Kernel Density Estimation (Gaussian KDE, Silverman bandwidth)
        kde = kernel_density_estimate(all_estimates)
        result["kde"] = kde

        # 13. MCMC Posterior Sampling (Metropolis-Hastings)
        mcmc = mcmc_posterior(all_estimates)
        result["mcmc"] = mcmc

        # 14. Conformal Prediction (distribution-free intervals)
        cal_history = [
            {"forecast": h["probability"], "outcome": h["outcome"]}
            for h in history if h.get("outcome") is not None
        ]
        conformal = conformal_prediction(all_estimates, history=cal_history or None)
        result["conformal"] = conformal

        # ── Evidence & Uncertainty ──
        # 15. Dempster-Shafer Evidence Theory (belief functions)
        ds = dempster_shafer_combine(all_estimates)
        result["dempster_shafer"] = ds

        # 16. Copula Dependency Analysis (Gaussian copula, Kish's n_eff)
        copula = copula_dependency_analysis(all_estimates)
        result["copula"] = copula

        # ── Game Theory & Diagnostics ──
        # 17. Herding detection (HHI-based)
        herding = detect_herding(all_estimates)
        result["herding"] = herding

        # 18. Information cascade detection
        cascade = None
        if self.debate_rounds > 1 and round1_estimates:
            cascade = compute_information_cascade(round1_estimates, all_estimates)
            result["cascade"] = cascade

        # 19. Nash equilibrium check
        nash = nash_equilibrium_check(all_estimates)
        result["nash_equilibrium"] = nash

        # 20. Coherence check (Mandel 2024)
        coherence = coherence_check(all_estimates)
        result["coherence"] = coherence

        # 21. Scoring Rule Analysis (Brier, incentive compatibility)
        scoring = scoring_rule_analysis(all_estimates)
        result["scoring_rules"] = scoring

        # ── Information Theory ──
        # 22. Information-theoretic analysis (MI, transfer entropy, redundancy)
        info_theory = information_analysis(
            all_estimates,
            round1_estimates if self.debate_rounds > 1 and round1_estimates else None,
        )
        result["information_theory"] = info_theory

        # ── Attribution & Meta-Analysis ──
        # 23. Shapley Value Attribution
        shapley = shapley_values(all_estimates)
        result["shapley"] = shapley

        # 24. Regime Detection (Hidden Markov Model)
        regime_input = {
            "mean_prob": result["probability"],
            "std_dev": result["std_dev"],
            "herding_score": herding["herding_score"],
            "cascade_rate": cascade["convergence_rate"] if cascade else 0,
        }
        regime = detect_regime(regime_input)
        result["regime"] = regime

        # 25. Optimal Transport — method distance & clustering
        method_probs = {
            "Weighted": result["probability"],
            "Bayesian": bayesian["bayesian_probability"],
            "Extremized": ext["extremized_probability"],
            "Surp. Popular": sp["sp_adjusted_probability"],
            "Log Opinion": logop["logop_probability"],
            "Cooke's": cooke["cooke_probability"],
            "Meta-Prob": mpw["mpw_probability"],
            "Neutral Pivot": pivot["pivoted_probability"],
            "Monte Carlo": mc["percentiles"]["p50"],
            "DS Pignistic": ds["pignistic_probability"],
            "Copula-Adj": copula["dependency_adjusted_probability"],
            "MCMC Post.": mcmc["posterior_median"],
            "KDE Mode": kde["mode"],
        }
        transport = method_distance_analysis(method_probs)
        result["optimal_transport"] = transport

        # 26. Calibration Curve (isotonic + Platt scaling)
        cal_curve = calibrate_probability(result["probability"], history=cal_history or None)
        result["calibration_curve"] = cal_curve

        # ── Stacking Ensemble (meta-learner across all methods) ──
        stacking = stacking_aggregate(method_probs)
        result["stacking"] = stacking

        save_swarm_forecast(question, result["probability"], result["consensus_score"], market_odds)

        # ══════════════════════════════════════════════════════════
        #  DISPLAY
        # ══════════════════════════════════════════════════════════
        self._print_agent_table(result, shapley)
        self._print_methods(result, bayesian, bootstrap, mc, ext, sp, logop, cooke, mpw, pivot)
        self._print_advanced_methods(ds, copula, mcmc, kde, conformal, transport, cal_curve, stacking)
        self._print_diagnostics(herding, nash, coherence, cascade, scoring, info_theory)
        self._print_regime(regime)
        self._print_agent_attribution(shapley, info_theory)
        self._print_final(result, market_odds, bayesian, bootstrap, conformal, ds, transport, regime)

        if market_odds is not None:
            edge = result["probability"] - market_odds
            bayesian_edge = bayesian["bayesian_probability"] - market_odds
            result["market_odds"] = market_odds
            result["edge"] = round(edge, 4)
            result["edge_pct"] = f"{edge:+.1%}"
            result["bayesian_edge"] = round(bayesian_edge, 4)
            result["bayesian_edge_pct"] = f"{bayesian_edge:+.1%}"

        return result

    # ══════════════════════════════════════════════════════════
    #  DISPLAY METHODS
    # ══════════════════════════════════════════════════════════

    def _print_agent_table(self, result: dict, shapley: dict):
        """Print agent estimates with Shapley attribution."""
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
        table.add_column("Shapley", justify="right", width=7)
        table.add_column("Key Factors", style=COLORS["dim"], max_width=34)

        shapley_map = shapley.get("per_agent_shapley", {})

        for i, est in enumerate(result["individual_estimates"], 1):
            p = est["probability"]
            p_color = probability_color(p)
            bar = progress_bar(p, width=12, filled_color=p_color)

            # Shapley value for this agent
            sv = shapley_map.get(est["agent_id"], {})
            sv_val = sv.get("shapley_value", 0)
            sv_color = COLORS["positive"] if sv_val > 0 else COLORS["negative"] if sv_val < 0 else COLORS["dim"]

            table.add_row(
                str(i),
                est["persona"],
                f"[{p_color}]{p:.0%}[/]",
                bar,
                f"{est['confidence']:.0%}",
                f"[{sv_color}]{sv_val:+.3f}[/]",
                " | ".join(est["key_factors"][:2]),
            )
        console.print(table)

    def _print_methods(self, result, bayesian, bootstrap, mc, ext, sp, logop, cooke, mpw, pivot):
        """Print classical aggregation methods."""
        console.print()
        console.print(f"  [bold {COLORS['accent2']}]{'━' * 3} CLASSICAL AGGREGATION {'━' * 30}[/]")
        console.print()

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

        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), show_edge=False)
        table.add_column("Method", style="bold", min_width=20)
        table.add_column("Prob", justify="right", width=6)
        table.add_column("", width=16)
        table.add_column("Detail", style=COLORS["dim"])

        for name, prob, detail in methods:
            p_color = probability_color(prob)
            bar = progress_bar(prob, width=14, filled_color=p_color)
            table.add_row(f"  {name}", f"[bold {p_color}]{prob:.1%}[/]", bar, detail)

        console.print(table)

    def _print_advanced_methods(self, ds, copula, mcmc, kde, conformal, transport, cal_curve, stacking):
        """Print advanced mathematical analysis methods."""
        console.print()
        console.print(f"  [bold {COLORS['accent']}]{'━' * 3} ADVANCED MATHEMATICAL ANALYSIS {'━' * 22}[/]")
        console.print()

        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), show_edge=False)
        table.add_column("Method", style="bold", min_width=20)
        table.add_column("Prob", justify="right", width=6)
        table.add_column("", width=16)
        table.add_column("Detail", style=COLORS["dim"])

        advanced = [
            ("Dempster-Shafer",    ds["pignistic_probability"],
                f"belief={ds['belief_yes']:.2f}  plaus={ds['plausibility_yes']:.2f}  uncert={ds['uncertainty_gap']:.2f}"),
            ("Copula-Adjusted",    copula["dependency_adjusted_probability"],
                f"eff. N={copula['effective_n']:.1f}/{copula['n_agents']}  indep={copula['independence_ratio']:.0%}"),
            ("MCMC Posterior",     mcmc["posterior_median"],
                f"HDI=[{mcmc['hdi_lower']:.1%}, {mcmc['hdi_upper']:.1%}]  accept={mcmc['acceptance_rate']:.0%}"),
            ("KDE Mode",           kde["mode"],
                f"bw={kde['bandwidth_used']:.3f}  bimodal={'yes' if kde.get('bimodal') else 'no'}"),
            ("Conformal",          (conformal["conformal_lower"] + conformal["conformal_upper"]) / 2,
                f"[{conformal['conformal_lower']:.1%}, {conformal['conformal_upper']:.1%}]  {conformal['method_used']}"),
            ("Robust Consensus",   transport["robust_consensus"],
                f"{transport['largest_cluster_size']}/{len(transport.get('pairwise_distances', {}))} methods agree"),
            ("Stacking Ensemble",  stacking["stacking_probability"],
                f"method: {stacking['method_used']}"),
        ]

        if cal_curve.get("calibration_method", "none") != "none":
            advanced.append(
                ("Calibrated",     cal_curve["calibrated_probability"],
                    f"ECE={cal_curve.get('expected_calibration_error', 0):.3f}  method={cal_curve['calibration_method']}")
            )

        for name, prob, detail in advanced:
            p_color = probability_color(prob)
            bar = progress_bar(prob, width=14, filled_color=p_color)
            table.add_row(f"  {name}", f"[bold {p_color}]{prob:.1%}[/]", bar, detail)

        console.print(table)

        # DS uncertainty visualization
        if ds["uncertainty_gap"] > 0.3:
            console.print()
            console.print(f"  [{COLORS['warning']}]⚠  High evidential uncertainty ({ds['uncertainty_gap']:.0%})[/]  "
                          f"[{COLORS['dim']}]— agents have conflicting evidence, consider sizing down[/]")
        if ds.get("should_abstain"):
            console.print(f"  [{COLORS['negative']}]✗  Dempster-Shafer recommends ABSTAIN[/]  "
                          f"[{COLORS['dim']}]— conflict={ds['conflict_coefficient']:.2f}[/]")

    def _print_diagnostics(self, herding, nash, coherence, cascade, scoring, info_theory):
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
            h_text = f"[{COLORS['dim']}]No herding  score={herding['herding_score']:.2f}[/]"
        console.print(f"  {h_icon}  Herding     {h_text}")

        # Nash
        if nash["stable"]:
            console.print(f"  [{COLORS['positive']}]OK[/]  Nash        [{COLORS['dim']}]Equilibrium stable — no agent has incentive to deviate[/]")
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
                console.print(f"  [{COLORS['positive']}]OK[/]  Cascade     [{COLORS['dim']}]No cascade  convergence={cascade.get('convergence_rate', 0):.0%}[/]")

        # Scoring rules
        n_strategic = scoring.get("n_strategic", 0)
        if n_strategic > 0:
            strat_agents = ", ".join(scoring.get("strategic_agents", [])[:3])
            console.print(f"  [{COLORS['warning']}]![/]   Incentives  [{COLORS['warning']}]{n_strategic} strategic[/]  [{COLORS['dim']}]{strat_agents}[/]")
        else:
            console.print(f"  [{COLORS['positive']}]OK[/]  Incentives  [{COLORS['dim']}]All agents incentive-compatible (truthful={scoring.get('mean_truthfulness', 1):.0%})[/]")

        # Information theory
        redundancy = info_theory.get("redundancy_ratio", 0)
        if redundancy > 0.7:
            console.print(f"  [{COLORS['warning']}]![/]   Redundancy  [{COLORS['warning']}]{redundancy:.0%} shared info[/]  [{COLORS['dim']}]agents mostly agree — limited diversity[/]")
        else:
            console.print(f"  [{COLORS['positive']}]OK[/]  Redundancy  [{COLORS['dim']}]{redundancy:.0%} shared  diversity={info_theory.get('diversity_index', 0):.2f}[/]")

    def _print_regime(self, regime: dict):
        """Print regime detection result."""
        console.print()
        console.print(f"  [bold {COLORS['accent2']}]{'━' * 3} REGIME DETECTION {'━' * 36}[/]")
        console.print()

        regime_name = regime.get("current_regime", "unknown").upper()
        regime_colors = {
            "CONSENSUS": COLORS["positive"],
            "DEBATE": COLORS["warning"],
            "CHAOS": COLORS["negative"],
        }
        r_color = regime_colors.get(regime_name, COLORS["dim"])
        confidence = regime.get("confidence", 0)

        console.print(f"  [{r_color}]■[/]  Regime      [bold {r_color}]{regime_name}[/]  "
                      f"[{COLORS['dim']}]confidence={confidence:.0%}[/]")

        desc = regime.get("regime_description", "")
        if desc:
            console.print(f"                  [{COLORS['dim']}]{desc}[/]")

        strategy = regime.get("recommended_strategy", "")
        if strategy:
            console.print(f"                  [{COLORS['dim']}]Strategy: {strategy}[/]")

    def _print_agent_attribution(self, shapley: dict, info_theory: dict):
        """Print agent-level insights — Shapley values and information flow."""
        console.print()
        console.print(f"  [bold {COLORS['accent']}]{'━' * 3} AGENT ATTRIBUTION (SHAPLEY VALUES) {'━' * 17}[/]")
        console.print()

        per_agent = shapley.get("per_agent_shapley", {})
        if not per_agent:
            return

        # Sort by Shapley value
        ranked = sorted(per_agent.values(), key=lambda x: x.get("shapley_value", 0), reverse=True)

        # Find max absolute value for bar scaling
        max_abs = max(abs(a.get("shapley_value", 0)) for a in ranked) or 1

        for agent_data in ranked:
            sv = agent_data.get("shapley_value", 0)
            persona = agent_data.get("persona", "Unknown")[:22].ljust(22)
            rank = agent_data.get("rank", 0)

            # Bidirectional bar: negative goes left, positive goes right
            bar_width = 12
            bar_fill = int(abs(sv) / max_abs * bar_width)

            if sv >= 0:
                sv_color = COLORS["positive"]
                bar = f"[{COLORS['dim']}]{'·' * bar_width}[/][{sv_color}]{'█' * bar_fill}{'·' * (bar_width - bar_fill)}[/]"
            else:
                sv_color = COLORS["negative"]
                bar = f"[{sv_color}]{'·' * (bar_width - bar_fill)}{'█' * bar_fill}[/][{COLORS['dim']}]{'·' * bar_width}[/]"

            console.print(f"  [{COLORS['dim']}]#{rank:<2}[/] {persona}  [{sv_color}]{sv:+.4f}[/]  {bar}")

        # Summary
        mvp = shapley.get("most_valuable_agent", "")
        redundant = shapley.get("redundant_agents", [])
        console.print()
        console.print(f"  [{COLORS['dim']}]MVP: {mvp}  |  "
                      f"Redundant: {len(redundant)} agents  |  "
                      f"Concentration: {shapley.get('concentration_index', 0):.2f}[/]")

    def _print_final(self, result, market_odds, bayesian, bootstrap, conformal, ds, transport, regime):
        """Print the final result panel."""
        prob = result["probability"]
        p_color = probability_color(prob)
        b_prob = bayesian["bayesian_probability"]
        b_color = probability_color(b_prob)
        robust = transport["robust_consensus"]
        r_color = probability_color(robust)

        # Build the big final display
        big_bar = progress_bar(prob, width=30, filled_color=p_color)

        lines = []
        lines.append(f"  [bold {p_color}]{prob:.1%}[/]  {big_bar}")
        lines.append("")
        lines.append(f"  [{COLORS['dim']}]Weighted[/]    [bold {p_color}]{prob:.1%}[/]     "
                     f"[{COLORS['dim']}]Bayesian[/]   [bold {b_color}]{b_prob:.1%}[/]     "
                     f"[{COLORS['dim']}]Robust[/]     [bold {r_color}]{robust:.1%}[/]")
        lines.append(f"  [{COLORS['dim']}]95% CI[/]     [{COLORS['dim']}][{bootstrap['ci_lower']:.1%}, {bootstrap['ci_upper']:.1%}][/]   "
                     f"[{COLORS['dim']}]Conformal[/]  [{COLORS['dim']}][{conformal['conformal_lower']:.1%}, {conformal['conformal_upper']:.1%}][/]   "
                     f"[{COLORS['dim']}]Consensus[/]  [bold]{result['consensus_score']:.0%}[/]")
        lines.append(f"  [{COLORS['dim']}]Entropy[/]    [{COLORS['dim']}]{bayesian['entropy']:.3f} bits[/]   "
                     f"[{COLORS['dim']}]DS Uncert[/]  [{COLORS['dim']}]{ds['uncertainty_gap']:.0%}[/]   "
                     f"[{COLORS['dim']}]Regime[/]     [{COLORS['dim']}]{regime.get('current_regime', '?').upper()}[/]")

        if market_odds is not None:
            edge = prob - market_odds
            b_edge = b_prob - market_odds
            r_edge = robust - market_odds
            e_color = edge_color(edge)
            be_color = edge_color(b_edge)
            re_color = edge_color(r_edge)
            lines.append("")
            lines.append(f"  [{COLORS['dim']}]Market[/]     [bold]{market_odds:.0%}[/]")
            lines.append(f"  [{COLORS['dim']}]Edge[/]       [bold {e_color}]{edge:+.1%}[/]       "
                         f"[{COLORS['dim']}]Bayesian[/]  [bold {be_color}]{b_edge:+.1%}[/]       "
                         f"[{COLORS['dim']}]Robust[/]  [bold {re_color}]{r_edge:+.1%}[/]")

            # Use robust consensus for signal
            avg_edge = (edge + b_edge + r_edge) / 3
            if abs(avg_edge) >= 0.05:
                direction = "LONG" if avg_edge > 0 else "SHORT"
                dir_color = COLORS["positive"] if avg_edge > 0 else COLORS["negative"]
                lines.append(f"  [{COLORS['dim']}]Signal[/]     [bold {dir_color}]{direction}[/] "
                             f"[{COLORS['dim']}]— swarm sees {abs(avg_edge):.0%} avg edge across 3 methods[/]")

            if ds.get("should_abstain"):
                lines.append(f"  [{COLORS['negative']}]⚠ CAUTION[/]  [{COLORS['dim']}]Dempster-Shafer flags high conflict — consider abstaining[/]")

        console.print()
        console.print(Panel(
            "\n".join(lines),
            border_style=COLORS["brand"],
            title=f"[bold {COLORS['brand']}]<<<>>>  RESULT  |  26 METHODS  |  {len(result['individual_estimates'])} AGENTS[/]",
            title_align="left",
            subtitle=f"[{COLORS['dim']}]powered by PolySwarm v1.0[/]",
            subtitle_align="right",
            padding=(1, 2),
        ))
