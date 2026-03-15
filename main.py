"""
PolySwarm -- Multi-agent AI forecasting engine for prediction markets.

Usage:
  python main.py forecast "Will BTC close above $100k on March 31 2026?"
  python main.py forecast "Will the Fed cut rates in June 2026?" --odds 0.35
  python main.py scenario "Elon Musk tweets that Tesla will accept Bitcoin again"
  python main.py scenario "SEC approves spot ETH ETF options" --context "ETH currently at $3,200"
  python main.py resolve "Will BTC close above $100k on March 31 2026?" --outcome 1.0
  python main.py serve
  python main.py calibration --export json
"""

import typer
from dotenv import load_dotenv
load_dotenv()

from core.theme import (
    console, header, section, footer, stat_row, stat_card,
    progress_bar, probability_color, edge_color, brier_color,
    quality_label, status_badge, category_style, COLORS, LOGO_SMALL,
)
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich import box

app = typer.Typer(
    help="PolySwarm -- Multi-agent prediction market forecasting & scenario simulation",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


# ═══════════════════════════════════════════
# Banner
# ═══════════════════════════════════════════

def _banner():
    """Print the startup banner."""
    console.print()
    console.print(f"  [bold #F59E0B]<<<>>>  P O L Y S W A R M[/]  [dim]v0.8.0[/dim]")
    console.print(f"  [{COLORS['dim']}]Multi-agent AI forecasting engine[/]")
    console.print(f"  [{COLORS['dim']}]{'─' * 42}[/]")


# ═══════════════════════════════════════════
# Forecast
# ═══════════════════════════════════════════

@app.command()
def forecast(
    question: str = typer.Argument(..., help="The question to forecast"),
    odds: float = typer.Option(None, "--odds", help="Current market odds (0.0-1.0) for edge calculation"),
    rounds: int = typer.Option(None, "--rounds", help="Number of debate rounds (default: 2)"),
    size: int = typer.Option(None, "--size", help="Number of agents to use (default: all 12)"),
):
    """Run a swarm forecast on a binary question."""
    import os
    if rounds:
        os.environ["DEBATE_ROUNDS"] = str(rounds)

    _banner()

    from core.swarm import Swarm
    from agents.personas import build_swarm
    swarm = Swarm(agents=build_swarm(size) if size else None)
    result = swarm.forecast(question, market_odds=odds)

    footer()


# ═══════════════════════════════════════════
# Scenario
# ═══════════════════════════════════════════

@app.command()
def scenario(
    description: str = typer.Argument(..., help="The scenario to simulate"),
    context: str = typer.Option("", "--context", help="Additional context for the simulation"),
):
    """Simulate crowd reactions to a scenario."""
    _banner()

    from core.scenario import ScenarioEngine
    engine = ScenarioEngine()
    engine.simulate(description, context)

    footer()


# ═══════════════════════════════════════════
# Resolve
# ═══════════════════════════════════════════

@app.command()
def resolve(
    question: str = typer.Argument(..., help="The question to resolve"),
    outcome: float = typer.Option(..., "--outcome", help="1.0 = YES resolved, 0.0 = NO resolved"),
):
    """Resolve a forecast and update calibration scores."""
    _banner()

    from core.calibration import resolve_forecast
    resolve_forecast(question, outcome)

    result_label = f"[{COLORS['positive']}]YES[/]" if outcome == 1.0 else f"[{COLORS['negative']}]NO[/]"
    console.print()
    console.print(Panel(
        f"  [{COLORS['positive']}]Resolved[/]  {question}\n"
        f"  Outcome: {result_label}",
        border_style=COLORS["positive"],
        padding=(1, 2),
    ))

    footer()


# ═══════════════════════════════════════════
# Serve
# ═══════════════════════════════════════════

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
):
    """Start the FastAPI server."""
    _banner()
    console.print()
    console.print(f"  [{COLORS['positive']}]Starting API server...[/]")
    console.print(f"  [{COLORS['dim']}]Docs:  http://{host}:{port}/docs[/]")
    console.print(f"  [{COLORS['dim']}]Health: http://{host}:{port}/health[/]")
    console.print()

    import uvicorn
    uvicorn.run("api.routes:app", host=host, port=port, reload=True)


# ═══════════════════════════════════════════
# Calibration
# ═══════════════════════════════════════════

@app.command()
def calibration(
    export: str = typer.Option(None, "--export", help="Export format: json or csv"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path (default: stdout)"),
):
    """Show current calibration scores across all agents."""
    from core.calibration import init_db, get_swarm_brier_score, get_agent_brier_scores, export_calibration
    init_db()

    if export:
        data = export_calibration(format=export)
        if output:
            with open(output, "w") as f:
                f.write(data)
            console.print(f"[{COLORS['positive']}]  Exported to {output}[/]")
        else:
            console.print(data)
        return

    _banner()
    section("Calibration")

    swarm_score = get_swarm_brier_score()
    agent_scores = get_agent_brier_scores()

    # Swarm-level score
    if swarm_score is not None:
        score_color = brier_color(swarm_score)
        console.print(f"  [{COLORS['dim']}]Swarm Brier Score[/]  [bold {score_color}]{swarm_score:.4f}[/]  {quality_label(swarm_score)}")
    else:
        console.print(f"  [{COLORS['dim']}]No resolved forecasts yet. Use 'resolve' to track accuracy.[/]")

    if agent_scores:
        console.print()
        table = Table(
            box=box.SIMPLE_HEAVY,
            border_style=COLORS["brand"],
            show_header=True,
            header_style=f"bold {COLORS['brand']}",
            padding=(0, 2),
        )
        table.add_column("#", style=COLORS["dim"], justify="right", width=3)
        table.add_column("Agent", style="bold", min_width=20)
        table.add_column("Brier", justify="right", width=8)
        table.add_column("", width=22)  # visual bar
        table.add_column("Rating", justify="center", width=12)

        for rank, (agent_id, score) in enumerate(sorted(agent_scores.items(), key=lambda x: x[1]), 1):
            score_color = brier_color(score)
            bar = progress_bar(1.0 - min(score * 4, 1.0), width=18, filled_color=score_color)
            table.add_row(
                str(rank),
                agent_id,
                f"[{score_color}]{score:.4f}[/]",
                bar,
                quality_label(score),
            )
        console.print(table)
        console.print()
        console.print(f"  [{COLORS['dim']}]Brier: 0.00 = perfect, 0.25 = random chance[/]")

    footer()


# ═══════════════════════════════════════════
# Context
# ═══════════════════════════════════════════

@app.command()
def context(
    question: str = typer.Argument("", help="Optional question for question-specific market search"),
):
    """Show all live data sources the agents see (debug/exploration)."""
    from data.context import build_context

    _banner()
    section("Live Data Context")

    console.print(f"  [{COLORS['dim']}]Fetching from all sources...[/]")
    console.print()
    ctx = build_context(question)

    console.print(Panel(
        ctx,
        border_style=COLORS["accent"],
        padding=(1, 2),
        title=f"[bold {COLORS['accent']}]Agent Context[/]",
        title_align="left",
    ))

    footer()


# ═══════════════════════════════════════════
# History
# ═══════════════════════════════════════════

@app.command()
def history(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of recent forecasts to show"),
):
    """Show past forecast history."""
    from core.calibration import init_db, get_forecast_history
    init_db()

    _banner()
    section("Forecast History")

    forecasts = get_forecast_history(limit=limit)
    if not forecasts:
        console.print(f"  [{COLORS['dim']}]No forecasts yet. Run a forecast to get started.[/]")
        footer()
        return

    table = Table(
        box=box.SIMPLE_HEAVY,
        border_style=COLORS["brand"],
        show_header=True,
        header_style=f"bold {COLORS['brand']}",
        padding=(0, 1),
    )
    table.add_column("Question", style="bold", max_width=45)
    table.add_column("Prob", justify="right", width=6)
    table.add_column("", width=12)  # mini bar
    table.add_column("Mkt", justify="right", width=5)
    table.add_column("Edge", justify="right", width=7)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Brier", justify="right", width=7)
    table.add_column("Date", style=COLORS["dim"], width=10)

    for f in forecasts:
        prob = f["probability"]
        p_color = probability_color(prob)
        bar = progress_bar(prob, width=10, filled_color=p_color)

        market = f"{f['market_odds']:.0%}" if f["market_odds"] else " --"
        if f["market_odds"]:
            edge_val = prob - f["market_odds"]
            e_color = edge_color(edge_val)
            edge = f"[{e_color}]{edge_val:+.1%}[/]"
        else:
            edge = f"[{COLORS['dim']}] --[/]"

        if f["status"] == "resolved":
            outcome_val = f["outcome"]
            status = f"[{COLORS['positive']}]YES[/]" if outcome_val == 1.0 else f"[{COLORS['negative']}]NO[/]"
        else:
            status = f"[{COLORS['dim']}]pending[/]"

        brier = f"[{brier_color(f['brier_score'])}]{f['brier_score']:.4f}[/]" if f["brier_score"] is not None else f"[{COLORS['dim']}] --[/]"
        date = f["created_at"][:10] if f["created_at"] else " --"

        table.add_row(
            f["question"][:45],
            f"[{p_color}]{prob:.0%}[/]",
            bar,
            market,
            edge,
            status,
            brier,
            date,
        )

    console.print(table)
    console.print()
    console.print(f"  [{COLORS['dim']}]Showing {len(forecasts)} of {limit} requested  |  Use --limit N to adjust[/]")

    footer()


# ═══════════════════════════════════════════
# Sources
# ═══════════════════════════════════════════

@app.command()
def sources():
    """List all registered data sources and their status."""
    from data.context import list_sources

    _banner()
    section("Data Sources")

    source_list = list_sources()
    available_count = sum(1 for s in source_list if s["available"])
    needs_key_count = sum(1 for s in source_list if not s["has_key"] and s["requires_key"])

    # Summary stats
    stat_row([
        ("Total", str(len(source_list)), "accent"),
        ("Live", str(available_count), "positive"),
        ("Need Key", str(needs_key_count), "warning"),
    ])
    console.print()

    table = Table(
        box=box.SIMPLE_HEAVY,
        border_style=COLORS["brand"],
        show_header=True,
        header_style=f"bold {COLORS['brand']}",
        padding=(0, 1),
    )
    table.add_column("", width=6, justify="center")  # status dot
    table.add_column("Source", style="bold", min_width=18)
    table.add_column("Category", width=18)
    table.add_column("Priority", justify="center", width=8)
    table.add_column("API Key", width=22)
    table.add_column("Description", style=COLORS["dim"])

    for s in source_list:
        badge = status_badge(s["available"], s["has_key"], s["requires_key"])
        cat = category_style(s["category"])
        priority_bar = progress_bar(s["priority"] / 100, width=6, filled_color=COLORS["accent"])
        key_display = f"[{COLORS['dim']}]--[/]" if not s["requires_key"] else (
            f"[{COLORS['positive']}]{s['requires_key']}[/]" if s["has_key"]
            else f"[{COLORS['warning']}]{s['requires_key']}[/]"
        )

        table.add_row(
            badge,
            s["name"],
            cat,
            priority_bar,
            key_display,
            s["description"],
        )

    console.print(table)
    console.print()
    console.print(f"  [{COLORS['dim']}]Add sources: drop a file in data/sources/ with @register_source[/]")
    console.print(f"  [{COLORS['dim']}]Filter:      POLYSWARM_SOURCES=name1,name2 env var[/]")

    footer()


if __name__ == "__main__":
    app()
