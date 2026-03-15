"""
PolySwarm — Multi-agent AI forecasting engine for prediction markets.

Usage:
  python main.py forecast "Will BTC close above $100k on March 31 2026?"
  python main.py forecast "Will the Fed cut rates in June 2026?" --odds 0.35
  python main.py scenario "Elon Musk tweets that Tesla will accept Bitcoin again"
  python main.py scenario "SEC approves spot ETH ETF options" --context "ETH currently at $3,200"
  python main.py resolve "Will BTC close above $100k on March 31 2026?" --outcome 1.0
  python main.py serve
  python main.py calibration
"""

import typer
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console

app = typer.Typer(help="PolySwarm — Multi-agent prediction market forecasting & scenario simulation")
console = Console()


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

    from core.swarm import Swarm
    from agents.personas import build_swarm
    swarm = Swarm(agents=build_swarm(size) if size else None)
    result = swarm.forecast(question, market_odds=odds)

    if odds is not None:
        console.print(f"\n[bold]Edge vs market:[/bold] {result.get('edge_pct', 'N/A')}")


@app.command()
def scenario(
    description: str = typer.Argument(..., help="The scenario to simulate"),
    context: str = typer.Option("", "--context", help="Additional context for the simulation"),
):
    """Simulate crowd reactions to a scenario (MiroFish-style)."""
    from core.scenario import ScenarioEngine
    engine = ScenarioEngine()
    engine.simulate(description, context)


@app.command()
def resolve(
    question: str = typer.Argument(..., help="The question to resolve"),
    outcome: float = typer.Option(..., "--outcome", help="1.0 = YES resolved, 0.0 = NO resolved"),
):
    """Resolve a forecast and update calibration scores."""
    from core.calibration import resolve_forecast
    resolve_forecast(question, outcome)
    console.print(f"[green]✓ Resolved:[/green] {question} → {'YES' if outcome == 1.0 else 'NO'}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
):
    """Start the FastAPI server."""
    import uvicorn
    uvicorn.run("api.routes:app", host=host, port=port, reload=True)


@app.command()
def calibration():
    """Show current calibration scores across all agents."""
    from core.calibration import get_swarm_brier_score, get_agent_brier_scores
    from rich.table import Table
    from rich import box

    swarm_score = get_swarm_brier_score()
    agent_scores = get_agent_brier_scores()

    console.print(f"\n[bold]Swarm Brier Score:[/bold] {swarm_score or 'No resolved forecasts yet'}")

    if agent_scores:
        table = Table(title="Agent Calibration (Brier Scores)", box=box.ROUNDED)
        table.add_column("Agent", style="cyan")
        table.add_column("Brier Score", justify="right")
        table.add_column("Quality")
        for agent_id, score in sorted(agent_scores.items(), key=lambda x: x[1]):
            quality = "🟢 Excellent" if score < 0.1 else "🟡 Good" if score < 0.2 else "🔴 Poor"
            table.add_row(agent_id, f"{score:.4f}", quality)
        console.print(table)


@app.command()
def context(
    question: str = typer.Argument("", help="Optional question for question-specific market search"),
):
    """Show all live data sources the agents see (debug/exploration)."""
    from data.context import build_context
    from rich.panel import Panel

    console.print("[bold]Fetching all data sources...[/bold]\n")
    ctx = build_context(question)
    console.print(Panel(ctx, title="Agent Context (Live Data)", border_style="cyan"))


if __name__ == "__main__":
    app()
