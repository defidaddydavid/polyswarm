"""
PolySwarm — Multi-agent AI forecasting engine for prediction markets.

Usage:
  python main.py forecast "Will BTC close above $100k on March 31 2026?"
  python main.py forecast "Will the Fed cut rates in June 2026?" --odds 0.35
  python main.py resolve "Will BTC close above $100k on March 31 2026?" --outcome 1.0
  python main.py serve
  python main.py calibration
"""

import typer
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich import print as rprint

app = typer.Typer(help="PolySwarm — Multi-agent prediction market forecasting")
console = Console()


@app.command()
def forecast(
    question: str = typer.Argument(..., help="The question to forecast"),
    odds: float = typer.Option(None, "--odds", help="Current market odds (0.0-1.0) for edge calculation"),
    rounds: int = typer.Option(None, "--rounds", help="Number of debate rounds"),
):
    """Run a swarm forecast on a question."""
    import os
    if rounds:
        os.environ["DEBATE_ROUNDS"] = str(rounds)

    from core.swarm import Swarm
    swarm = Swarm()
    result = swarm.forecast(question, market_odds=odds)

    if odds is not None:
        console.print(f"\n[bold]Edge vs market:[/bold] {result.get('edge_pct', 'N/A')}")


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
    """Show current calibration scores."""
    from core.calibration import get_swarm_brier_score, get_agent_brier_scores
    swarm_score = get_swarm_brier_score()
    agent_scores = get_agent_brier_scores()

    console.print("\n[bold]Swarm Brier Score:[/bold]", swarm_score or "No resolved forecasts yet")
    if agent_scores:
        console.print("\n[bold]Agent Brier Scores:[/bold] (lower = better)")
        for agent_id, score in sorted(agent_scores.items(), key=lambda x: x[1]):
            console.print(f"  {agent_id}: {score:.4f}")


if __name__ == "__main__":
    app()
