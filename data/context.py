"""
Context injector — assembles comprehensive, real-time market data
from all registered data sources.

Uses the modular SourceRegistry to discover and fetch from all
available data pipelines in parallel.

To add a new data source:
  1. Create a class in data/sources/ that extends DataSource
  2. Decorate with @register_source
  3. It will be auto-discovered and included in context

To add a source that requires an API key:
  1. Set requires_key = "YOUR_ENV_VAR_NAME" on the class
  2. The source will only run when that env var is set

Filter sources via env var:
  POLYSWARM_SOURCES=binance_spot,funding_rates,fear_greed
"""

from __future__ import annotations

# Import sources to trigger registration
import data.sources  # noqa: F401

from data.registry import registry


def build_context(question: str = "") -> str:
    """
    Build comprehensive context from all registered data sources.
    Uses ThreadPoolExecutor for parallel fetching.
    """
    return registry.build_context(question)


def list_sources() -> list[dict]:
    """List all registered data sources and their status."""
    return registry.status()
