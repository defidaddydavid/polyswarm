"""
Polymarket API integration.
Fetch market data, current odds, and volume.
"""

from __future__ import annotations
import httpx

POLYMARKET_API = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"


def search_markets(query: str, limit: int = 5) -> list[dict]:
    """Search Polymarket for markets matching a query."""
    try:
        resp = httpx.get(
            f"{GAMMA_API}/markets",
            params={"q": query, "limit": limit, "active": True},
            timeout=10,
        )
        return resp.json() if isinstance(resp.json(), list) else resp.json().get("markets", [])
    except Exception as e:
        return []


def get_market(condition_id: str) -> dict | None:
    """Get a specific market by condition ID."""
    try:
        resp = httpx.get(f"{GAMMA_API}/markets/{condition_id}", timeout=10)
        return resp.json()
    except Exception:
        return None


def format_market_summary(market: dict) -> str:
    """Format a market into a readable string for agent context."""
    return (
        f"Market: {market.get('question', 'N/A')}\n"
        f"Current YES odds: {float(market.get('outcomePrices', [0])[0])*100:.1f}%\n"
        f"Volume: ${float(market.get('volume', 0)):,.0f}\n"
        f"End date: {market.get('endDate', 'N/A')}"
    )
