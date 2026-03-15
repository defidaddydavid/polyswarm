"""
Manifold Markets API integration.
"""

from __future__ import annotations
import httpx

MANIFOLD_API = "https://api.manifold.markets/v0"


def search_markets(query: str, limit: int = 5) -> list[dict]:
    try:
        resp = httpx.get(f"{MANIFOLD_API}/search-markets", params={"term": query, "limit": limit}, timeout=10)
        return resp.json() if isinstance(resp.json(), list) else []
    except Exception:
        return []


def get_market(market_id: str) -> dict | None:
    try:
        resp = httpx.get(f"{MANIFOLD_API}/market/{market_id}", timeout=10)
        return resp.json()
    except Exception:
        return None


def format_market(market: dict) -> str:
    prob = market.get("probability", 0)
    return (
        f"[Manifold] {market.get('question', 'N/A')}\n"
        f"  Probability: {prob*100:.1f}%  |  Volume: ${market.get('volume', 0):,.0f}  |  "
        f"Closes: {market.get('closeTime', 'N/A')}"
    )
