"""
Unified prediction market data — Polymarket + Manifold.
Fetch live markets, odds, and volume for context injection.
"""

from __future__ import annotations
import httpx

GAMMA_API = "https://gamma-api.polymarket.com"
MANIFOLD_API = "https://api.manifold.markets/v0"


def fetch_polymarket_trending(limit: int = 5) -> str:
    """Fetch trending/active Polymarket markets."""
    try:
        resp = httpx.get(
            f"{GAMMA_API}/markets",
            params={"limit": limit, "active": True, "order": "volume", "ascending": False},
            timeout=10,
        )
        markets = resp.json() if isinstance(resp.json(), list) else []
        if not markets:
            return ""
        lines = []
        for m in markets[:limit]:
            q = m.get("question", "N/A")[:70]
            try:
                prices = m.get("outcomePrices", [])
                yes_price = float(prices[0]) * 100 if prices else 0
            except (IndexError, ValueError, TypeError):
                yes_price = 0
            vol = float(m.get("volume", 0))
            lines.append(f"  [{yes_price:5.1f}% YES | ${vol:>12,.0f} vol] {q}")
        return "Polymarket Trending:\n" + "\n".join(lines)
    except Exception:
        return ""


def fetch_manifold_trending(limit: int = 5) -> str:
    """Fetch trending Manifold Markets."""
    try:
        resp = httpx.get(
            f"{MANIFOLD_API}/search-markets",
            params={"term": "", "limit": limit, "sort": "liquidity"},
            timeout=10,
        )
        markets = resp.json() if isinstance(resp.json(), list) else []
        if not markets:
            return ""
        lines = []
        for m in markets[:limit]:
            q = m.get("question", "N/A")[:70]
            prob = m.get("probability", 0) * 100
            vol = m.get("volume", 0)
            lines.append(f"  [{prob:5.1f}% YES | ${vol:>10,.0f} vol] {q}")
        return "Manifold Trending:\n" + "\n".join(lines)
    except Exception:
        return ""


def fetch_market_for_question(question: str) -> str:
    """Search both Polymarket and Manifold for markets matching a question."""
    results = []

    # Polymarket
    try:
        resp = httpx.get(
            f"{GAMMA_API}/markets",
            params={"limit": 3, "active": True},
            timeout=10,
        )
        markets = resp.json() if isinstance(resp.json(), list) else []
        for m in markets:
            q = m.get("question", "").lower()
            if any(word in q for word in question.lower().split()[:3]):
                try:
                    prices = m.get("outcomePrices", [])
                    yes_price = float(prices[0]) * 100 if prices else 0
                except (IndexError, ValueError, TypeError):
                    yes_price = 0
                results.append(f"  [Polymarket] {m['question'][:70]}: {yes_price:.1f}% YES")
    except Exception:
        pass

    # Manifold
    try:
        resp = httpx.get(
            f"{MANIFOLD_API}/search-markets",
            params={"term": question[:50], "limit": 3},
            timeout=10,
        )
        markets = resp.json() if isinstance(resp.json(), list) else []
        for m in markets:
            prob = m.get("probability", 0) * 100
            results.append(f"  [Manifold] {m.get('question', 'N/A')[:70]}: {prob:.1f}% YES")
    except Exception:
        pass

    if results:
        return "Related prediction markets:\n" + "\n".join(results[:5])
    return ""
