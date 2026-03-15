"""Prediction market data sources — Polymarket, Manifold."""

from data.registry import DataSource, register_source
import json
import httpx


@register_source
class PolymarketTrending(DataSource):
    name = "polymarket"
    category = "prediction_markets"
    description = "Trending prediction markets on Polymarket"
    priority = 40

    def fetch(self) -> str:
        resp = self.http.get(
            "https://gamma-api.polymarket.com/markets",
            params={"limit": 5, "active": True, "order": "volume", "ascending": False},
        )
        markets = resp.json()
        if not markets:
            return ""
        lines = []
        for m in markets[:5]:
            question = m.get("question", "")[:70]
            outcome_prices = m.get("outcomePrices", "")
            try:
                prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
                yes_price = float(prices[0]) if prices else 0
                lines.append(f"  [{yes_price:.0%}] {question}")
            except Exception:
                lines.append(f"  {question}")
        return "Polymarket (trending):\n" + "\n".join(lines)

    def search(self, question: str) -> str:
        """Search for markets matching a question."""
        try:
            resp = self.http.get(
                "https://gamma-api.polymarket.com/markets",
                params={"limit": 3, "active": True, "order": "volume", "ascending": False,
                        "tag": question[:50]},
            )
            markets = resp.json()
            if not markets:
                return ""
            lines = []
            for m in markets[:3]:
                q = m.get("question", "")[:70]
                try:
                    prices = json.loads(m.get("outcomePrices", "[]"))
                    yes = float(prices[0]) if prices else 0
                    lines.append(f"  [{yes:.0%}] {q}")
                except Exception:
                    lines.append(f"  {q}")
            return "Related Polymarket markets:\n" + "\n".join(lines) if lines else ""
        except Exception:
            return ""


@register_source
class ManifoldTrending(DataSource):
    name = "manifold"
    category = "prediction_markets"
    description = "Trending markets on Manifold Markets"
    priority = 38

    def fetch(self) -> str:
        resp = self.http.get(
            "https://api.manifold.markets/v0/search-markets",
            params={"limit": 5, "sort": "liquidity", "filter": "open", "term": "crypto"},
        )
        markets = resp.json()
        if not markets:
            return ""
        lines = []
        for m in markets[:5]:
            q = m.get("question", "")[:70]
            prob = m.get("probability", 0)
            lines.append(f"  [{prob:.0%}] {q}")
        return "Manifold Markets (crypto):\n" + "\n".join(lines)

    def search(self, question: str) -> str:
        """Search for markets matching a question."""
        try:
            resp = self.http.get(
                "https://api.manifold.markets/v0/search-markets",
                params={"limit": 3, "sort": "liquidity", "filter": "open", "term": question[:80]},
            )
            markets = resp.json()
            if not markets:
                return ""
            lines = []
            for m in markets[:3]:
                q = m.get("question", "")[:70]
                prob = m.get("probability", 0)
                lines.append(f"  [{prob:.0%}] {q}")
            return "Related Manifold markets:\n" + "\n".join(lines) if lines else ""
        except Exception:
            return ""
