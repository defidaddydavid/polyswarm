"""
News context fetcher — pulls recent headlines relevant to a question.
Uses free APIs only (no API key required for basic usage).
"""

from __future__ import annotations
import httpx
from datetime import datetime, timedelta


def fetch_crypto_news(limit: int = 5) -> str:
    """Fetch recent crypto news from CryptoPanic (free tier)."""
    try:
        resp = httpx.get(
            "https://cryptopanic.com/api/v1/posts/",
            params={"auth_token": "free", "public": "true", "kind": "news", "filter": "hot"},
            timeout=8,
        )
        data = resp.json()
        results = data.get("results", [])[:limit]
        if not results:
            return ""
        headlines = "\n".join([f"  • {r['title']} ({r.get('source', {}).get('title', 'unknown')})" for r in results])
        return f"Recent crypto headlines:\n{headlines}"
    except Exception:
        return ""


def fetch_fear_greed_extended() -> str:
    """Fetch 7-day Fear & Greed history."""
    try:
        resp = httpx.get("https://api.alternative.me/fng/?limit=7", timeout=5)
        data = resp.json()["data"]
        lines = [f"  {d['value_classification']:20s} ({d['value']:>3s})" for d in data]
        return "Fear & Greed (7 days, newest first):\n" + "\n".join(lines)
    except Exception:
        return ""


def fetch_btc_dominance() -> str:
    """Fetch BTC dominance from CoinGecko (free)."""
    try:
        resp = httpx.get(
            "https://api.coingecko.com/api/v3/global",
            timeout=8,
        )
        data = resp.json().get("data", {})
        btc_dom = data.get("market_cap_percentage", {}).get("btc", 0)
        total_mcap = data.get("total_market_cap", {}).get("usd", 0)
        change_24h = data.get("market_cap_change_percentage_24h_usd", 0)
        return (
            f"BTC Dominance: {btc_dom:.1f}%\n"
            f"Total Crypto Market Cap: ${total_mcap/1e12:.2f}T ({change_24h:+.1f}% 24h)"
        )
    except Exception:
        return ""


def fetch_top_movers() -> str:
    """Fetch top gainers/losers from Binance."""
    try:
        resp = httpx.get("https://api.binance.com/api/v3/ticker/24hr", timeout=8)
        data = resp.json()
        usdt_pairs = [d for d in data if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > 10_000_000]
        sorted_by_change = sorted(usdt_pairs, key=lambda x: float(x["priceChangePercent"]))
        top_losers = sorted_by_change[:3]
        top_gainers = sorted_by_change[-3:][::-1]

        lines = ["Top gainers (24h): " + " | ".join(f"{d['symbol']} {float(d['priceChangePercent']):+.1f}%" for d in top_gainers)]
        lines.append("Top losers  (24h): " + " | ".join(f"{d['symbol']} {float(d['priceChangePercent']):+.1f}%" for d in top_losers))
        return "\n".join(lines)
    except Exception:
        return ""
