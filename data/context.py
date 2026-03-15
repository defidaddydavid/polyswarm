"""
Context injector — fetches relevant data for a question
and formats it as a string for agent consumption.
"""

from __future__ import annotations
import httpx
from datetime import datetime


def build_context(question: str) -> str:
    """Build context string for a given question."""
    sections = []

    sections.append(f"Current UTC time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")

    # Fear & Greed Index
    try:
        resp = httpx.get("https://api.alternative.me/fng/?limit=3", timeout=5)
        data = resp.json()["data"]
        fng_lines = [f"  {d['value_classification']} ({d['value']}) — {d['timestamp']}" for d in data]
        sections.append("Crypto Fear & Greed Index (last 3 days):\n" + "\n".join(fng_lines))
    except Exception:
        sections.append("Fear & Greed Index: unavailable")

    # BTC price (Binance)
    try:
        resp = httpx.get("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT", timeout=5)
        d = resp.json()
        sections.append(
            f"BTC/USDT 24h: Price=${float(d['lastPrice']):,.0f} | "
            f"Change={float(d['priceChangePercent']):+.2f}% | "
            f"Volume=${float(d['quoteVolume'])/1e9:.2f}B"
        )
    except Exception:
        sections.append("BTC price: unavailable")

    # ETH price
    try:
        resp = httpx.get("https://api.binance.com/api/v3/ticker/24hr?symbol=ETHUSDT", timeout=5)
        d = resp.json()
        sections.append(
            f"ETH/USDT 24h: Price=${float(d['lastPrice']):,.0f} | "
            f"Change={float(d['priceChangePercent']):+.2f}%"
        )
    except Exception:
        pass

    # BTC funding rate (Binance perpetual)
    try:
        resp = httpx.get(
            "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=3", timeout=5
        )
        rates = resp.json()
        avg_rate = sum(float(r["fundingRate"]) for r in rates) / len(rates)
        sections.append(f"BTC Funding Rate (avg last 3): {avg_rate*100:.4f}% per 8h")
    except Exception:
        pass

    return "\n\n".join(sections)
