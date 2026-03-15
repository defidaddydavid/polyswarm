"""
Context injector — fetches relevant data for a question
and formats it as a string for agent consumption.
"""

from __future__ import annotations
import httpx
from datetime import datetime
from data.news import fetch_crypto_news, fetch_fear_greed_extended, fetch_btc_dominance, fetch_top_movers


def build_context(question: str) -> str:
    """Build rich context string for a given question."""
    sections = []
    sections.append(f"Current UTC time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")

    # BTC price (Binance)
    try:
        resp = httpx.get("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT", timeout=5)
        d = resp.json()
        sections.append(
            f"BTC/USDT: ${float(d['lastPrice']):,.0f}  |  24h: {float(d['priceChangePercent']):+.2f}%  |  "
            f"Volume: ${float(d['quoteVolume'])/1e9:.2f}B  |  High: ${float(d['highPrice']):,.0f}  Low: ${float(d['lowPrice']):,.0f}"
        )
    except Exception:
        pass

    # ETH price
    try:
        resp = httpx.get("https://api.binance.com/api/v3/ticker/24hr?symbol=ETHUSDT", timeout=5)
        d = resp.json()
        sections.append(f"ETH/USDT: ${float(d['lastPrice']):,.0f}  |  24h: {float(d['priceChangePercent']):+.2f}%")
    except Exception:
        pass

    # BTC funding rate
    try:
        resp = httpx.get("https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=3", timeout=5)
        rates = resp.json()
        avg_rate = sum(float(r["fundingRate"]) for r in rates) / len(rates)
        sentiment = "bullish" if avg_rate > 0.0001 else "bearish" if avg_rate < -0.0001 else "neutral"
        sections.append(f"BTC Funding Rate (avg 3): {avg_rate*100:.4f}%/8h ({sentiment})")
    except Exception:
        pass

    # Extended data
    for fn in [fetch_fear_greed_extended, fetch_btc_dominance, fetch_top_movers, fetch_crypto_news]:
        try:
            result = fn()
            if result:
                sections.append(result)
        except Exception:
            pass

    return "\n\n".join(sections)
