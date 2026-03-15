"""
Social and sentiment data from free APIs.
"""

from __future__ import annotations
import httpx


def fetch_reddit_trending() -> str:
    """Fetch trending crypto posts from Reddit."""
    try:
        headers = {"User-Agent": "PolySwarm/0.2.0"}
        resp = httpx.get(
            "https://www.reddit.com/r/cryptocurrency/hot.json?limit=5",
            headers=headers,
            timeout=10,
        )
        data = resp.json()
        posts = data.get("data", {}).get("children", [])
        if not posts:
            return ""
        lines = []
        for p in posts[:5]:
            d = p.get("data", {})
            title = d.get("title", "")[:80]
            score = d.get("score", 0)
            comments = d.get("num_comments", 0)
            lines.append(f"  [{score:>5} pts, {comments:>3} comments] {title}")
        return "Reddit r/cryptocurrency (hot):\n" + "\n".join(lines)
    except Exception:
        return ""


def fetch_google_trends_proxy() -> str:
    """Simple proxy for crypto search interest via CoinGecko trending."""
    try:
        resp = httpx.get("https://api.coingecko.com/api/v3/search/trending", timeout=8)
        data = resp.json()
        coins = data.get("coins", [])[:5]
        lines = [f"  #{i+1} {c['item']['name']} ({c['item']['symbol']})" for i, c in enumerate(coins)]
        return "Trending on CoinGecko:\n" + "\n".join(lines)
    except Exception:
        return ""


def fetch_crypto_market_overview() -> str:
    """Market overview — dominance, volume, active coins."""
    try:
        resp = httpx.get("https://api.coingecko.com/api/v3/global", timeout=8)
        data = resp.json().get("data", {})
        active = data.get("active_cryptocurrencies", 0)
        markets = data.get("markets", 0)
        total_vol = data.get("total_volume", {}).get("usd", 0)
        btc_dom = data.get("market_cap_percentage", {}).get("btc", 0)
        eth_dom = data.get("market_cap_percentage", {}).get("eth", 0)
        mcap_change = data.get("market_cap_change_percentage_24h_usd", 0)
        return (
            f"Crypto Market Overview:\n"
            f"  Active coins: {active:,} | Exchanges: {markets}\n"
            f"  24h Volume: ${total_vol/1e9:.1f}B\n"
            f"  BTC dom: {btc_dom:.1f}% | ETH dom: {eth_dom:.1f}% | Market cap 24h: {mcap_change:+.1f}%"
        )
    except Exception:
        return ""
