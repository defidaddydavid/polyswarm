"""Sentiment & social data sources — Fear/Greed, Reddit, news, trending."""

from data.registry import DataSource, register_source


@register_source
class FearAndGreed(DataSource):
    name = "fear_greed"
    category = "sentiment"
    description = "Crypto Fear & Greed Index (7-day)"
    priority = 90

    def fetch(self) -> str:
        resp = self.http.get("https://api.alternative.me/fng/?limit=7")
        data = resp.json()["data"]
        lines = [f"  {d['value_classification']:20s} ({d['value']:>3s})" for d in data]
        return "Fear & Greed (7 days, newest first):\n" + "\n".join(lines)


@register_source
class RedditCrypto(DataSource):
    name = "reddit"
    category = "social"
    description = "r/cryptocurrency hot posts"
    priority = 45

    def fetch(self) -> str:
        headers = {"User-Agent": "PolySwarm/0.7.0"}
        resp = self.http.get("https://www.reddit.com/r/cryptocurrency/hot.json?limit=5", headers=headers)
        posts = resp.json().get("data", {}).get("children", [])
        if not posts:
            return ""
        lines = []
        for p in posts[:5]:
            d = p.get("data", {})
            lines.append(f"  [{d.get('score', 0):>5} pts, {d.get('num_comments', 0):>3} comments] {d.get('title', '')[:80]}")
        return "Reddit r/cryptocurrency (hot):\n" + "\n".join(lines)


@register_source
class CoinGeckoTrending(DataSource):
    name = "trending"
    category = "social"
    description = "Trending coins on CoinGecko"
    priority = 48

    def fetch(self) -> str:
        resp = self.http.get("https://api.coingecko.com/api/v3/search/trending")
        coins = resp.json().get("coins", [])[:5]
        lines = [f"  #{i+1} {c['item']['name']} ({c['item']['symbol']})" for i, c in enumerate(coins)]
        return "Trending on CoinGecko:\n" + "\n".join(lines)


@register_source
class CryptoPanicNews(DataSource):
    name = "news"
    category = "sentiment"
    description = "Latest crypto news headlines"
    requires_key = "CRYPTOPANIC_API_KEY"  # works with "free" token too
    priority = 60

    @property
    def has_key(self) -> bool:
        return True  # works with free public token

    def fetch(self) -> str:
        import os
        token = os.getenv("CRYPTOPANIC_API_KEY", "free")
        resp = self.http.get(
            "https://cryptopanic.com/api/v1/posts/",
            params={"auth_token": token, "public": "true", "kind": "news", "filter": "hot"},
        )
        results = resp.json().get("results", [])[:5]
        if not results:
            return ""
        headlines = "\n".join([f"  • {r['title']} ({r.get('source', {}).get('title', '')})" for r in results])
        return f"Recent crypto headlines:\n{headlines}"
