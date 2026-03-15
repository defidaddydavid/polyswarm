"""Market data sources — spot prices, top movers, market overview."""

from data.registry import DataSource, register_source


@register_source
class BinanceSpotPrices(DataSource):
    name = "binance_spot"
    category = "market"
    description = "BTC, ETH, SOL spot prices from Binance"
    priority = 95

    def fetch(self) -> str:
        sections = []
        for sym, name in [("BTCUSDT", "BTC"), ("ETHUSDT", "ETH"), ("SOLUSDT", "SOL")]:
            try:
                resp = self.http.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={sym}")
                d = resp.json()
                price = float(d['lastPrice'])
                change = float(d['priceChangePercent'])
                vol = float(d['quoteVolume'])
                if name == "BTC":
                    sections.append(
                        f"{name}/USDT: ${price:,.0f}  |  24h: {change:+.2f}%  |  "
                        f"Volume: ${vol/1e9:.2f}B  |  High: ${float(d['highPrice']):,.0f}  Low: ${float(d['lowPrice']):,.0f}"
                    )
                else:
                    fmt = f"${price:,.0f}" if price > 100 else f"${price:,.2f}"
                    sections.append(f"{name}/USDT: {fmt}  |  24h: {change:+.2f}%  |  Volume: ${vol/1e9:.2f}B")
            except Exception:
                pass
        return "\n".join(sections)


@register_source
class BinanceTopMovers(DataSource):
    name = "top_movers"
    category = "market"
    description = "Top gainers and losers by 24h change"
    priority = 80

    def fetch(self) -> str:
        resp = self.http.get("https://api.binance.com/api/v3/ticker/24hr")
        data = resp.json()
        usdt_pairs = [d for d in data if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > 10_000_000]
        sorted_by_change = sorted(usdt_pairs, key=lambda x: float(x["priceChangePercent"]))
        top_losers = sorted_by_change[:3]
        top_gainers = sorted_by_change[-3:][::-1]
        lines = ["Top gainers (24h): " + " | ".join(f"{d['symbol']} {float(d['priceChangePercent']):+.1f}%" for d in top_gainers)]
        lines.append("Top losers  (24h): " + " | ".join(f"{d['symbol']} {float(d['priceChangePercent']):+.1f}%" for d in top_losers))
        return "\n".join(lines)


@register_source
class CoinGeckoMarketOverview(DataSource):
    name = "market_overview"
    category = "market"
    description = "Global crypto market cap, dominance, volume"
    priority = 88

    def fetch(self) -> str:
        resp = self.http.get("https://api.coingecko.com/api/v3/global")
        data = resp.json().get("data", {})
        active = data.get("active_cryptocurrencies", 0)
        total_vol = data.get("total_volume", {}).get("usd", 0)
        btc_dom = data.get("market_cap_percentage", {}).get("btc", 0)
        eth_dom = data.get("market_cap_percentage", {}).get("eth", 0)
        mcap_change = data.get("market_cap_change_percentage_24h_usd", 0)
        total_mcap = data.get("total_market_cap", {}).get("usd", 0)
        return (
            f"Crypto Market Overview:\n"
            f"  Total Market Cap: ${total_mcap/1e12:.2f}T ({mcap_change:+.1f}% 24h)\n"
            f"  24h Volume: ${total_vol/1e9:.1f}B | Active coins: {active:,}\n"
            f"  BTC dom: {btc_dom:.1f}% | ETH dom: {eth_dom:.1f}%"
        )
