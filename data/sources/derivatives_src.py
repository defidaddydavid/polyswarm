"""Derivatives data sources — funding rates, OI, liquidations, options."""

from data.registry import DataSource, register_source


@register_source
class BinanceFundingRates(DataSource):
    name = "funding_rates"
    category = "derivatives"
    description = "Perpetual funding rates for top assets"
    priority = 85

    def fetch(self) -> str:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
        results = []
        for sym in symbols:
            try:
                resp = self.http.get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}&limit=1")
                data = resp.json()
                if data:
                    rate = float(data[0]["fundingRate"]) * 100
                    indicator = "+" if rate > 0.01 else "-" if rate < -0.01 else "~"
                    results.append(f"  {indicator} {sym.replace('USDT',''):>5}: {rate:+.4f}%")
            except Exception:
                pass
        return "Funding Rates (8h):\n" + "\n".join(results) if results else ""


@register_source
class BinanceOpenInterest(DataSource):
    name = "open_interest"
    category = "derivatives"
    description = "BTC/ETH open interest from Binance Futures"
    priority = 82

    def fetch(self) -> str:
        lines = []
        for sym in ["BTCUSDT", "ETHUSDT"]:
            resp = self.http.get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={sym}")
            oi = float(resp.json().get("openInterest", 0))
            resp2 = self.http.get(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={sym}")
            price = float(resp2.json().get("price", 0))
            lines.append(f"  {sym.replace('USDT','')}: {oi:,.0f} contracts (${oi * price/1e9:.2f}B)")
        return "Open Interest:\n" + "\n".join(lines)


@register_source
class BinanceLongShort(DataSource):
    name = "long_short_ratio"
    category = "derivatives"
    description = "Account long/short ratios"
    priority = 78

    def fetch(self) -> str:
        lines = []
        for sym in ["BTCUSDT", "ETHUSDT"]:
            resp = self.http.get(f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={sym}&period=1h&limit=1")
            data = resp.json()
            if data:
                ratio = float(data[0]["longShortRatio"])
                long_pct = float(data[0]["longAccount"]) * 100
                short_pct = float(data[0]["shortAccount"]) * 100
                bias = "LONG bias" if ratio > 1.2 else "SHORT bias" if ratio < 0.8 else "balanced"
                lines.append(f"  {sym.replace('USDT','')}: L={long_pct:.1f}% / S={short_pct:.1f}% ({bias})")
        return "Long/Short Ratios:\n" + "\n".join(lines)


@register_source
class BinanceTopTraders(DataSource):
    name = "top_traders"
    category = "derivatives"
    description = "Top trader position ratios"
    priority = 75

    def fetch(self) -> str:
        lines = []
        for sym in ["BTCUSDT", "ETHUSDT"]:
            resp = self.http.get(f"https://fapi.binance.com/futures/data/topLongShortPositionRatio?symbol={sym}&period=1h&limit=1")
            data = resp.json()
            if data:
                ratio = float(data[0]["longShortRatio"])
                long_pct = float(data[0]["longAccount"]) * 100
                short_pct = float(data[0]["shortAccount"]) * 100
                lines.append(f"  {sym.replace('USDT','')}: Top traders L={long_pct:.1f}% / S={short_pct:.1f}% (ratio={ratio:.2f})")
        return "Top Trader Positions:\n" + "\n".join(lines)


@register_source
class BinanceLiquidations(DataSource):
    name = "liquidations"
    category = "derivatives"
    description = "Recent forced liquidations"
    priority = 72

    def fetch(self) -> str:
        resp = self.http.get("https://fapi.binance.com/fapi/v1/allForceOrders?limit=50", timeout=8)
        data = resp.json()
        if not data:
            return ""
        total_long = sum(float(o.get("origQty", 0)) * float(o.get("price", 0)) for o in data if o.get("side") == "SELL")
        total_short = sum(float(o.get("origQty", 0)) * float(o.get("price", 0)) for o in data if o.get("side") == "BUY")
        return f"Recent Liquidations (last 50):\n  Longs liquidated: ${total_long:,.0f}\n  Shorts liquidated: ${total_short:,.0f}"


@register_source
class DeribitOptions(DataSource):
    name = "btc_options"
    category = "derivatives"
    description = "BTC options OI and put/call ratio from Deribit"
    priority = 83

    def fetch(self) -> str:
        resp = self.http.get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option")
        data = resp.json().get("result", [])
        if not data:
            return ""
        total_oi = sum(d.get("open_interest", 0) for d in data)
        total_volume = sum(d.get("volume", 0) for d in data)
        puts = [d for d in data if d.get("instrument_name", "").endswith("P")]
        calls = [d for d in data if d.get("instrument_name", "").endswith("C")]
        put_oi = sum(d.get("open_interest", 0) for d in puts)
        call_oi = sum(d.get("open_interest", 0) for d in calls)
        pcr = put_oi / call_oi if call_oi > 0 else 0
        return (
            f"BTC Options (Deribit):\n"
            f"  Total OI: {total_oi:,.0f} BTC | 24h Volume: {total_volume:,.0f} BTC\n"
            f"  Put/Call OI Ratio: {pcr:.2f} ({'bearish' if pcr > 0.7 else 'bullish' if pcr < 0.4 else 'neutral'})\n"
            f"  Calls OI: {call_oi:,.0f} | Puts OI: {put_oi:,.0f}"
        )


@register_source
class DeribitVolatility(DataSource):
    name = "btc_vol"
    category = "derivatives"
    description = "BTC historical volatility from Deribit"
    priority = 70

    def fetch(self) -> str:
        resp = self.http.get("https://www.deribit.com/api/v2/public/get_historical_volatility?currency=BTC")
        vol_data = resp.json().get("result", [])
        if vol_data:
            latest = vol_data[-1]
            if isinstance(latest, list) and len(latest) >= 2:
                return f"BTC Historical Volatility: {latest[1]:.1f}%"
        return ""
