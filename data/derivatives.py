"""
Derivatives market data — funding rates, open interest, liquidations, options.
Uses free APIs from Binance, CoinGlass proxies, and Deribit.
"""

from __future__ import annotations
import httpx


def fetch_multi_funding_rates() -> str:
    """Fetch funding rates for top assets from Binance Futures."""
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
    results = []
    try:
        for sym in symbols:
            resp = httpx.get(
                f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={sym}&limit=1",
                timeout=5,
            )
            data = resp.json()
            if data:
                rate = float(data[0]["fundingRate"]) * 100
                indicator = "+" if rate > 0.01 else "-" if rate < -0.01 else "~"
                results.append(f"  {indicator} {sym.replace('USDT',''):>5}: {rate:+.4f}%")
        if results:
            return "Funding Rates (8h):\n" + "\n".join(results)
    except Exception:
        pass
    return ""


def fetch_open_interest() -> str:
    """Fetch open interest for BTC and ETH from Binance Futures."""
    try:
        lines = []
        for sym in ["BTCUSDT", "ETHUSDT"]:
            resp = httpx.get(
                f"https://fapi.binance.com/fapi/v1/openInterest?symbol={sym}",
                timeout=5,
            )
            data = resp.json()
            oi = float(data.get("openInterest", 0))

            # get price for USD value
            resp2 = httpx.get(
                f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={sym}",
                timeout=5,
            )
            price = float(resp2.json().get("price", 0))
            oi_usd = oi * price
            lines.append(f"  {sym.replace('USDT','')}: {oi:,.0f} contracts (${oi_usd/1e9:.2f}B)")
        return "Open Interest:\n" + "\n".join(lines)
    except Exception:
        return ""


def fetch_long_short_ratio() -> str:
    """Fetch long/short ratio from Binance Futures."""
    try:
        lines = []
        for sym in ["BTCUSDT", "ETHUSDT"]:
            resp = httpx.get(
                f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={sym}&period=1h&limit=1",
                timeout=5,
            )
            data = resp.json()
            if data:
                ratio = float(data[0]["longShortRatio"])
                long_pct = float(data[0]["longAccount"]) * 100
                short_pct = float(data[0]["shortAccount"]) * 100
                bias = "LONG bias" if ratio > 1.2 else "SHORT bias" if ratio < 0.8 else "balanced"
                lines.append(f"  {sym.replace('USDT','')}: L={long_pct:.1f}% / S={short_pct:.1f}% ({bias})")
        return "Long/Short Ratios:\n" + "\n".join(lines)
    except Exception:
        return ""


def fetch_top_trader_positions() -> str:
    """Fetch top trader long/short ratio from Binance Futures."""
    try:
        lines = []
        for sym in ["BTCUSDT", "ETHUSDT"]:
            resp = httpx.get(
                f"https://fapi.binance.com/futures/data/topLongShortPositionRatio?symbol={sym}&period=1h&limit=1",
                timeout=5,
            )
            data = resp.json()
            if data:
                ratio = float(data[0]["longShortRatio"])
                long_pct = float(data[0]["longAccount"]) * 100
                short_pct = float(data[0]["shortAccount"]) * 100
                lines.append(f"  {sym.replace('USDT','')}: Top traders L={long_pct:.1f}% / S={short_pct:.1f}% (ratio={ratio:.2f})")
        return "Top Trader Positions:\n" + "\n".join(lines)
    except Exception:
        return ""


def fetch_liquidations_24h() -> str:
    """Fetch recent liquidations from Binance Futures."""
    try:
        resp = httpx.get(
            "https://fapi.binance.com/fapi/v1/allForceOrders?limit=50",
            timeout=8,
        )
        data = resp.json()
        if not data:
            return ""

        total_long_liq = 0
        total_short_liq = 0
        for order in data:
            qty = float(order.get("origQty", 0))
            price = float(order.get("price", 0))
            val = qty * price
            if order.get("side") == "SELL":  # long liquidated
                total_long_liq += val
            else:  # short liquidated
                total_short_liq += val

        return (
            f"Recent Liquidations (last 50 orders):\n"
            f"  Longs liquidated: ${total_long_liq:,.0f}\n"
            f"  Shorts liquidated: ${total_short_liq:,.0f}"
        )
    except Exception:
        return ""


def fetch_btc_options_oi() -> str:
    """Fetch BTC options data from Deribit (public, no auth needed)."""
    try:
        resp = httpx.get(
            "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option",
            timeout=10,
        )
        data = resp.json().get("result", [])
        if not data:
            return ""

        total_oi = sum(d.get("open_interest", 0) for d in data)
        total_volume = sum(d.get("volume", 0) for d in data)

        # count puts vs calls
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
    except Exception:
        return ""


def fetch_deribit_iv_index() -> str:
    """Fetch Deribit DVOL (implied volatility index)."""
    try:
        resp = httpx.get(
            "https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_usd",
            timeout=8,
        )
        price_data = resp.json().get("result", {})

        resp2 = httpx.get(
            "https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&resolution=3600&start_timestamp=0&end_timestamp=999999999999999",
            timeout=8,
        )
        # simpler approach: get historical vol
        resp3 = httpx.get(
            "https://www.deribit.com/api/v2/public/get_historical_volatility?currency=BTC",
            timeout=8,
        )
        vol_data = resp3.json().get("result", [])
        if vol_data:
            latest = vol_data[-1]
            if isinstance(latest, list) and len(latest) >= 2:
                return f"BTC Historical Volatility: {latest[1]:.1f}%"
        return ""
    except Exception:
        return ""
