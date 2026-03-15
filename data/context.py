"""
Context injector — assembles comprehensive, real-time market data
from multiple sources for agent consumption.

Data sources:
  - Binance (spot + futures)
  - Deribit (options)
  - DeFi Llama (TVL, protocols)
  - CoinGecko (market overview, trending)
  - Fear & Greed Index
  - Reddit (r/cryptocurrency)
  - Polymarket / Manifold (prediction markets)
  - Mempool.space (BTC network)
  - Blockchain.info (BTC on-chain)
"""

from __future__ import annotations
import httpx
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from data.news import fetch_crypto_news, fetch_fear_greed_extended, fetch_btc_dominance, fetch_top_movers
from data.onchain import (
    fetch_btc_mempool, fetch_btc_fees, fetch_gas_prices,
    fetch_defi_tvl, fetch_top_protocols_tvl, fetch_stablecoin_supply,
)
from data.derivatives import (
    fetch_multi_funding_rates, fetch_open_interest,
    fetch_long_short_ratio, fetch_top_trader_positions,
    fetch_btc_options_oi, fetch_deribit_iv_index,
    fetch_liquidations_24h,
)
from data.social import fetch_reddit_trending, fetch_google_trends_proxy, fetch_crypto_market_overview
from data.prediction_markets import fetch_polymarket_trending, fetch_manifold_trending, fetch_market_for_question


def _fetch_core_prices() -> list[str]:
    """Fetch BTC and ETH spot prices from Binance."""
    sections = []
    try:
        resp = httpx.get("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT", timeout=5)
        d = resp.json()
        sections.append(
            f"BTC/USDT: ${float(d['lastPrice']):,.0f}  |  24h: {float(d['priceChangePercent']):+.2f}%  |  "
            f"Volume: ${float(d['quoteVolume'])/1e9:.2f}B  |  High: ${float(d['highPrice']):,.0f}  Low: ${float(d['lowPrice']):,.0f}"
        )
    except Exception:
        pass
    try:
        resp = httpx.get("https://api.binance.com/api/v3/ticker/24hr?symbol=ETHUSDT", timeout=5)
        d = resp.json()
        sections.append(
            f"ETH/USDT: ${float(d['lastPrice']):,.0f}  |  24h: {float(d['priceChangePercent']):+.2f}%  |  "
            f"Volume: ${float(d['quoteVolume'])/1e9:.2f}B"
        )
    except Exception:
        pass
    try:
        resp = httpx.get("https://api.binance.com/api/v3/ticker/24hr?symbol=SOLUSDT", timeout=5)
        d = resp.json()
        sections.append(f"SOL/USDT: ${float(d['lastPrice']):,.2f}  |  24h: {float(d['priceChangePercent']):+.2f}%")
    except Exception:
        pass
    return sections


def build_context(question: str = "") -> str:
    """
    Build comprehensive context from 15+ live data sources.
    Uses ThreadPoolExecutor for parallel fetching.
    """
    sections = [f"Current UTC time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"]

    # Core prices (synchronous, fast)
    sections.extend(_fetch_core_prices())

    # All other data sources — fetch in parallel
    fetchers = {
        "Fear & Greed": fetch_fear_greed_extended,
        "Market Overview": fetch_crypto_market_overview,
        "Top Movers": fetch_top_movers,
        "Funding Rates": fetch_multi_funding_rates,
        "Open Interest": fetch_open_interest,
        "Long/Short Ratio": fetch_long_short_ratio,
        "Top Traders": fetch_top_trader_positions,
        "Liquidations": fetch_liquidations_24h,
        "BTC Options": fetch_btc_options_oi,
        "BTC IV": fetch_deribit_iv_index,
        "DeFi TVL": fetch_defi_tvl,
        "Top Protocols": fetch_top_protocols_tvl,
        "Stablecoins": fetch_stablecoin_supply,
        "BTC Mempool": fetch_btc_mempool,
        "BTC Fees": fetch_btc_fees,
        "ETH Gas": fetch_gas_prices,
        "Reddit": fetch_reddit_trending,
        "Trending": fetch_google_trends_proxy,
        "News": fetch_crypto_news,
        "Polymarket": fetch_polymarket_trending,
        "Manifold": fetch_manifold_trending,
    }

    results = {}
    with ThreadPoolExecutor(max_workers=12) as executor:
        future_to_name = {executor.submit(fn): name for name, fn in fetchers.items()}
        for future in as_completed(future_to_name, timeout=15):
            name = future_to_name[future]
            try:
                result = future.result(timeout=10)
                if result:
                    results[name] = result
            except Exception:
                pass

    # Append in logical order
    order = [
        "Fear & Greed", "Market Overview", "Top Movers",
        "Funding Rates", "Open Interest", "Long/Short Ratio", "Top Traders", "Liquidations",
        "BTC Options", "BTC IV",
        "DeFi TVL", "Top Protocols", "Stablecoins",
        "BTC Mempool", "BTC Fees", "ETH Gas",
        "Reddit", "Trending", "News",
        "Polymarket", "Manifold",
    ]
    for key in order:
        if key in results:
            sections.append(results[key])

    # Question-specific market search
    if question:
        try:
            market_data = fetch_market_for_question(question)
            if market_data:
                sections.append(market_data)
        except Exception:
            pass

    return "\n\n".join(sections)
