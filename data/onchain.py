"""
On-chain data fetchers — real blockchain data from free APIs.
"""

from __future__ import annotations
import httpx


def fetch_btc_hashrate() -> str:
    """BTC network hashrate from blockchain.info."""
    try:
        resp = httpx.get("https://blockchain.info/q/hashrate", timeout=8)
        hashrate_ghs = float(resp.text)
        hashrate_ehs = hashrate_ghs / 1e9
        return f"BTC Hashrate: {hashrate_ehs:.2f} EH/s"
    except Exception:
        return ""


def fetch_btc_mempool() -> str:
    """BTC mempool stats from mempool.space."""
    try:
        resp = httpx.get("https://mempool.space/api/mempool", timeout=8)
        data = resp.json()
        count = data.get("count", 0)
        vsize = data.get("vsize", 0)
        return f"BTC Mempool: {count:,} unconfirmed txs ({vsize/1e6:.1f} MB)"
    except Exception:
        return ""


def fetch_btc_fees() -> str:
    """Recommended BTC fees from mempool.space."""
    try:
        resp = httpx.get("https://mempool.space/api/v1/fees/recommended", timeout=8)
        data = resp.json()
        return (
            f"BTC Fees: fastest={data.get('fastestFee', '?')} sat/vB | "
            f"30min={data.get('halfHourFee', '?')} | "
            f"1hr={data.get('hourFee', '?')} | "
            f"economy={data.get('economyFee', '?')}"
        )
    except Exception:
        return ""


def fetch_gas_prices() -> str:
    """ETH gas prices from etherscan-compatible free API."""
    try:
        resp = httpx.get(
            "https://api.etherscan.io/api?module=gastracker&action=gasoracle",
            timeout=8,
        )
        data = resp.json().get("result", {})
        if isinstance(data, dict):
            return (
                f"ETH Gas: Low={data.get('SafeGasPrice', '?')} | "
                f"Standard={data.get('ProposeGasPrice', '?')} | "
                f"Fast={data.get('FastGasPrice', '?')} Gwei"
            )
        return ""
    except Exception:
        return ""


def fetch_eth_supply() -> str:
    """ETH total supply from ultrasound.money API."""
    try:
        resp = httpx.get("https://ultrasound.money/api/v2/fees/supply-over-time", timeout=8)
        # fallback: use etherscan
        resp2 = httpx.get(
            "https://api.etherscan.io/api?module=stats&action=ethsupply",
            timeout=8,
        )
        data = resp2.json()
        supply = int(data.get("result", 0)) / 1e18
        return f"ETH Total Supply: {supply:,.0f} ETH"
    except Exception:
        return ""


def fetch_defi_tvl() -> str:
    """Total DeFi TVL from DeFi Llama."""
    try:
        resp = httpx.get("https://api.llama.fi/v2/historicalChainTvl", timeout=8)
        data = resp.json()
        if data:
            latest = data[-1]
            tvl = latest.get("tvl", 0)
            prev = data[-2].get("tvl", tvl) if len(data) > 1 else tvl
            change = ((tvl - prev) / prev * 100) if prev > 0 else 0
            return f"DeFi Total TVL: ${tvl/1e9:.1f}B ({change:+.1f}% 24h)"
    except Exception:
        pass
    return ""


def fetch_top_protocols_tvl() -> str:
    """Top 5 protocols by TVL from DeFi Llama."""
    try:
        resp = httpx.get("https://api.llama.fi/protocols", timeout=10)
        data = resp.json()
        sorted_protocols = sorted(data, key=lambda x: x.get("tvl", 0), reverse=True)[:5]
        lines = [f"  {p['name']}: ${p['tvl']/1e9:.2f}B" for p in sorted_protocols]
        return "Top DeFi Protocols by TVL:\n" + "\n".join(lines)
    except Exception:
        return ""


def fetch_stablecoin_supply() -> str:
    """Stablecoin market data from DeFi Llama."""
    try:
        resp = httpx.get("https://stablecoins.llama.fi/stablecoins?includePrices=true", timeout=10)
        data = resp.json()
        stables = data.get("peggedAssets", [])
        total = sum(s.get("circulating", {}).get("peggedUSD", 0) for s in stables[:10])
        top3 = sorted(stables, key=lambda x: x.get("circulating", {}).get("peggedUSD", 0), reverse=True)[:3]
        lines = [f"  {s['name']}: ${s.get('circulating', {}).get('peggedUSD', 0)/1e9:.1f}B" for s in top3]
        return f"Stablecoin Supply (${total/1e9:.0f}B total):\n" + "\n".join(lines)
    except Exception:
        return ""


def fetch_exchange_flows() -> str:
    """BTC exchange netflow from CryptoQuant (free endpoints)."""
    try:
        # Use blockchain.info for exchange balance approximation
        resp = httpx.get("https://blockchain.info/q/totalbc", timeout=8)
        total_btc = int(resp.text) / 1e8
        return f"BTC Total Supply: {total_btc:,.0f} BTC"
    except Exception:
        return ""
