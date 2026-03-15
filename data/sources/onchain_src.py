"""On-chain data sources — BTC mempool, ETH gas, DeFi TVL, stablecoins."""

from data.registry import DataSource, register_source


@register_source
class BTCMempool(DataSource):
    name = "btc_mempool"
    category = "onchain"
    description = "BTC mempool unconfirmed transactions"
    priority = 55

    def fetch(self) -> str:
        resp = self.http.get("https://mempool.space/api/mempool")
        data = resp.json()
        count = data.get("count", 0)
        vsize = data.get("vsize", 0)
        return f"BTC Mempool: {count:,} unconfirmed txs ({vsize/1e6:.1f} MB)"


@register_source
class BTCFees(DataSource):
    name = "btc_fees"
    category = "onchain"
    description = "BTC recommended transaction fees"
    priority = 53

    def fetch(self) -> str:
        resp = self.http.get("https://mempool.space/api/v1/fees/recommended")
        d = resp.json()
        return (
            f"BTC Fees: fastest={d.get('fastestFee', '?')} sat/vB | "
            f"30min={d.get('halfHourFee', '?')} | 1hr={d.get('hourFee', '?')} | "
            f"economy={d.get('economyFee', '?')}"
        )


@register_source
class BTCHashrate(DataSource):
    name = "btc_hashrate"
    category = "onchain"
    description = "BTC network hashrate"
    priority = 50

    def fetch(self) -> str:
        resp = self.http.get("https://blockchain.info/q/hashrate")
        hashrate_ghs = float(resp.text)
        return f"BTC Hashrate: {hashrate_ghs / 1e9:.2f} EH/s"


@register_source
class ETHGas(DataSource):
    name = "eth_gas"
    category = "onchain"
    description = "ETH gas prices"
    requires_key = "ETHERSCAN_API_KEY"  # works without key but rate-limited
    priority = 52

    @property
    def has_key(self) -> bool:
        return True  # works without key, just slower

    def fetch(self) -> str:
        import os
        params = {"module": "gastracker", "action": "gasoracle"}
        key = os.getenv("ETHERSCAN_API_KEY")
        if key:
            params["apikey"] = key
        resp = self.http.get("https://api.etherscan.io/api", params=params)
        data = resp.json().get("result", {})
        if isinstance(data, dict):
            return (
                f"ETH Gas: Low={data.get('SafeGasPrice', '?')} | "
                f"Standard={data.get('ProposeGasPrice', '?')} | "
                f"Fast={data.get('FastGasPrice', '?')} Gwei"
            )
        return ""


@register_source
class DefiTVL(DataSource):
    name = "defi_tvl"
    category = "defi"
    description = "Total DeFi TVL from DeFi Llama"
    priority = 65

    def fetch(self) -> str:
        resp = self.http.get("https://api.llama.fi/v2/historicalChainTvl")
        data = resp.json()
        if data:
            latest = data[-1]
            tvl = latest.get("tvl", 0)
            prev = data[-2].get("tvl", tvl) if len(data) > 1 else tvl
            change = ((tvl - prev) / prev * 100) if prev > 0 else 0
            return f"DeFi Total TVL: ${tvl/1e9:.1f}B ({change:+.1f}% 24h)"
        return ""


@register_source
class TopProtocols(DataSource):
    name = "top_protocols"
    category = "defi"
    description = "Top DeFi protocols by TVL"
    priority = 62
    timeout = 12

    def fetch(self) -> str:
        resp = self.http.get("https://api.llama.fi/protocols")
        data = resp.json()
        top5 = sorted(data, key=lambda x: x.get("tvl", 0), reverse=True)[:5]
        lines = [f"  {p['name']}: ${p['tvl']/1e9:.2f}B" for p in top5]
        return "Top DeFi Protocols by TVL:\n" + "\n".join(lines)


@register_source
class StablecoinSupply(DataSource):
    name = "stablecoins"
    category = "defi"
    description = "Stablecoin supply breakdown"
    priority = 60
    timeout = 12

    def fetch(self) -> str:
        resp = self.http.get("https://stablecoins.llama.fi/stablecoins?includePrices=true")
        data = resp.json()
        stables = data.get("peggedAssets", [])
        total = sum(s.get("circulating", {}).get("peggedUSD", 0) for s in stables[:10])
        top3 = sorted(stables, key=lambda x: x.get("circulating", {}).get("peggedUSD", 0), reverse=True)[:3]
        lines = [f"  {s['name']}: ${s.get('circulating', {}).get('peggedUSD', 0)/1e9:.1f}B" for s in top3]
        return f"Stablecoin Supply (${total/1e9:.0f}B total):\n" + "\n".join(lines)
