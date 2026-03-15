<div align="center">

# 🐝 PolySwarm

### Multi-agent AI forecasting engine for prediction markets

*Spawn a swarm of 12 AI agents with distinct personas, let them debate, and get calibrated probability estimates — with edge calculation vs live market odds.*

[![Python](https://img.shields.io/badge/python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Anthropic](https://img.shields.io/badge/Claude-powered-D4691C?style=for-the-badge&logo=anthropic&logoColor=white)](https://anthropic.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

</div>

---

## What is PolySwarm?

PolySwarm runs two modes:

**🗳️ Forecast Mode** — Feed it any resolvable question. A swarm of 12 AI agents — each with different expertise, memory, and known biases — independently form probability estimates, debate each other across multiple rounds, then converge on a calibrated consensus probability. Compare against live Polymarket / Manifold odds to find your edge.

**🎭 Scenario Mode** — Feed it a scenario ("Elon tweets Bitcoin is dead", "Fed announces emergency rate cut"). The swarm simulates how each market participant archetype immediately reacts, what they do, and what the crowd narrative becomes. Second-order effects included.

---

## Demo

```
$ python main.py forecast "Will BTC close above $100k before June 2026?" --odds 0.42

  ╔══════════════════════════════════════════════════════════════════╗
  ║  Question: Will BTC close above $100k before June 2026?         ║
  ╚══════════════════════════════════════════════════════════════════╝

  Fetching context... BTC $87,240 (+2.1%) | F&G: Greed (71) | Funding: +0.012%/8h

  ── Round 1 ──
    Macro Analyst        →  34%  (confidence: 72%)
    Crypto Native        →  61%  (confidence: 75%)
    Quant Trader         →  48%  (confidence: 80%)
    Retail Participant   →  63%  (confidence: 55%)
    Contrarian Skeptic   →  27%  (confidence: 65%)
    On-Chain Analyst     →  55%  (confidence: 70%)
    Institutional Desk   →  39%  (confidence: 78%)
    Event Specialist     →  52%  (confidence: 73%)
    DeFi Specialist      →  49%  (confidence: 68%)
    Options Trader       →  44%  (confidence: 76%)
    Geopolitical Analyst →  38%  (confidence: 69%)
    Social Sentiment     →  58%  (confidence: 62%)

  ── Round 2 ── (agents update after seeing each other's reasoning)
    ...

  ╔══════════════════════════════════════════════════════════════════╗
  ║  Swarm Probability:  48.7%                                      ║
  ║  Consensus: 58%  |  Agents: 12  |  Std Dev: 0.112              ║
  ║  Market odds: 42%  →  Edge: +6.7%                              ║
  ╚══════════════════════════════════════════════════════════════════╝
```

```
$ python main.py scenario "Elon Musk tweets Tesla will accept Bitcoin again"

  Macro Analyst        🟡 sentiment=+0.18  impact=+3.2%  · monitors positioning · waits for confirmation
  Crypto Native        🟢 sentiment=+0.74  impact=+8.5%  · buys spot BTC · posts bullish CT thread
  Quant Trader         🟡 sentiment=+0.22  impact=+2.1%  · adjusts momentum signals · checks options flow
  Retail Participant   🟢 sentiment=+0.81  impact=+12.0% · market buys BTC · tweets "LFG"
  Contrarian Skeptic   🔴 sentiment=-0.15  impact=-1.0%  · checks if rumour verified · prepares short
  Options Trader       🟢 sentiment=+0.45  impact=+5.5%  · buys near-term calls · sells puts
  ...

  ╔══════════════════════════════════════════════════════════════════╗
  ║  BULLISH  |  Sentiment: +0.41  |  Price Impact: +6.2%          ║
  ║  "Retail FOMO dominates initial reaction. Institutions wait     ║
  ║   for confirmation before deploying capital."                   ║
  ║  → Short squeeze on BTC perpetuals within 15 minutes           ║
  ║  → TSLA stock gaps up at open                                  ║
  ║  → Altcoins follow with 30-min lag                             ║
  ╚══════════════════════════════════════════════════════════════════╝
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/defidaddydavid/polyswarm.git
cd polyswarm
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY
```

Get a free key at [console.anthropic.com](https://console.anthropic.com)

### 3. Run

```bash
# Forecast mode — binary probability
python main.py forecast "Will BTC close above $100k before June 2026?" --odds 0.42

# Scenario mode — crowd simulation
python main.py scenario "SEC announces emergency crypto trading ban"

# Scenario with context
python main.py scenario "Binance is insolvent" --context "BTC currently at $87k, market is euphoric"

# More agents, more rounds
python main.py forecast "Will ETH flip BTC in 2026?" --rounds 3

# Start REST API
python main.py serve
```

---

## Docker

```bash
# One command deploy
ANTHROPIC_API_KEY=your_key docker compose up
```

API available at `http://localhost:8000` — full docs at `/docs`

---

## REST API

```bash
# Forecast
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"question": "Will BTC hit $150k in 2026?", "market_odds": 0.25}'

# Scenario simulation
curl -X POST http://localhost:8000/scenario \
  -d '{"scenario": "Elon Musk tweets Dogecoin will replace USD", "context": "DOGE currently $0.18"}'

# Resolve for calibration
curl -X POST http://localhost:8000/resolve \
  -d '{"question": "Will BTC hit $150k in 2026?", "outcome": 1.0}'

# Calibration scores
curl http://localhost:8000/calibration

# List all agents
curl http://localhost:8000/agents
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         PolySwarm                               │
├───────────────────────┬─────────────────────────────────────────┤
│   FORECAST MODE       │   SCENARIO MODE                         │
│                       │                                         │
│  Question input       │  Scenario input                         │
│       ↓               │       ↓                                 │
│  Context injection    │  Context injection                      │
│  (live market data)   │  (live market data + news)              │
│       ↓               │       ↓                                 │
│  Round 1: All 12      │  All 12 agents simulate                 │
│  agents form          │  immediate reactions,                   │
│  independent          │  sentiment shifts, and                  │
│  probability          │  specific actions                       │
│  estimates            │       ↓                                 │
│       ↓               │  Aggregate sentiment vector             │
│  Round 2: Agents      │       ↓                                 │
│  see each other,      │  Narrative generation                   │
│  update beliefs       │  + second-order effects                 │
│       ↓               │                                         │
│  Weighted             │                                         │
│  aggregation          │                                         │
│  (calibration-adj)    │                                         │
│       ↓               │                                         │
│  Final probability    │                                         │
│  + edge vs market     │                                         │
├───────────────────────┴─────────────────────────────────────────┤
│                    CALIBRATION LAYER                            │
│  Brier score tracking per agent → weights update on resolution  │
│  Better-calibrated agents gain influence over time              │
├─────────────────────────────────────────────────────────────────┤
│                    DATA LAYER                                   │
│  Binance · Polymarket · Manifold · Fear&Greed · CryptoNews     │
│  CoinGecko · Funding Rates · On-chain (via context)            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Live Data Sources

PolySwarm injects **real-time market data** from 15+ free APIs into every agent's context. No API keys required for most sources.

| Category | Source | Data |
|----------|--------|------|
| **Price** | Binance | BTC, ETH, SOL spot + 24h stats |
| **Derivatives** | Binance Futures | Funding rates (6 assets), OI, long/short ratios, top trader positions, liquidations |
| **Options** | Deribit | BTC options OI, put/call ratio, historical vol |
| **DeFi** | DeFi Llama | Total TVL, top protocols, stablecoin supply |
| **On-Chain** | Mempool.space | BTC mempool, fee estimates |
| **On-Chain** | Blockchain.info | BTC hashrate, supply |
| **Gas** | Etherscan | ETH gas prices |
| **Sentiment** | Alternative.me | Fear & Greed (7-day history) |
| **Social** | Reddit | r/cryptocurrency hot posts |
| **Social** | CoinGecko | Trending coins, market overview, BTC dominance |
| **News** | CryptoPanic | Latest crypto headlines |
| **Markets** | Polymarket | Trending prediction markets + odds |
| **Markets** | Manifold | Trending markets + probabilities |

Run `python main.py context` to see all live data:

```
$ python main.py context

  BTC/USDT: $87,240  |  24h: +2.12%  |  Volume: $28.5B
  ETH/USDT: $3,180   |  24h: +1.85%

  Fear & Greed (7 days): Greed (71) → Greed (68) → Neutral (55) → ...

  Funding Rates:
    + BTC: +0.0105%
    + ETH: +0.0082%
    - SOL: -0.0034%

  Open Interest:
    BTC: 125,420 contracts ($10.94B)
    ETH: 2,340,120 contracts ($7.43B)

  BTC Options (Deribit):
    Total OI: 42,500 BTC | Put/Call: 0.58 (bullish)

  DeFi Total TVL: $142.8B (+0.3% 24h)
  Stablecoin Supply ($185B total):
    USDT: $142.1B | USDC: $32.4B | DAI: $5.2B

  Reddit r/cryptocurrency (hot):
    [1,240 pts] "BTC just broke through $87k resistance..."

  Polymarket Trending:
    [72.5% YES | $12,450,000 vol] Will Trump win 2028?
```

---

## The 12 Agents

| # | Agent | Focus | Known Bias |
|---|-------|-------|------------|
| 1 | 📊 Macro Analyst | Fed policy, rates, global liquidity | Too conservative on crypto upside |
| 2 | ₿ Crypto Native | On-chain, CT narrative, funding rates | Structurally bullish |
| 3 | 📉 Quant Trader | Statistical base rates, vol surface | Dismissive of narrative |
| 4 | 📱 Retail Participant | Social sentiment, recent price action | FOMO/panic prone |
| 5 | 🐻 Contrarian Skeptic | Tail risks, overcrowded trades | Structurally bearish |
| 6 | 🔍 On-Chain Analyst | Whale flows, exchange reserves | Can lag price action |
| 7 | 🏦 Institutional Desk | ETF flows, regulatory, risk metrics | Conservative, slow to move |
| 8 | 📅 Event Specialist | Catalysts, FOMC, halvings, upgrades | Overweights known events |
| 9 | 🌐 DeFi Specialist | TVL, yields, protocol governance | Overweights DeFi signals |
| 10 | ⚡ Options Trader | IV rank, skew, gamma, term structure | Overweights derivatives signals |
| 11 | 🌍 Geopolitical Analyst | Regulation, sanctions, nation-state | Behind curve on market implications |
| 12 | 📡 Social Sentiment | Reddit, Twitter, Trends, influencers | Reactive not predictive |

---

## Calibration

PolySwarm tracks prediction accuracy using **Brier scores** (lower = better).

```bash
# After a market resolves:
python main.py resolve "Will BTC close above $100k before June 2026?" --outcome 1.0

# Check who's most accurate:
python main.py calibration
```

Better-calibrated agents automatically receive higher weight in future aggregations. The swarm gets smarter over time.

---

## Use Cases

| Use Case | Mode | Example |
|----------|------|---------|
| Prediction market edge | Forecast | Find +EV bets vs Polymarket odds |
| Event trading | Forecast | FOMC, ETF decisions, halvings |
| Black swan simulation | Scenario | "Exchange X is insolvent" |
| Options positioning | Scenario | Validate directional bias before buying calls/puts |
| Market research | Scenario | "New stablecoin regulation passed" |
| Risk management | Scenario | Simulate tail events before they happen |

---

## Roadmap

- [ ] Live Polymarket sync + auto-compare (leaderboard)
- [ ] Web UI with real-time debate viewer
- [ ] Agent memory persistence across sessions (Redis)
- [ ] Streaming API (watch agents think in real-time)
- [ ] OpenAI / local LLM support (Ollama)
- [ ] Telegram & Discord bot
- [ ] More data sources (Glassnode, Santiment, Nansen)
- [ ] Custom persona builder

---

## License

MIT — use it, fork it, build on it.

---

<div align="center">

Built by [@defidaddydavid](https://github.com/defidaddydavid) · Powered by [Claude](https://anthropic.com)

*If this is useful, drop a ⭐ — it helps more people find it.*

</div>
