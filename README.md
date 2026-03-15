# PolySwarm 🐝

> **Multi-agent AI forecasting engine for prediction markets.**
> Spawn a swarm of AI agents with distinct personas, let them debate, and get a calibrated probability estimate — with edge calculation vs live market odds.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Anthropic](https://img.shields.io/badge/powered%20by-Claude-orange.svg)](https://anthropic.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What is PolySwarm?

PolySwarm is a digital forecasting sandbox. Feed it any resolvable question and a swarm of 8 AI agents — each with different expertise, memory, and known biases — independently form probability estimates, then debate each other across multiple rounds.

The result: a calibrated, consensus-adjusted probability with full reasoning transparency. Compare it against live Polymarket odds to find your edge.

```
Question: "Will BTC close above $100k before June 2026?"

── Round 1 ──
  Macro Analyst       →  34%  (confidence: 72%)
  Crypto Native       →  61%  (confidence: 75%)
  Quant Trader        →  45%  (confidence: 80%)
  Retail Participant  →  58%  (confidence: 55%)
  Contrarian Skeptic  →  28%  (confidence: 65%)
  On-Chain Analyst    →  52%  (confidence: 70%)
  Institutional Desk  →  38%  (confidence: 78%)
  Event Specialist    →  49%  (confidence: 73%)

── Round 2 ── (agents update after seeing each other's reasoning)
  ...

  Swarm Probability:  46.8%
  Consensus:          61%   |   Agents: 8
  Market odds:        38%   |   Edge:   +8.8%
```

---

## Features

- **8 distinct agent personas** — macro analyst, crypto native, quant trader, retail participant, contrarian skeptic, on-chain analyst, institutional desk, event specialist
- **Multi-round debate** — agents see each other's reasoning and update beliefs before the final aggregation
- **Live context injection** — Fear & Greed Index, BTC/ETH price, funding rates pulled automatically on each run
- **Calibration tracking** — Brier scores per agent, stored in SQLite. Better-calibrated agents get higher weight over time
- **Edge calculator** — pass current market odds and get the swarm's edge vs the market
- **FastAPI server** — programmatic access via REST API
- **Polymarket integration** — search and fetch live markets
- **Docker ready** — one command deploy

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
# Add your ANTHROPIC_API_KEY to .env
```

Get a free Anthropic API key at [console.anthropic.com](https://console.anthropic.com)

### 3. Run your first forecast

```bash
# Basic forecast
python main.py forecast "Will BTC close above $100k before June 2026?"

# With market odds for edge calculation
python main.py forecast "Will the Fed cut rates in June 2026?" --odds 0.35

# More debate rounds = more nuanced output
python main.py forecast "Will ETH flip BTC in 2026?" --odds 0.12 --rounds 3
```

### 4. Start the API server

```bash
python main.py serve
# → http://localhost:8000/docs
```

---

## Docker

```bash
docker compose up
```

```yaml
# docker-compose.yml
services:
  polyswarm:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./data:/app/data
```

---

## API

```bash
# Run a forecast
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"question": "Will BTC hit $150k in 2026?", "market_odds": 0.25}'

# Resolve a forecast (for calibration tracking)
curl -X POST http://localhost:8000/resolve \
  -d '{"question": "Will BTC hit $150k in 2026?", "outcome": 1.0}'

# Get calibration scores
curl http://localhost:8000/calibration
```

---

## Calibration

PolySwarm tracks forecast accuracy using **Brier scores** (lower = better, 0.0 = perfect, 0.25 = random).

When a market resolves, call `resolve` and calibration weights automatically update — better-performing agents gain influence in future aggregations.

```bash
# Resolve a forecast
python main.py resolve "Will BTC close above $100k before June 2026?" --outcome 1.0

# View calibration scores
python main.py calibration
```

---

## Agent Personas

| Agent | Focus | Known Bias |
|---|---|---|
| Macro Analyst | Fed policy, rates, global liquidity | Underweights crypto-native factors |
| Crypto Native | On-chain data, CT narrative, funding rates | Structurally bullish |
| Quant Trader | Statistical base rates, market microstructure | Dismissive of qualitative factors |
| Retail Participant | Recent price action, social sentiment | Momentum-chasing, FOMO-prone |
| Contrarian Skeptic | Overcrowded trades, tail risks | Structurally bearish |
| On-Chain Analyst | Wallet flows, exchange reserves, smart money | Can lag price action |
| Institutional Desk | ETF flows, regulatory, risk-adjusted metrics | Conservative, slow to move |
| Event Specialist | FOMC, halvings, protocol upgrades, catalysts | Overweights known events |

---

## Use Cases

- **Prediction markets** — find edge vs Polymarket, Manifold, Metaculus odds
- **Crypto trading signals** — feed swarm output into a signal engine as a sentiment layer
- **Event forecasting** — FOMC decisions, ETF approvals, protocol upgrades
- **Options strategy** — confirm directional bias before sizing straddles or directional plays
- **Research** — study how different agent personas disagree and why

---

## Roadmap

- [ ] Polymarket live market sync + auto-compare
- [ ] Public leaderboard (swarm accuracy vs market resolution)
- [ ] Web UI with live debate viewer
- [ ] Additional agent personas (geopolitical analyst, DeFi specialist)
- [ ] OpenAI / local LLM support
- [ ] Telegram / Discord bot integration

---

## License

MIT — use it, fork it, build on it.

---

Built by [@defidaddydavid](https://github.com/defidaddydavid) · Powered by [Claude](https://anthropic.com)
