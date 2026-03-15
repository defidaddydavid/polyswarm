<div align="center">

<br />

<img src="https://img.shields.io/badge/⟐-PolySwarm-000000?style=for-the-badge&labelColor=F59E0B&color=000000" alt="PolySwarm" height="40" />

<br /><br />

# Multi-Agent AI Forecasting Engine

### Predict markets. Simulate crowds. Find your edge.

<br />

*12 AI agents with distinct personas debate in real-time, informed by 15+ live data feeds,*
*to produce calibrated probability estimates via 7 aggregation methods and crowd simulations.*

<br />

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

**Works with:** &nbsp; Claude · GPT-4o · Llama · Mistral · Any OpenAI-compatible API

<br />

[**Get Started**](#-quickstart) · [**API Docs**](#-rest-api) · [**Data Sources**](#-live-data-sources) · [**How It Works**](#-how-it-works) · [**Live Site**](https://defidaddydavid.github.io/polyswarm/)

<br />

---

</div>

<br />

## Two Powerful Modes

<table>
<tr>
<td width="50%" valign="top">

### Forecast Mode

Ask any binary question. The swarm debates and returns a calibrated probability.

```bash
python main.py forecast \
  "Will BTC close above $100k before June?" \
  --odds 0.42
```

**Output:**
- Probability estimate from 12 independent agents
- Multi-round debate (agents update after seeing others)
- 7 aggregation methods: Bayesian, extremized, surprisingly popular, LogOP, Cooke's, MC, bootstrap CI
- Herding detection, Nash equilibrium, information cascades
- Edge calculation vs live market odds

</td>
<td width="50%" valign="top">

### Scenario Mode

Simulate how markets react to any event — before it happens.

```bash
python main.py scenario \
  "Elon tweets Tesla will accept Bitcoin again"
```

**Output:**
- Each agent's immediate reaction & actions
- Sentiment shift per persona (-1.0 to +1.0)
- Estimated price impact
- Crowd narrative + second-order effects

</td>
</tr>
</table>

<br />

## Quickstart

```bash
# Clone
git clone https://github.com/defidaddydavid/polyswarm.git && cd polyswarm

# Install
pip install -r requirements.txt

# Configure (add your API key)
cp .env.example .env

# Run your first forecast
python main.py forecast "Will BTC hit $100k before July 2026?" --odds 0.45

# Or simulate a scenario
python main.py scenario "SEC bans crypto staking in the US"

# Or just see the live data feed
python main.py context
```

**Docker:**
```bash
ANTHROPIC_API_KEY=your_key docker compose up
# API at http://localhost:8000/docs
```

<br />

## Multi-LLM Support

PolySwarm is provider-agnostic. Use whatever LLM you want.

```bash
# Anthropic Claude (default)
LLM_PROVIDER=anthropic MODEL_FAST=claude-sonnet-4-20250514

# OpenAI
LLM_PROVIDER=openai MODEL_FAST=gpt-4o-mini

# Local via Ollama (free, private, no API key)
LLM_PROVIDER=ollama MODEL_FAST=llama3.1:8b

# Any OpenAI-compatible API (Groq, Together, etc.)
LLM_PROVIDER=openai OPENAI_API_KEY=your_key OPENAI_BASE_URL=https://api.groq.com/openai/v1
```

<br />

## How It Works

```
                    ┌──────────────────┐
                    │   Your Question  │
                    │  or Scenario     │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Context Engine   │
                    │  (15+ live APIs)  │
                    └────────┬─────────┘
                             │
              ┌──────────────▼──────────────┐
              │                             │
    ┌─────────▼─────────┐        ┌─────────▼─────────┐
    │   FORECAST MODE    │        │   SCENARIO MODE    │
    │                    │        │                    │
    │  Round 1: 12 agents│        │  12 agents react   │
    │  form estimates    │        │  independently     │
    │       ↓            │        │       ↓            │
    │  Round 2: Debate   │        │  Aggregate         │
    │  Update beliefs    │        │  sentiment vector  │
    │       ↓            │        │       ↓            │
    │  7x Aggregation    │        │  Generate crowd    │
    │  + Game Theory     │        │  narrative +       │
    │       ↓            │        │  2nd-order effects │
    │  Final probability │        │                    │
    │  + edge vs market  │        │                    │
    └────────────────────┘        └────────────────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼─────────┐
                    │  Calibration DB   │
                    │  Brier scores     │
                    │  Agent weights    │
                    │  update over time │
                    └──────────────────┘
```

<br />

## Statistical Analysis

Every forecast runs through a full statistical analysis pipeline — 7 aggregation methods plus game-theoretic analysis:

### Aggregation Methods

| Method | Module | What It Does |
|--------|--------|-------------|
| **Bayesian Updating** | `core/bayesian.py` | Treats each agent's estimate as evidence. Updates beliefs via Bayes' theorem with KL-divergence weighting. More surprising estimates carry more information. |
| **Extremized Aggregation** | `core/extremize.py` | Based on IARPA ACE research (Satopää/Tetlock). Corrects systematic under-confidence by pushing averages away from 50% in log-odds space. |
| **Surprisingly Popular** | `core/surprisingly_popular.py` | Prelec et al. (2017, *Nature*). Exploits meta-cognitive information — the correct answer is often "more popular than people predict." |
| **Log Opinion Pool** | `core/opinion_pool.py` | Multiplicative combination in log space. Satisfies external Bayesianity — theoretically optimal when agents share likelihoods. |
| **Cooke's Classical Model** | `core/opinion_pool.py` | Performance-based weighting from expert elicitation theory. Weights by calibration AND informativeness. Unqualified agents are pruned. |
| **Monte Carlo Simulation** | `core/statistics.py` | 5,000 simulations treating each agent as a beta distribution. Produces percentiles, skew, and threshold probabilities. |
| **Bootstrap CI** | `core/statistics.py` | Resamples agent estimates 1,000 times to produce 95% confidence intervals. Quantifies uncertainty in the consensus. |

### Game Theory & Analysis

| Method | Module | What It Does |
|--------|--------|-------------|
| **Herding Detection** | `core/game_theory.py` | HHI-based clustering analysis. Detects when agents converge suspiciously. Flags contrarian signals. |
| **Information Cascades** | `core/game_theory.py` | Tracks belief shifts between debate rounds. Detects when agents flip sides and whether convergence was genuine. |
| **Nash Equilibrium** | `core/game_theory.py` | Checks if consensus is stable — would any agent benefit from deviating? Unstable = low-confidence forecast. |
| **Agent Agreement** | `core/bayesian.py` | Pairwise Jensen-Shannon divergence matrix. Identifies most aligned and most divergent agent pairs. |

<br />

## The 12 Agents

Every agent has a unique lens, information focus, and documented bias — creating genuine disagreement, not echo-chamber consensus.

| | Agent | What They Watch | Known Blind Spot |
|:---:|-------|----------------|-----------------|
| 📊 | **Macro Analyst** | Fed policy, rates, DXY, global liquidity | Underweights crypto-native catalysts |
| ₿ | **Crypto Native** | On-chain flows, funding rates, CT narrative | Structurally bullish, narrative-captured |
| 📉 | **Quant Trader** | Base rates, vol surface, mean-reversion | Ignores qualitative regime shifts |
| 📱 | **Retail Participant** | Price action, Reddit, influencers | FOMO/panic driven, momentum-chasing |
| 🐻 | **Contrarian Skeptic** | Tail risks, crowded trades, leverage | Sees crashes everywhere |
| 🔍 | **On-Chain Analyst** | Whale wallets, exchange flows, miners | Can lag price action |
| 🏦 | **Institutional Desk** | ETF flows, regulatory, risk metrics | Conservative, slow to move |
| 📅 | **Event Specialist** | FOMC, halvings, upgrades, catalysts | Overweights scheduled events |
| 🌐 | **DeFi Specialist** | TVL, yields, governance, contagion | Overweights DeFi signals |
| ⚡ | **Options Trader** | IV rank, skew, gamma, term structure | Over-indexes on derivatives |
| 🌍 | **Geopolitical Analyst** | Regulation, sanctions, CBDCs | Behind the curve on markets |
| 📡 | **Social Sentiment** | Reddit, Twitter, Trends, search volume | Reactive, not predictive |

<br />

## Live Data Sources

Every forecast is grounded in **real-time data from 23 sources** — fetched in parallel via a **modular plugin system**. Most sources work without API keys; optional keys unlock higher rate limits.

<table>
<tr>
<td width="33%" valign="top">

**Market Data**
- Binance — BTC, ETH, SOL spot
- Binance — Top movers (24h)
- CoinGecko — Market overview
- Binance Futures — funding rates (6 assets)
- Binance Futures — open interest
- Binance Futures — long/short ratios
- Binance Futures — top trader positions
- Binance Futures — liquidations

</td>
<td width="33%" valign="top">

**On-Chain & DeFi**
- Deribit — BTC options + volatility
- DeFi Llama — total TVL + top protocols
- DeFi Llama — stablecoin supply
- Mempool.space — BTC mempool + fees
- Blockchain.info — BTC hashrate
- Etherscan — ETH gas prices

</td>
<td width="33%" valign="top">

**Sentiment & Prediction**
- Fear & Greed Index (7-day)
- Reddit r/cryptocurrency
- CoinGecko trending coins
- CryptoPanic headlines
- Polymarket trending + search
- Manifold Markets trending + search

</td>
</tr>
</table>

### Adding Your Own Data Source

PolySwarm uses a **plugin-based registry** — add a new source with zero changes to existing code:

```python
# data/sources/my_source.py
from data.registry import DataSource, register_source

@register_source
class MySource(DataSource):
    name = "my_source"
    category = "sentiment"
    description = "My custom data feed"
    requires_key = "MY_API_KEY"  # None if no key needed

    def fetch(self) -> str:
        resp = self.http.get("https://api.example.com/data")
        return f"My Data: {resp.json()['value']}"
```

That's it. Drop the file in `data/sources/`, and it's automatically discovered and included.

```bash
# See everything the agents see:
python main.py context

# List all sources and their status:
python main.py sources

# Filter to specific sources:
POLYSWARM_SOURCES=binance_spot,funding_rates,fear_greed python main.py forecast "..."
```

<br />

## REST API

```bash
# Start server
python main.py serve

# Forecast (with optional API key auth)
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_key" \
  -d '{"question": "Will BTC hit $150k in 2026?", "market_odds": 0.25}'

# Scenario simulation
curl -X POST http://localhost:8000/scenario \
  -H "Content-Type: application/json" \
  -d '{"scenario": "Binance announces insolvency", "context": "BTC at $87k"}'

# Resolve & track accuracy
curl -X POST http://localhost:8000/resolve \
  -H "Content-Type: application/json" \
  -d '{"question": "Will BTC hit $150k in 2026?", "outcome": 1.0}'

# Forecast history
curl http://localhost:8000/forecasts?limit=20

# Calibration leaderboard
curl http://localhost:8000/calibration

# List all agents
curl http://localhost:8000/agents

# Data source status
curl http://localhost:8000/sources
```

Set `POLYSWARM_API_KEY` in `.env` to require `X-API-Key` header on protected endpoints.

Full interactive docs at `http://localhost:8000/docs`

<br />

## Calibration

PolySwarm gets smarter over time. Every forecast is tracked. When markets resolve, Brier scores update per agent — and better-calibrated agents automatically receive more weight.

```bash
python main.py resolve "Will BTC close above $100k?" --outcome 1.0
python main.py calibration
```

| Score | Meaning |
|-------|---------|
| 0.00 – 0.10 | Excellent calibration |
| 0.10 – 0.20 | Good |
| 0.20 – 0.25 | Average (random = 0.25) |
| 0.25+ | Poor — agent gets downweighted |

<br />

## Use Cases

| | Use Case | Mode | Example |
|:---:|----------|------|---------|
| 💰 | **Prediction market edge** | Forecast | Find +EV bets vs Polymarket odds |
| 📅 | **Event trading** | Forecast | FOMC, ETF decisions, halvings |
| 💥 | **Black swan simulation** | Scenario | "Exchange X is insolvent" |
| 📈 | **Options positioning** | Scenario | Validate bias before buying calls/puts |
| 🔬 | **Market research** | Scenario | "New stablecoin regulation passed" |
| 🛡️ | **Risk management** | Scenario | Simulate tail events before they happen |
| 🤖 | **Signal pipeline** | API | Feed swarm output into trading algos |

<br />

## Roadmap

- [x] Modular plugin-based data pipeline
- [x] API key authentication
- [x] Forecast history & calibration export
- [x] 7 peer-reviewed aggregation methods
- [ ] Live Polymarket sync + auto-compare leaderboard
- [ ] Web UI with real-time debate viewer
- [ ] Agent memory persistence (Redis)
- [ ] Streaming API — watch agents think live
- [ ] Telegram & Discord bot
- [ ] More data sources (Glassnode, Santiment, Nansen — just drop a plugin!)
- [ ] Custom persona builder
- [ ] Backtesting against historical market resolutions

<br />

---

<div align="center">

<br />

**MIT License** — use it, fork it, build on it.

Built by [@defidaddydavid](https://github.com/defidaddydavid)

*If this helps you find edge, drop a star*

⟐ PolySwarm

<br />

</div>
