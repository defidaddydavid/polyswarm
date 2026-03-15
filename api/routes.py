"""
FastAPI routes for PolySwarm.
"""

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional
import os
from core.swarm import Swarm
from core.calibration import (
    get_swarm_brier_score,
    get_agent_brier_scores,
    resolve_forecast,
    get_forecast_history,
)

app = FastAPI(
    title="PolySwarm",
    description="Multi-agent AI forecasting engine for prediction markets",
    version="0.6.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional API key authentication
API_KEY = os.getenv("POLYSWARM_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """If POLYSWARM_API_KEY is set, require it in X-API-Key header."""
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


swarm = Swarm()


class ForecastRequest(BaseModel):
    question: str
    market_odds: Optional[float] = None
    rounds: Optional[int] = None


class ScenarioRequest(BaseModel):
    scenario: str
    context: Optional[str] = ""


class ResolveRequest(BaseModel):
    question: str
    outcome: float


@app.post("/forecast")
async def forecast(req: ForecastRequest, _=Depends(verify_api_key)):
    try:
        if req.rounds:
            os.environ["DEBATE_ROUNDS"] = str(req.rounds)
        result = swarm.forecast(req.question, req.market_odds)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scenario")
async def scenario(req: ScenarioRequest, _=Depends(verify_api_key)):
    try:
        from core.scenario import ScenarioEngine
        engine = ScenarioEngine()
        result = engine.simulate(req.scenario, req.context or "")
        return {
            "scenario": result.scenario,
            "aggregate_sentiment": result.aggregate_sentiment,
            "aggregate_price_impact": result.aggregate_price_impact,
            "consensus": result.consensus,
            "narrative": result.narrative,
            "secondary_effects": result.secondary_effects,
            "reactions": [
                {
                    "agent_id": r.agent_id,
                    "persona": r.persona,
                    "immediate_reaction": r.immediate_reaction,
                    "sentiment_shift": r.sentiment_shift,
                    "price_impact_estimate": r.price_impact_estimate,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                    "actions": r.actions,
                }
                for r in result.reactions
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resolve")
async def resolve(req: ResolveRequest):
    resolve_forecast(req.question, req.outcome)
    return {"status": "resolved", "question": req.question, "outcome": req.outcome}


@app.get("/calibration")
async def calibration():
    return {
        "swarm_brier_score": get_swarm_brier_score(),
        "agent_brier_scores": get_agent_brier_scores(),
        "note": "Brier score: lower is better. Perfect = 0.0, random = 0.25",
    }


@app.get("/agents")
async def agents():
    from agents.personas import PERSONA_DEFINITIONS
    return {"agents": PERSONA_DEFINITIONS, "count": len(PERSONA_DEFINITIONS)}


@app.get("/forecasts")
async def forecasts(limit: int = 50, _=Depends(verify_api_key)):
    """Retrieve past forecast history from the calibration DB."""
    try:
        history = get_forecast_history(limit=limit)
        return {"forecasts": history, "count": len(history)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sources")
async def sources():
    """List all registered data sources and their status."""
    from data.context import list_sources
    source_list = list_sources()
    available = sum(1 for s in source_list if s["available"])
    return {
        "sources": source_list,
        "total": len(source_list),
        "available": available,
    }


@app.get("/health")
async def health():
    from data.context import list_sources
    source_list = list_sources()
    return {
        "status": "ok",
        "version": "0.7.0",
        "agents": 12,
        "data_sources": sum(1 for s in source_list if s["available"]),
    }
