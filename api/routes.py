"""
FastAPI routes for PolySwarm.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.swarm import Swarm
from core.calibration import (
    get_swarm_brier_score,
    get_agent_brier_scores,
    resolve_forecast,
)

app = FastAPI(
    title="PolySwarm",
    description="Multi-agent AI forecasting engine for prediction markets",
    version="0.1.0",
)

swarm = Swarm()


class ForecastRequest(BaseModel):
    question: str
    market_odds: float | None = None


class ResolveRequest(BaseModel):
    question: str
    outcome: float  # 1.0 = YES, 0.0 = NO


@app.post("/forecast")
async def forecast(req: ForecastRequest):
    try:
        result = swarm.forecast(req.question, req.market_odds)
        return result
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


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
