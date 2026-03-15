"""
Base agent class with multi-LLM support.
Supports: Anthropic Claude, OpenAI GPT, Ollama (local models).
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel


class AgentEstimate(BaseModel):
    agent_id: str
    persona: str
    probability: float
    confidence: float
    reasoning: str
    key_factors: list[str]
    round: int


def _get_llm_client():
    """Factory for LLM client based on LLM_PROVIDER env var."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Add it to .env or run:\n"
                "  export ANTHROPIC_API_KEY=your_key_here\n"
                "Or switch providers: LLM_PROVIDER=ollama (free, local)"
            )
        import anthropic
        return "anthropic", anthropic.Anthropic(api_key=key)
    elif provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY not set. Add it to .env or run:\n"
                "  export OPENAI_API_KEY=your_key_here\n"
                "Or switch providers: LLM_PROVIDER=ollama (free, local)"
            )
        import openai
        base_url = os.getenv("OPENAI_BASE_URL")
        return "openai", openai.OpenAI(api_key=key, **({"base_url": base_url} if base_url else {}))
    elif provider == "ollama":
        import httpx
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return "ollama", httpx.Client(base_url=base_url, timeout=60)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use 'anthropic', 'openai', or 'ollama'.")


def _get_model_name() -> str:
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model_env = os.getenv("MODEL_FAST")
    if model_env:
        return model_env
    defaults = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o-mini",
        "ollama": "llama3.1:8b",
    }
    return defaults.get(provider, "claude-sonnet-4-20250514")


def _call_llm(provider: str, client, system: str, user: str, max_tokens: int = 512) -> str:
    """Unified LLM call across providers."""
    model = _get_model_name()

    if provider == "anthropic":
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()

    elif provider == "openai":
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content.strip()

    elif provider == "ollama":
        resp = client.post("/api/chat", json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        })
        return resp.json()["message"]["content"].strip()

    raise ValueError(f"Unknown provider: {provider}")


def _parse_json(raw: str) -> dict:
    """Parse JSON from LLM response, handling code fences."""
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


@dataclass
class Agent:
    agent_id: str
    persona: str
    description: str
    information_focus: str
    bias_profile: str
    base_confidence: float = 0.7
    memory: list[str] = field(default_factory=list)
    estimates_history: list[AgentEstimate] = field(default_factory=list)

    def __post_init__(self):
        self._provider, self._client = _get_llm_client()

    def _build_system_prompt(self) -> str:
        return f"""You are a {self.persona} participating in a prediction market forecasting exercise.

Your profile:
{self.description}

Information focus: {self.information_focus}
Known biases: {self.bias_profile}

Your job is to estimate the probability that a given event will resolve YES.
Be honest, calibrated, and reason carefully. Do not be overconfident.
Always output a JSON object with these exact fields:
- probability: float between 0.0 and 1.0
- confidence: float between 0.0 and 1.0 (how confident you are in your estimate)
- reasoning: string (2-4 sentences explaining your thinking)
- key_factors: list of 3-5 strings (most important factors driving your estimate)

Output ONLY valid JSON, no other text."""

    def estimate(
        self,
        question: str,
        context: str,
        debate_round: int = 1,
        other_estimates: Optional[list[AgentEstimate]] = None,
    ) -> AgentEstimate:
        user_content = f"Question: {question}\n\nContext:\n{context}\n"

        if other_estimates and debate_round > 1:
            others_summary = "\n".join([
                f"- {e.persona}: {e.probability:.0%} confidence={e.confidence:.0%} | {e.reasoning[:150]}"
                for e in other_estimates
                if e.agent_id != self.agent_id
            ])
            user_content += f"\n--- Other agents' estimates (Round {debate_round - 1}) ---\n{others_summary}\n\nConsider their perspectives. You may update your estimate or defend your original position.\n"

        if self.memory:
            memory_str = "\n".join(self.memory[-5:])
            user_content += f"\nYour relevant past observations:\n{memory_str}"

        raw = _call_llm(self._provider, self._client, self._build_system_prompt(), user_content)
        data = _parse_json(raw)

        estimate = AgentEstimate(
            agent_id=self.agent_id,
            persona=self.persona,
            probability=float(data["probability"]),
            confidence=float(data["confidence"]),
            reasoning=data["reasoning"],
            key_factors=data.get("key_factors", []),
            round=debate_round,
        )
        self.estimates_history.append(estimate)
        return estimate

    def add_memory(self, memory: str):
        self.memory.append(memory)
