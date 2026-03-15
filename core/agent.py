"""
Base agent class. Each agent has a persona, memory, and reasoning style.
They take a question + context, form a probability estimate, then can
update after seeing other agents' reasoning (debate rounds).
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Optional
import anthropic
from pydantic import BaseModel


class AgentEstimate(BaseModel):
    agent_id: str
    persona: str
    probability: float          # 0.0 - 1.0
    confidence: float           # 0.0 - 1.0 (self-reported)
    reasoning: str
    key_factors: list[str]
    round: int


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
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self._model = os.getenv("MODEL_FAST", "claude-3-5-haiku-20241022")

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
        """Form a probability estimate, optionally after seeing others' views."""

        user_content = f"""Question: {question}

Context:
{context}
"""

        if other_estimates and debate_round > 1:
            others_summary = "\n".join([
                f"- {e.persona}: {e.probability:.0%} confidence={e.confidence:.0%} | {e.reasoning[:150]}"
                for e in other_estimates
                if e.agent_id != self.agent_id
            ])
            user_content += f"""
--- Other agents' estimates (Round {debate_round - 1}) ---
{others_summary}

Consider their perspectives. You may update your estimate or defend your original position.
"""

        if self.memory:
            memory_str = "\n".join(self.memory[-5:])  # last 5 memories
            user_content += f"\nYour relevant past observations:\n{memory_str}"

        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=self._build_system_prompt(),
            messages=[{"role": "user", "content": user_content}],
        )

        import json
        raw = response.content[0].text.strip()
        # strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())

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
