"""LLM Agent implementation for forecasting."""

import asyncio
import json
from typing import Optional, Any
from dataclasses import dataclass

import httpx

from config import config
from bot.prompts import AGENT_PROMPTS, FORECAST_TEMPLATE, PEER_REVIEW_TEMPLATE
from bot.utils import parse_probability, extract_reasoning
from models.schemas import AgentForecast, ResearchContext


@dataclass
class AgentConfig:
    name: str
    system_prompt: str
    weight: float


AGENT_CONFIGS = {
    "base_rate_analyst": AgentConfig(
        name="base_rate_analyst",
        system_prompt=AGENT_PROMPTS["base_rate_analyst"],
        weight=1.0,
    ),
    "market_prior": AgentConfig(
        name="market_prior",
        system_prompt=AGENT_PROMPTS["market_prior"],
        weight=1.5,
    ),
    "status_quo_anchor": AgentConfig(
        name="status_quo_anchor",
        system_prompt=AGENT_PROMPTS["status_quo_anchor"],
        weight=1.0,
    ),
    "devils_advocate": AgentConfig(
        name="devils_advocate",
        system_prompt=AGENT_PROMPTS["devils_advocate"],
        weight=1.0,
    ),
    "synthesizer": AgentConfig(
        name="synthesizer",
        system_prompt=AGENT_PROMPTS["synthesizer"],
        weight=1.5,
    ),
}


class LLMAgent:
    """Agent that makes forecasts via Metaculus LLM proxy."""

    def __init__(self, agent_config: AgentConfig):
        self.name = agent_config.name
        self.system_prompt = agent_config.system_prompt
        self.weight = agent_config.weight
        self.proxy_url = config.metaculus_llm_proxy
        self.timeout = config.agent_timeout

    async def forecast(
        self,
        question_text: str,
        resolution_criteria: str,
        research_context: ResearchContext,
        model: str = "claude-sonnet-4-20250514",
    ) -> AgentForecast:
        research_section = research_context.to_prompt_section()

        user_message = FORECAST_TEMPLATE.format(
            question_text=question_text,
            resolution_criteria=resolution_criteria,
            research_section=research_section,
        )

        response = await self._call_llm(
            messages=[{"role": "user", "content": user_message}],
            model=model,
        )

        probability = parse_probability(response) or 0.5
        reasoning = extract_reasoning(response)

        return AgentForecast(
            agent_name=self.name,
            weight=self.weight,
            initial_reasoning=reasoning,
            initial_probability=probability,
            final_probability=probability,
            final_reasoning=reasoning,
        )

    async def peer_review(
        self,
        question_text: str,
        other_agent: AgentForecast,
        model: str = "claude-sonnet-4-20250514",
    ) -> str:
        user_message = f"""## Question
{question_text}

## Another Agent's Forecast
Agent: {other_agent.agent_name}
Probability: {other_agent.final_probability:.2%}
Reasoning: {other_agent.final_reasoning}

## Your Task
Provide a brief critique of this forecast. Consider:
1. Are there weaknesses in the reasoning?
2. What factors might they be missing?
3. Is their probability well-calibrated?

Keep your critique constructive and under 200 words."""

        response = await self._call_llm(
            messages=[{"role": "user", "content": user_message}],
            model=model,
        )

        return response

    async def revise_forecast(
        self,
        question_text: str,
        initial_forecast: AgentForecast,
        peer_critique: str,
        model: str = "claude-sonnet-4-20250514",
    ) -> AgentForecast:
        user_message = PEER_REVIEW_TEMPLATE.format(
            question_text=question_text,
            initial_probability=initial_forecast.initial_probability,
            initial_reasoning=initial_forecast.initial_reasoning,
            peer_critique=peer_critique,
        )

        response = await self._call_llm(
            messages=[{"role": "user", "content": user_message}],
            model=model,
        )

        probability = (
            parse_probability(response) or initial_forecast.initial_probability
        )
        reasoning = extract_reasoning(response)

        return AgentForecast(
            agent_name=self.name,
            weight=self.weight,
            initial_reasoning=initial_forecast.initial_reasoning,
            initial_probability=initial_forecast.initial_probability,
            peer_critique=peer_critique,
            final_probability=probability,
            final_reasoning=reasoning,
        )

    async def _call_llm(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 2000,
    ) -> str:
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "system": self.system_prompt,
            "messages": messages,
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": config.metaculus_token,
        }

        for attempt in range(config.agent_max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.proxy_url,
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["content"][0]["text"]
            except Exception as e:
                if attempt == config.agent_max_retries - 1:
                    raise RuntimeError(
                        f"LLM call failed after {config.agent_max_retries} attempts: {e}"
                    )
                await asyncio.sleep(2**attempt)

        return ""


def create_committee(
    agent_names: Optional[list[str]] = None,
) -> list[LLMAgent]:
    if agent_names is None:
        agent_names = list(AGENT_CONFIGS.keys())

    agents = []
    for name in agent_names:
        if name in AGENT_CONFIGS:
            agents.append(LLMAgent(AGENT_CONFIGS[name]))

    return agents


async def run_initial_forecasts(
    agents: list[LLMAgent],
    question_text: str,
    resolution_criteria: str,
    research_context: ResearchContext,
    model: str = "claude-sonnet-4-20250514",
) -> list[AgentForecast]:
    tasks = [
        agent.forecast(question_text, resolution_criteria, research_context, model)
        for agent in agents
    ]

    return await asyncio.gather(*tasks)


async def run_peer_reviews(
    agents: list[LLMAgent],
    forecasts: list[AgentForecast],
    question_text: str,
    model: str = "claude-sonnet-4-20250514",
) -> list[str]:
    n = len(agents)
    critiques = []

    for i, agent in enumerate(agents):
        reviewed_idx = (i + 1) % n
        critique = await agent.peer_review(
            question_text, forecasts[reviewed_idx], model
        )
        critiques.append(critique)

    return critiques


async def run_revisions(
    agents: list[LLMAgent],
    forecasts: list[AgentForecast],
    critiques: list[str],
    question_text: str,
    model: str = "claude-sonnet-4-20250514",
) -> list[AgentForecast]:
    tasks = [
        agent.revise_forecast(question_text, forecast, critique, model)
        for agent, forecast, critique in zip(agents, forecasts, critiques)
    ]

    return await asyncio.gather(*tasks)
