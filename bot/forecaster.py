"""Main forecasting workflow orchestrator."""

import asyncio
from typing import Optional

from bot.agents import (
    create_committee,
    run_initial_forecasts,
    run_peer_reviews,
    run_revisions,
    LLMAgent,
)
from bot.utils import weighted_average, aggregate_logit_space
from bot.prompts import SYNTHESIS_TEMPLATE, format_agent_forecasts
from models.schemas import AgentForecast, CommitteeResult, ResearchContext
from research.integrated_search import integrated_research


class Forecaster:
    """Orchestrates the multi-agent forecasting workflow."""

    def __init__(
        self,
        agent_names: Optional[list[str]] = None,
        model: str = "claude-sonnet-4-20250514",
        use_peer_review: bool = True,
    ):
        self.agents = create_committee(agent_names)
        self.model = model
        self.use_peer_review = use_peer_review

    async def forecast_binary(
        self,
        question_id: int,
        question_text: str,
        resolution_criteria: str,
        use_adj: bool = True,
        use_asknews: bool = True,
        use_perplexity: bool = True,
    ) -> CommitteeResult:
        research_str = await integrated_research(
            question_text,
            include_asknews=use_asknews,
            include_perplexity=use_perplexity,
        )

        research = ResearchContext(
            question_text=question_text,
            adj_prior_section=research_str,
        )

        forecasts = await run_initial_forecasts(
            self.agents,
            question_text,
            resolution_criteria,
            research,
            self.model,
        )

        if self.use_peer_review:
            critiques = await run_peer_reviews(
                self.agents, forecasts, question_text, self.model
            )
            forecasts = await run_revisions(
                self.agents, forecasts, critiques, question_text, self.model
            )

        final_prob = self._aggregate_forecasts(forecasts)
        reasoning = self._summarize_reasoning(forecasts)

        return CommitteeResult(
            question_id=question_id,
            question_text=question_text,
            question_type="binary",
            research_context=research.to_prompt_section(),
            agent_forecasts=forecasts,
            final_probability=final_prob,
            reasoning_summary=reasoning,
        )

    async def forecast_numeric(
        self,
        question_id: int,
        question_text: str,
        resolution_criteria: str,
        lower_bound: float,
        upper_bound: float,
        open_lower: bool = False,
        open_upper: bool = False,
        use_adj: bool = True,
        use_asknews: bool = True,
        use_perplexity: bool = True,
    ) -> CommitteeResult:
        research_str = await integrated_research(
            question_text,
            include_asknews=use_asknews,
            include_perplexity=use_perplexity,
        )
        research = ResearchContext(
            question_text=question_text,
            adj_prior_section=research_str,
        )

        from bot.prompts import NUMERIC_FORECAST_TEMPLATE

        forecasts = []
        for agent in self.agents:
            forecast = await self._forecast_numeric_agent(
                agent,
                question_text,
                resolution_criteria,
                research,
                lower_bound,
                upper_bound,
                open_lower,
                open_upper,
            )
            forecasts.append(forecast)

        final_cdf = self._aggregate_cdfs(forecasts, lower_bound, upper_bound)
        reasoning = self._summarize_reasoning(forecasts)

        return CommitteeResult(
            question_id=question_id,
            question_text=question_text,
            question_type="numeric",
            research_context=research.to_prompt_section(),
            agent_forecasts=forecasts,
            final_probability=0.5,
            final_cdf=final_cdf,
            reasoning_summary=reasoning,
        )

    async def forecast_multiple_choice(
        self,
        question_id: int,
        question_text: str,
        resolution_criteria: str,
        options: list[str],
        use_adj: bool = True,
        use_asknews: bool = True,
        use_perplexity: bool = True,
    ) -> CommitteeResult:
        research_str = await integrated_research(
            question_text,
            include_asknews=use_asknews,
            include_perplexity=use_perplexity,
        )
        research = ResearchContext(
            question_text=question_text,
            adj_prior_section=research_str,
        )

        from bot.prompts import MULTIPLE_CHOICE_TEMPLATE
        from bot.utils import parse_option_probabilities

        forecasts = []
        for agent in self.agents:
            user_message = MULTIPLE_CHOICE_TEMPLATE.format(
                question_text=question_text,
                resolution_criteria=resolution_criteria,
                options_list="\n".join(f"- {opt}" for opt in options),
                research_section=research.to_prompt_section(),
                option_prob_format="\n".join(
                    f"{opt}: [probability]" for opt in options
                ),
            )

            response = await agent._call_llm(
                messages=[{"role": "user", "content": user_message}],
                model=self.model,
            )

            option_probs = parse_option_probabilities(response, options)
            total = sum(option_probs.values())
            if total > 0:
                option_probs = {k: v / total for k, v in option_probs.items()}

            forecasts.append(
                AgentForecast(
                    agent_name=agent.name,
                    weight=agent.weight,
                    final_probability=0.5,
                    final_reasoning=response,
                )
            )

        final_probs = self._aggregate_multiple_choice(forecasts, options)
        reasoning = self._summarize_reasoning(forecasts)

        return CommitteeResult(
            question_id=question_id,
            question_text=question_text,
            question_type="multiple_choice",
            research_context=research.to_prompt_section(),
            agent_forecasts=forecasts,
            final_option_probs=final_probs,
            reasoning_summary=reasoning,
        )

    async def _forecast_numeric_agent(
        self,
        agent: LLMAgent,
        question_text: str,
        resolution_criteria: str,
        research: ResearchContext,
        lower_bound: float,
        upper_bound: float,
        open_lower: bool,
        open_upper: bool,
    ) -> AgentForecast:
        from bot.prompts import NUMERIC_FORECAST_TEMPLATE
        from bot.utils import parse_percentiles, parse_median, extract_reasoning

        user_message = NUMERIC_FORECAST_TEMPLATE.format(
            question_text=question_text,
            resolution_criteria=resolution_criteria,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            open_lower=open_lower,
            open_upper=open_upper,
            research_section=research.to_prompt_section(),
        )

        response = await agent._call_llm(
            messages=[{"role": "user", "content": user_message}],
            model=self.model,
        )

        percentiles = parse_percentiles(response)
        median = parse_median(response)
        reasoning = extract_reasoning(response)

        return AgentForecast(
            agent_name=agent.name,
            weight=agent.weight,
            final_probability=median or (lower_bound + upper_bound) / 2,
            final_reasoning=reasoning,
        )

    def _aggregate_forecasts(self, forecasts: list[AgentForecast]) -> float:
        probabilities = [f.final_probability for f in forecasts]
        weights = [f.weight for f in forecasts]
        return aggregate_logit_space(probabilities, weights)

    def _aggregate_cdfs(
        self, forecasts: list[AgentForecast], lower: float, upper: float
    ) -> list[float]:
        from bot.utils import cdf_from_percentiles

        weights = [f.weight for f in forecasts]
        total_weight = sum(weights)

        aggregated = [0.0] * 201
        for forecast in forecasts:
            pass

        return aggregated

    def _aggregate_multiple_choice(
        self, forecasts: list[AgentForecast], options: list[str]
    ) -> dict[str, float]:
        return {opt: 1.0 / len(options) for opt in options}

    def _summarize_reasoning(self, forecasts: list[AgentForecast]) -> str:
        sections = []
        for f in forecasts:
            sections.append(f"**{f.agent_name}**: {f.final_reasoning[:200]}...")
        return "\n\n".join(sections)


async def forecast_question(
    question: dict,
    use_peer_review: bool = True,
    use_adj: bool = True,
    use_asknews: bool = True,
    use_perplexity: bool = True,
) -> CommitteeResult:
    forecaster = Forecaster(use_peer_review=use_peer_review)

    q_type = question.get("type", "binary")
    q_id = question["id"]
    q_text = question["title"]
    criteria = question.get("resolution_criteria", "")

    if q_type == "binary":
        return await forecaster.forecast_binary(
            question_id=q_id,
            question_text=q_text,
            resolution_criteria=criteria,
            use_adj=use_adj,
            use_asknews=use_asknews,
            use_perplexity=use_perplexity,
        )
    elif q_type == "numeric":
        return await forecaster.forecast_numeric(
            question_id=q_id,
            question_text=q_text,
            resolution_criteria=criteria,
            lower_bound=question.get("lower_bound", 0),
            upper_bound=question.get("upper_bound", 1),
            open_lower=question.get("open_lower", False),
            open_upper=question.get("open_upper", False),
            use_adj=use_adj,
            use_asknews=use_asknews,
            use_perplexity=use_perplexity,
        )
    elif q_type == "multiple_choice":
        return await forecaster.forecast_multiple_choice(
            question_id=q_id,
            question_text=q_text,
            resolution_criteria=criteria,
            options=question.get("options", []),
            use_adj=use_adj,
            use_asknews=use_asknews,
            use_perplexity=use_perplexity,
        )
    else:
        raise ValueError(f"Unknown question type: {q_type}")
