"""Vox forecasting bot with multi-agent committee architecture.

Uses:
- 5-agent committee with base-rate focused personas
- ADJ prediction market priors via REST API
- AskNews and Perplexity for research
- Metaculus LLM proxy for Anthropic models
"""

import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
)

from config import config
from bot.forecaster import Forecaster, forecast_question
from bot.utils import weighted_average
from models.schemas import CommitteeResult, ResearchContext
from research.integrated_search import integrated_research

logger = logging.getLogger(__name__)


class VoxForecaster(ForecastBot):
    """Multi-agent committee forecasting bot.

    Overrides the research and forecast methods to use:
    - ADJ prediction market priors
    - 5-agent committee with peer review
    """

    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._committee_forecaster = Forecaster(
            use_peer_review=True,
            model="claude-sonnet-4-20250514",
        )

    async def run_research(self, question: MetaculusQuestion) -> str:
        """Run integrated research from ADJ, AskNews, and Perplexity."""
        async with self._concurrency_limiter:
            research = await integrated_research(
                question=question.question_text,
                include_asknews=bool(
                    config.asknews_client_id and config.asknews_secret
                ),
                include_perplexity=bool(config.perplexity_api_key),
            )
            logger.info(f"Research for {question.page_url}:\n{research[:500]}...")
            return research

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        """Forecast binary question using multi-agent committee."""
        async with self._concurrency_limiter:
            context = ResearchContext(
                question_text=question.question_text,
                adj_prior_section=research,
            )

            result = await self._committee_forecaster.forecast_binary(
                question_id=question.id,
                question_text=question.question_text,
                resolution_criteria=self._format_criteria(question),
                use_adj=True,
                use_asknews=False,
                use_perplexity=False,
            )

            logger.info(
                f"Committee forecast for {question.page_url}: "
                f"{result.final_probability:.2%}"
            )

            return ReasonedPrediction(
                prediction_value=result.final_probability,
                reasoning=result.reasoning_summary,
            )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        """Forecast multiple choice question using committee."""
        async with self._concurrency_limiter:
            result = await self._committee_forecaster.forecast_multiple_choice(
                question_id=question.id,
                question_text=question.question_text,
                resolution_criteria=self._format_criteria(question),
                options=question.options,
                use_adj=True,
                use_asknews=False,
                use_perplexity=False,
            )

            if result.final_option_probs:
                predicted_options = PredictedOptionList(
                    predicted_option_list=[
                        {"option_name": k, "probability": v}
                        for k, v in result.final_option_probs.items()
                    ]
                )
            else:
                uniform_prob = 1.0 / len(question.options)
                predicted_options = PredictedOptionList(
                    predicted_option_list=[
                        {"option_name": opt, "probability": uniform_prob}
                        for opt in question.options
                    ]
                )

            logger.info(
                f"Committee MC forecast for {question.page_url}: {result.final_option_probs}"
            )

            return ReasonedPrediction(
                prediction_value=predicted_options,
                reasoning=result.reasoning_summary,
            )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        """Forecast numeric question using committee."""
        async with self._concurrency_limiter:
            result = await self._committee_forecaster.forecast_numeric(
                question_id=question.id,
                question_text=question.question_text,
                resolution_criteria=self._format_criteria(question),
                lower_bound=question.lower_bound,
                upper_bound=question.upper_bound,
                open_lower=question.open_lower_bound,
                open_upper=question.open_upper_bound,
                use_adj=True,
                use_asknews=False,
                use_perplexity=False,
            )

            if result.final_cdf:
                numeric_dist = NumericDistribution(
                    declared_density_values=result.final_cdf
                )
            else:
                median = result.final_probability
                numeric_dist = NumericDistribution.from_central_prediction(
                    central_prediction=median,
                    question=question,
                    spread=0.2,
                )

            logger.info(
                f"Committee numeric forecast for {question.page_url}: median={result.final_probability}"
            )

            return ReasonedPrediction(
                prediction_value=numeric_dist,
                reasoning=result.reasoning_summary,
            )

    def _format_criteria(self, question: MetaculusQuestion) -> str:
        """Format resolution criteria for prompt."""
        parts = []
        if hasattr(question, "resolution_criteria") and question.resolution_criteria:
            parts.append(question.resolution_criteria)
        if hasattr(question, "fine_print") and question.fine_print:
            parts.append(question.fine_print)
        if hasattr(question, "background_info") and question.background_info:
            parts.append(f"Background: {question.background_info}")
        return "\n\n".join(parts) if parts else "No specific criteria provided."


class SimpleVoxForecaster(ForecastBot):
    """Simplified single-agent forecaster for quick testing.

    Uses ADJ research but simpler forecasting without full committee.
    """

    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        """Run integrated research."""
        async with self._concurrency_limiter:
            return await integrated_research(
                question=question.question_text,
                include_asknews=bool(config.asknews_client_id),
                include_perplexity=bool(config.perplexity_api_key),
            )

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster using base-rate thinking.

            Question: {question.question_text}

            Background: {question.background_info or "N/A"}

            Resolution Criteria: {question.resolution_criteria or "N/A"}

            Research Data:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Instructions:
            1. Consider base rates from historical data
            2. Weight prediction market prices heavily if available
            3. Consider the status quo bias
            4. Provide your probability as: Probability: XX%

            Provide your reasoning first, then the probability.
            """
        )

        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for {question.page_url}:\n{reasoning}")

        import re

        match = re.search(r"Probability:\s*([\d.]+)%", reasoning, re.IGNORECASE)
        if match:
            prob = float(match.group(1)) / 100
            prob = max(0.01, min(0.99, prob))
        else:
            prob = 0.5

        return ReasonedPrediction(prediction_value=prob, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster.

            Question: {question.question_text}

            Options: {question.options}

            Research: {research}

            Provide probabilities for each option. Format:
            Option: Probability%
            """
        )

        reasoning = await self.get_llm("default", "llm").invoke(prompt)

        import re

        probs = {}
        for opt in question.options:
            match = re.search(
                rf"{re.escape(opt)}:\s*([\d.]+)%", reasoning, re.IGNORECASE
            )
            if match:
                probs[opt] = float(match.group(1)) / 100

        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        else:
            probs = {opt: 1.0 / len(question.options) for opt in question.options}

        predicted_options = PredictedOptionList(
            predicted_option_list=[
                {"option_name": k, "probability": v} for k, v in probs.items()
            ]
        )

        return ReasonedPrediction(
            prediction_value=predicted_options, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster.

            Question: {question.question_text}

            Range: {question.lower_bound} to {question.upper_bound}

            Research: {research}

            Provide your median estimate. Format: Median: XX
            """
        )

        reasoning = await self.get_llm("default", "llm").invoke(prompt)

        import re

        match = re.search(r"Median:\s*([\d.]+)", reasoning, re.IGNORECASE)
        if match:
            median = float(match.group(1))
        else:
            median = (question.lower_bound + question.upper_bound) / 2

        numeric_dist = NumericDistribution.from_central_prediction(
            central_prediction=median,
            question=question,
            spread=0.2,
        )

        return ReasonedPrediction(prediction_value=numeric_dist, reasoning=reasoning)


def create_bot(use_committee: bool = True, **kwargs) -> ForecastBot:
    """Create forecaster bot instance.

    Args:
        use_committee: If True, use full multi-agent committee.
                      If False, use simpler single-agent forecaster.
    """
    defaults = {
        "research_reports_per_question": 1,
        "predictions_per_research_report": 1,
        "use_research_summary_to_forecast": True,
        "publish_reports_to_metaculus": True,
        "folder_to_save_reports_to": "forecasts",
        "skip_previously_forecasted_questions": True,
        "llms": {
            "default": GeneralLlm(
                model="anthropic/claude-sonnet-4-20250514",
                temperature=0.3,
                timeout=120,
                allowed_tries=3,
            ),
        },
    }
    defaults.update(kwargs)

    if use_committee:
        return VoxForecaster(**defaults)
    else:
        return SimpleVoxForecaster(**defaults)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(description="Run Vox forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Run mode (default: tournament)",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple single-agent forecaster instead of committee",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Don't publish forecasts to Metaculus",
    )
    args = parser.parse_args()

    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    bot = create_bot(
        use_committee=not args.simple,
        publish_reports_to_metaculus=not args.no_publish,
    )

    if run_mode == "tournament":
        main_reports = asyncio.run(
            bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID,
                return_exceptions=True,
            )
        )
        forecast_reports = main_reports

    elif run_mode == "metaculus_cup":
        bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            bot.forecast_on_tournament(
                MetaculusApi.CURRENT_METACULUS_CUP_ID,
                return_exceptions=True,
            )
        )

    elif run_mode == "test_questions":
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/36248/who-will-be-the-first-to-leave-the-trump-cabinet/",
        ]
        bot.skip_previously_forecasted_questions = False
        questions = [MetaculusApi.get_question_by_url(url) for url in EXAMPLE_QUESTIONS]
        forecast_reports = asyncio.run(
            bot.forecast_questions(questions, return_exceptions=True)
        )

    ForecastBot.log_report_summary(forecast_reports)
