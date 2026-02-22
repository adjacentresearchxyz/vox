"""Bot module - agents, prompts, and forecasting utilities."""

from bot.agents import LLMAgent, create_committee, AGENT_CONFIGS
from bot.utils import parse_probability, normalize_probability, weighted_average
from bot.forecaster import Forecaster, forecast_question

__all__ = [
    "LLMAgent",
    "create_committee",
    "AGENT_CONFIGS",
    "parse_probability",
    "normalize_probability",
    "weighted_average",
    "Forecaster",
    "forecast_question",
]
