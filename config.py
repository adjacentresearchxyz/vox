"""Configuration and environment variables for Vox forecasting bot."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """Centralized configuration loaded from environment variables."""

    # Metaculus
    metaculus_token: str = ""
    metaculus_llm_proxy: str = (
        "https://llm-proxy.metaculus.com/proxy/anthropic/v1/messages/"
    )

    # ADJ API
    adj_api_base: str = "https://v2.api.adj.news/api/v1"
    adj_api_key: str = ""

    # Research providers
    asknews_client_id: str = ""
    asknews_secret: str = ""
    perplexity_api_key: str = ""
    exa_api_key: str = ""
    openrouter_api_key: str = ""

    # Anthropic (for direct calls if needed)
    anthropic_api_key: str = ""

    # Cache settings
    cache_ttl_seconds: int = 300
    cache_max_size: int = 1000

    # Agent settings
    committee_size: int = 5
    agent_timeout: int = 120
    agent_max_retries: int = 3

    # Rate limiting
    max_concurrent_questions: int = 2
    requests_per_second: float = 1.0

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            metaculus_token=os.getenv("METACULUS_TOKEN", ""),
            adj_api_key=os.getenv("ADJ_API_KEY", ""),
            asknews_client_id=os.getenv("ASKNEWS_CLIENT_ID", ""),
            asknews_secret=os.getenv("ASKNEWS_SECRET", ""),
            perplexity_api_key=os.getenv("PERPLEXITY_API_KEY", ""),
            exa_api_key=os.getenv("EXA_API_KEY", ""),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "300")),
            max_concurrent_questions=int(os.getenv("MAX_CONCURRENT_QUESTIONS", "2")),
        )


# Global config instance
config = Config.from_env()
