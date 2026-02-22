"""Research module for multi-source information retrieval."""

from .adj_client import AdjClient
from .integrated_search import integrated_research, get_adj_prior_section

__all__ = [
    "AdjClient",
    "integrated_research",
    "get_adj_prior_section",
]
