"""Pydantic schemas for ADJ API responses."""

from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field


class OHLC(BaseModel):
    """Open-High-Low-Close price data."""

    open: float
    high: float
    low: float
    close: float


class PricePoint(BaseModel):
    """A single price point with optional OHLC data."""

    timestamp: datetime
    price: float
    ohlc: Optional[OHLC] = None


class Trade(BaseModel):
    """A single trade record."""

    trade_id: str
    timestamp: datetime
    price: float
    count: Optional[int] = None
    side: Optional[str] = None
    volume: Optional[float] = None


class Market(BaseModel):
    """A prediction market from Kalshi, Polymarket, etc."""

    market_id: str
    ticker: Optional[str] = None
    platform: str
    question: str
    description: Optional[str] = None
    probability: Optional[float] = None
    volume: Optional[float] = None
    liquidity: Optional[float] = None
    status: str = "active"
    end_date: Optional[datetime] = None
    link: Optional[str] = None


class Outcome(BaseModel):
    """An outcome within an event, linked to multiple markets."""

    outcome_id: str
    event_id: Optional[str] = None
    event_name: Optional[str] = None
    name: str
    resolved_value: Optional[bool] = None
    markets: list[Market] = Field(default_factory=list)
    latest_price: Optional[float] = None
    price_spread: Optional[float] = None


class Event(BaseModel):
    """An election or predictable contest."""

    event_id: str
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    region: Optional[str] = None
    election_date: Optional[datetime] = None
    status: str = "active"
    outcomes: list[Outcome] = Field(default_factory=list)
    outcomes_count: int = 0


class Constituent(BaseModel):
    """A constituent market within an index."""

    ticker: str
    platform: str
    weight: float
    price: Optional[float] = None
    volume: Optional[float] = None
    contribution: Optional[float] = None


class Index(BaseModel):
    """A weighted composite of market prices (UPFI methodology)."""

    index_id: str
    name: str
    description: Optional[str] = None
    constituents_count: int = 0
    latest_price: Optional[float] = None
    price_change_1d: Optional[float] = None
    price_change_7d: Optional[float] = None
    updated_at: Optional[datetime] = None


class ReferenceRate(BaseModel):
    """Cross-platform benchmark aggregating prices for same outcome."""

    rate_id: str
    name: str
    description: Optional[str] = None
    methodology: str = "volume_weighted_average"
    sources: list[dict] = Field(default_factory=list)
    sources_count: int = 0
    latest_price: Optional[float] = None
    spread: Optional[float] = None


class SearchMatch(BaseModel):
    """A single match from semantic search."""

    entity_type: str
    entity_id: str
    similarity: float
    name: str
    description: Optional[str] = None
    latest_price: Optional[float] = None
    category: Optional[str] = None
    platform: Optional[str] = None


class SearchResult(BaseModel):
    """Result from unified search across entities."""

    query: str
    type: Optional[str] = None
    total_results: int
    results: list[SearchMatch] = Field(default_factory=list)


# Request/Response models for internal use


class ResearchContext(BaseModel):
    """Compiled research context for a question."""

    question_text: str
    adj_markets: list[dict] = Field(default_factory=list)
    adj_prior: Optional[float] = None
    adj_prior_section: str = ""
    asknews_summary: str = ""
    perplexity_research: str = ""
    compiled_at: datetime = Field(default_factory=datetime.now)

    def to_prompt_section(self) -> str:
        """Format research as a prompt section."""
        sections = []

        if self.adj_prior_section:
            sections.append(f"### Prediction Market Data\n{self.adj_prior_section}")

        if self.asknews_summary:
            sections.append(f"### Recent News\n{self.asknews_summary}")

        if self.perplexity_research:
            sections.append(f"### Deep Research\n{self.perplexity_research}")

        return "\n\n".join(sections)


class AgentForecast(BaseModel):
    """A forecast from a single agent."""

    agent_name: str
    weight: float
    initial_reasoning: str = ""
    initial_probability: Optional[float] = None
    peer_critique: str = ""
    final_probability: float = 0.5
    final_reasoning: str = ""


class CommitteeResult(BaseModel):
    """Final result from committee forecasting."""

    question_id: int
    question_text: str
    question_type: str
    research_context: str
    agent_forecasts: list[AgentForecast] = Field(default_factory=list)
    final_probability: float = 0.5
    final_cdf: Optional[list[float]] = None
    final_option_probs: Optional[dict[str, float]] = None
    reasoning_summary: str = ""
