"""ADJ Political Index API client with caching.

API Base: https://v2.api.adj.news/api/v1
"""

import asyncio
import hashlib
import time
from typing import Optional, Any
import aiohttp
import logging

from config import config

logger = logging.getLogger(__name__)


class ResponseCache:
    """In-memory cache with TTL for API response deduplication."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        self.cache: dict[str, dict[str, Any]] = {}
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._lock = asyncio.Lock()

    def _hash(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()

    async def get(self, key: str) -> Optional[Any]:
        """Get cached response if not expired."""
        async with self._lock:
            hashed = self._hash(key)
            if hashed in self.cache:
                entry = self.cache[hashed]
                if time.time() - entry["timestamp"] < self.ttl:
                    logger.debug(f"Cache HIT for {key[:50]}...")
                    return entry["data"]
                else:
                    del self.cache[hashed]
            logger.debug(f"Cache MISS for {key[:50]}...")
            return None

    async def set(self, key: str, data: Any) -> None:
        """Cache response with timestamp."""
        async with self._lock:
            if len(self.cache) >= self.max_size:
                # Remove oldest entries
                sorted_keys = sorted(
                    self.cache.keys(), key=lambda k: self.cache[k]["timestamp"]
                )
                for old_key in sorted_keys[: len(self.cache) // 4]:
                    del self.cache[old_key]

            hashed = self._hash(key)
            self.cache[hashed] = {"data": data, "timestamp": time.time()}
            logger.debug(f"Cache SET for {key[:50]}...")

    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            self.cache.clear()
            logger.info("Cache cleared")


class AdjClient:
    """REST client for ADJ Political Index API.

    Provides access to:
    - Markets (Kalshi, Polymarket prediction contracts)
    - Events & Outcomes (elections, contests)
    - Indices (UPFI weighted composites)
    - Reference Rates (cross-platform benchmarks)
    - Semantic Search (natural language queries)
    """

    BASE_URL = "https://v2.api.adj.news/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl: int = 300,
        timeout: int = 30,
    ):
        self.api_key = api_key or config.adj_api_key
        self.cache = ResponseCache(ttl_seconds=cache_ttl)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "AdjClient":
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    def _get_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        json_data: Optional[dict] = None,
        use_cache: bool = True,
    ) -> Any:
        """Make HTTP request with caching."""
        cache_key = f"{method}:{endpoint}:{params}:{json_data}"

        if use_cache:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                return cached

        url = f"{self.BASE_URL}{endpoint}"
        headers = {}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        session = self._get_session()

        try:
            async with session.request(
                method,
                url,
                params=params,
                json=json_data,
                headers=headers,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if use_cache:
                        await self.cache.set(cache_key, data)
                    return data
                elif response.status == 404:
                    logger.warning(f"ADJ API 404: {endpoint}")
                    return None
                else:
                    text = await response.text()
                    logger.error(f"ADJ API error {response.status}: {text[:200]}")
                    raise Exception(f"ADJ API error {response.status}: {text[:200]}")
        except asyncio.TimeoutError:
            logger.error(f"ADJ API timeout for {endpoint}")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"ADJ API client error: {e}")
            raise

    # ==================== SEARCH ====================

    async def search(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10,
        min_similarity: float = 0.0,
        max_similarity: float = 1.0,
    ) -> dict:
        """Unified semantic search across all entity types.

        Args:
            query: Natural language search query
            entity_type: Filter by 'market', 'event', 'outcome', 'index', 'rate'
            limit: Max results (default 10)
            min_similarity: Minimum similarity threshold (0.0-1.0)
            max_similarity: Maximum similarity threshold (0.0-1.0)

        Returns:
            SearchResult with matched entities
        """
        params = {
            "q": query,
            "limit": limit,
            "min_similarity": min_similarity,
            "max_similarity": max_similarity,
        }
        if entity_type:
            params["type"] = entity_type

        return await self._request("GET", "/search", params=params)

    # ==================== MARKETS ====================

    async def list_markets(
        self,
        platform: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List markets with optional filtering.

        Args:
            platform: Filter by 'kalshi' or 'polymarket'
            status: Filter by 'active', 'finalized', 'cancelled'
            limit: Max results
        """
        params: dict[str, Any] = {"limit": limit}
        if platform:
            params["platform"] = platform
        if status:
            params["status"] = status

        return await self._request("GET", "/markets", params=params)

    async def get_market(self, market_id: str) -> Optional[dict]:
        """Get market details by ID.

        Args:
            market_id: Market ticker or ID (e.g., "SENATECO-26-R")
        """
        return await self._request("GET", f"/markets/{market_id}")

    async def get_market_prices(
        self,
        market_id: str,
        limit: int = 100,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1day",
    ) -> list[dict]:
        """Get price history for a market.

        Args:
            market_id: Market ticker or ID
            limit: Max data points
            start: Start time (ISO 8601)
            end: End time (ISO 8601)
            interval: '1min', '5min', '1hour', '1day', 'daily'
        """
        params = {"limit": limit, "interval": interval}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        return await self._request("GET", f"/markets/{market_id}/prices", params=params)

    async def get_market_trades(
        self,
        market_id: str,
        limit: int = 100,
    ) -> list[dict]:
        """Get recent trades for a market."""
        return await self._request(
            "GET", f"/markets/{market_id}/trades", params={"limit": limit}
        )

    async def get_similar_markets(
        self,
        market_id: str,
        limit: int = 10,
        min_similarity: float = 0.0,
    ) -> list[dict]:
        """Find markets semantically similar to a given market."""
        return await self._request(
            "GET",
            f"/markets/{market_id}/similar",
            params={"limit": limit, "min_similarity": min_similarity},
        )

    # ==================== EVENTS & OUTCOMES ====================

    async def list_events(
        self,
        category: Optional[str] = None,
    ) -> list[dict]:
        """List all events.

        Args:
            category: Filter by 'presidential', 'senate', 'house', 'governor', 'mayoral'
        """
        params = {}
        if category:
            params["category"] = category

        return await self._request("GET", "/events", params=params)

    async def get_event(self, event_id: str) -> Optional[dict]:
        """Get event details with all outcomes."""
        return await self._request("GET", f"/events/{event_id}")

    async def get_outcome(self, outcome_id: str) -> Optional[dict]:
        """Get outcome details with all linked markets.

        This is the key endpoint for cross-platform aggregation.
        """
        return await self._request("GET", f"/outcomes/{outcome_id}")

    # ==================== INDICES ====================

    async def list_indices(self) -> list[dict]:
        """List all available indices."""
        return await self._request("GET", "/indices")

    async def get_index(self, index_id: str) -> Optional[dict]:
        """Get index details."""
        return await self._request("GET", f"/indices/{index_id}")

    async def get_index_constituents(self, index_id: str) -> list[dict]:
        """Get index constituents with weights and prices."""
        return await self._request("GET", f"/indices/{index_id}/constituents")

    async def get_index_prices(
        self,
        index_id: str,
        limit: int = 100,
        interval: str = "raw",
    ) -> list[dict]:
        """Get index price history.

        Args:
            index_id: Index identifier (e.g., "upfi", "upfi_senate_kalshi")
            limit: Max data points
            interval: 'raw', '1day', 'daily'
        """
        return await self._request(
            "GET",
            f"/indices/{index_id}/prices",
            params={"limit": limit, "interval": interval},
        )

    async def get_index_settlement(
        self,
        index_id: str,
        limit: int = 30,
    ) -> list[dict]:
        """Get daily settlement values (12:00 UTC snapshots)."""
        return await self._request(
            "GET", f"/indices/{index_id}/settlement", params={"limit": limit}
        )

    # ==================== REFERENCE RATES ====================

    async def list_rates(self) -> list[dict]:
        """List all reference rates."""
        return await self._request("GET", "/rates")

    async def get_rate(self, rate_id: str) -> Optional[dict]:
        """Get reference rate with contributing sources."""
        return await self._request("GET", f"/rates/{rate_id}")

    async def get_rate_prices(
        self,
        rate_id: str,
        limit: int = 100,
        interval: str = "raw",
    ) -> list[dict]:
        """Get reference rate price history."""
        return await self._request(
            "GET",
            f"/rates/{rate_id}/prices",
            params={"limit": limit, "interval": interval},
        )

    # ==================== CONVENIENCE METHODS ====================

    async def find_related_markets(
        self, question: str, threshold: float = 0.5
    ) -> list[dict]:
        """Find markets related to a forecasting question.

        Args:
            question: The forecasting question text
            threshold: Minimum similarity to include (0.0-1.0)

        Returns:
            List of related markets with similarity scores
        """
        result = await self.search(
            query=question,
            entity_type="market",
            limit=10,
            min_similarity=threshold,
        )

        if result and "results" in result:
            return result["results"]
        return []

    async def get_market_prior(
        self,
        question: str,
        direct_match_threshold: float = 0.7,
    ) -> tuple[Optional[float], str]:
        """Get prediction market prior for a question.

        Args:
            question: The forecasting question
            direct_match_threshold: Similarity threshold for "direct match"

        Returns:
            Tuple of (prior_probability or None, formatted_section)
        """
        results = await self.find_related_markets(question)

        if not results:
            return None, ""

        # Check for direct match
        for match in results:
            if match.get("similarity", 0) >= direct_match_threshold:
                market_id = match.get("entity_id")
                if market_id:
                    market = await self.get_market(market_id)
                    if market and market.get("probability") is not None:
                        prob = market["probability"]
                        section = f"""PREDICTION MARKET PRIOR (Direct Match):
Platform: {market.get("platform", "unknown")}
Question: {market.get("question", "N/A")}
Current Probability: {prob}%
Volume: ${market.get("volume", 0):,.0f}
Similarity: {match.get("similarity", 0):.0%}

This is a DIRECT MATCH. Use this as your primary anchor."""
                        return prob / 100.0, section

        # No direct match, format similar markets
        section_lines = ["RELATED PREDICTION MARKETS (for reference):"]
        for match in results[:5]:
            market_id = match.get("entity_id")
            if market_id:
                market = await self.get_market(market_id)
                if market:
                    sim = match.get("similarity", 0)
                    section_lines.append(
                        f"- [{sim:.0%} similar] {market.get('question', 'N/A')}"
                    )
                    if market.get("probability") is not None:
                        section_lines.append(
                            f"  {market.get('platform', 'unknown')}: {market['probability']}%"
                        )

        return None, "\n".join(section_lines)


# Singleton instance for easy access
_adj_client: Optional[AdjClient] = None


def get_adj_client() -> AdjClient:
    """Get or create the global ADJ client instance."""
    global _adj_client
    if _adj_client is None:
        _adj_client = AdjClient()
    return _adj_client
