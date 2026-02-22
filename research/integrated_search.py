"""Integrated multi-source research for forecasting.

Combines:
1. ADJ prediction market data (priors, similar markets)
2. AskNews (recent news summaries)
3. Perplexity (deep research with citations)
"""

import asyncio
import logging
from typing import Optional

from config import config
from .adj_client import AdjClient, get_adj_client

logger = logging.getLogger(__name__)


async def get_adj_prior_section(
    question: str,
    adj_client: Optional[AdjClient] = None,
    direct_match_threshold: float = 0.7,
) -> tuple[Optional[float], str]:
    """Get ADJ prediction market prior for a question.

    Args:
        question: The forecasting question text
        adj_client: ADJ client instance (uses global if None)
        direct_match_threshold: Similarity for "direct match"

    Returns:
        Tuple of (prior_probability or None, formatted_section)
    """
    client = adj_client or get_adj_client()

    try:
        return await client.get_market_prior(question, direct_match_threshold)
    except Exception as e:
        logger.error(f"ADJ prior lookup failed: {e}")
        return None, ""


async def get_asknews_summary(question: str) -> str:
    """Get recent news summary from AskNews.

    Args:
        question: The forecasting question

    Returns:
        Formatted news summary or empty string on failure
    """
    if not config.asknews_client_id or not config.asknews_secret:
        logger.warning("AskNews credentials not configured, skipping")
        return ""

    try:
        from asknews_sdk import AsyncAskNewsSDK

        async with AsyncAskNewsSDK(
            client_id=config.asknews_client_id,
            client_secret=config.asknews_secret,
            scopes={"news"},
        ) as ask:
            # Get recent news
            hot_response = await ask.news.search_news(
                query=question,
                n_articles=5,
                return_type="both",
                strategy="latest news",
            )

            # Get historical context
            historical_response = await ask.news.search_news(
                query=question,
                n_articles=8,
                return_type="both",
                strategy="news knowledge",
            )

            # Format results
            sections = []

            hot_articles = getattr(hot_response, "as_dicts", [])
            if hot_articles:
                sections.append("### Recent News (Past 48 Hours)")
                for article in hot_articles[:5]:
                    title = getattr(article, "eng_title", str(article))
                    summary = getattr(article, "summary", "")
                    pub_date = getattr(article, "pub_date", None)
                    date_str = (
                        pub_date.strftime("%b %d, %Y")
                        if pub_date and hasattr(pub_date, "strftime")
                        else ""
                    )
                    source = getattr(article, "source_id", "")
                    url = getattr(article, "article_url", "")

                    sections.append(f"**{title}** ({date_str})")
                    if summary:
                        sections.append(
                            summary[:300] + ("..." if len(summary) > 300 else "")
                        )
                    if source and url:
                        sections.append(f"Source: [{source}]({url})")
                    sections.append("")

            historical_articles = getattr(historical_response, "as_dicts", [])
            if historical_articles:
                sections.append("### Historical Context")
                for article in historical_articles[:5]:
                    title = getattr(article, "eng_title", str(article))
                    summary = getattr(article, "summary", "")

                    sections.append(f"**{title}**")
                    if summary:
                        sections.append(
                            summary[:200] + ("..." if len(summary) > 200 else "")
                        )
                    sections.append("")

            return "\n".join(sections)

    except Exception as e:
        logger.error(f"AskNews lookup failed: {e}")
        return ""


async def get_perplexity_research(question: str) -> str:
    """Get deep research from Perplexity.

    Args:
        question: The forecasting question

    Returns:
        Research summary with citations or empty string on failure
    """
    if not config.perplexity_api_key:
        logger.warning("Perplexity API key not configured, skipping")
        return ""

    try:
        import aiohttp

        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {config.perplexity_api_key}",
            "content-type": "application/json",
        }
        payload = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant to a superforecaster. "
                        "Generate a concise but detailed rundown of the most relevant "
                        "information for forecasting. Include citations with URLs. "
                        "Check prediction markets (Metaculus, Kalshi, Polymarket, Polymarket) for relevant forecasts. "
                        "Do not produce forecasts yourself."
                    ),
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
        }

        timeout = aiohttp.ClientTimeout(total=60)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    content = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    return content
                else:
                    text = await response.text()
                    logger.error(
                        f"Perplexity API error {response.status}: {text[:200]}"
                    )
                    return ""

    except Exception as e:
        logger.error(f"Perplexity lookup failed: {e}")
        return ""


async def integrated_research(
    question: str,
    adj_client: Optional[AdjClient] = None,
    include_asknews: bool = True,
    include_perplexity: bool = True,
) -> str:
    """Run integrated research from multiple sources.

    Combines:
    1. ADJ prediction market data (always included if available)
    2. AskNews recent news (optional)
    3. Perplexity deep research (optional)

    Args:
        question: The forecasting question
        adj_client: ADJ client instance
        include_asknews: Whether to include AskNews
        include_perplexity: Whether to include Perplexity

    Returns:
        Compiled research context string
    """
    sections = []

    # Always try ADJ first (most valuable for forecasting)
    adj_prior, adj_section = await get_adj_prior_section(question, adj_client)
    if adj_section:
        sections.append(adj_section)

    # Run AskNews and Perplexity in parallel
    tasks = []
    if include_asknews:
        tasks.append(get_asknews_summary(question))
    else:
        tasks.append(asyncio.sleep(0))  # Placeholder

    if include_perplexity:
        tasks.append(get_perplexity_research(question))
    else:
        tasks.append(asyncio.sleep(0))  # Placeholder

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process AskNews result
    if include_asknews and len(results) >= 1:
        asknews_result = results[0]
        if isinstance(asknews_result, str) and asknews_result:
            sections.append(asknews_result)
        elif isinstance(asknews_result, Exception):
            logger.error(f"AskNews error: {asknews_result}")

    # Process Perplexity result
    if include_perplexity and len(results) >= 2:
        perplexity_result = results[1]
        if isinstance(perplexity_result, str) and perplexity_result:
            sections.append("### Deep Research\n" + perplexity_result)
        elif isinstance(perplexity_result, Exception):
            logger.error(f"Perplexity error: {perplexity_result}")

    return "\n\n".join(sections)
