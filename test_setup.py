#!/usr/bin/env python3
"""Test script to verify Vox bot setup and configuration.

Run with: python test_setup.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from config import config

        print("  ✓ config")
    except Exception as e:
        print(f"  ✗ config: {e}")
        return False

    try:
        from models import (
            Market,
            Outcome,
            ResearchContext,
            AgentForecast,
            CommitteeResult,
        )

        print("  ✓ models")
    except Exception as e:
        print(f"  ✗ models: {e}")
        return False

    try:
        from research import AdjClient, integrated_research

        print("  ✓ research")
    except Exception as e:
        print(f"  ✗ research: {e}")
        return False

    try:
        from bot import LLMAgent, create_committee, Forecaster

        print("  ✓ bot")
    except Exception as e:
        print(f"  ✗ bot: {e}")
        return False

    return True


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    from config import config

    print(f"  ADJ API Base: {config.adj_api_base}")
    print(f"  Committee Size: {config.committee_size}")
    print(f"  Cache TTL: {config.cache_ttl_seconds}s")

    has_metaculus = bool(config.metaculus_token)
    has_adj = bool(config.adj_api_key)

    print(f"  Metaculus Token: {'✓ set' if has_metaculus else '✗ not set'}")
    print(f"  ADJ API Key: {'✓ set' if has_adj else '✗ not set'}")

    if not has_metaculus:
        print("\n  ⚠ Warning: METACULUS_TOKEN not set - bot cannot submit forecasts")

    return True


def test_committee():
    """Test committee creation."""
    print("\nTesting committee creation...")
    from bot import create_committee, AGENT_CONFIGS

    agents = create_committee()

    print(f"  Created {len(agents)} agents:")
    for agent in agents:
        print(f"    - {agent.name} (weight: {agent.weight})")

    expected = len(AGENT_CONFIGS)
    if len(agents) != expected:
        print(f"  ✗ Expected {expected} agents, got {len(agents)}")
        return False

    return True


async def test_adj_client():
    """Test ADJ client initialization."""
    print("\nTesting ADJ client...")
    from research import AdjClient

    client = AdjClient()
    print(f"  ✓ AdjClient created")
    print(f"  Base URL: {client.BASE_URL}")

    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Vox Bot Setup Test")
    print("=" * 50)

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_config()
    all_passed &= test_committee()
    all_passed &= asyncio.run(test_adj_client())

    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed!")
        print("\nNext steps:")
        print("  1. Copy .env.template to .env")
        print("  2. Add your API keys to .env")
        print("  3. Run: poetry install")
        print("  4. Run: poetry run python main.py --mode test_questions")
    else:
        print("✗ Some tests failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
