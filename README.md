# Vox - Multi-Agent Committee Forecasting Bot

Vox is a sophisticated forecasting bot for the [Metaculus AI Tournament](https://www.metaculus.com/aib/) that uses a **5-agent committee architecture** with prediction market priors from the ADJ Political Index API.

## Key Features

- **Multi-Agent Committee**: 5 specialized forecasting agents with distinct personas
- **Prediction Market Priors**: Integrates ADJ API for real-time market prices as Bayesian priors
- **Peer Review Workflow**: Agents critique and revise forecasts before aggregation
- **Multi-Source Research**: Combines ADJ markets, AskNews, and Perplexity
- **Logit-Space Aggregation**: Final forecasts aggregated in probability space for better calibration

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Vox Forecaster                          │
├─────────────────────────────────────────────────────────────────┤
│  Research Layer                                                 │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                    │
│  │  ADJ API  │  │  AskNews  │  │ Perplexity│                    │
│  │ (priors)  │  │  (news)   │  │ (research)│                    │
│  └───────────┘  └───────────┘  └───────────┘                    │
├─────────────────────────────────────────────────────────────────┤
│  Agent Committee (5 agents with peer review)                    │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │ base_rate_analyst│  │   market_prior   │                     │
│  │    (weight 1.0)  │  │   (weight 1.5)   │                     │
│  └──────────────────┘  └──────────────────┘                     │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │ status_quo_anchor│  │ devils_advocate  │                     │
│  │    (weight 1.0)  │  │   (weight 1.0)   │                     │
│  └──────────────────┘  └──────────────────┘                     │
│  ┌──────────────────┐                                           │
│  │   synthesizer    │                                           │
│  │   (weight 1.5)   │                                           │
│  └──────────────────┘                                           │
├─────────────────────────────────────────────────────────────────┤
```

## Agent Personas

| Agent | Weight | Focus |
|-------|--------|-------|
| **base_rate_analyst** | 1.0 | Historical frequency, reference class forecasting |
| **market_prior** | 1.5 | Prediction market prices as Bayesian priors |
| **status_quo_anchor** | 1.0 | Conservative default, regression to mean |
| **devils_advocate** | 1.0 | Challenge consensus, find overlooked risks |
| **synthesizer** | 1.5 | Combine perspectives, resolve disagreements |

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/your-username/vox.git
cd vox
poetry install
```

### 2. Configure Environment

```bash
cp .env.template .env
# Edit .env with your API keys
```

Required:
- `METACULUS_TOKEN` - Your bot's Metaculus API token

Recommended:
- `ADJ_API_KEY` - For prediction market priors
- `ASKNEWS_CLIENT_ID` + `ASKNEWS_SECRET` - For news research
- `PERPLEXITY_API_KEY` - For deep research

### 3. Run the Bot

```bash
# Test on example questions
poetry run python main.py --mode test_questions

# Run on tournament questions
poetry run python main.py --mode tournament

# Use simpler single-agent mode (faster)
poetry run python main.py --simple --mode test_questions

# Run without submitting to Metaculus
poetry run python main.py --no-publish --mode test_questions
```

### 4. Verify Setup

```bash
poetry run python test_setup.py
```

## ADJ API Integration

Vox uses the [ADJ Political Index API](https://v2.api.adj.news) for prediction market data:

- **Semantic Search**: Find markets related to forecasting questions
- **Direct Match Priors**: When similarity > 70%, use market price as primary prior
- **Cross-Platform Aggregation**: Reference rates combine Kalshi, Polymarket, etc.
- **Caching**: 5-minute TTL with request deduplication

Example:
```python
from research import AdjClient

async with AdjClient() as client:
    # Find related markets
    results = await client.search("Will Trump win 2024?", entity_type="market")
    
    # Get market prior for a question
    prior, section = await client.get_market_prior(question)
```

## Forecasting Workflow

1. **Research Phase**
   - Query ADJ for prediction market priors
   - Fetch recent news from AskNews
   - Get deep research from Perplexity
   - All sources run in parallel

2. **Initial Forecasts**
   - All 5 agents generate independent forecasts
   - Each follows their persona's approach

3. **Peer Review**
   - Each agent reviews another agent's forecast
   - Provides constructive critique

4. **Revision**
   - Agents revise based on peer feedback
   - May maintain or adjust their forecast

5. **Aggregation**
   - Forecasts combined in logit space
   - Weighted by agent weights (market_prior, synthesizer = 1.5x)

## GitHub Actions Automation

The included workflow runs every 30 minutes:

1. Fork this repository
2. Go to Settings → Secrets and variables → Actions
3. Add your API keys as repository secrets:
   - `METACULUS_TOKEN`
   - `ADJ_API_KEY`
   - `ASKNEWS_CLIENT_ID`, `ASKNEWS_SECRET`
   - `PERPLEXITY_API_KEY`
4. Enable Actions in the Actions tab

## Getting API Keys

### Metaculus Token
1. Go to https://metaculus.com/aib
2. Create a bot account
3. Click "Show My Token"

### ADJ API Key
Contact the ADJ team for API access to prediction market data.

### AskNews (Free for Tournament)
1. Create account at https://my.asknews.app
2. Email `rob [at] asknews [.app]` with your bot username
3. Generate credentials at https://my.asknews.app/en/settings/api-credentials

### Perplexity
1. Create account at https://perplexity.ai
2. Go to Settings → Account → API
3. Generate API key and add credits

### Metaculus LLM Proxy (Free Anthropic/OpenAI Credits)
Email `ben [at] metaculus [.com]` with:
- Your bot username
- Description of your bot
- Requested models and budget

## Configuration Options

Key settings in `config.py`:

```python
cache_ttl_seconds: int = 300        # Cache TTL for API responses
committee_size: int = 5             # Number of agents
agent_timeout: int = 120            # LLM call timeout
max_concurrent_questions: int = 2   # Parallel question limit
```

## Extending the Bot

### Add a New Agent

1. Add system prompt to `bot/prompts.py`:
```python
NEW_AGENT_SYSTEM = """Your agent's approach..."""
```

2. Register in `bot/agents.py`:
```python
AGENT_CONFIGS["new_agent"] = AgentConfig(
    name="new_agent",
    system_prompt=AGENT_PROMPTS["new_agent"],
    weight=1.0,
)
```

### Add a New Research Source

1. Create client in `research/new_source.py`
2. Integrate in `research/integrated_search.py`
3. Add to `integrated_research()` function

## Ideas for Improvement

- **Fine-tune on Metaculus data**: Train calibration on historical forecasts
- **Dynamic agent weights**: Adjust weights based on past accuracy
- **Question decomposition**: Break complex questions into simpler ones
- **Monte Carlo simulations**: Combine with scenario modeling
- **Extremize predictions**: Apply calibration transforms based on bot history

## Resources

- [Metaculus AI Tournament](https://www.metaculus.com/aib/)
- [forecasting-tools package](https://github.com/Metaculus/forecasting-tools)
- [Metaculus Discord](https://discord.com/invite/NJgCC2nDfh) - #build-a-forecasting-bot

## License

MIT License - See LICENSE file for details.
