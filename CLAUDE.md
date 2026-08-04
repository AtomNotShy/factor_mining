# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Factor Mining System is a quantitative trading platform for US equity ETF strategy backtesting and live trading, built on Interactive Brokers. It features a unified event-driven architecture with Freqtrade-style strategy support (v2/v3 protocols).

## Build / Lint / Test Commands

### Backend (Python)

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Run API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Lint (line-length: 100)
black src/ && flake8 src/ && mypy src/

# Test
pytest                                            # All tests
pytest tests/test_file.py                         # Single file
pytest tests/test_file.py::TestClass::test_method # Single test
pytest --cov=src --cov-report=term-missing        # With coverage
```

### Frontend

```bash
cd frontend
pnpm install
pnpm dev      # Port 3000
pnpm build    # tsc && vite build
pnpm lint     # ESLint
```

### CLI Backtest

```bash
python3 src/main.py backtest -c config.yaml
```

## Architecture

### Event-Driven Flow (v3)

```
MarketEvent → Strategy.generate_signals() → SignalEvent
    ↓
SignalEvent → _check_risk() → RiskEvent
    ↓
RiskEvent → OrderCreatedEvent
    ↓
OrderCreatedEvent → ExecutionProvider.submit_order() → OrderFilledEvent
    ↓
OrderFilledEvent → PortfolioState.update() → 记录成交
```

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/api/routers/` | FastAPI endpoints with `/api/v1` prefix |
| `src/core/types.py` | Domain models: Signal, OrderIntent, Fill, PortfolioState |
| `src/core/events/` | UnifiedEventEngine for event-driven processing |
| `src/data/collectors/` | Data sources: IB (default), Polygon, CCXT |
| `src/data/providers/` | DataFeed interfaces: HistoricalDataFeed, DataFeedFactory |
| `src/evaluation/backtesting/` | Dual engines: `engine.py`, `freqtrade_engine.py` |
| `src/execution/providers/` | Execution providers: SimulatedExecutionProvider, IB execution |
| `src/strategies/base/` | Strategy base classes, templates, 100+ indicators |
| `src/strategies/user_strategies/` | User-created strategies (auto-registered) |

### Strategy System

Two protocols supported:

**v2 (Strategy base class):** Uses `generate_signals()` method

**v3 (FreqtradeStrategy):** Uses Freqtrade-style lifecycle:
- `populate_indicators()` → `populate_entry_trend()` → `populate_exit_trend()`
- Optional callbacks: `custom_stoploss()`, `confirm_trade_entry/exit()`

Strategy templates: RSI, MACD, Bollinger, Stochastic, Momentum, MeanReversion, TrendFollowing, MultiTimeframe

### Configuration

- Pydantic `BaseSettings` with `env_prefix` (e.g., `IB__HOST`)
- YAML config files (`config.yaml`) for strategy parameters
- `.env` for environment variables

## Code Conventions

### Python

- **Imports**: Standard library → Third-party → Local, alphabetical
- **Line length**: 100 characters
- **Types**: Required on all functions and variables
- **Logging**: Use `get_logger(__name__)` from `src/utils/logger.py`, never `print()`
- **Errors**: Catch specific exceptions, never bare `except:`
- **Models**: Dataclasses for data models, `str, Enum` for string enums

### Anti-Patterns to Avoid

- **Global variables** like `_ib_broker`, `_db_store` - Use dependency injection
- **Silent failures** with bare `except:` - Catch specific exception types
- **Print statements** in production - Use `logger.info()`/`logger.error()`
- **代码不用考虑向后兼容**，一次性写好，不用考虑兼容性 

### Strategy Registration

Import in `src/strategies/__init__.py` for auto-registration.

### Adding Factors

Extend `TechnicalFactor` in `src/factors/technical/`, register via `FactorRegistry`.

## Quantitative Research Guidelines

When working on strategies or backtesting:

- Distinguish between alpha (idiosyncratic returns) and beta (market exposure)
- Watch for look-ahead bias, survivorship bias, data leakage, overfitting
- Treat unusually stable or high-performing backtest results with skepticism
- Always specify: signal definition, asset universe, data frequency, holding period, trading rules
- Consider: volatility targeting, position sizing, max drawdown, tail risk

## Key Files

| File | Purpose |
|------|---------|
| `src/evaluation/backtesting/engine.py` | Main event-driven backtest engine |
| `src/strategies/base/freqtrade_interface.py` | FreqtradeStrategy protocol |
| `src/strategies/base/indicators.py` | 100+ technical indicators |
| `src/core/types.py` | Core domain types (Signal, Order, etc.) |
| `src/core/events/engine.py` | UnifiedEventEngine for event orchestration |
| `src/config/settings.py` | Pydantic settings with nested models |
