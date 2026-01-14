# AGENTS.md - Factor Mining System Development Guide

**Generated:** 2025-01-10 | **Commit:** Current working tree

---

## BUILD / LINT / TEST COMMANDS

### Backend
```bash
# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev]"  # With dev dependencies

# Run API server
python3 -m src.api.main
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Lint (line-length: 100)
black src/ && flake8 src/ && mypy src/

# Test
pytest                          # All tests
pytest tests/test_informative.py                # Single file
pytest tests/test_informative.py::TestClass     # Single class
pytest tests/test_informative.py::TestClass::test_method  # Single test
pytest --cov=src --cov-report=term-missing  # With coverage
```

### Frontend
```bash
cd frontend
pnpm install

pnpm dev           # Port 3000
pnpm build         # tsc && vite build
pnpm lint          # ESLint
```

### Docker
```bash
docker-compose up --build
docker-compose up -d && docker-compose logs -f
docker-compose down
```

---

## CODE STYLE GUIDELINES

### Python (src/)
- **Imports**: Standard library → Third-party → Local, alphabetical within groups
- **Line length**: 100 characters (configured in `pyproject.toml`)
- **Types**: Required on all functions and variables
- **Models**: Use `dataclass` for data models (see `src/core/types.py`)
- **Enums**: Extend `str, Enum` for string-based serialization
- **Config**: Pydantic `BaseSettings` with `env_prefix` (e.g., `IB__HOST`)
- **Logging**: Use `get_logger(__name__)` from `src/utils/logger.py`
- **Async**: Use `asynccontextmanager` for FastAPI lifespan
- **Error handling**: NO bare `except:` or `except Exception`. Catch specific exceptions.
- **Logging vs print**: Use `logger` everywhere. No `print()` in production code.

### Frontend (frontend/src/)
- **Imports**: React → Third-party → Local, use `@/*` alias for relative imports
- **Components**: Functional components with hooks, TypeScript interfaces for props
- **State**: Zustand for global state, `useState`/`useReducer` for local state
- **Styling**: Tailwind CSS, dark mode via `'class'` strategy
- **Design**: Typography-first, minimal icons (Stripe/Vercel/Linear style)
- **TypeScript**: Strict mode enabled, no `any` unless absolutely necessary

---

## PROJECT STRUCTURE

```
factor_mining/
├── src/                    # Python backend (FastAPI)
│   ├── api/               # Routers (7 modules)
│   ├── config/            # Pydantic settings (env_prefix)
│   ├── core/              # Domain types (Signal, Order, PortfolioState)
│   ├── data/              # Collectors (IB, Polygon, CCXT)
│   ├── evaluation/        # Dual backtest engines + metrics
│   ├── execution/         # Broker implementations
│   ├── factors/           # 40+ technical factors
│   ├── strategies/        # Strategy implementations (v2)
│   └── utils/             # Logger
├── frontend/               # React/TypeScript + Vite
│   └── src/
│       ├── components/     # Charts, Layout
│       ├── pages/         # Dashboard, Backtest, History, Monitoring
│       ├── services/      # Axios API
│       └── stores/       # Zustand
├── examples/              # 13 demo scripts
├── tests/                 # Unit tests (pytest + unittest)
└── data/                  # Parquet cache, OHLCV
```

---

## KEY CONVENTIONS

| Pattern | Location | Convention |
|---------|----------|------------|
| **Add strategy** | `src/strategies/` | Import in `__init__.py` for auto-registration |
| **Add factor** | `src/factors/technical/` | Extend `TechnicalFactor`, register via `FactorRegistry` |
| **API routes** | `src/api/routers/` | Use `/api/v1` prefix, factory pattern via `create_app()` |
| **Settings** | `src/config/settings.py` | Nested Pydantic models, `env_prefix` per section |
| **Core types** | `src/core/types.py` | Dataclass enums (Signal, Order, OrderStatus) |

---

## ANTI-PATTERNS (AVOID)

- **Tests at root**: `test_*.py` in root instead of `tests/`. Fix: Put in `tests/`.
- **Global variables**: `_ib_broker`, `_db_store`. Use dependency injection.
- **Silent failures**: Bare `except:` / `except Exception`. Catch specific types.
- **Print statements**: Replace `print()` with `logger.info()`/`logger.error()`.
- **Type suppression**: Never use `as any`, `@ts-ignore`, `@ts-expect-error`.
- **Frontend visual changes**: Delegate to `frontend-ui-ux-engineer` agent.

---

## NOTES

- **Strategy auto-registration**: Import in `src/strategies/__init__.py` triggers registry
- **v2 migration**: v1 API removed. Use `examples/usage_example.py` as v2 reference
- **Coverage configured but unused**: `pytest-cov` installed, reports not generated
- **USE IB for data source**: Default data source is Interactive Brokers

---

## STRATEGY SYSTEM (v2)

### Strategy Templates

The system provides Freqtrade-style strategy templates that you can inherit and customize:

```python
from src.strategies.base import RSIStrategy, TrendFollowingStrategy, sma, rsi

# 方式1: 继承策略模板
class MyRSIStrategy(RSIStrategy):
    strategy_name = "My RSI Strategy"
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30

# 方式2: 使用策略工厂
from src.strategies.base import StrategyTemplateFactory
strategy = StrategyTemplateFactory.create(
    'rsi',
    strategy_id='my_rsi',
    rsi_period=10
)
```

### Available Templates

| Template | Description | Key Parameters |
|----------|-------------|----------------|
| `TrendFollowingStrategy` | Dual MA crossover | fast_period, slow_period, ma_type |
| `RSIStrategy` | RSI overbought/oversold | rsi_period, rsi_overbought, rsi_oversold |
| `MACDStrategy` | MACD crossover | macd_fast, macd_slow, macd_signal |
| `BollingerBandStrategy` | Bollinger band breakout | bb_period, bb_std |
| `StochasticStrategy` | Stochastic oscillator | k_period, d_period, overbought, oversold |
| `MomentumStrategy` | Momentum ranking | lookback_period, momentum_type |
| `MeanReversionStrategy` | Mean reversion | band_std, lookback |
| `MultiTimeframeStrategy` | Multi-timeframe analysis | entry_timeframe, exit_timeframe |

### Technical Indicators (100+)

```python
from src.strategies.base import sma, ema, rsi, macd, bollinger_bands, atr

# Moving averages
data['sma_20'] = sma(data['close'], 20)
data['ema_12'] = ema(data['close'], 12)

# Momentum
data['rsi_14'] = rsi(data['close'], 14)
macd_result = macd(data['close'], 12, 26, 9)
data['macd'] = macd_result.macd
data['macd_signal'] = macd_result.signal

# Volatility
bb = bollinger_bands(data['close'], 20, 2)
data['bb_upper'] = bb.upper
data['bb_lower'] = bb.lower
data['atr_14'] = atr(data['high'], data['low'], data['close'], 14)
```

### Running Demo

```bash
python3 examples/strategy_template_demo.py
```

---

## UNIFIED EVENT-DRIVEN ARCHITECTURE (v3)

### New Architecture Components

The system now has a unified event-driven architecture for backtesting and live trading:

```
src/data/providers/
├── base.py              # DataFeed抽象接口
│   ├── DataFeed         # 抽象基类
│   ├── HistoricalDataFeed  # 历史回放（回测用）
│   └── DataFeedFactory  # 数据源工厂
│
src/execution/providers/
├── base.py              # ExecutionProvider抽象接口
│   ├── ExecutionProvider    # 抽象基类
│   ├── SimulatedExecutionProvider  # 模拟撮合（回测用）
│   └── ExecutionProviderFactory  # 执行器工厂
│
src/evaluation/backtesting/
├── unified_backtest_engine.py  # 统一回测引擎
│   └── UnifiedBacktestEngine   # 事件驱动回测
```

### Event Flow (Complete Chain)

```
MarketEvent → 策略.generate_signals() → SignalEvent
    ↓
SignalEvent → _check_risk() → RiskEvent
    ↓
RiskEvent → OrderCreatedEvent
    ↓
OrderCreatedEvent → ExecutionProvider.submit_order() → OrderFilledEvent
    ↓
OrderFilledEvent → PortfolioState.update() → 记录成交
```

### Key Improvements (vs v2)

| Issue | Before | After |
|-------|--------|-------|
| **Fixed price 100.0** | Hardcoded in `execution_manager.py` | Real prices from `DataFeed` |
| **Empty market data** | `current_prices = {}` in `event_backtest_engine.py` | Real data from `bars_map` |
| **Empty risk check** | `pass` in `_check_risk()` | Full risk limits implementation |
| **Two event engines** | `event_engine.py` + `event_backtest_engine.py` | Single `UnifiedEventEngine` |
| **Non-deterministic** | Async parallel processing in `engine.py` | Controlled event flow |

### Usage Example

```python
from src.evaluation.backtesting.unified_backtest_engine import UnifiedBacktestEngine, BacktestConfig
from src.data.providers.base import HistoricalDataFeed, DataFeedFactory
from src.execution.providers.base import ExecutionProviderFactory
from src.strategies.base.unified_strategy import UnifiedStrategy

# 1. Create config
config = BacktestConfig(
    initial_capital=100000,
    commission_rate=0.001,
    slippage_rate=0.0005,
    fill_price_type="close",
    max_position_size=0.1,
    daily_loss_limit=0.05,
)

# 2. Create data feed and execution provider
data_feed = DataFeedFactory.create_historical_feed(warmup_days=260)
execution_provider = ExecutionProviderFactory.create_simulated_provider(
    commission_rate=0.001,
    slippage_rate=0.0005,
    fill_price_type="close",
    initial_capital=100000,
)

# 3. Create unified backtest engine
engine = UnifiedBacktestEngine(
    config=config,
    data_feed=data_feed,
    execution_provider=execution_provider,
)

# 4. Run backtest
results = await engine.run(
    strategies=[my_strategy],
    universe=["AAPL", "MSFT", "GOOGL"],
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
)

# 5. Check results
print(f"Total Return: {results['results']['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {results['results']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['results']['max_drawdown_pct']:.2f}%")
```

### Running Tests

```bash
# Run unified backtest engine tests
pytest tests/test_unified_backtest_engine.py -v

# Run all tests
pytest tests/ -v
```

---

# FREQTRADE-STYLE STRATEGY SYSTEM (v3)

## Overview

The system now supports Freqtrade-style strategies with a complete protocol system. This enables:
- Vectorized indicator calculation
- Freqtrade lifecycle callbacks
- Custom stoploss/ROI handling
- Trade confirmation callbacks

## New Architecture Components

```
src/strategies/base/
├── freqtrade_interface.py    # FreqtradeStrategy base class + Protocol
├── lifecycle.py              # FreqtradeLifecycleMixin for v2 strategies
└── templates.py              # Strategy templates (RSI, MACD, etc.)

src/evaluation/backtesting/
├── freqtrade_engine.py       # FreqtradeBacktestEngine
├── freqtrade_config.py       # FreqtradeBacktestConfig
├── freqtrade_report.py       # FreqtradeReportGenerator
└── stoploss_manager.py       # StoplossManager + ExitReason

src/execution/broker/
├── simulated_freqtrade.py    # SimulatedFreqtradeBroker
└── simulated_freqtrade_protections.py  # FreqtradeProtections
```

## Creating a Freqtrade-style Strategy

### Basic Structure

```python
from src.strategies.base.freqtrade_interface import FreqtradeStrategy
from src.strategies.base.lifecycle import FreqtradeLifecycleMixin
import pandas as pd

class MyFreqtradeStrategy(FreqtradeStrategy, FreqtradeLifecycleMixin):
    # Strategy Configuration
    strategy_name = "My Strategy"
    strategy_id = "my_strategy"
    timeframe = "1d"
    
    # Trading Configuration
    stoploss = -0.10
    trailing_stop = False
    minimal_roi = {0: 0.02, "60": 0.01}
    
    # Entry/Exit Configuration
    use_exit_signal = True
    exit_profit_only = False
    
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Calculate technical indicators."""
        # Add your indicators here
        return dataframe
    
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Generate entry signals."""
        dataframe['enter_long'] = False  # or your signal
        dataframe['enter_tag'] = ''      # optional signal tag
        return dataframe
    
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Generate exit signals."""
        dataframe['exit_long'] = False   # or your signal
        dataframe['exit_tag'] = ''       # optional exit reason
        return dataframe
```

### Optional Callbacks

```python
# Custom stoploss - return your stoploss rate
def custom_stoploss(self, pair, current_profit, current_rate, current_time, **kwargs) -> float:
    return self.stoploss

# Custom sell logic - return reason string to trigger sell
def custom_sell(self, pair, current_profit, current_rate, current_time, **kwargs) -> Optional[str]:
    return None  # Use default logic

# Custom buy logic - return reason string
def custom_buy(self, pair, current_rate, current_time, **kwargs) -> Optional[str]:
    return None

# Trade confirmation - return False to cancel order
def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force, current_time, **kwargs) -> bool:
    return True

def confirm_trade_exit(self, pair, order_type, amount, rate, time_in_force, current_time, **kwargs) -> bool:
    return True

# Post-fill handling
def order_filled(self, trade, order, current_time, **kwargs) -> None:
    pass

# Lifecycle hooks
async def bot_start(self, **kwargs) -> None:
    pass

async def bot_loop_start(self, **kwargs) -> None:
    pass

async def botShutdown(self, **kwargs) -> None:
    pass
```

### Migrating from v2 Strategy

If you have an existing `Strategy` class (v2), add the mixin to support Freqtrade callbacks:

```python
from src.strategies.base.strategy import Strategy
from src.strategies.base.lifecycle import FreqtradeLifecycleMixin

class MyStrategy(Strategy, FreqtradeLifecycleMixin):
    # Your existing v2 code...
    
    # Add Freqtrade callbacks as needed
    def populate_indicators(self, dataframe, metadata) -> pd.DataFrame:
        return dataframe
    
    def populate_entry_trend(self, dataframe, metadata) -> pd.DataFrame:
        dataframe['enter_long'] = dataframe['rsi'] < 30
        return dataframe
```

## Freqtrade Backtest Configuration

```python
from src.evaluation.backtesting.freqtrade_config import FreqtradeBacktestConfig

config = FreqtradeBacktestConfig(
    timeframe="1d",
    pairs=["SPY", "QQQ", "IWM"],
    stake_amount=10000.0,
    max_open_trades=3,
    fee=0.001,  # 0.1% commission
    dry_run_wallet=100000.0,  # Initial capital
    enable_protections=True,
)
```

## Simulated Broker with Protections

```python
from src.execution.broker.simulated_freqtrade import SimulatedFreqtradeBroker
from src.execution.broker.simulated_freqtrade_protections import FreqtradeProtections

protections = FreqtradeProtections(
    max_drawdown_protection={"enabled": True, "max_drawdown": 0.15, "lookback_days": 30},
    stoploss_on_exchange={"enabled": False},
    max_open_trades={"enabled": True, "max_open_trades": 3},
    cooldown_protection={"enabled": True, "cooldown_duration": 60},
    blacklist_protection={"enabled": False},
)

broker = SimulatedFreqtradeBroker(
    commission_rate=0.001,
    slippage_rate=0.0005,
    fill_price_type='close',
    initial_capital=100000.0,
    protections=protections,
)
```

## Strategy Templates

Available templates in `StrategyTemplateFactory`:

| Template | Description | Key Parameters |
|----------|-------------|----------------|
| `TrendFollowingStrategy` | Dual MA crossover | fast_period, slow_period, ma_type |
| `RSIStrategy` | RSI overbought/oversold | rsi_period, rsi_overbought, rsi_oversold |
| `MACDStrategy` | MACD crossover | macd_fast, macd_slow, macd_signal |
| `BollingerBandStrategy` | Bollinger band breakout | bb_period, bb_std |
| `StochasticStrategy` | Stochastic oscillator | k_period, d_period, overbought, oversold |
| `MomentumStrategy` | Momentum ranking | lookback_period, momentum_type |
| `MeanReversionStrategy` | Mean reversion | band_std, lookback |
| `MultiTimeframeStrategy` | Multi-timeframe analysis | entry_timeframe, exit_timeframe |

Example with template:

```python
from src.strategies.base import RSIStrategy

class MyRSIStrategy(RSIStrategy):
    strategy_name = "My RSI Strategy"
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30
```

## ETF Momentum Strategy Example

The `ETFMomentumJoinQuantStrategy` demonstrates the full Freqtrade protocol:

```python
from src.strategies.user_strategies.etf_momentum_joinquant import ETFMomentumJoinQuantStrategy

# Default configuration
strategy = ETFMomentumJoinQuantStrategy()

# Custom configuration
class CustomStrategy(ETFMomentumJoinQuantStrategy):
    strategy_id = 'custom_etf'
    etf_pool = ['QQQ', 'SPY', 'TLT', 'GLD']
    lookback_days = 30
    r2_threshold = 0.6
    stoploss = -0.08
    trailing_stop = True
```

Run the example:
```bash
python3 examples/etf_momentum_joinquant_example.py
```

## Key Differences from v2

| Aspect | v2 (Strategy) | v3 (FreqtradeStrategy) |
|--------|---------------|------------------------|
| Signal Generation | `generate_signals()` | `populate_entry_trend()` / `populate_exit_trend()` |
| Indicators | Inline in signals | `populate_indicators()` first |
| Position Sizing | `size_positions()` | Use `stake_amount` + `max_position_size` |
| Stoploss | `stoploss_pct` | `custom_stoploss()` + `stoploss` attribute |
| ROI | `take_profit_pct` | `minimal_roi` dict |
| Confirmation | Not available | `confirm_trade_entry/exit()` |
| Lifecycle | Limited | Full lifecycle (`bot_start`, `botShutdown`, etc.) |

## Import Paths

```python
# Core
from src.strategies.base.freqtrade_interface import FreqtradeStrategy
from src.strategies.base.lifecycle import FreqtradeLifecycleMixin
from src.strategies.base.templates import StrategyTemplateFactory

# Backtest
from src.evaluation.backtesting.freqtrade_engine import FreqtradeBacktestEngine
from src.evaluation.backtesting.freqtrade_config import FreqtradeBacktestConfig
from src.evaluation.backtesting.freqtrade_report import FreqtradeReportGenerator
from src.evaluation.backtesting.stoploss_manager import StoplossManager, ExitReason

# Broker
from src.execution.broker.simulated_freqtrade import SimulatedFreqtradeBroker
from src.execution.broker.simulated_freqtrade_protections import FreqtradeProtections

# Strategies
from src.strategies.user_strategies.etf_momentum_joinquant import ETFMomentumJoinQuantStrategy
```
