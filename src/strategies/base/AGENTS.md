# src/strategies/base/ - Strategy Development Guide

**Generated:** 2025-01-10
**Commit:** Current working tree
**Branch:** N/A

---

## OVERVIEW

Freqtrade-style strategy framework with vectorized indicators and template system.

---

## STRUCTURE

```
base/
├── strategy.py      # Strategy (v2) + StrategyConfig + StrategyRegistry
├── interface.py     # IStrategy protocol (Freqtrade-style)
├── templates.py     # Strategy templates + StrategyTemplateFactory
├── indicators.py    # 100+ technical indicators
└── parameters.py    # Parameter classes for hyperopt
```

---

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| **Strategy base** | `strategy.py` | Strategy with generate_signals/size_positions |
| **Freqtrade interface** | `interface.py` | IStrategy with populate_indicators/entry/exit |
| **Templates** | `templates.py` | RSI, TrendFollowing, MACD, Bollinger, etc. |
| **Indicators** | `indicators.py` | sma, ema, rsi, macd, bollinger_bands, etc. |
| **Registry** | `strategy.py` | StrategyRegistry singleton |
| **Auto-registration** | `../__init__.py:17` | Import triggers registry.register() |

---

## CONVENTIONS

### Strategy Interface (v2)

```python
from src.strategies.base import Strategy

class MyStrategy(Strategy):
    def generate_signals(self, md, ctx) -> List[Signal]:
        # Generate trading signals
        pass
    
    def size_positions(self, signals, portfolio, risk, ctx) -> List[OrderIntent]:
        # Calculate position sizes
        pass
```

### Freqtrade-style Interface (IStrategy)

```python
from src.strategies.base import IStrategy

class MyStrategy(IStrategy):
    strategy_name = "My Strategy"
    timeframe = "1d"
    
    def populate_indicators(self, dataframe, metadata) -> pd.DataFrame:
        dataframe['rsi'] = rsi(dataframe['close'], 14)
        return dataframe
    
    def populate_entry_trend(self, dataframe, metadata) -> pd.DataFrame:
        dataframe['enter_long'] = dataframe['rsi'] < 30
        return dataframe
    
    def populate_exit_trend(self, dataframe, metadata) -> pd.DataFrame:
        dataframe['exit_long'] = dataframe['rsi'] > 70
        return dataframe
```

### Strategy Templates

```python
from src.strategies.base import RSIStrategy, TrendFollowingStrategy

class MyRSIStrategy(RSIStrategy):
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30

# Or use factory
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
| `TrendFollowingStrategy` | Dual moving average crossover | fast_period, slow_period, ma_type |
| `RSIStrategy` | RSI overbought/oversold | rsi_period, rsi_overbought, rsi_oversold |
| `MACDStrategy` | MACD crossover | macd_fast, macd_slow, macd_signal |
| `BollingerBandStrategy` | Bollinger band breakout | bb_period, bb_std |
| `StochasticStrategy` | Stochastic oscillator | k_period, d_period, overbought, oversold |
| `MomentumStrategy` | Momentum ranking | lookback_period, momentum_type |
| `MeanReversionStrategy` | Mean reversion | band_std, lookback |
| `MultiTimeframeStrategy` | Multi-timeframe analysis | entry_timeframe, exit_timeframe |

### Available Indicators

**Moving Averages**: `sma`, `ema`, `wma`, `dema`, `tema`, `vwap`, `vwma`, `hma`, `zlema`, `kama`

**Trend**: `adx`, `Aroon`, `cci`, `dmi`

**Momentum**: `rsi`, `stoch_rsi`, `macd`, `roc`, `mom`, `williams_r`, `ultimate_oscillator`

**Volatility**: `atr`, `bollinger_bands`, `kc`, `donchian`

**Volume**: `obv`, `ad`, `adosc`, `mfi`, `vwap_band`

**Statistical**: `zscore`, `percentile`, `entropy`, `correlation`

---

## CONFIGURATION

```python
from src.strategies.base.interface import StrategyConfig

config = StrategyConfig(
    strategy_name="My Strategy",
    strategy_id="my_strategy",
    timeframe="1d",
    stoploss=-0.05,
    trailing_stop=False,
    trailing_stop_positive=0.02,
    minimal_roi={"0": 0.05, "60": 0.03},
    order_types={"entry": "limit", "exit": "limit", "stoploss": "market"},
)
```

---

## ANTI-PATTERNS

- **Global registry**: `strategy_registry` is a singleton (no DI)
- **Large strategy files**: Keep strategy modules small and focused
- **Type suppression**: Never use `as any`, `@ts-ignore`
- **Hardcoded magic numbers**: Use configurable parameters
