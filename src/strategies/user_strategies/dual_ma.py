"""
Dual Moving Average Crossover Strategy (Freqtrade-style)

A classic trend-following strategy that uses two moving averages:
- Fast MA (short period) - more responsive to price changes
- Slow MA (long period) - filters out noise

Entry Signals:
- Golden Cross (金叉): Fast MA crosses above Slow MA → BUY
- Death Cross (死叉): Fast MA crosses below Slow MA → SELL

Features:
- Vectorized indicator calculation
- Configurable MA type (SMA, EMA, WMA, etc.)
- Optional exit signal filtering
- Custom stoploss and ROI support
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np

from src.strategies.base.freqtrade_interface import FreqtradeStrategy
from src.strategies.base.lifecycle import FreqtradeLifecycleMixin
from src.utils.logger import get_logger


logger = get_logger("strategy.dual_ma")


class DualMAStrategy(FreqtradeStrategy, FreqtradeLifecycleMixin):
    """
    Dual Moving Average Crossover Strategy
    
    A simple yet effective trend-following strategy that generates signals
    based on the relationship between fast and slow moving averages.
    
    Parameters:
    -----------
    fast_period : int
        Period for the fast moving average (default: 10)
    slow_period : int
        Period for the slow moving average (default: 30)
    ma_type : str
        Type of moving average (sma, ema, wma, dema, tema, vwap)
    """

    # Strategy Configuration
    strategy_name = "Dual MA Crossover"
    strategy_id = "dual_ma"
    timeframe = "1d"
    startup_candle_count = 50

    # Trading Configuration
    stoploss = -0.10
    trailing_stop = False
    minimal_roi = {
        0: 0.05,      # 5% profit immediately
        60: 0.02,     # 2% profit after 1 hour
        180: 0.01,    # 1% profit after 3 hours
    }

    # Entry/Exit Configuration
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Strategy Parameters
    fast_period: int = 10
    slow_period: int = 30
    ma_type: str = "sma"  # sma, ema, wma, dema, tema, vwap, vwma, hma, kama, zlema

    def __init__(self):
        super().__init__()
        self.logger = get_logger(f"strategy.{self.strategy_id}")

    def _calculate_ma(self, series: pd.Series, period: int, ma_type: str) -> pd.Series:
        """
        Calculate moving average of specified type.
        
        Args:
            series: Price series
            period: MA period
            ma_type: Type of MA
            
        Returns:
            Moving average series
        """
        if ma_type == "sma":
            return series.rolling(period).mean()
        elif ma_type == "ema":
            return series.ewm(span=period, adjust=False).mean()
        elif ma_type == "wma":
            weights = np.arange(1, period + 1)
            return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        elif ma_type == "dema":
            ema1 = series.ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            return 2 * ema1 - ema2
        elif ma_type == "tema":
            ema1 = series.ewm(span=period, adjust=False).mean()
            ema2 = ema1.ewm(span=period, adjust=False).mean()
            ema3 = ema2.ewm(span=period, adjust=False).mean()
            return 3 * ema1 - 3 * ema2 + ema3
        elif ma_type == "vwap":
            return (series * series).rolling(period).sum() / series.rolling(period).sum()
        elif ma_type == "vwma":
            return (series * series).rolling(period).sum() / series.rolling(period).sum()
        elif ma_type == "hma":
            half = int(period / 2)
            wma_half = self._calculate_ma(series, half, "wma")
            full_ma = self._calculate_ma(series, period, "wma")
            return 2 * wma_half - full_ma
        elif ma_type == "kama":
            # Kaufman Adaptive Moving Average
            ma = series.rolling(period).mean()
            volatility = series.diff().abs().rolling(period).sum()
            efficiency = series.diff().abs().rolling(period).sum() / volatility
            fast = 2 / (2 + 1)
            slow = 2 / (30 + 1)
            alpha = efficiency * (fast - slow) + slow
            return ma + alpha * (series - ma)
        elif ma_type == "zlema":
            # Zero Lag Exponential Moving Average
            ema = series.ewm(span=period, adjust=False).mean()
            lag = int((period - 1) / 2)
            return ema + ema - ema.shift(lag)
        else:
            # Default to SMA
            return series.rolling(period).mean()

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate technical indicators for the strategy.
        
        This method adds:
        - fast_ma: Fast moving average
        - slow_ma: Slow moving average
        - ma_cross: Difference between fast and slow MA
        - ma_direction: Direction of MA crossover (1=golden, -1=dead, 0=none)
        """
        pair = metadata.get('pair', 'unknown') if metadata else 'unknown'

        # Calculate moving averages
        dataframe['fast_ma'] = self._calculate_ma(
            dataframe['close'], self.fast_period, self.ma_type
        )
        dataframe['slow_ma'] = self._calculate_ma(
            dataframe['close'], self.slow_period, self.ma_type
        )

        # Calculate MA crossover
        dataframe['ma_cross'] = dataframe['fast_ma'] - dataframe['slow_ma']

        # Calculate previous crossover for signal detection
        dataframe['ma_cross_prev'] = dataframe['ma_cross'].shift(1)

        # Determine crossover direction
        dataframe['ma_direction'] = 0
        # Golden Cross: fast crosses above slow (was below, now above)
        dataframe.loc[
            (dataframe['ma_cross'] > 0) & (dataframe['ma_cross_prev'] <= 0),
            'ma_direction'
        ] = 1
        # Death Cross: fast crosses below slow (was above, now below)
        dataframe.loc[
            (dataframe['ma_cross'] < 0) & (dataframe['ma_cross_prev'] >= 0),
            'ma_direction'
        ] = -1

        # MA trend (for additional filtering)
        dataframe['ma_trend'] = (dataframe['fast_ma'] > dataframe['slow_ma']).astype(int) * 2 - 1

        self.logger.debug(
            f"{pair}: fast_ma={dataframe['fast_ma'].iloc[-1]:.2f}, "
            f"slow_ma={dataframe['slow_ma'].iloc[-1]:.2f}, "
            f"direction={dataframe['ma_direction'].iloc[-1]}"
        )

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate entry signals based on MA crossover.
        
        Entry conditions:
        - Golden Cross (ma_direction == 1): BUY signal
        - Or: Fast MA > Slow MA with strong momentum
        """
        pair = metadata.get('pair', 'unknown') if metadata else 'unknown'

        # Initialize columns
        dataframe['enter_long'] = False
        dataframe['enter_tag'] = ''

        # Golden Cross entry signal
        golden_cross = dataframe['ma_direction'] == 1
        dataframe.loc[golden_cross, 'enter_long'] = True
        dataframe.loc[golden_cross, 'enter_tag'] = 'golden_cross'

        # Additional entry: Strong uptrend (optional filter)
        # Uncomment to require both golden cross AND uptrend confirmation
        # dataframe['enter_long'] = golden_cross & (dataframe['ma_trend'] == 1)

        enter_count = dataframe['enter_long'].sum()
        if enter_count > 0:
            self.logger.info(f"{pair}: 生成 {enter_count} 个进场信号")

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate exit signals based on MA crossover.
        
        Exit conditions:
        - Death Cross (ma_direction == -1): SELL signal
        - Or: Price closes below fast MA (stop loss filter)
        """
        pair = metadata.get('pair', 'unknown')

        # Initialize columns
        dataframe['exit_long'] = False
        dataframe['exit_tag'] = ''

        # Death Cross exit signal
        death_cross = dataframe['ma_direction'] == -1
        dataframe.loc[death_cross, 'exit_long'] = True
        dataframe.loc[death_cross, 'exit_tag'] = 'death_cross'

        # Optional: Price below fast MA exit
        price_below_fast = dataframe['close'] < dataframe['fast_ma']
        dataframe.loc[price_below_fast, 'exit_long'] = True
        dataframe.loc[price_below_fast, 'exit_tag'] = 'price_below_ma'

        exit_count = dataframe['exit_long'].sum()
        if exit_count > 0:
            self.logger.info(f"{pair}: 生成 {exit_count} 个离场信号")

        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        current_profit: float,
        current_rate: float,
        current_time,
        **kwargs
    ) -> float:
        """
        Custom stoploss logic.
        
        Dynamic stoploss based on profit level:
        - Small profit: Tight stop
        - Large profit: Wider stop to allow for continuation
        """
        if current_profit > 0.05:
            return -0.05  # 5% stop if up 5%
        elif current_profit > 0.02:
            return -0.03  # 3% stop if up 2%
        else:
            return self.stoploss  # Use default stoploss

    def custom_sell(
        self,
        pair: str,
        current_profit: float,
        current_rate: float,
        current_time,
        **kwargs
    ) -> Optional[str]:
        """
        Custom sell logic for additional exit conditions.
        """
        # Exit if profit drops significantly from peak
        trade = kwargs.get('trade')
        if trade and current_profit < -0.02:
            return 'profit_dropoff'

        return None  # Use default logic


# ============================================================================
# Strategy Factory Support
# ============================================================================

def create_dual_ma_strategy(
    fast_period: int = 10,
    slow_period: int = 30,
    ma_type: str = "sma",
    strategy_id: Optional[str] = None,
    **kwargs
) -> DualMAStrategy:
    """
    Factory function to create a DualMA strategy with custom parameters.
    
    Args:
        fast_period: Fast MA period
        slow_period: Slow MA period
        ma_type: Moving average type
        strategy_id: Custom strategy ID
        **kwargs: Additional strategy parameters
        
    Returns:
        Configured DualMAStrategy instance
    """
    class CustomDualMAStrategy(DualMAStrategy):
        pass

    # Set custom class attributes
    CustomDualMAStrategy.fast_period = fast_period
    CustomDualMAStrategy.slow_period = slow_period
    CustomDualMAStrategy.ma_type = ma_type
    if strategy_id:
        CustomDualMAStrategy.strategy_id = strategy_id
        CustomDualMAStrategy.__name__ = f"{strategy_id.title().replace('_', '')}Strategy"

    # Apply additional kwargs
    valid_attrs = ['stoploss', 'trailing_stop', 'minimal_roi', 'use_exit_signal']
    for key, value in kwargs.items():
        if key in valid_attrs:
            setattr(CustomDualMAStrategy, key, value)

    return CustomDualMAStrategy()


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    # Create default strategy
    strategy = DualMAStrategy()
    print(f"Strategy: {strategy.strategy_name}")
    print(f"Fast Period: {strategy.fast_period}")
    print(f"Slow Period: {strategy.slow_period}")
    print(f"MA Type: {strategy.ma_type}")
    print(f"Stoploss: {strategy.stoploss}")
    print()

    # Create custom strategy
    custom = create_dual_ma_strategy(
        fast_period=5,
        slow_period=20,
        ma_type="ema",
        strategy_id="fast_dual_ma"
    )
    print(f"Custom Strategy ID: {custom.strategy_id}")
    print(f"Custom Fast Period: {custom.fast_period}")
    print(f"Custom Slow Period: {custom.slow_period}")
    print(f"Custom MA Type: {custom.ma_type}")
    print()

    # Test with sample data
    import pandas as pd
    import numpy as np

    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'high': 105 + np.cumsum(np.random.randn(100) * 0.5),
        'low': 95 + np.cumsum(np.random.randn(100) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
        'volume': np.random.randint(1_000_000, 10_000_000, 100)
    }, index=dates)

    # Populate indicators
    df = strategy.populate_indicators(data.copy(), {'pair': 'AAPL'})
    print(f"Indicators: fast_ma={df['fast_ma'].iloc[-1]:.2f}, slow_ma={df['slow_ma'].iloc[-1]:.2f}")

    # Generate signals
    entry = strategy.populate_entry_trend(df.copy(), {'pair': 'AAPL'})
    exit = strategy.populate_exit_trend(df.copy(), {'pair': 'AAPL'})

    print(f"Entry signals: {entry['enter_long'].sum()}")
    print(f"Exit signals: {exit['exit_long'].sum()}")
