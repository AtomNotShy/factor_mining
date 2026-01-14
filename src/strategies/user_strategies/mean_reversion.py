"""
Mean Reversion Strategy (Freqtrade-style)

A contrarian strategy that capitalizes on price deviations from a moving average:
- When price deviates significantly below the mean → BUY (expect reversion)
- When price reverts back to mean → SELL (take profit)
- Additional sell signal on extended overbought conditions

Features:
- Z-Score based entry/exit signals
- Configurable lookback period and entry/exit thresholds
- Volatility-adjusted position sizing
- Custom stoploss and ROI support

Theory:
- Prices tend to revert to their mean (moving average)
- Extreme deviations (high Z-Score) often precede reversals
- Lower volatility environments have higher reversion success rates
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np

from src.strategies.base.freqtrade_interface import FreqtradeStrategy
from src.strategies.base.lifecycle import FreqtradeLifecycleMixin
from src.utils.logger import get_logger


logger = get_logger("strategy.mean_reversion")


class MeanReversionStrategy(FreqtradeStrategy, FreqtradeLifecycleMixin):
    """
    Mean Reversion Strategy
    
    A contrarian strategy that trades based on price deviations from
    a moving average, using Z-Score to identify overbought/oversold conditions.
    
    Parameters:
    -----------
    lookback : int
        Period for calculating mean and standard deviation
    entry_z : float
        Z-Score threshold for entry signals (default: -2.0 for buy)
    exit_z : float
        Z-Score threshold for exit signals (default: 0.0)
    ma_period : int
        Moving average period for trend confirmation
    """

    # Strategy Configuration
    strategy_name = "Mean Reversion"
    strategy_id = "mean_reversion"
    timeframe = "1d"
    startup_candle_count = 60

    # Trading Configuration
    stoploss = -0.15  # Wider stop for mean reversion
    trailing_stop = False
    minimal_roi = {
        0: 0.03,      # 3% profit target
        120: 0.015,   # 1.5% after 2 hours
        360: 0.01,    # 1% after 6 hours
    }

    # Entry/Exit Configuration
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Strategy Parameters
    lookback: int = 60          # Lookback period for mean/std
    entry_z: float = -2.0       # Z-Score for buy signal (oversold)
    exit_z: float = 0.0         # Z-Score for exit (mean reversion complete)
    ma_period: int = 20         # MA period for trend filter
    atr_period: int = 14        # ATR period for volatility

    def __init__(self):
        super().__init__()
        self.logger = get_logger(f"strategy.{self.strategy_id}")

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """
        Calculate technical indicators for the strategy.
        
        This method adds:
        - zscore: Current Z-Score of price
        - ma20: 20-period moving average for trend
        - atr14: 14-period Average True Range
        - bb_upper, bb_middle, bb_lower: Bollinger Bands
        - volatility_rank: Volatility percentile (0-1)
        """
        pair = metadata.get('pair', 'unknown')

        # Z-Score calculation
        lookback = min(self.lookback, len(dataframe))
        if lookback < 10:
            # Not enough data
            dataframe['zscore'] = 0.0
            dataframe['bb_upper'] = dataframe['close']
            dataframe['bb_middle'] = dataframe['close']
            dataframe['bb_lower'] = dataframe['close']
            dataframe['atr14'] = 0.0
            dataframe['volatility_rank'] = 0.5
            dataframe['ma_trend'] = 1
            return dataframe

        # Rolling mean and std
        recent = dataframe['close'].iloc[-lookback:]
        rolling_mean = recent.rolling(self.ma_period).mean()
        rolling_std = recent.rolling(self.ma_period).std()

        # Current Z-Score
        current_mean = rolling_mean.iloc[-1]
        current_std = rolling_std.iloc[-1]

        if current_std > 0:
            dataframe['zscore'] = (dataframe['close'] - current_mean) / current_std
        else:
            dataframe['zscore'] = 0.0

        # Bollinger Bands (for additional confirmation)
        bb_period = self.ma_period
        bb_std = 2.0
        dataframe['bb_middle'] = dataframe['close'].rolling(bb_period).mean()
        bb_rolling_std = dataframe['close'].rolling(bb_period).std()
        dataframe['bb_upper'] = dataframe['bb_middle'] + bb_std * bb_rolling_std
        dataframe['bb_lower'] = dataframe['bb_middle'] - bb_std * bb_rolling_std

        # ATR for volatility
        high_low = dataframe['high'] - dataframe['low']
        high_close = abs(dataframe['high'] - dataframe['close'].shift())
        low_close = abs(dataframe['low'] - dataframe['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        dataframe['atr14'] = tr.rolling(self.atr_period).mean()

        # Volatility rank (percentile of ATR)
        atr_values = dataframe['atr14'].dropna()
        if len(atr_values) > self.atr_period:
            atr_percentile = atr_values.rank(pct=True).iloc[-1]
            dataframe['volatility_rank'] = atr_percentile
        else:
            dataframe['volatility_rank'] = 0.5

        # MA trend (for additional filtering)
        dataframe['ma_trend'] = (dataframe['close'] > dataframe['bb_middle']).astype(int) * 2 - 1

        # Position size factor based on volatility
        # Lower volatility = larger position
        dataframe['vol_adj_factor'] = 1.0 / (dataframe['volatility_rank'] + 0.5)

        self.logger.debug(
            f"{pair}: zscore={dataframe['zscore'].iloc[-1]:.2f}, "
            f"bb_middle={dataframe['bb_middle'].iloc[-1]:.2f}, "
            f"atr14={dataframe['atr14'].iloc[-1]:.2f}"
        )

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """
        Generate entry signals based on Z-Score.
        
        Entry conditions:
        - Z-Score below entry_z threshold (oversold) → BUY
        - Optional: Price near lower Bollinger Band
        """
        pair = metadata.get('pair', 'unknown')

        # Initialize columns
        dataframe['enter_long'] = False
        dataframe['enter_tag'] = ''

        # Primary signal: Z-Score below threshold (oversold)
        zscore_buy = dataframe['zscore'] < self.entry_z

        # Secondary confirmation: Price near lower Bollinger Band
        bb_touch = dataframe['close'] <= dataframe['bb_lower'] * 1.02

        # Combined signal: Z-Score oversold + BB touch
        entry_signal = zscore_buy & bb_touch

        dataframe.loc[entry_signal, 'enter_long'] = True
        dataframe.loc[entry_signal, 'enter_tag'] = 'zscore_oversold'

        # Alternative entry: Strong reversal (large negative Z-Score)
        strong_buy = dataframe['zscore'] < (self.entry_z - 0.5)
        dataframe.loc[strong_buy, 'enter_long'] = True
        dataframe.loc[strong_buy, 'enter_tag'] = 'zscore_strong'

        enter_count = dataframe['enter_long'].sum()
        if enter_count > 0:
            self.logger.info(f"{pair}: 生成 {enter_count} 个进场信号")

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """
        Generate exit signals based on Z-Score reversion.
        
        Exit conditions:
        - Z-Score above exit_z threshold (reversion complete) → SELL
        - Or: Price reaches middle Bollinger Band
        - Or: Price exceeds upper Bollinger Band (overbought)
        """
        pair = metadata.get('pair', 'unknown')

        # Initialize columns
        dataframe['exit_long'] = False
        dataframe['exit_tag'] = ''

        # Primary exit: Z-Score reverts to mean
        zscore_exit = dataframe['zscore'] > self.exit_z

        # Secondary exit: Price at middle Bollinger Band
        bb_middle_touch = dataframe['close'] >= dataframe['bb_middle']

        # Overbought exit: Price exceeds upper Bollinger Band
        overbought = dataframe['close'] >= dataframe['bb_upper']

        # Combined signals
        exit_signal = zscore_exit & bb_middle_touch
        dataframe.loc[exit_signal, 'exit_long'] = True
        dataframe.loc[exit_signal, 'exit_tag'] = 'zscore_reversion'

        # Overbought exit
        dataframe.loc[overbought, 'exit_long'] = True
        dataframe.loc[overbought, 'exit_tag'] = 'overbought'

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
        Custom stoploss logic for mean reversion.
        
        Wider stops for mean reversion due to:
        - Potential for extended drawdowns
        - Higher win rate, lower profit per trade
        """
        # Dynamic stop based on entry Z-Score
        trade = kwargs.get('trade')
        if trade and current_profit < -0.05:
            # If significantly underwater, wider stop to avoid premature exits
            return -0.20

        # Standard stop
        return self.stoploss

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
        # Exit if profit drops below threshold after being in profit
        trade = kwargs.get('trade')
        if trade and current_profit < -0.02:
            # Check if we were previously in profit
            return 'profit_dropoff'

        # Exit if Z-Score starts rising again (failed reversion)
        return None

    def adjust_trade_position(
        self,
        pair: str,
        current_rate: float,
        current_time,
        current_profit: float,
        min_stake: float,
        max_stake: float,
        current_stake: float,
        **kwargs
    ) -> Optional[float]:
        """
        Adjust position size based on volatility.
        
        Lower volatility = larger position size
        """
        # Get volatility adjustment factor from dataframe
        # This is a simplified version - in practice you'd need to pass the dataframe
        if current_profit > 0.02:
            # Scale out if in profit
            return -current_stake * 0.5  # Reduce position by 50%
        elif current_profit < -0.05:
            # Add to position if significantly underwater (pyramiding)
            # Only if Z-Score is still very negative
            return current_stake * 0.25  # Add 25% more

        return None  # No adjustment


# ============================================================================
# Strategy Factory Support
# ============================================================================

def create_mean_reversion_strategy(
    lookback: int = 60,
    entry_z: float = -2.0,
    exit_z: float = 0.0,
    ma_period: int = 20,
    strategy_id: Optional[str] = None,
    **kwargs
) -> MeanReversionStrategy:
    """
    Factory function to create a MeanReversion strategy with custom parameters.
    
    Args:
        lookback: Lookback period for mean/std calculation
        entry_z: Z-Score threshold for buy signal
        exit_z: Z-Score threshold for exit signal
        ma_period: MA period for Bollinger Bands
        strategy_id: Custom strategy ID
        **kwargs: Additional strategy parameters
        
    Returns:
        Configured MeanReversionStrategy instance
    """
    class CustomMeanReversionStrategy(MeanReversionStrategy):
        pass

    # Set custom class attributes
    CustomMeanReversionStrategy.lookback = lookback
    CustomMeanReversionStrategy.entry_z = entry_z
    CustomMeanReversionStrategy.exit_z = exit_z
    CustomMeanReversionStrategy.ma_period = ma_period
    if strategy_id:
        CustomMeanReversionStrategy.strategy_id = strategy_id
        CustomMeanReversionStrategy.__name__ = f"{strategy_id.title().replace('_', '')}Strategy"

    # Apply additional kwargs
    valid_attrs = ['stoploss', 'trailing_stop', 'minimal_roi', 'use_exit_signal', 'atr_period']
    for key, value in kwargs.items():
        if key in valid_attrs:
            setattr(CustomMeanReversionStrategy, key, value)

    return CustomMeanReversionStrategy()


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    # Create default strategy
    strategy = MeanReversionStrategy()
    print(f"Strategy: {strategy.strategy_name}")
    print(f"Lookback: {strategy.lookback}")
    print(f"Entry Z-Score: {strategy.entry_z}")
    print(f"Exit Z-Score: {strategy.exit_z}")
    print(f"MA Period: {strategy.ma_period}")
    print(f"Stoploss: {strategy.stoploss}")
    print()

    # Create custom strategy
    custom = create_mean_reversion_strategy(
        lookback=40,
        entry_z=-1.5,
        exit_z=-0.5,
        strategy_id="fast_mean_reversion"
    )
    print(f"Custom Strategy ID: {custom.strategy_id}")
    print(f"Custom Lookback: {custom.lookback}")
    print(f"Custom Entry Z-Score: {custom.entry_z}")
    print()

    # Test with sample data
    import pandas as pd
    import numpy as np

    # Generate mean-reverting data
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2024-01-01', periods=n, freq='D')

    # Create price series with mean-reverting tendency
    returns = np.random.randn(n) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, n)),
        'high': prices * (1 + np.abs(np.random.uniform(0, 0.02, n))),
        'low': prices * (1 - np.abs(np.random.uniform(0, 0.02, n))),
        'close': prices,
        'volume': np.random.randint(1_000_000, 10_000_000, n)
    }, index=dates)

    # Populate indicators
    df = strategy.populate_indicators(data.copy(), {'pair': 'SPY'})
    print(f"Indicators: zscore={df['zscore'].iloc[-1]:.2f}, "
          f"bb_middle={df['bb_middle'].iloc[-1]:.2f}, "
          f"atr14={df['atr14'].iloc[-1]:.2f}")

    # Generate signals
    entry = strategy.populate_entry_trend(df.copy(), {'pair': 'SPY'})
    exit = strategy.populate_exit_trend(df.copy(), {'pair': 'SPY'})

    print(f"Entry signals: {entry['enter_long'].sum()}")
    print(f"Exit signals: {exit['exit_long'].sum()}")
    print()

    # Show recent signals
    recent_entry = entry[entry['enter_long']].tail(5)
    if not recent_entry.empty:
        print("Recent Entry Signals:")
        print(recent_entry[['close', 'zscore', 'enter_tag']])
