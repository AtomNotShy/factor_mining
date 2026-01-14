"""
Example: ETF Momentum JoinQuant Strategy

This example demonstrates how to:
1. Create an ETF momentum strategy using the Freqtrade-style framework
2. Generate trading signals with the strategy
3. Populate technical indicators
4. Configure strategy parameters

The ETF Momentum strategy:
- Calculates weighted linear regression momentum for each ETF
- Filters by R² (trend stability)
- Picks top N ETFs with highest momentum scores
- Uses volatility-based position sizing
"""

import pandas as pd
import numpy as np
from datetime import date
from loguru import logger

from src.strategies.user_strategies.etf_momentum_joinquant import ETFMomentumJoinQuantStrategy


def generate_mock_data(
    symbol: str,
    start_date: date,
    end_date: date,
    initial_price: float = 100.0,
    volatility: float = 0.02,
    drift: float = 0.0005
) -> pd.DataFrame:
    """
    Generate realistic OHLCV data for testing.

    Args:
        symbol: ETF symbol
        start_date: Start date
        end_date: End date
        initial_price: Starting price
        volatility: Daily volatility (std of returns)
        drift: Daily drift (mean return)
    """
    # Generate dates (excluding weekends)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n = len(all_dates)

    # Generate returns
    returns = np.random.normal(drift, volatility, n)

    # Calculate prices
    prices = initial_price * (1 + returns).cumprod()

    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
        'high': prices * (1 + np.abs(np.random.uniform(0, 0.01, n))),
        'low': prices * (1 - np.abs(np.random.uniform(0, 0.01, n))),
        'close': prices,
        'volume': np.random.uniform(1_000_000, 10_000_000, n).astype(int)
    }, index=all_dates)

    # Ensure high > low
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)

    return data


def run_strategy_demo():
    """Demonstrate the ETF Momentum strategy."""
    logger.remove()
    logger.add(lambda msg: print(msg, end=""))

    print("\n" + "=" * 70)
    print("ETF Momentum JoinQuant Strategy Demo")
    print("=" * 70 + "\n")

    # 1. Create strategy instance with default parameters
    print("1. Creating ETF Momentum Strategy...")
    strategy = ETFMomentumJoinQuantStrategy()
    print(f"   Strategy Name: {strategy.strategy_name}")
    print(f"   Strategy ID: {strategy.strategy_id}")
    print(f"   ETF Pool: {strategy.etf_pool}")
    print(f"   Lookback Days: {strategy.lookback_days}")
    print(f"   R² Threshold: {strategy.r2_threshold}")
    print(f"   Stoploss: {strategy.stoploss}")
    print(f"   Trailing Stop: {strategy.trailing_stop}")
    print(f"   Target Positions: {strategy.target_positions}")
    print(f"   Minimal ROI: {strategy.minimal_roi}\n")

    # 2. Create custom strategy with different parameters
    print("2. Creating Custom Strategy...")

    class CustomETFMomentumStrategy(ETFMomentumJoinQuantStrategy):
        strategy_id = 'custom_etf_momentum'
        strategy_name = 'Custom ETF Momentum'
        etf_pool = ['QQQ', 'SPY', 'IWM', 'TLT', 'GLD']
        lookback_days = 30
        r2_threshold = 0.6
        stoploss = -0.08
        trailing_stop = True
        trailing_stop_positive = 0.02
        target_positions = 3
        minimal_roi = {0: 0.03, "60": 0.015, "120": 0.005}

    custom_strategy = CustomETFMomentumStrategy()
    print(f"   Custom Strategy Name: {custom_strategy.strategy_name}")
    print(f"   Custom Lookback Days: {custom_strategy.lookback_days}")
    print(f"   Custom R² Threshold: {custom_strategy.r2_threshold}")
    print(f"   Custom Stoploss: {custom_strategy.stoploss}\n")

    # 3. Generate mock market data for each ETF
    print("3. Generating Mock Market Data...")
    start_date = date(2024, 1, 1)
    end_date = date(2024, 12, 31)
    universe = ['QQQ', 'SPY', 'IWM', 'TLT', 'GLD']

    market_data = {}
    for symbol in universe:
        # Different characteristics for each ETF
        if symbol == 'QQQ':
            price, vol, drift = 380.0, 0.025, 0.0008  # Tech - high growth
        elif symbol == 'SPY':
            price, vol, drift = 450.0, 0.015, 0.0005  # S&P 500 - moderate
        elif symbol == 'IWM':
            price, vol, drift = 200.0, 0.020, 0.0004  # Russell 2000 - small cap
        elif symbol == 'TLT':
            price, vol, drift = 100.0, 0.012, -0.0002  # Bonds - inverse correlation
        elif symbol == 'GLD':
            price, vol, drift = 180.0, 0.018, 0.0003  # Gold - inflation hedge
        else:
            price, vol, drift = 100.0, 0.020, 0.0005

        market_data[symbol] = generate_mock_data(symbol, start_date, end_date, price, vol, drift)
        print(f"   {symbol}: {len(market_data[symbol])} trading days, "
              f"price ${market_data[symbol]['close'].iloc[-1]:.2f}")

    print()

    # 4. Demonstrate indicator population for each ETF
    print("4. Populating Indicators for Each ETF...")
    for symbol in universe:
        df = strategy.populate_indicators(market_data[symbol].copy(), {'pair': symbol})
        last_row = df.iloc[-1]
        print(f"   {symbol}: momentum={last_row.get('momentum', 'N/A'):.4f}, "
              f"r²={last_row.get('momentum_r2', 'N/A'):.4f}, "
              f"volatility={last_row.get('momentum_volatility', 'N/A'):.4f}")

    print()

    # 5. Generate entry signals
    print("5. Generating Entry Signals...")
    entry_signals = []
    for symbol in universe:
        df = market_data[symbol].copy()
        indicators = strategy.populate_indicators(df, {'pair': symbol})
        entry = strategy.populate_entry_trend(indicators, {'pair': symbol})
        last_row = entry.iloc[-1]
        if last_row.get('enter_long', False):
            entry_signals.append({
                'symbol': symbol,
                'momentum': last_row.get('momentum', 0),
                'momentum_score': last_row.get('momentum_score', 0),
                'enter_tag': last_row.get('enter_tag', ''),
            })
            print(f"   BUY {symbol}: momentum={last_row.get('momentum', 0):.4f}, "
                  f"score={last_row.get('momentum_score', 0):.4f}")

    if not entry_signals:
        print("   No entry signals generated")

    print()

    # 6. Generate exit signals
    print("6. Generating Exit Signals...")
    for symbol in universe:
        df = market_data[symbol].copy()
        indicators = strategy.populate_indicators(df, {'pair': symbol})
        exit_df = strategy.populate_exit_trend(indicators, {'pair': symbol})
        last_row = exit_df.iloc[-1]
        if last_row.get('exit_long', False):
            print(f"   SELL {symbol}: reason={last_row.get('exit_tag', 'N/A')}")

    print()

    # 7. Test custom stoploss
    print("7. Testing Custom Stoploss...")
    from datetime import datetime
    current_time = datetime.now()
    stoploss = strategy.custom_stoploss(
        pair='QQQ',
        current_profit=0.03,
        current_rate=380.0,
        current_time=current_time,
    )
    print(f"   Custom stoploss rate: {stoploss:.4f}")
    print(f"   Stop price: ${380.0 * (1 + stoploss):.2f}")

    print()

    # 8. Test custom sell
    print("8. Testing Custom Sell...")
    from datetime import datetime
    current_time = datetime.now()
    reason = strategy.custom_sell(
        pair='QQQ',
        current_profit=0.02,
        current_rate=380.0,
        current_time=current_time,
    )
    print(f"   Custom sell reason: {reason}")

    print()

    # 9. Test confirmation callbacks
    print("9. Testing Trade Confirmation...")
    pair = 'QQQ'
    order_type = 'limit'
    amount = 10.0
    rate = 380.0
    time_in_force = 'gtc'
    current_time = datetime.now()

    entry_ok = strategy.confirm_trade_entry(
        pair, order_type, amount, rate, time_in_force, current_time
    )
    print(f"   Trade entry confirmed: {entry_ok}")

    exit_ok = strategy.confirm_trade_exit(
        pair, order_type, amount, rate, time_in_force, current_time
    )
    print(f"   Trade exit confirmed: {exit_ok}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")


def run_signal_generation_demo():
    """Demonstrate detailed signal generation."""
    print("\n" + "=" * 70)
    print("DETAILED SIGNAL GENERATION DEMO")
    print("=" * 70 + "\n")

    strategy = ETFMomentumJoinQuantStrategy()

    # Generate sample data
    dates = pd.date_range('2024-01-01', periods=252, freq='B')
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(252) * 0.5),
        'high': 105 + np.cumsum(np.random.randn(252) * 0.5),
        'low': 95 + np.cumsum(np.random.randn(252) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(252) * 0.5),
        'volume': np.random.randint(1_000_000, 10_000_000, 252)
    }, index=dates)

    print("1. Populating indicators...")
    df = strategy.populate_indicators(data.copy(), {})
    indicator_cols = ['momentum', 'momentum_r2', 'momentum_volatility', 'momentum_valid', 
                      'momentum_score', 'sma_20', 'sma_50', 'price_vs_sma20', 'price_vs_sma50']
    print(f"   Added columns: {', '.join([c for c in indicator_cols if c in df.columns])}")

    print("\n2. Indicator statistics...")
    print(f"   Momentum: min={df['momentum'].min():.4f}, max={df['momentum'].max():.4f}, "
          f"mean={df['momentum'].mean():.4f}")
    print(f"   Momentum R²: min={df['momentum_r2'].min():.4f}, max={df['momentum_r2'].max():.4f}, "
          f"mean={df['momentum_r2'].mean():.4f}")
    print(f"   Valid signals: {df['momentum_valid'].sum()} / {len(df)}")

    print("\n3. Generating entry signals...")
    entry_df = strategy.populate_entry_trend(df.copy(), {})
    entry_signals = entry_df[entry_df['enter_long'] == True]
    print(f"   Entry signals: {len(entry_signals)}")
    if len(entry_signals) > 0:
        print(f"   Entry tags: {entry_signals['enter_tag'].unique().tolist()}")

    print("\n4. Generating exit signals...")
    exit_df = strategy.populate_exit_trend(df.copy(), {})
    exit_signals = exit_df[exit_df['exit_long'] == True]
    print(f"   Exit signals: {len(exit_signals)}")
    if len(exit_signals) > 0:
        print(f"   Exit tags: {exit_signals['exit_tag'].unique().tolist()}")

    print("\n" + "=" * 70)
    print("SIGNAL DEMO COMPLETED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_strategy_demo()
    run_signal_generation_demo()
