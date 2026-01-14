"""
UnifiedBacktestEngine 集成测试
"""

import pytest
from datetime import date, datetime, timedelta
from typing import List

from src.evaluation.backtesting import (
    UnifiedBacktestEngine, 
    UnifiedConfig,
    FeatureFlag,
    TradeConfig, TimeConfig, FillConfig, StoplossConfig, ProtectionConfig,
    BacktestResult,
)
from src.strategies.base import Strategy
from src.core.types import Signal, OrderIntent, ActionType, OrderSide, OrderType


class DummyStrategy(Strategy):
    """测试用简单策略"""
    
    def __init__(self):
        super().__init__()
        self.call_count = 0
    
    def generate_signals(self, md, ctx) -> List[Signal]:
        self.call_count += 1
        
        signals = []
        if md.bars.empty:
            return signals
            
        for symbol in md.bars["symbol"].unique():
            if self.call_count % 5 == 0:
                signals.append(Signal(
                    ts_utc=ctx.now_utc,
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    action=ActionType.LONG,
                    strength=1.0,
                ))
        return signals


@pytest.fixture
def simple_config():
    return UnifiedConfig(
        trade=TradeConfig(initial_capital=100000),
        time=TimeConfig(clock_mode="daily"),
        features=FeatureFlag.NONE,
    )


@pytest.fixture
def freqtrade_config():
    return UnifiedConfig(
        trade=TradeConfig(initial_capital=100000),
        time=TimeConfig(clock_mode="hybrid", execution_time="16:00"),
        stoploss=StoplossConfig(stoploss=-0.05, trailing_stop=True),
        features=FeatureFlag.STOPLOSS_MANAGER | FeatureFlag.VECTORIZED,
    )


@pytest.fixture
def sample_market_data():
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    
    data = []
    for symbol in ["AAPL", "MSFT"]:
        for d in dates:
            base_price = 100 if symbol == "AAPL" else 200
            data.append({
                "timestamp": d,
                "open": base_price + np.random.randn() * 2,
                "high": base_price + np.random.randn() * 3,
                "low": base_price + np.random.randn() * 3,
                "close": base_price + np.random.randn() * 2,
                "volume": 1000000,
                "symbol": symbol,
            })
    
    df = pd.DataFrame(data)
    df = df.set_index("timestamp")
    return df


class TestUnifiedConfig:
    """UnifiedConfig 测试类"""
    
    def test_default_config(self):
        config = UnifiedConfig()
        
        assert config.trade.initial_capital == 100000.0
        assert config.trade.commission_rate == 0.001
        assert config.time.signal_timeframe == "1d"
        assert config.time.clock_mode == "daily"
        assert FeatureFlag.ALL in config.features
    
    def test_feature_flags(self):
        config = UnifiedConfig(
            features=FeatureFlag.VECTORIZED | FeatureFlag.STOPLOSS_MANAGER
        )
        
        assert FeatureFlag.VECTORIZED in config.features
        assert FeatureFlag.STOPLOSS_MANAGER in config.features
        assert FeatureFlag.PROTECTIONS not in config.features
    
    def test_validate_valid(self):
        config = UnifiedConfig()
        config.validate()
    
    def test_validate_invalid_capital(self):
        config = UnifiedConfig()
        config.trade.initial_capital = 0
        
        with pytest.raises(AssertionError):
            config.validate()
    
    def test_validate_invalid_stoploss(self):
        config = UnifiedConfig()
        config.stoploss.stoploss = 0.5
        
        with pytest.raises(AssertionError):
            config.validate()


class TestUnifiedBacktestEngine:
    """UnifiedBacktestEngine 测试类"""
    
    @pytest.mark.asyncio
    async def test_engine_creation(self, simple_config):
        engine = UnifiedBacktestEngine(config=simple_config)
        
        assert engine.config == simple_config
        assert engine._stoploss_manager is None
        assert engine._cost_model is not None
    
    @pytest.mark.asyncio
    async def test_result_structure(self, simple_config):
        result = BacktestResult(
            run_id="test-123",
            strategy_name="test_strategy",
            initial_capital=100000,
            final_equity=110000,
            total_return=0.1,
            total_return_pct=10.0,
        )
        
        assert result.run_id == "test-123"
        assert result.total_return_pct == 10.0
        assert result.to_dict()["total_return_pct"] == 10.0
    
    @pytest.mark.asyncio
    async def test_engine_with_stoploss(self, freqtrade_config):
        engine = UnifiedBacktestEngine(config=freqtrade_config)
        
        assert engine._stoploss_manager is not None


class TestFeatureFlags:
    """特性开关测试类"""
    
    def test_stoploass_manager_flag(self):
        config = UnifiedConfig(features=FeatureFlag.STOPLOSS_MANAGER)
        engine = UnifiedBacktestEngine(config=config)
        
        assert engine._stoploss_manager is not None
    
    def test_no_stoploss_manager_flag(self):
        config = UnifiedConfig(features=FeatureFlag.NONE)
        engine = UnifiedBacktestEngine(config=config)
        
        assert engine._stoploss_manager is None
    
    def test_all_features(self):
        config = UnifiedConfig(features=FeatureFlag.ALL)
        engine = UnifiedBacktestEngine(config=config)
        
        assert engine._stoploss_manager is not None


class TestBacktestContext:
    """BacktestContext 测试类"""
    
    def test_context_creation(self):
        from src.evaluation.backtesting.unified_engine import BacktestContext
        
        ctx = BacktestContext(
            run_id="test-123",
            strategy_id="test_strategy",
        )
        
        assert ctx.run_id == "test-123"
        assert ctx.strategy_id == "test_strategy"
        assert ctx.all_signals == []
        assert ctx.all_orders == []
        assert ctx.all_fills == []
        assert ctx.portfolio_daily == []
    
    def test_context_defaults(self):
        from src.evaluation.backtesting.unified_engine import BacktestContext
        
        ctx = BacktestContext()
        
        assert ctx.run_id == ""
        assert ctx.current_ts is None
        assert ctx.trading_days == []
        assert ctx.timeline == []
