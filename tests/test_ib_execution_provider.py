"""
IBExecutionProvider Tests
"""

import asyncio
from datetime import datetime, timezone
import pytest

from src.core.types import Fill, OrderIntent, OrderSide, OrderType
from src.execution.providers.ib import (
    IBExecutionProvider,
    IBExecutionProviderConfig,
)


class TestIBExecutionProviderConfig:
    """Config tests"""
    
    def test_defaults(self):
        config = IBExecutionProviderConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 7497
        assert config.client_id == 1
        assert config.paper_trading is False
        assert config.commission_rate == 0.001
    
    def test_custom(self):
        config = IBExecutionProviderConfig(
            host="192.168.1.100",
            port=4002,
            client_id=2,
            account_id="DU1234567",
            paper_trading=True,
        )
        assert config.host == "192.168.1.100"
        assert config.paper_trading is True


class TestIBExecutionProvider:
    """Provider tests"""
    
    @pytest.fixture
    def provider(self):
        config = IBExecutionProviderConfig(
            host="127.0.0.1",
            port=7497,
            client_id=1,
            account_id="DU1234567",
            paper_trading=True,
        )
        return IBExecutionProvider(config=config)
    
    def test_initialization(self, provider):
        assert provider._connected is False
        assert provider._ib is None
        assert provider._portfolio.cash == 100000.0
    
    def test_create_contract(self, provider):
        """测试创建合约"""
        contract = provider._create_contract("SPY")
        
        assert contract.symbol == "SPY"
        assert contract.secType == "STK"
        assert contract.currency == "USD"
    
    def test_create_contract_caching(self, provider):
        """测试合约缓存"""
        contract1 = provider._create_contract("AAPL")
        contract2 = provider._create_contract("AAPL")
        
        assert contract1 is contract2  # 应该是同一个对象
    
    def test_to_ib_order_market(self, provider):
        """测试市价单转换"""
        intent = OrderIntent(
            order_id="test_001",
            symbol="SPY",
            side=OrderSide.BUY,
            qty=100,
            order_type=OrderType.MKT,
        )
        
        order = provider._to_ib_order(intent)
        
        assert order.action == "BUY"
        assert order.totalQuantity == 100
        # paper_trading 会添加前缀
        assert "test_001" in order.orderId
    
    def test_to_ib_order_limit(self, provider):
        """测试限价单转换"""
        intent = OrderIntent(
            order_id="test_002",
            symbol="SPY",
            side=OrderSide.SELL,
            qty=50,
            order_type=OrderType.LMT,
            limit_price=500.0,
        )
        
        order = provider._to_ib_order(intent)
        
        assert order.action == "SELL"
        assert order.totalQuantity == 50
        assert order.lmtPrice == 500.0
    
    def test_to_ib_order_stop(self, provider):
        """测试止损单转换"""
        intent = OrderIntent(
            order_id="test_003",
            symbol="SPY",
            side=OrderSide.BUY,
            qty=100,
            order_type=OrderType.STP,
            stop_price=450.0,
        )
        
        order = provider._to_ib_order(intent)
        
        assert order.action == "BUY"
        assert order.totalQuantity == 100
        assert order.auxPrice == 450.0
    
    def test_to_ib_order_invalid_limit(self, provider):
        """测试限价单缺少limit_price"""
        intent = OrderIntent(
            order_id="test_004",
            symbol="SPY",
            side=OrderSide.BUY,
            qty=100,
            order_type=OrderType.LMT,
        )
        
        with pytest.raises(ValueError, match="限价单需要指定limit_price"):
            provider._to_ib_order(intent)
    
    def test_to_ib_order_invalid_stop(self, provider):
        """测试止损单缺少stop_price"""
        intent = OrderIntent(
            order_id="test_005",
            symbol="SPY",
            side=OrderSide.BUY,
            qty=100,
            order_type=OrderType.STP,
        )
        
        with pytest.raises(ValueError, match="止损单需要指定stop_price"):
            provider._to_ib_order(intent)
    
    @pytest.mark.asyncio
    async def test_submit_order_paper_trading(self, provider):
        """测试模拟交易订单提交"""
        # paper_trading 模式会自动跳过真实连接
        
        intent = OrderIntent(
            symbol="SPY",
            side=OrderSide.BUY,
            qty=100,
            order_type=OrderType.MKT,
            strategy_id="test_strategy",
        )
        
        result = await provider.submit_order(intent)
        
        assert result["success"] is True
        assert "order_id" in result
        assert "fill_id" in result
        assert result["fill_quantity"] == 100
        assert result["commission"] > 0
    
    @pytest.mark.asyncio
    async def test_submit_order_limit_paper(self, provider):
        """测试模拟交易限价单"""
        intent = OrderIntent(
            symbol="AAPL",
            side=OrderSide.SELL,
            qty=50,
            order_type=OrderType.LMT,
            limit_price=180.0,
        )
        
        result = await provider.submit_order(intent)
        
        assert result["success"] is True
        assert result["fill_quantity"] == 50
        # 模拟成交价应该接近限价
        assert abs(result["fill_price"] - 180.0) < 1.0
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, provider):
        """测试取消订单"""
        # 先提交一个订单
        intent = OrderIntent(
            symbol="SPY",
            side=OrderSide.BUY,
            qty=100,
            order_type=OrderType.MKT,
        )
        submit_result = await provider.submit_order(intent)
        order_id = submit_result["order_id"]
        
        # 取消订单
        result = await provider.cancel_order(order_id, "测试取消")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_order_status(self, provider):
        """测试获取订单状态"""
        # 先提交一个订单
        intent = OrderIntent(
            symbol="SPY",
            side=OrderSide.BUY,
            qty=100,
            order_type=OrderType.MKT,
        )
        submit_result = await provider.submit_order(intent)
        order_id = submit_result["order_id"]
        
        status = await provider.get_order_status(order_id)
        
        assert status is not None
        assert status["order_id"] == order_id
        assert status["symbol"] == "SPY"
        assert status["side"] == "BUY"
        assert status["quantity"] == 100
    
    @pytest.mark.asyncio
    async def test_get_order_status_nonexistent(self, provider):
        """测试获取不存在的订单状态"""
        status = await provider.get_order_status("nonexistent")
        
        assert status is None
    
    @pytest.mark.asyncio
    async def test_poll_fills(self, provider):
        """测试轮询成交"""
        # 提交订单
        intent = OrderIntent(
            symbol="SPY",
            side=OrderSide.BUY,
            qty=100,
            order_type=OrderType.MKT,
        )
        await provider.submit_order(intent)
        
        fills = await provider.poll_fills()
        
        assert len(fills) > 0
        assert fills[0].symbol == "SPY"
    
    @pytest.mark.asyncio
    async def test_get_portfolio_state(self, provider):
        """测试获取组合状态"""
        # 模拟交易模式应该使用配置的初始资金
        portfolio = await provider.get_portfolio_state()
        
        assert portfolio.cash <= 100000.0  # 买入后现金减少
        assert portfolio.equity <= 100000.0
    
    def test_reset(self, provider):
        """测试重置"""
        # 先提交订单
        intent = OrderIntent(
            symbol="SPY",
            side=OrderSide.BUY,
            qty=100,
            order_type=OrderType.MKT,
        )
        asyncio.get_event_loop().run_until_complete(provider.submit_order(intent))
        
        # 重置
        provider.reset(capital=50000.0)
        
        assert provider._portfolio.cash == 50000.0
        assert provider._portfolio.equity == 50000.0
        assert len(provider._orders) == 0
        assert len(provider._fills) == 0
    
    def test_update_portfolio_buy(self, provider):
        """测试更新组合（买入）"""
        fill = Fill(
            fill_id="test_fill_001",
            order_id="test_order",
            ts_fill_utc=datetime.now(timezone.utc),
            symbol="SPY",
            side=OrderSide.BUY,
            qty=100,
            price=450.0,
            fee=0.45,
        )
        
        provider._update_portfolio(fill)
        
        assert provider._portfolio.positions["SPY"] == 100
        assert provider._portfolio.avg_price["SPY"] == 450.0
    
    def test_update_portfolio_sell(self, provider):
        """测试更新组合（卖出）"""
        # 先买入
        buy_fill = Fill(
            fill_id="test_fill_001",
            order_id="test_order_1",
            ts_fill_utc=datetime.now(timezone.utc),
            symbol="SPY",
            side=OrderSide.BUY,
            qty=100,
            price=450.0,
            fee=0.45,
        )
        provider._update_portfolio(buy_fill)
        
        # 再卖出
        sell_fill = Fill(
            fill_id="test_fill_002",
            order_id="test_order_2",
            ts_fill_utc=datetime.now(timezone.utc),
            symbol="SPY",
            side=OrderSide.SELL,
            qty=50,
            price=455.0,
            fee=0.23,
        )
        provider._update_portfolio(sell_fill)
        
        assert provider._portfolio.positions["SPY"] == 50
        # 平均成本应该保持不变
        assert provider._portfolio.avg_price["SPY"] == 450.0
    
    @pytest.mark.asyncio
    async def test_close(self, provider):
        """测试关闭"""
        await provider.close()
        
        assert provider._closed is True
        assert provider._connected is False
        assert provider._ib is None


class TestIBProviderFactory:
    """IBProviderFactory 测试"""
    
    def test_create(self):
        """测试工厂创建"""
        from src.execution.providers.ib import IBProviderFactory
        
        provider = IBProviderFactory.create(
            host="127.0.0.1",
            port=7497,
            client_id=1,
            account_id="DU1234567",
            paper_trading=True,
        )
        
        assert isinstance(provider, IBExecutionProvider)
        assert provider.config.host == "127.0.0.1"
        assert provider.config.paper_trading is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
