"""
成本模型
模拟美股交易的完整成本结构，包括：
1. 佣金（基于订单金额的阶梯费率）
2. SEC 费用
3. TAF 费用
4. 滑点（基于流动性）
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class TradeInfo:
    """交易信息"""
    symbol: str
    side: Side
    quantity: float
    fill_price: float
    commission: float = 0.0
    sec_fee: float = 0.0
    taf_fee: float = 0.0
    slippage: float = 0.0
    liquidity_flag: str = "none"  # "add" or "remove" (maker/taker)


class USStockCostModel:
    """
    美股成本模型
    
    模拟 Interactive Brokers 的费率结构：
    - 阶梯式佣金（基于月度交易额）
    - SEC Transaction Fee（卖出时收取）
    - TAF Fee（每股收取）
    - 自适应滑点模型
    """
    
    # IB 佣金阶梯（美股现货）
    # 格式：(月度交易额阈值美元, 每股费率美分, 最低佣金)
    IB_COMMISSION_TIERS = [
        (0, 0.005, 0.35),          # $0 - $25,000: 0.5¢/股，最低 $0.35
        (25000, 0.004, 0.35),      # $25,000 - $100,000: 0.4¢/股
        (100000, 0.003, 0.35),     # $100,000 - $500,000: 0.3¢/股
        (500000, 0.002, 0.35),     # $500,000 - $1,000,000: 0.2¢/股
        (1000000, 0.001, 0.35),    # $1,000,000+: 0.1¢/股
    ]
    
    # SEC Transaction Fee (卖出时收取)
    SEC_FEE_RATE = 0.000027  # 0.0027%
    
    # TAF Fee (每股收取，美股)
    TAF_FEE_PER_SHARE = 0.000166  # 0.0166¢/股
    
    # 基础滑点（按资产类型）
    BASE_SLIPPAGE = {
        "SPY": 0.0001,      # 0.01% - 高流动性
        "QQQ": 0.0001,
        "IWM": 0.00015,
        "TLT": 0.00015,
        "GLD": 0.0002,
        "default": 0.0005,   # 0.05% - 默认
    }
    
    def __init__(
        self,
        monthly_volume: float = 100000.0,  # 默认月度交易额
        default_slippage: Optional[float] = None,
    ):
        """
        初始化成本模型
        
        Args:
            monthly_volume: 预估月度交易额（用于确定佣金阶梯）
            default_slippage: 默认滑点率（可选）
        """
        self.monthly_volume = monthly_volume
        self.default_slippage = default_slippage
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        import logging
        logger = logging.getLogger("cost_model")
        return logger
    
    def get_commission_rate(self, order_value: float) -> float:
        """
        获取佣金费率
        
        Args:
            order_value: 订单金额（美元）
            
        Returns:
            佣金金额（美元）
        """
        # 确定适用的佣金阶梯
        applicable_tier = self.IB_COMMISSION_TIERS[0]
        for threshold, per_share, min_commission in self.IB_COMMISSION_TIERS:
            if self.monthly_volume >= threshold:
                applicable_tier = (threshold, per_share, min_commission)
        
        _, per_share_cents, min_commission = applicable_tier
        
        # 计算佣金
        commission = order_value * (per_share_cents / 100.0)
        
        # 应用最低佣金
        commission = max(commission, min_commission)
        
        # 封顶佣金（订单金额的 1%）
        commission = min(commission, order_value * 0.01)
        
        return commission
    
    def calculate_commission(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        fill_price: float,
    ) -> float:
        """
        计算佣金
        
        Args:
            symbol: 股票代码
            side: 买卖方向
            quantity: 数量
            fill_price: 成交价格
            
        Returns:
            佣金金额
        """
        order_value = quantity * fill_price
        commission = self.get_commission_rate(order_value)
        
        self.logger.debug(
            f"Commission: {symbol} {side.value} {quantity} @ {fill_price} = ${commission:.2f}"
        )
        
        return commission
    
    def calculate_sec_fee(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        fill_price: float,
    ) -> float:
        """
        计算 SEC Transaction Fee（仅卖出时收取）
        
        Args:
            symbol: 股票代码
            side: 买卖方向
            quantity: 数量
            fill_price: 成交价格
            
        Returns:
            SEC 费用
        """
        if side == Side.BUY:
            return 0.0
        
        order_value = quantity * fill_price
        sec_fee = order_value * self.SEC_FEE_RATE
        
        self.logger.debug(f"SEC Fee: {symbol} = ${sec_fee:.4f}")
        
        return sec_fee
    
    def calculate_taf_fee(
        self,
        symbol: str,
        side: Side,
        quantity: float,
    ) -> float:
        """
        计算 TAF Fee（每股收取）
        
        Args:
            symbol: 股票代码
            side: 买卖方向
            quantity: 数量
            
        Returns:
            TAF 费用
        """
        taf_fee = quantity * self.TAF_FEE_PER_SHARE
        
        self.logger.debug(f"TAF Fee: {symbol} = ${taf_fee:.4f}")
        
        return taf_fee
    
    def get_base_slippage(self, symbol: str) -> float:
        """
        获取标的的基础滑点
        
        Args:
            symbol: 股票代码
            
        Returns:
            基础滑点率
        """
        if self.default_slippage is not None:
            return self.default_slippage
        
        # 检查特殊代码
        for prefix in self.BASE_SLIPPAGE.keys():
            if symbol.upper() == prefix.upper():
                return self.BASE_SLIPPAGE[prefix]
        
        return self.BASE_SLIPPAGE["default"]
    
    def calculate_slippage(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        fill_price: float,
        avg_daily_volume: Optional[float] = None,
        order_value: Optional[float] = None,
    ) -> float:
        """
        计算滑点
        
        滑点模型：
        - 基础滑点（基于资产类型）
        - 订单规模因子（订单越大滑点越大）
        - 流动性因子（相对于日成交量）
        
        Args:
            symbol: 股票代码
            side: 买卖方向
            quantity: 数量
            fill_price: 成交价格
            avg_daily_volume: 平均日成交量（可选）
            order_value: 订单金额（可选）
            
        Returns:
            滑点金额
        """
        if order_value is None:
            order_value = quantity * fill_price
        
        base_slippage = self.get_base_slippage(symbol)
        
        # 订单规模因子（订单金额超过 10 万美元时增加）
        size_factor = 1.0
        if order_value > 100000:
            size_factor = 1.0 + (order_value - 100000) / 100000 * 0.5
        
        # 流动性因子
        liquidity_factor = 1.0
        if avg_daily_volume is not None and avg_daily_volume > 0:
            volume_ratio = order_value / avg_daily_volume
            if volume_ratio > 0.01:  # 订单超过日成交量的 1%
                liquidity_factor = 1.0 + (volume_ratio - 0.01) * 10
        
        total_slippage_rate = base_slippage * size_factor * liquidity_factor
        slippage_amount = order_value * total_slippage_rate
        
        self.logger.debug(
            f"Slippage: {symbol} {side.value} = ${slippage_amount:.4f} "
            f"(rate={total_slippage_rate*100:.3f}%, size_factor={size_factor:.2f})"
        )
        
        return slippage_amount
    
    def calculate_total_cost(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        fill_price: float,
        avg_daily_volume: Optional[float] = None,
    ) -> TradeInfo:
        """
        计算总成本
        
        Args:
            symbol: 股票代码
            side: 买卖方向
            quantity: 数量
            fill_price: 成交价格
            avg_daily_volume: 平均日成交量（可选）
            
        Returns:
            TradeInfo 包含所有成本信息
        """
        order_value = quantity * fill_price
        
        # 计算各项费用
        commission = self.calculate_commission(symbol, side, quantity, fill_price)
        sec_fee = self.calculate_sec_fee(symbol, side, quantity, fill_price)
        taf_fee = self.calculate_taf_fee(symbol, side, quantity)
        slippage = self.calculate_slippage(
            symbol, side, quantity, fill_price, avg_daily_volume, order_value
        )
        
        total_cost = commission + sec_fee + taf_fee + slippage
        
        self.logger.info(
            f"Total Cost: {symbol} {side.value} {quantity} @ {fill_price} = "
            f"${total_cost:.2f} (comm=${commission:.2f}, sec=${sec_fee:.4f}, "
            f"taf=${taf_fee:.4f}, slippage=${slippage:.4f})"
        )
        
        # 确定流动性标记
        liquidity_flag = "remove" if side == Side.SELL else "add"
        
        return TradeInfo(
            symbol=symbol,
            side=side,
            quantity=quantity,
            fill_price=fill_price,
            commission=commission,
            sec_fee=sec_fee,
            taf_fee=taf_fee,
            slippage=slippage,
            liquidity_flag=liquidity_flag,
        )
    
    def estimate_commission_for_backtest(
        self,
        trade_value: float,
        tier_multiplier: float = 1.0,
    ) -> float:
        """
        回测用佣金估算（简化版）
        
        Args:
            trade_value: 交易金额
            tier_multiplier: 佣金阶梯系数（0.5-1.0 之间）
            
        Returns:
            估算的佣金
        """
        # 默认使用 0.4¢/股 的费率
        rate = 0.004 * tier_multiplier
        commission = trade_value * rate
        
        # 最低佣金
        commission = max(commission, 0.35)
        
        # 封顶
        commission = min(commission, trade_value * 0.01)
        
        return commission


# 便捷函数
def calculate_us_stock_commission(
    trade_value: float,
    monthly_volume: float = 100000.0,
) -> float:
    """计算美股佣金（便捷函数）"""
    model = USStockCostModel(monthly_volume=monthly_volume)
    return model.get_commission_rate(trade_value)


def estimate_total_cost(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    avg_volume: Optional[float] = None,
) -> Dict[str, float]:
    """
    估算总成本（便捷函数）
    
    Returns:
        包含各项费用的字典
    """
    model = USStockCostModel()
    side_enum = Side.SELL if side.lower() == "sell" else Side.BUY
    
    trade_info = model.calculate_total_cost(
        symbol=symbol,
        side=side_enum,
        quantity=quantity,
        fill_price=price,
        avg_daily_volume=avg_volume,
    )
    
    return {
        "commission": trade_info.commission,
        "sec_fee": trade_info.sec_fee,
        "taf_fee": trade_info.taf_fee,
        "slippage": trade_info.slippage,
        "total": trade_info.commission + trade_info.sec_fee + trade_info.taf_fee + trade_info.slippage,
    }


# 向后兼容：保留原有的简单 CostModel
class CostModel:
    """简单成本模型（向后兼容）"""

    def __init__(self, commission_rate: float = 0.001, slippage_rate: float = 0.0005):
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

    def estimate_fee(self, order, fill_price: float) -> float:
        """估算手续费"""
        from src.core.types import OrderIntent
        if hasattr(order, 'qty'):
            return abs(order.qty) * fill_price * self.commission_rate
        return 0.0

    def estimate_slippage(self, order, fill_price: float) -> float:
        """估算滑点"""
        from src.core.types import OrderIntent
        if hasattr(order, 'qty'):
            return abs(order.qty) * fill_price * self.slippage_rate
        return 0.0


if __name__ == "__main__":
    # 测试成本模型
    print("=" * 60)
    print("US Stock Cost Model Test")
    print("=" * 60)
    
    model = USStockCostModel(monthly_volume=100000)
    
    # 测试 1: 买入 1000 股 SPY @ $450
    print("\n1. Buy 1000 SPY @ $450")
    cost = model.calculate_total_cost(
        symbol="SPY",
        side=Side.BUY,
        quantity=1000,
        fill_price=450.0,
        avg_daily_volume=50000000,
    )
    print(f"   Commission: ${cost.commission:.2f}")
    print(f"   SEC Fee: ${cost.sec_fee:.4f}")
    print(f"   TAF Fee: ${cost.taf_fee:.4f}")
    print(f"   Slippage: ${cost.slippage:.4f}")
    print(f"   Total Cost: ${cost.commission + cost.sec_fee + cost.taf_fee + cost.slippage:.2f}")
    
    # 测试 2: 卖出 500 股 QQQ @ $380
    print("\n2. Sell 500 QQQ @ $380")
    cost = model.calculate_total_cost(
        symbol="QQQ",
        side=Side.SELL,
        quantity=500,
        fill_price=380.0,
        avg_daily_volume=40000000,
    )
    print(f"   Commission: ${cost.commission:.2f}")
    print(f"   SEC Fee: ${cost.sec_fee:.4f}")
    print(f"   TAF Fee: ${cost.taf_fee:.4f}")
    print(f"   Slippage: ${cost.slippage:.4f}")
    print(f"   Total Cost: ${cost.commission + cost.sec_fee + cost.taf_fee + cost.slippage:.2f}")
    
    # 测试 3: 大订单
    print("\n3. Buy 10000 AAPL @ $150 (Large Order)")
    cost = model.calculate_total_cost(
        symbol="AAPL",
        side=Side.BUY,
        quantity=10000,
        fill_price=150.0,
        avg_daily_volume=60000000,
    )
    print(f"   Commission: ${cost.commission:.2f}")
    print(f"   Slippage: ${cost.slippage:.4f}")
    print(f"   Total Cost: ${cost.commission + cost.sec_fee + cost.taf_fee + cost.slippage:.2f}")
    
    # 测试 4: 小订单
    print("\n4. Buy 10 TSLA @ $250 (Small Order)")
    cost = model.calculate_total_cost(
        symbol="TSLA",
        side=Side.BUY,
        quantity=10,
        fill_price=250.0,
        avg_daily_volume=30000000,
    )
    print(f"   Commission: ${cost.commission:.2f}")
    print(f"   Slippage: ${cost.slippage:.4f}")
    print(f"   Total Cost: ${cost.commission + cost.sec_fee + cost.taf_fee + cost.slippage:.2f}")
