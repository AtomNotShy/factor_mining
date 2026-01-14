"""
多个演示策略示例
- simple_momentum: 简单动量策略
- simple_ma: 简单均线策略
- multi_tf_momentum: 多时间框架动量策略

多时间框架策略使用说明：
1. 在策略中使用@informative装饰器定义辅助时间框架
2. 装饰器会自动将数据对齐到主时间框架
3. 通过self.get_informative_pair(timeframe, column)访问已对齐的数据
4. 配置TimeConfig.informative_timeframes列表以指定需要的时间框架

示例配置：
{
  "signal_timeframe": "1h",
  "informative_timeframes": ["4h", "15m"],
  ...
}
"""

from .simple_momentum import SimpleMomentumStrategy
from .simple_ma import SimpleMAStrategy
from .etf_rotation import ETFRotationStrategy

__all__ = ["SimpleMomentumStrategy", "SimpleMAStrategy", "MultiTfMomentumStrategy", "ETFRotationStrategy"]
