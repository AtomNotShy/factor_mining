"""多时间框架informative功能单元测试
测试@informative装饰器、merge_informative_pair等核心功能
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategies.base.informative import (
    informative, 
    merge_informative_pair, 
    merge_informative_pairs,
    resample_to_interval,
    _timeframe_to_minutes
)


class TestTimeframeConversion(unittest.TestCase):
    
    def test_timeframe_to_minutes_minutes(self):
        assert _timeframe_to_minutes('1m') == 1
        assert _timeframe_to_minutes('5m') == 5
        assert _timeframe_to_minutes('15m') == 15
        assert _timeframe_to_minutes('30m') == 30
        assert _timeframe_to_minutes('1h') == 60
        assert _timeframe_to_minutes('2h') == 120
        assert _timeframe_to_minutes('4h') == 240
        assert _timeframe_to_minutes('1d') == 1440
        assert _timeframe_to_minutes('1w') == 10080
    
    def test_timeframe_to_minutes_invalid(self):
        with self.assertRaises(ValueError):
            _timeframe_to_minutes('invalid')


class TestResampleToInterval(unittest.TestCase):
    
    def test_resample_5m_to_1h(self):
        dates = pd.date_range(start='2024-01-01', periods=480, freq='5min', tz='UTC')
        df_5m = pd.DataFrame({
            'open': np.random.randn(480).cumsum() + 100,
            'high': np.random.randn(480).cumsum() + 102,
            'low': np.random.randn(480).cumsum() + 98,
            'close': np.random.randn(480).cumsum() + 100,
            'volume': np.random.randint(100, 1000, 480),
        }, index=dates)
        
        df_1h = resample_to_interval(df_5m, 60)
        
        self.assertEqual(len(df_1h), 40)
        self.assertIn('open', df_1h.columns)
        self.assertIn('high', df_1h.columns)
        self.assertIn('low', df_1h.columns)
        self.assertIn('close', df_1h.columns)
        self.assertIn('volume', df_1h.columns)
    
    def test_resample_empty_dataframe(self):
        df = pd.DataFrame()
        result = resample_to_interval(df, 60)
        self.assertTrue(result.empty)
    
    def test_resample_custom_aggregation(self):
        dates = pd.date_range(start='2024-01-01', periods=240, freq='5min', tz='UTC')
        df = pd.DataFrame({
            'close': np.random.randn(240).cumsum() + 100,
        }, index=dates)
        
        agg = {'close': 'mean'}
        result = resample_to_interval(df, 60, agg=agg)
        
        self.assertIn('close', result.columns)
        self.assertEqual(len(result), 20)


class TestInformativeDecorator(unittest.TestCase):
    
    def test_informative_decorator_marking(self):
        @informative('1h')
        def test_method(self, dataframe, metadata):
            dataframe['test'] = 1
            return dataframe
        
        self.assertTrue(hasattr(test_method, '_is_informative'))


class TestMergeInformativePair(unittest.TestCase):
    
    def test_merge_informative_pair_basic(self):
        dates_main = pd.date_range(start='2024-01-01 09:00', periods=12, freq='5min', tz='UTC')
        df_main = pd.DataFrame({
            'close': [100.0 + i * 0.5 for i in range(12)],
        }, index=dates_main)
        
        dates_inf = pd.date_range(start='2024-01-01 09:00', periods=3, freq='1h', tz='UTC')
        df_inf = pd.DataFrame({
            'rsi': [30.0, 45.0, 50.0],
        }, index=dates_inf)
        
        merged = merge_informative_pair(df_main, df_inf, '5m', '1h', ffill=True)
        
        self.assertIn('rsi_1h', merged.columns)
        self.assertIn('close', merged.columns)
        self.assertEqual(len(merged), 12)
    
    def test_merge_informative_pair_ffill(self):
        dates_main = pd.date_range(start='2024-01-01 09:00', periods=12, freq='5min', tz='UTC')
        df_main = pd.DataFrame({
            'close': [100.0 + i * 0.5 for i in range(12)],
        }, index=dates_main)
        
        dates_inf = pd.date_range(start='2024-01-01 09:00', periods=3, freq='1h', tz='UTC')
        df_inf = pd.DataFrame({
            'value': [10, 20, 30],
        }, index=dates_inf)
        
        merged = merge_informative_pair(df_main, df_inf, '5m', '1h', ffill=True)
        
        self.assertTrue(pd.isna(merged['value_1h'].iloc[0]))
    
    def test_merge_informative_pair_empty_inputs(self):
        dates_main = pd.date_range(start='2024-01-01', periods=3, freq='5min', tz='UTC')
        df_main = pd.DataFrame({
            'close': [1.0, 2.0, 3.0],
        }, index=dates_main)
        
        df_inf = pd.DataFrame()
        
        merged = merge_informative_pair(df_main, df_inf, '5m', '1h')
        
        self.assertIn('close', merged.columns)
        self.assertEqual(len(merged), 3)


class TestMergeInformativePairs(unittest.TestCase):
    
    def test_merge_informative_pairs_multiple(self):
        dates = pd.date_range(start='2024-01-01', periods=12, freq='5min', tz='UTC')
        df_main = pd.DataFrame({
            'close': [100 + i for i in range(12)],
        }, index=dates)
        
        df_1h = pd.DataFrame({
            'rsi': np.random.rand(2) * 50,
        }, index=pd.date_range(start='2024-01-01', periods=2, freq='1h', tz='UTC'))
        
        df_4h = pd.DataFrame({
            'ema': np.random.rand(1) * 100,
        }, index=pd.date_range(start='2024-01-01', periods=1, freq='4h', tz='UTC'))
        
        merged = merge_informative_pairs(
            df_main, 
            {'1h': df_1h, '4h': df_4h},
            '5m',
            ffill=True
        )
        
        self.assertIn('rsi_1h', merged.columns)
        self.assertIn('ema_4h', merged.columns)
        self.assertIn('close', merged.columns)
        self.assertEqual(len(merged), 12)
    
    def test_merge_informative_pairs_empty_dict(self):
        df_main = pd.DataFrame({'close': [1, 2, 3]}, index=pd.DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-03']))
        
        merged = merge_informative_pairs(df_main, {}, '5m')
        
        self.assertEqual(len(merged), 3)
        self.assertIn('close', merged.columns)


if __name__ == '__main__':
    unittest.main()
