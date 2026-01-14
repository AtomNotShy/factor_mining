import React, { useState, useMemo, useCallback } from 'react'
import { format } from 'date-fns'
import ReactECharts from 'echarts-for-react'
import { useThemeStore } from '../../stores/themeStore'
import { formatTimestamp, resolveTimestampSeries } from '../../utils/formatTimestamp'

interface BacktestData {
  strategy_name: string
  symbol?: string
  universe?: string[]
  backtest_period: {
    start_date: string
    end_date: string
    days: number
  }
  config: {
    initial_capital: number
    commission_rate: number
    slippage_rate: number
  }
  price_data: {
    timestamps: string[]
    close: number[]
    volume: number[]
  }
  trades: any[]
  results: any
  enhanced_metrics: any
  equity_comparison: {
    strategy_equity: number[]
    benchmark_equity: number[]
    excess_returns: number[]
    timestamps: string[]
  }
  benchmark_data: any
  timestamp: string
}

interface BacktestDetailsProps {
  data: BacktestData
  onBack: () => void
  onRerun: () => void
}

type MetricCategory = 'returns' | 'risk' | 'benchmark' | 'trades'

const METRIC_CONFIG = {
  returns: {
    label: 'Returns',
    icon: '📈',
    color: '#089981',
    metrics: [
      { key: 'total_return', label: 'Total Return', format: 'percent' },
      { key: 'benchmark_return', label: 'Benchmark Return', format: 'percent' },
      { key: 'annual_return', label: 'Annual Return', format: 'percent' },
      { key: 'excess_return', label: 'Excess Return', format: 'percent' },
      { key: 'best_day', label: 'Best Day', format: 'percent' },
      { key: 'worst_day', label: 'Worst Day', format: 'percent' },
    ]
  },
  risk: {
    label: 'Risk',
    icon: '⚠️',
    color: '#f23645',
    metrics: [
      { key: 'max_drawdown', label: 'Max Drawdown', format: 'percent' },
      { key: 'annual_volatility', label: 'Volatility', format: 'percent' },
      { key: 'benchmark_volatility', label: 'Benchmark Vol', format: 'percent' },
      { key: 'sharpe_ratio', label: 'Sharpe Ratio', format: 'number' },
      { key: 'sortino_ratio', label: 'Sortino Ratio', format: 'number' },
      { key: 'calmar_ratio', label: 'Calmar Ratio', format: 'number' },
    ]
  },
  benchmark: {
    label: 'Benchmark',
    icon: '🎯',
    color: '#2962ff',
    metrics: [
      { key: 'alpha', label: 'Alpha', format: 'number' },
      { key: 'beta', label: 'Beta', format: 'number' },
      { key: 'information_ratio', label: 'Info Ratio', format: 'number' },
      { key: 'r_squared', label: 'R-Squared', format: 'number' },
      { key: 'tracking_error', label: 'Tracking Error', format: 'number' },
    ]
  },
  trades: {
    label: 'Trades',
    icon: '📊',
    color: '#e91e63',
    metrics: [
      { key: 'total_trades', label: 'Total Trades', format: 'integer' },
      { key: 'winning_trades', label: 'Winning Trades', format: 'integer' },
      { key: 'losing_trades', label: 'Losing Trades', format: 'integer' },
      { key: 'win_rate', label: 'Win Rate', format: 'percent' },
      { key: 'daily_win_rate', label: 'Daily Win Rate', format: 'percent' },
      { key: 'profit_loss_ratio', label: 'P/L Ratio', format: 'number' },
      { key: 'profit_factor', label: 'Profit Factor', format: 'number' },
      { key: 'expectancy', label: 'Expectancy', format: 'number' },
      { key: 'consecutive_wins', label: 'Consecutive Wins', format: 'integer' },
      { key: 'consecutive_losses', label: 'Consecutive Losses', format: 'integer' },
    ]
  }
}

const formatValue = (value: number | undefined, format: string): string => {
  if (value === undefined || value === null || !Number.isFinite(value)) return '-'
  if (format === 'percent') {
    const percentVal = value >= 1 ? value : value * 100
    return `${percentVal.toFixed(0)}%`
  }
  if (format === 'currency') return `${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  if (format === 'integer') return value.toFixed(0)
  return value.toFixed(2)
}

const getValueColor = (value: number | undefined, key: string): string => {
  if (value === undefined || value === null) return 'inherit'
  const positiveKeys = ['total_return', 'annual_return', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'win_rate', 'profit_factor', 'expectancy', 'excess_return', 'alpha', 'benchmark_return', 'daily_win_rate', 'winning_trades', 'total_trades']
  if (positiveKeys.includes(key)) {
    return value >= 0 ? '#089981' : '#f23645'
  }
  if (key === 'max_drawdown' || key === 'annual_volatility' || key === 'benchmark_volatility' || key === 'beta') {
    return value > 1 ? '#f23645' : value > 0.8 ? '#ff9800' : '#089981'
  }
  if (key === 'losing_trades') {
    return '#f23645'
  }
  if (key === 'profit_loss_ratio' || key === 'information_ratio') {
    return value >= 0.5 ? '#089981' : value >= 0 ? '#ff9800' : '#f23645'
  }
  return 'inherit'
}

export default function BacktestDetails({ data, onBack, onRerun }: BacktestDetailsProps) {
  const { theme } = useThemeStore()
  const isDark = theme === 'dark'

  const [activeTab, setActiveTab] = useState<'overview' | 'trades' | 'positions' | 'analysis'>('overview')
  const [showBenchmark, setShowBenchmark] = useState(true)
  const [selectedTradePage, setSelectedTradePage] = useState(1)

  // Guard against invalid data
  if (!data || !data.backtest_period || !data.config) {
    return (
      <div className="p-8 text-center text-red-500">
        Error: Invalid backtest data format. Missing critical fields.
        <button onClick={onBack} className="ml-4 underline">Go Back</button>
      </div>
    )
  }

  const m = data.enhanced_metrics || {}
  const eq = data.equity_comparison || {}
  const benchmarkLabel = data.benchmark_data?.symbol || 'Benchmark'
  const resolvedTimestamps = useMemo(() => {
    return resolveTimestampSeries(eq.timestamps || [], data.price_data?.timestamps || [])
  }, [eq.timestamps, data.price_data?.timestamps])

  const tabs = [
    { id: 'overview' as const, label: 'Overview', icon: '📊' },
    { id: 'trades' as const, label: 'Trades', icon: '📋' },
    { id: 'positions' as const, label: 'Positions', icon: '💼' },
    { id: 'analysis' as const, label: 'Analysis', icon: '📈' },
  ]

  const equityChartOption = useMemo(() => {
    const xData = resolvedTimestamps.map(formatTimestamp)
    const strategyData = eq.strategy_equity || []
    const benchmarkData = showBenchmark ? eq.benchmark_equity || [] : []
    const formatValue = (value: number) => {
      if (!Number.isFinite(value)) return '-'
      return value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    }

    return {
      tooltip: {
        trigger: 'axis',
        backgroundColor: isDark ? '#1e222d' : '#ffffff',
        borderColor: isDark ? '#2a2e39' : '#e5e5e5',
        textStyle: { color: isDark ? '#d1d4dc' : '#1a1a1a' },
        axisPointer: { type: 'cross' },
        formatter: (params: any) => {
          if (!Array.isArray(params) || params.length === 0) return ''
          const title = params[0].axisValueLabel ?? params[0].name ?? ''
          const lines = params.map((item: any) => {
            const raw = Array.isArray(item.value) ? item.value[1] : item.value
            return `${item.marker || ''} ${item.seriesName}: ${formatValue(Number(raw))}`
          })
          return [title, ...lines].join('<br/>')
        },
      },
      legend: {
        data: ['Strategy', benchmarkLabel],
        top: 8,
        left: 12,
        textStyle: { color: isDark ? '#d1d4dc' : '#1a1a1a' },
      },
      grid: {
        left: 60,
        right: 20,
        top: 48,
        bottom: 30,
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: xData,
        axisLine: { lineStyle: { color: isDark ? '#2a2e39' : '#e5e5e5' } },
        axisLabel: { color: isDark ? '#787b86' : '#6b7280', fontSize: 10 },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value',
        position: 'left',
        axisLabel: { formatter: '${value}', color: isDark ? '#d1d4dc' : '#1a1a1a', fontSize: 10 },
        splitLine: { lineStyle: { color: isDark ? '#2a2e39' : '#f0f0f0' } },
        scale: true,
      },
      series: [
        {
          name: 'Strategy',
          type: 'line',
          smooth: true,
          symbol: 'none',
          data: strategyData,
          lineStyle: { color: '#089981', width: 2 },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(8, 153, 129, 0.3)' },
                { offset: 1, color: 'rgba(8, 153, 129, 0)' },
              ],
            },
          },
        },
        ...(showBenchmark ? [{
          name: benchmarkLabel,
          type: 'line',
          smooth: true,
          symbol: 'none',
          data: benchmarkData,
          lineStyle: { color: '#787b86', width: 2, type: 'dashed' },
        }] : []),
      ],
      dataZoom: [
        { type: 'inside', start: 0, end: 100 },
        { type: 'slider', bottom: 0, height: 20, borderColor: 'transparent', backgroundColor: isDark ? '#2a2e39' : '#f0f0f0' },
      ],
    }
  }, [eq, showBenchmark, isDark, benchmarkLabel, resolvedTimestamps])

  const drawdownChartOption = useMemo(() => {
    const drawdownSeries = m.drawdown_series || []
    const xData = resolvedTimestamps.map(formatTimestamp)
    const formatPercent = (value: number) => {
      if (!Number.isFinite(value)) return '-'
      return `${value.toFixed(2)}%`
    }

    return {
      tooltip: {
        trigger: 'axis',
        backgroundColor: isDark ? '#1e222d' : '#ffffff',
        borderColor: isDark ? '#2a2e39' : '#e5e5e5',
        textStyle: { color: isDark ? '#d1d4dc' : '#1a1a1a' },
        formatter: (params: any) => {
          if (!Array.isArray(params) || params.length === 0) return ''
          const title = params[0].axisValueLabel ?? params[0].name ?? ''
          const lines = params.map((item: any) => {
            const raw = Array.isArray(item.value) ? item.value[1] : item.value
            return `${item.marker || ''} ${item.seriesName || 'Drawdown'}: ${formatPercent(Number(raw))}`
          })
          return [title, ...lines].join('<br/>')
        },
      },
      grid: { left: 50, right: 20, top: 20, bottom: 30 },
      xAxis: {
        type: 'category',
        data: xData,
        axisLine: { lineStyle: { color: isDark ? '#2a2e39' : '#e5e5e5' } },
        axisLabel: { color: isDark ? '#787b86' : '#6b7280', fontSize: 9 },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value',
        axisLabel: { formatter: '{value}%', color: isDark ? '#787b86' : '#6b7280', fontSize: 9 },
        splitLine: { lineStyle: { color: isDark ? '#2a2e39' : '#f0f0f0' } },
        max: 5,
      },
      series: [{
        type: 'line',
        data: drawdownSeries.map((v: number) => (v * 100).toFixed(2)),
        symbol: 'none',
        lineStyle: { color: '#f23645', width: 1.5 },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(242, 54, 69, 0.3)' },
              { offset: 1, color: 'rgba(242, 54, 69, 0)' },
            ],
          },
        },
      }],
    }
  }, [m, eq, isDark])

  const tradePageSize = 20
  const totalTradePages = Math.ceil((data.trades?.length || 0) / tradePageSize)
  const paginatedTrades = data.trades?.slice(
    (selectedTradePage - 1) * tradePageSize,
    selectedTradePage * tradePageSize
  ) || []

  const containerStyle: React.CSSProperties = {
    height: '100vh',
    backgroundColor: isDark ? '#131722' : '#ffffff',
    color: isDark ? '#d1d4dc' : '#1a1a1a',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  }

  const panelStyle: React.CSSProperties = {
    backgroundColor: isDark ? '#1e222d' : '#ffffff',
    border: `1px solid ${isDark ? '#2a2e39' : '#e5e5e5'}`,
    borderRadius: 8,
  }

  return (
    <div style={containerStyle}>
      {/* Header */}
      <div
        style={{
          height: 56,
          padding: '0 20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: `1px solid ${isDark ? '#2a2e39' : '#e5e5e5'}`,
          backgroundColor: isDark ? '#1e222d' : '#ffffff',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <button
            onClick={onBack}
            style={{
              padding: '6px 12px',
              backgroundColor: 'transparent',
              border: `1px solid ${isDark ? '#2a2e39' : '#e5e5e5'}`,
              borderRadius: 4,
              color: 'inherit',
              cursor: 'pointer',
              fontSize: 13,
            }}
          >
            ← Back
          </button>
          <div>
            <div style={{ fontSize: 16, fontWeight: 600 }}>{data.strategy_name}</div>
            <div style={{ fontSize: 12, color: isDark ? '#787b86' : '#6b7280' }}>
              {(() => {
                const symbols = data.universe && data.universe.length > 0
                  ? data.universe
                  : (data.symbol ? [data.symbol] : [])
                if (symbols.length <= 3) {
                  return symbols.join(', ')
                }
                return `${symbols.slice(0, 3).join(', ')} +${symbols.length - 3}`
              })()} · {data.backtest_period.days} days · {format(new Date(data.backtest_period.start_date), 'MMM d')} - {format(new Date(data.backtest_period.end_date), 'MMM d, yyyy')}
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            onClick={onRerun}
            style={{
              padding: '6px 16px',
              backgroundColor: '#2962ff',
              border: 'none',
              borderRadius: 4,
              color: '#ffffff',
              cursor: 'pointer',
              fontSize: 12,
              fontWeight: 500,
              zIndex: 10,
              position: 'relative',
            }}
          >
            Re-run
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Left Panel - Metrics */}
        <div
          style={{
            width: 320,
            padding: 16,
            overflow: 'auto',
            borderRight: `1px solid ${isDark ? '#2a2e39' : '#e5e5e5'}`,
            backgroundColor: isDark ? '#131722' : '#f8f9fa',
          }}
        >
          {/* Strategy Config */}
          <div style={{ ...panelStyle, padding: 16, marginBottom: 16 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: isDark ? '#787b86' : '#6b7280', marginBottom: 12, textTransform: 'uppercase' }}>
              Configuration
            </div>
            <div style={{ display: 'grid', gap: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                <span style={{ color: isDark ? '#787b86' : '#6b7280' }}>Initial Capital</span>
                <span style={{ fontWeight: 500 }}>${data.config.initial_capital.toLocaleString()}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                <span style={{ color: isDark ? '#787b86' : '#6b7280' }}>Commission</span>
                <span style={{ fontWeight: 500 }}>{(data.config.commission_rate * 100).toFixed(3)}%</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                <span style={{ color: isDark ? '#787b86' : '#6b7280' }}>Slippage</span>
                <span style={{ fontWeight: 500 }}>{(data.config.slippage_rate * 100).toFixed(3)}%</span>
              </div>
            </div>
          </div>

          {/* Metrics by Category */}
          {(Object.keys(METRIC_CONFIG) as MetricCategory[]).map((category) => {
            const config = METRIC_CONFIG[category]
            return (
              <div key={category} style={{ ...panelStyle, padding: 16, marginBottom: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                  <span>{config.icon}</span>
                  <span style={{ fontSize: 11, fontWeight: 600, color: config.color, textTransform: 'uppercase' }}>
                    {config.label}
                  </span>
                </div>
                <div style={{ display: 'grid', gap: 10 }}>
                  {config.metrics.map((metric) => {
                    const value = m[metric.key]
                    const color = getValueColor(value, metric.key)
                    return (
                      <div key={metric.key} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: 12, color: isDark ? '#787b86' : '#6b7280' }}>
                          {metric.label}
                        </span>
                        <span style={{ fontSize: 13, fontWeight: 600, color }}>
                          {formatValue(value, metric.format)}
                        </span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>

        {/* Right Panel - Charts & Data */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Tabs */}
          <div
            style={{
              display: 'flex',
              padding: '0 20px',
              borderBottom: `1px solid ${isDark ? '#2a2e39' : '#e5e5e5'}`,
              backgroundColor: isDark ? '#1e222d' : '#ffffff',
            }}
          >
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  padding: '12px 20px',
                  fontSize: 13,
                  fontWeight: 500,
                  backgroundColor: activeTab === tab.id ? 'transparent' : 'transparent',
                  color: activeTab === tab.id ? '#2962ff' : isDark ? '#787b86' : '#6b7280',
                  border: 'none',
                  borderBottom: activeTab === tab.id ? '2px solid #2962ff' : '2px solid transparent',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                }}
              >
                <span>{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div style={{ flex: 1, overflow: 'auto', padding: 16 }}>
            {activeTab === 'overview' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                {/* Equity Chart */}
                <div style={{ ...panelStyle, height: 400, padding: 16 }}>
                  <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Equity Curve</div>
                  <div style={{ height: 340 }}>
                    <ReactECharts option={equityChartOption} style={{ width: '100%', height: '100%' }} opts={{ renderer: 'canvas' }} />
                  </div>
                </div>

                {/* Drawdown Chart */}
                <div style={{ ...panelStyle, height: 200, padding: 16 }}>
                  <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Drawdown</div>
                  <div style={{ height: 140 }}>
                    <ReactECharts option={drawdownChartOption} style={{ width: '100%', height: '100%' }} opts={{ renderer: 'canvas' }} />
                  </div>
                </div>

                {/* Performance Summary - All Key Metrics */}
                <div style={{ ...panelStyle, padding: 16 }}>
                  <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 16 }}>Performance Summary</div>

                  {/* Return & Benchmark Metrics */}
                  <div style={{ fontSize: 12, fontWeight: 600, color: isDark ? '#787b86' : '#6b7280', marginBottom: 8, textTransform: 'uppercase' }}>Returns</div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
                    {[
                      { label: 'Strategy Return', value: m.total_return, format: 'percent', key: 'total_return' },
                      { label: 'Benchmark Return', value: m.benchmark_return, format: 'percent', key: 'benchmark_return' },
                      { label: 'Annual Return', value: m.annual_return, format: 'percent', key: 'annual_return' },
                      { label: 'Excess Return', value: m.excess_return, format: 'percent', key: 'excess_return' },
                    ].map((item) => (
                      <div key={item.label} style={{ padding: 12, backgroundColor: isDark ? '#2a2e39' : '#f5f5f5', borderRadius: 6 }}>
                        <div style={{ fontSize: 11, color: isDark ? '#787b86' : '#6b7280', marginBottom: 4 }}>{item.label}</div>
                        <div style={{ fontSize: 16, fontWeight: 600, color: getValueColor(item.value, item.key) }}>
                          {formatValue(item.value, item.format)}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Risk & Alpha/Beta Metrics */}
                  <div style={{ fontSize: 12, fontWeight: 600, color: isDark ? '#787b86' : '#6b7280', marginBottom: 8, textTransform: 'uppercase' }}>Risk & Alpha/Beta</div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
                    {[
                      { label: 'Alpha', value: m.alpha, format: 'number', key: 'alpha' },
                      { label: 'Beta', value: m.beta, format: 'number', key: 'beta' },
                      { label: 'Sharpe', value: m.sharpe_ratio, format: 'number', key: 'sharpe_ratio' },
                      { label: 'Sortino', value: m.sortino_ratio, format: 'number', key: 'sortino_ratio' },
                    ].map((item) => (
                      <div key={item.label} style={{ padding: 12, backgroundColor: isDark ? '#2a2e39' : '#f5f5f5', borderRadius: 6 }}>
                        <div style={{ fontSize: 11, color: isDark ? '#787b86' : '#6b7280', marginBottom: 4 }}>{item.label}</div>
                        <div style={{ fontSize: 16, fontWeight: 600, color: getValueColor(item.value, item.key) }}>
                          {formatValue(item.value, item.format)}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Volatility & Drawdown */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
                    {[
                      { label: 'Info Ratio', value: m.information_ratio, format: 'number', key: 'information_ratio' },
                      { label: 'Volatility', value: m.annual_volatility, format: 'percent', key: 'annual_volatility' },
                      { label: 'Bench Vol', value: m.benchmark_volatility, format: 'percent', key: 'benchmark_volatility' },
                      { label: 'Max Drawdown', value: m.max_drawdown, format: 'percent', key: 'max_drawdown' },
                    ].map((item) => (
                      <div key={item.label} style={{ padding: 12, backgroundColor: isDark ? '#2a2e39' : '#f5f5f5', borderRadius: 6 }}>
                        <div style={{ fontSize: 11, color: isDark ? '#787b86' : '#6b7280', marginBottom: 4 }}>{item.label}</div>
                        <div style={{ fontSize: 16, fontWeight: 600, color: getValueColor(item.value, item.key) }}>
                          {formatValue(item.value, item.format)}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Win Rate & Trade Metrics */}
                  <div style={{ fontSize: 12, fontWeight: 600, color: isDark ? '#787b86' : '#6b7280', marginBottom: 8, textTransform: 'uppercase' }}>Trade Statistics</div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
                    {[
                      { label: 'Win Rate', value: m.win_rate, format: 'percent', key: 'win_rate' },
                      { label: 'Daily Win Rate', value: m.daily_win_rate, format: 'percent', key: 'daily_win_rate' },
                      { label: 'P/L Ratio', value: m.profit_loss_ratio, format: 'number', key: 'profit_loss_ratio' },
                      { label: 'Winning Trades', value: m.winning_trades, format: 'integer', key: 'winning_trades' },
                    ].map((item) => (
                      <div key={item.label} style={{ padding: 12, backgroundColor: isDark ? '#2a2e39' : '#f5f5f5', borderRadius: 6 }}>
                        <div style={{ fontSize: 11, color: isDark ? '#787b86' : '#6b7280', marginBottom: 4 }}>{item.label}</div>
                        <div style={{ fontSize: 16, fontWeight: 600, color: getValueColor(item.value, item.key) }}>
                          {formatValue(item.value, item.format)}
                        </div>
                      </div>
                    ))}
                    {[
                      { label: 'Losing Trades', value: m.losing_trades, format: 'integer', key: 'losing_trades' },
                      { label: 'Total Trades', value: m.total_trades, format: 'integer', key: 'total_trades' },
                    ].map((item) => (
                      <div key={item.label} style={{ padding: 12, backgroundColor: isDark ? '#2a2e39' : '#f5f5f5', borderRadius: 6 }}>
                        <div style={{ fontSize: 11, color: isDark ? '#787b86' : '#6b7280', marginBottom: 4 }}>{item.label}</div>
                        <div style={{ fontSize: 16, fontWeight: 600, color: item.label === 'Losing Trades' ? '#f23645' : 'inherit' }}>
                          {formatValue(item.value, item.format)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'trades' && (
              <div style={{ ...panelStyle, padding: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column', height: '100%' }}>
                {/* Trade Table Header */}
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '140px 80px 100px 100px 120px 100px',
                    padding: '10px 16px',
                    backgroundColor: isDark ? '#2a2e39' : '#f5f5f5',
                    fontSize: 11,
                    fontWeight: 600,
                    color: isDark ? '#787b86' : '#6b7280',
                    textTransform: 'uppercase',
                  }}
                >
                  <span>Time</span>
                  <span>Symbol</span>
                  <span>Type</span>
                  <span style={{ textAlign: 'right' }}>Price</span>
                  <span style={{ textAlign: 'right' }}>Size</span>
                  <span style={{ textAlign: 'right' }}>Amount</span>
                </div>

                {/* Trade List */}
                <div style={{ flex: 1, overflow: 'auto' }}>
                  {paginatedTrades.map((trade: any, i: number) => (
                    <div
                      key={i}
                      style={{
                        display: 'grid',
                        gridTemplateColumns: '140px 80px 100px 100px 120px 100px',
                        padding: '10px 16px',
                        borderBottom: `1px solid ${isDark ? '#2a2e39' : '#e5e5e5'}`,
                        fontSize: 12,
                        alignItems: 'center',
                      }}
                    >
                      <span style={{ color: isDark ? '#d1d4dc' : '#1a1a1a' }}>
                        {trade.timestamp ? format(new Date(trade.timestamp), 'yyyy-MM-dd HH:mm') : '-'}
                      </span>
                      <span style={{ color: isDark ? '#d1d4dc' : '#1a1a1a' }}>{trade.symbol || '-'}</span>
                      <span>
                        <span
                          style={{
                            padding: '2px 6px',
                            borderRadius: 2,
                            fontSize: 10,
                            fontWeight: 600,
                            backgroundColor: trade.order_type === 'buy' ? 'rgba(8, 153, 129, 0.2)' : 'rgba(242, 54, 69, 0.2)',
                            color: trade.order_type === 'buy' ? '#089981' : '#f23645',
                          }}
                        >
                          {trade.order_type?.toUpperCase() || '-'}
                        </span>
                      </span>
                      <span style={{ textAlign: 'right', color: isDark ? '#d1d4dc' : '#1a1a1a' }}>
                        ${trade.price?.toFixed(2) || '0.00'}
                      </span>
                      <span style={{ textAlign: 'right', color: isDark ? '#d1d4dc' : '#1a1a1a' }}>
                        {Math.round(trade.size) || '0'}
                      </span>
                      <span style={{ textAlign: 'right', color: isDark ? '#d1d4dc' : '#1a1a1a' }}>
                        ${(trade.price * trade.size).toFixed(2)}
                      </span>
                    </div>
                  ))}
                </div>

                {/* Pagination */}
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '10px 16px',
                    borderTop: `1px solid ${isDark ? '#2a2e39' : '#e5e5e5'}`,
                    fontSize: 12,
                  }}
                >
                  <span style={{ color: isDark ? '#787b86' : '#6b7280' }}>
                    {data.trades?.length || 0} trades
                  </span>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <button
                      onClick={() => setSelectedTradePage(Math.max(1, selectedTradePage - 1))}
                      disabled={selectedTradePage === 1}
                      style={{
                        padding: '4px 8px',
                        backgroundColor: 'transparent',
                        border: `1px solid ${isDark ? '#2a2e39' : '#e5e5e5'}`,
                        borderRadius: 4,
                        color: 'inherit',
                        cursor: selectedTradePage === 1 ? 'not-allowed' : 'pointer',
                        opacity: selectedTradePage === 1 ? 0.5 : 1,
                      }}
                    >
                      Prev
                    </button>
                    <span style={{ color: isDark ? '#d1d4dc' : '#1a1a1a' }}>
                      {selectedTradePage} / {totalTradePages || 1}
                    </span>
                    <button
                      onClick={() => setSelectedTradePage(Math.min(totalTradePages, selectedTradePage + 1))}
                      disabled={selectedTradePage === totalTradePages}
                      style={{
                        padding: '4px 8px',
                        backgroundColor: 'transparent',
                        border: `1px solid ${isDark ? '#2a2e39' : '#e5e5e5'}`,
                        borderRadius: 4,
                        color: 'inherit',
                        cursor: selectedTradePage === totalTradePages ? 'not-allowed' : 'pointer',
                        opacity: selectedTradePage === totalTradePages ? 0.5 : 1,
                      }}
                    >
                      Next
                    </button>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'positions' && (
              <div style={{ ...panelStyle, padding: 40, textAlign: 'center' }}>
                <div style={{ fontSize: 48, marginBottom: 16 }}>💼</div>
                <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>No Active Positions</div>
                <div style={{ fontSize: 13, color: isDark ? '#787b86' : '#6b7280' }}>
                  Positions will appear here when trades are executed.
                </div>
              </div>
            )}

            {activeTab === 'analysis' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <div style={{ ...panelStyle, padding: 16 }}>
                  <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 16 }}>Risk Metrics</div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
                    {[
                      { label: 'Ulcer Index', value: m.ulcer_index },
                      { label: 'Burke Ratio', value: m.burke_ratio },
                      { label: 'Time in Market', value: m.time_in_market },
                      { label: 'Avg Drawdown', value: m.avg_drawdown },
                      { label: 'Avg Duration', value: m.avg_drawdown_duration },
                      { label: 'Recovery Days', value: m.max_drawdown_window?.recovery_days },
                    ].map((item) => (
                      <div key={item.label} style={{ padding: 12, backgroundColor: isDark ? '#2a2e39' : '#f5f5f5', borderRadius: 6 }}>
                        <div style={{ fontSize: 11, color: isDark ? '#787b86' : '#6b7280', marginBottom: 4 }}>{item.label}</div>
                        <div style={{ fontSize: 16, fontWeight: 600 }}>
                          {item.value != null ? item.value.toFixed(3) : '-'}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {m.max_drawdown_window && (
                  <div style={{ ...panelStyle, padding: 16 }}>
                    <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 16 }}>Maximum Drawdown Period</div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
                      <div>
                        <div style={{ fontSize: 11, color: isDark ? '#787b86' : '#6b7280', marginBottom: 4 }}>Period</div>
                        <div style={{ fontSize: 13, fontWeight: 500 }}>
                          {m.max_drawdown_window.start_date?.split('T')[0]} → {m.max_drawdown_window.end_date?.split('T')[0]}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 11, color: isDark ? '#787b86' : '#6b7280', marginBottom: 4 }}>Drawdown</div>
                        <div style={{ fontSize: 13, fontWeight: 600, color: '#f23645' }}>
                          {(m.max_drawdown_window.drawdown_pct * 100).toFixed(2)}%
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 11, color: isDark ? '#787b86' : '#6b7280', marginBottom: 4 }}>Duration</div>
                        <div style={{ fontSize: 13, fontWeight: 500 }}>
                          {m.max_drawdown_window.duration_days} days
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: 11, color: isDark ? '#787b86' : '#6b7280', marginBottom: 4 }}>Recovery</div>
                        <div style={{ fontSize: 13, fontWeight: 500 }}>
                          {m.max_drawdown_window.recovery_days ? `${m.max_drawdown_window.recovery_days} days` : 'Not recovered'}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
