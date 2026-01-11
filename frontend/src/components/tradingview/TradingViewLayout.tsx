import React, { useState, useCallback, useMemo } from 'react'
import { useThemeStore } from '../../stores/themeStore'
import { resolveTimestampSeries } from '../../utils/formatTimestamp'
import ChartToolbar from './ChartToolbar'
import LeftPanel from './LeftPanel'
import MainChart from './MainChart'
import SubChart from './SubChart'
import BottomPanel from './BottomPanel'

export type TimeRange = '1D' | '1W' | '1M' | '3M' | '6M' | 'YTD' | '1Y' | '2Y' | '5Y' | 'ALL'
export type ChartType = 'area' | 'line'
export type SubChartType = 'drawdown' | 'returns' | 'distribution' | 'monthly' | 'rolling'
export type BottomTab = 'trades' | 'positions' | 'summary'

interface BacktestData {
  strategy_name: string
  symbol: string
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
  enhanced_metrics: any
  equity_comparison: {
    strategy_equity: number[]
    benchmark_equity: number[]
    excess_returns: number[]
    timestamps: string[]
  }
  benchmark_data: any
  trades: any[]
  results: any
  price_data?: {
    timestamps: string[]
  }
}

interface TradingViewLayoutProps {
  data: BacktestData | null
  loading: boolean
  onRunBacktest: (params: any) => void
}

const TIME_RANGES = [
  { value: '1D', label: '1D' },
  { value: '1W', label: '1W' },
  { value: '1M', label: '1M' },
  { value: '3M', label: '3M' },
  { value: '6M', label: '6M' },
  { value: 'YTD', label: 'YTD' },
  { value: '1Y', label: '1Y' },
  { value: '2Y', label: '2Y' },
  { value: '5Y', label: '5Y' },
  { value: 'ALL', label: 'ALL' },
]

const SUB_CHART_OPTIONS = [
  { value: 'drawdown', label: '回撤', icon: '📉' },
  { value: 'returns', label: '收益', icon: '📊' },
  { value: 'distribution', label: '分布', icon: '📐' },
  { value: 'monthly', label: '月度', icon: '🗓️' },
  { value: 'rolling', label: '滚动', icon: '📈' },
]

export default function TradingViewLayout({ data, loading, onRunBacktest }: TradingViewLayoutProps) {
  const { theme } = useThemeStore()
  const isDark = theme === 'dark'

  const [leftPanelCollapsed, setLeftPanelCollapsed] = useState(false)
  const [bottomPanelVisible, setBottomPanelVisible] = useState(true)
  const [bottomPanelHeight, setBottomPanelHeight] = useState(280)
  const [leftPanelWidth, setLeftPanelWidth] = useState(300)
  const [isResizingLeft, setIsResizingLeft] = useState(false)
  const [isResizingBottom, setIsResizingBottom] = useState(false)

  const [timeRange, setTimeRange] = useState('ALL' as string)
  const [chartType, setChartType] = useState('area' as string)
  const [subChartType, setSubChartType] = useState('drawdown' as string)
  const [bottomTab, setBottomTab] = useState('trades' as string)
  const [showBenchmark, setShowBenchmark] = useState(true)
  const [showExcess, setShowExcess] = useState(true)
  const [useLogScale, setUseLogScale] = useState(false)

  const handleResizeLeft = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    setIsResizingLeft(true)
    
    const handleMouseMove = (e: MouseEvent) => {
      const newWidth = Math.max(250, Math.min(500, e.clientX))
      setLeftPanelWidth(newWidth)
    }
    
    const handleMouseUp = () => {
      setIsResizingLeft(false)
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
    
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }, [])

  const handleResizeBottom = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    setIsResizingBottom(true)
    
    const handleMouseMove = (e: MouseEvent) => {
      const newHeight = Math.max(150, Math.min(500, window.innerHeight - e.clientY))
      setBottomPanelHeight(newHeight)
    }
    
    const handleMouseUp = () => {
      setIsResizingBottom(false)
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
    
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }, [])

  const filteredData = useMemo(() => {
    if (!data) return null

    const { equity_comparison } = data
    const { timestamps } = equity_comparison
    const resolvedTimestamps = resolveTimestampSeries(
      timestamps || [],
      data.price_data?.timestamps || []
    )

    if (timeRange === 'ALL') {
      return {
        ...data,
        equity_comparison: {
          ...equity_comparison,
          timestamps: resolvedTimestamps,
        },
      }
    }

    const now = new Date()
    let startDate = new Date()
    
    switch (timeRange) {
      case '1D': startDate.setDate(now.getDate() - 1); break
      case '1W': startDate.setDate(now.getDate() - 7); break
      case '1M': startDate.setMonth(now.getMonth() - 1); break
      case '3M': startDate.setMonth(now.getMonth() - 3); break
      case '6M': startDate.setMonth(now.getMonth() - 6); break
      case 'YTD': startDate = new Date(now.getFullYear(), 0, 1); break
      case '1Y': startDate.setFullYear(now.getFullYear() - 1); break
      case '2Y': startDate.setFullYear(now.getFullYear() - 2); break
      case '5Y': startDate.setFullYear(now.getFullYear() - 5); break
    }

    const startTime = startDate.getTime()
    const indices = resolvedTimestamps.map((_, i) => i).filter((i) => {
      const ts = new Date(resolvedTimestamps[i]).getTime()
      return ts >= startTime
    })

    if (indices.length === 0) {
      return data
    }

    const startIdx = indices[0]
    const endIdx = indices[indices.length - 1] + 1

    return {
      ...data,
      equity_comparison: {
        ...equity_comparison,
        strategy_equity: equity_comparison.strategy_equity.slice(startIdx, endIdx),
        benchmark_equity: equity_comparison.benchmark_equity.slice(startIdx, endIdx),
        excess_returns: equity_comparison.excess_returns.slice(startIdx, endIdx),
        timestamps: resolvedTimestamps.slice(startIdx, endIdx),
      },
    }
  }, [data, timeRange])

  const containerStyle: React.CSSProperties = {
    backgroundColor: isDark ? '#131722' : '#ffffff',
    color: isDark ? '#d1d4dc' : '#1a1a1a',
    height: '100vh',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  }

  const contentStyle: React.CSSProperties = {
    flex: 1,
    display: 'flex',
    overflow: 'hidden',
  }

  const chartAreaStyle: React.CSSProperties = {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  }

  if (!data) {
    return (
      <div style={containerStyle}>
        <ChartToolbar
          strategyName=""
          symbol=""
          timeRange={timeRange}
          chartType={chartType}
          showBenchmark={showBenchmark}
          showExcess={showExcess}
          useLogScale={useLogScale}
          onTimeRangeChange={setTimeRange}
          onChartTypeChange={setChartType}
          onShowBenchmarkChange={setShowBenchmark}
          onShowExcessChange={setShowExcess}
          onUseLogScaleChange={setUseLogScale}
          onFullScreen={() => {}}
          onSettings={() => {}}
        />
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ textAlign: 'center', color: isDark ? '#787b86' : '#6b7280' }}>
            <p style={{ fontSize: 18, marginBottom: 16 }}>No backtest data</p>
            <p style={{ fontSize: 14 }}>Run a backtest to see results</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div style={containerStyle}>
      <ChartToolbar
        strategyName={data.strategy_name}
        symbol={data.symbol}
        timeRange={timeRange}
        chartType={chartType}
        showBenchmark={showBenchmark}
        showExcess={showExcess}
        useLogScale={useLogScale}
        onTimeRangeChange={(range) => setTimeRange(range)}
        onChartTypeChange={(type) => setChartType(type)}
        onShowBenchmarkChange={setShowBenchmark}
        onShowExcessChange={setShowExcess}
        onUseLogScaleChange={setUseLogScale}
        onFullScreen={() => {}}
        onSettings={() => {}}
      />

      <div style={contentStyle}>
        <LeftPanel
          data={data}
          collapsed={leftPanelCollapsed}
          width={leftPanelWidth}
          isDark={isDark}
          onToggle={() => setLeftPanelCollapsed(!leftPanelCollapsed)}
        />

        <div
          style={{
            width: 4,
            backgroundColor: isDark ? '#2a2e39' : '#e5e5e5',
            cursor: 'col-resize',
            transition: 'background-color 0.2s',
          }}
          onMouseDown={handleResizeLeft}
          onMouseEnter={(e) => e.currentTarget.style.backgroundColor = isDark ? '#3a3f4b' : '#d1d4dc'}
          onMouseLeave={(e) => e.currentTarget.style.backgroundColor = isDark ? '#2a2e39' : '#e5e5e5'}
        />

        <div style={chartAreaStyle}>
          <div style={{ flex: bottomPanelVisible ? `calc(100% - ${bottomPanelHeight}px)` : 1, overflow: 'hidden' }}>
          <MainChart
            data={filteredData!}
            chartType={chartType as 'area' | 'line'}
            showBenchmark={showBenchmark}
            showExcess={showExcess}
            useLogScale={useLogScale}
            isDark={isDark}
          />
          </div>

          {bottomPanelVisible && (
            <>
              <div
                style={{
                  height: 4,
                  backgroundColor: isDark ? '#2a2e39' : '#e5e5e5',
                  cursor: 'row-resize',
                  transition: 'background-color 0.2s',
                }}
                onMouseDown={handleResizeBottom}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = isDark ? '#3a3f4b' : '#d1d4dc'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = isDark ? '#2a2e39' : '#e5e5e5'}
              />
      <BottomPanel
        data={filteredData!}
        activeTab={bottomTab as any}
        height={bottomPanelHeight}
        isDark={isDark}
        onTabChange={setBottomTab as any}
        onClose={() => setBottomPanelVisible(false)}
      />
            </>
          )}
        </div>
      </div>

      {isResizingLeft && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            zIndex: 9999,
            cursor: 'col-resize',
          }}
        />
      )}

      {isResizingBottom && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            zIndex: 9999,
            cursor: 'row-resize',
          }}
        />
      )}
    </div>
  )
}
