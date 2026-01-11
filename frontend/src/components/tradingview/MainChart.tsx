import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { formatTimestamp } from '../../utils/formatTimestamp'

interface BacktestData {
  strategy_name: string
  symbol: string
  enhanced_metrics: any
  equity_comparison: {
    strategy_equity: number[]
    benchmark_equity: number[]
    excess_returns: number[]
    timestamps: string[]
  }
}

interface MainChartProps {
  data: BacktestData
  chartType: 'area' | 'line'
  showBenchmark: boolean
  showExcess: boolean
  useLogScale: boolean
  isDark: boolean
}

export default function MainChart({
  data,
  chartType,
  showBenchmark,
  showExcess,
  useLogScale,
  isDark,
}: MainChartProps) {
  const { equity_comparison } = data
  const { strategy_equity, benchmark_equity, excess_returns, timestamps } = equity_comparison

  const option = useMemo(() => {
    const xData = timestamps.map(formatTimestamp)
    const strategyData = strategy_equity.map((v, i) => [xData[i], v])
    const benchmarkData = showBenchmark && benchmark_equity.length > 0 
      ? benchmark_equity.map((v, i) => [xData[i], v])
      : []
    const excessData = showExcess && excess_returns.length > 0
      ? excess_returns.map((v, i) => [xData[i], v])
      : []

    const series: any[] = []
    const yAxis: any[] = []

    // Strategy line
    series.push({
      name: 'Strategy',
      type: chartType === 'area' ? 'line' : 'line',
      smooth: true,
      symbol: 'none',
      data: strategyData,
      lineStyle: { color: '#089981', width: 2 },
      areaStyle: chartType === 'area' ? {
        color: {
          type: 'linear',
          x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(8, 153, 129, 0.3)' },
            { offset: 1, color: 'rgba(8, 153, 129, 0)' },
          ],
        },
      } : undefined,
      yAxisIndex: 0,
      emphasis: { focus: 'series' },
    })

    // Benchmark line
    if (showBenchmark && benchmarkData.length > 0) {
      series.push({
        name: 'QQQ',
        type: 'line',
        smooth: true,
        symbol: 'none',
        data: benchmarkData,
        lineStyle: { color: '#787b86', width: 2, type: 'dashed' },
        yAxisIndex: 0,
        emphasis: { focus: 'series' },
      })
    }

    // Excess returns bar (right Y-axis)
    if (showExcess && excessData.length > 0) {
      series.push({
        name: 'Excess %',
        type: 'bar',
        data: excessData,
        itemStyle: {
          color: (params: any) => {
            const value = params.value?.[1]
            return value >= 0 ? '#089981' : '#f23645'
          },
        },
        yAxisIndex: 1,
        emphasis: { focus: 'series' },
      })

      yAxis.push({
        type: 'value',
        name: 'Excess %',
        position: 'right',
        axisLabel: { formatter: '{value}%', color: '#787b86' },
        splitLine: { show: false },
        axisLine: { show: false },
      })
    }

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
        axisPointer: {
          type: 'cross',
          label: { backgroundColor: isDark ? '#2a2e39' : '#e5e5e5' },
        },
        formatter: (params: any) => {
          if (!Array.isArray(params) || params.length === 0) return ''
          const title = params[0].axisValueLabel ?? params[0].name ?? ''
          const lines = params.map((item: any) => {
            const raw = Array.isArray(item.value) ? item.value[1] : item.value
            const numeric = Number(raw)
            const label = item.seriesName || ''
            const isPercent = label.toLowerCase().includes('excess')
            const formatted = isPercent ? `${formatValue(numeric)}%` : formatValue(numeric)
            return `${item.marker || ''} ${label}: ${formatted}`
          })
          return [title, ...lines].join('<br/>')
        },
      },
      legend: {
        data: ['Strategy', 'QQQ', 'Excess %'],
        top: 8,
        left: 12,
        textStyle: { color: isDark ? '#d1d4dc' : '#1a1a1a' },
      },
      grid: {
        left: 60,
        right: showExcess ? 60 : 20,
        top: 48,
        bottom: 20,
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: xData,
        axisLine: { lineStyle: { color: isDark ? '#2a2e39' : '#e5e5e5' } },
        axisLabel: { 
          color: isDark ? '#787b86' : '#6b7280',
          formatter: (value: string) => {
            const idx = xData.indexOf(value)
            if (idx === 0 || idx === xData.length - 1 || idx % Math.ceil(xData.length / 8) === 0) {
              return value
            }
            return ''
          },
        },
        splitLine: { show: false },
      },
      yAxis: [
        {
          type: 'value',
          position: 'left',
          axisLabel: { 
            formatter: '${value}',
            color: isDark ? '#d1d4dc' : '#1a1a1a',
          },
          splitLine: { lineStyle: { color: isDark ? '#2a2e39' : '#f0f0f0' } },
          axisLine: { show: false },
          scale: useLogScale,
        },
        ...yAxis,
      ],
      series,
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100,
        },
        {
          type: 'slider',
          bottom: 0,
          height: 20,
          borderColor: 'transparent',
          backgroundColor: isDark ? '#2a2e39' : '#f0f0f0',
          fillerColor: 'rgba(41, 98, 255, 0.2)',
          handleStyle: {
            color: '#2962ff',
          },
          textStyle: { color: 'transparent' },
        },
      ],
    }
  }, [strategy_equity, benchmark_equity, excess_returns, timestamps, chartType, showBenchmark, showExcess, useLogScale, isDark])

  if (!strategy_equity || strategy_equity.length === 0) {
    return (
      <div style={{ 
        flex: 1, 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        color: isDark ? '#787b86' : '#6b7280',
      }}>
        No data available
      </div>
    )
  }

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <ReactECharts
        option={option}
        style={{ width: '100%', height: '100%' }}
        theme={isDark ? 'dark' : 'light'}
        opts={{
          renderer: 'canvas',
        }}
      />
    </div>
  )
}
