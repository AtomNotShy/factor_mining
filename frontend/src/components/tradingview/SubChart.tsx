import React, { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { formatTimestamp } from '../../utils/formatTimestamp'

interface BacktestData {
  enhanced_metrics: any
  equity_comparison: {
    strategy_equity: number[]
    timestamps: string[]
  }
  results: any
}

interface SubChartProps {
  data: BacktestData
  type: 'drawdown' | 'returns' | 'distribution' | 'monthly' | 'rolling'
  isDark: boolean
}

const CHART_TITLES = {
  drawdown: 'Drawdown',
  returns: 'Daily Returns',
  distribution: 'Return Distribution',
  monthly: 'Monthly Returns',
  rolling: 'Rolling Sharpe',
}

export default function SubChart({ data, type, isDark }: SubChartProps) {
  const { enhanced_metrics, equity_comparison, results } = data
  const { timestamps } = equity_comparison

  const option = useMemo(() => {
    switch (type) {
      case 'drawdown': {
        const drawdownSeries = enhanced_metrics?.drawdown_series || []
        const xData = timestamps.map(formatTimestamp)
        const chartData = drawdownSeries.map((v: number, i: number) => [xData[i], (v * 100).toFixed(2)])
        
        return {
          title: {
            text: CHART_TITLES.drawdown,
            left: 8,
            top: 4,
            textStyle: { fontSize: 11, color: isDark ? '#787b86' : '#6b7280' },
          },
          tooltip: {
            trigger: 'axis',
            backgroundColor: isDark ? '#1e222d' : '#ffffff',
            borderColor: isDark ? '#2a2e39' : '#e5e5e5',
            textStyle: { color: isDark ? '#d1d4dc' : '#1a1a1a' },
          },
          grid: { left: 50, right: 20, top: 28, bottom: 20 },
          xAxis: {
            type: 'category',
            data: xData,
            axisLine: { lineStyle: { color: isDark ? '#2a2e39' : '#e5e5e5' } },
            axisLabel: { 
              color: isDark ? '#787b86' : '#6b7280',
              fontSize: 9,
              show: false,
            },
            splitLine: { show: false },
          },
          yAxis: {
            type: 'value',
            axisLabel: { formatter: '{value}%', color: isDark ? '#787b86' : '#6b7280', fontSize: 9 },
            splitLine: { lineStyle: { color: isDark ? '#2a2e39' : '#f0f0f0' } },
          },
          series: [{
            name: 'Drawdown',
            type: 'line',
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
            data: chartData,
            lineStyle: { color: '#f23645', width: 1 },
            symbol: 'none',
          }],
        }
      }

      case 'returns': {
        const returns_series = results?.returns_distribution?.returns || []
        const xData = timestamps.map(formatTimestamp)
        const chartData = returns_series.map((v: number, i: number) => [xData[i], (v * 100).toFixed(2)])
        
        return {
          title: {
            text: CHART_TITLES.returns,
            left: 8,
            top: 4,
            textStyle: { fontSize: 11, color: isDark ? '#787b86' : '#6b7280' },
          },
          tooltip: {
            trigger: 'axis',
            backgroundColor: isDark ? '#1e222d' : '#ffffff',
            borderColor: isDark ? '#2a2e39' : '#e5e5e5',
            textStyle: { color: isDark ? '#d1d4dc' : '#1a1a1a' },
          },
          grid: { left: 50, right: 20, top: 28, bottom: 20 },
          xAxis: {
            type: 'category',
            data: xData,
            axisLine: { lineStyle: { color: isDark ? '#2a2e39' : '#e5e5e5' } },
            axisLabel: { 
              color: isDark ? '#787b86' : '#6b7280',
              fontSize: 9,
              show: false,
            },
            splitLine: { show: false },
          },
          yAxis: {
            type: 'value',
            axisLabel: { formatter: '{value}%', color: isDark ? '#787b86' : '#6b7280', fontSize: 9 },
            splitLine: { lineStyle: { color: isDark ? '#2a2e39' : '#f0f0f0' } },
          },
          series: [{
            name: 'Returns',
            type: 'bar',
            data: chartData,
            itemStyle: {
              color: (params: any) => {
                const value = params.value?.[1]
                return value >= 0 ? '#089981' : '#f23645'
              },
            },
          }],
        }
      }

      case 'distribution': {
        const distribution = results?.returns_distribution?.distribution || []
        const bins = Object.keys(distribution).map(Number).sort((a, b) => a - b)
        const values = bins.map(b => ({ value: distribution[b], bin: b }))
        
        return {
          title: {
            text: CHART_TITLES.distribution,
            left: 8,
            top: 4,
            textStyle: { fontSize: 11, color: isDark ? '#787b86' : '#6b7280' },
          },
          tooltip: {
            trigger: 'axis',
            backgroundColor: isDark ? '#1e222d' : '#ffffff',
            borderColor: isDark ? '#2a2e39' : '#e5e5e5',
            textStyle: { color: isDark ? '#d1d4dc' : '#1a1a1a' },
          },
          grid: { left: 50, right: 20, top: 28, bottom: 20 },
          xAxis: {
            type: 'value',
            axisLabel: { formatter: '{value}%', color: isDark ? '#787b86' : '#6b7280', fontSize: 9 },
            splitLine: { lineStyle: { color: isDark ? '#2a2e39' : '#f0f0f0' } },
          },
          yAxis: {
            type: 'value',
            axisLabel: { color: isDark ? '#787b86' : '#6b7280', fontSize: 9 },
            splitLine: { show: false },
          },
          series: [{
            type: 'bar',
            data: values,
            itemStyle: {
              color: (params: any) => {
                return params.data.bin >= 0 ? '#089981' : '#f23645'
              },
            },
          }],
        }
      }

      case 'monthly': {
        const monthly = results?.monthly_returns || {}
        const years = Object.keys(monthly).sort()
        
        return {
          title: {
            text: CHART_TITLES.monthly,
            left: 8,
            top: 4,
            textStyle: { fontSize: 11, color: isDark ? '#787b86' : '#6b7280' },
          },
          tooltip: {
            trigger: 'item',
            backgroundColor: isDark ? '#1e222d' : '#ffffff',
            borderColor: isDark ? '#2a2e39' : '#e5e5e5',
            textStyle: { color: isDark ? '#d1d4dc' : '#1a1a1a' },
            formatter: (params: any) => {
              return `${params.data[0]} ${params.data[1]}<br/>${params.data[2].toFixed(2)}%`
            },
          },
          grid: { left: 40, right: 20, top: 28, bottom: 20 },
          xAxis: {
            type: 'value',
            axisLabel: { color: isDark ? '#787b86' : '#6b7280', fontSize: 9 },
            splitLine: { lineStyle: { color: isDark ? '#2a2e39' : '#f0f0f0' } },
          },
          yAxis: {
            type: 'category',
            data: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            axisLabel: { color: isDark ? '#787b86' : '#6b7280', fontSize: 9 },
            splitLine: { show: false },
          },
          series: [{
            type: 'scatter',
            symbolSize: (val: any) => Math.max(6, Math.min(20, Math.abs(val[2]) * 3)),
            data: years.flatMap(year => 
              [1,2,3,4,5,6,7,8,9,10,11,12].map(month => [
                month, 
                month - 1, 
                monthly[year]?.[month] || 0
              ])
            ),
            itemStyle: {
              color: (params: any) => {
                return params.data[2] >= 0 ? '#089981' : '#f23645'
              },
            },
          }],
        }
      }

      case 'rolling': {
        const rolling = results?.rolling_metrics?.rolling_sharpe || {}
        const dates = Object.keys(rolling).sort()
        const chartData = dates.map(d => [d, rolling[d]])
        
        return {
          title: {
            text: CHART_TITLES.rolling,
            left: 8,
            top: 4,
            textStyle: { fontSize: 11, color: isDark ? '#787b86' : '#6b7280' },
          },
          tooltip: {
            trigger: 'axis',
            backgroundColor: isDark ? '#1e222d' : '#ffffff',
            borderColor: isDark ? '#2a2e39' : '#e5e5e5',
            textStyle: { color: isDark ? '#d1d4dc' : '#1a1a1a' },
          },
          grid: { left: 50, right: 20, top: 28, bottom: 20 },
          xAxis: {
            type: 'category',
            data: dates.map(formatTimestamp),
            axisLine: { lineStyle: { color: isDark ? '#2a2e39' : '#e5e5e5' } },
            axisLabel: { 
              color: isDark ? '#787b86' : '#6b7280',
              fontSize: 9,
              show: false,
            },
            splitLine: { show: false },
          },
          yAxis: {
            type: 'value',
            axisLabel: { color: isDark ? '#787b86' : '#6b7280', fontSize: 9 },
            splitLine: { lineStyle: { color: isDark ? '#2a2e39' : '#f0f0f0' } },
          },
          series: [{
            name: 'Rolling Sharpe',
            type: 'line',
            data: chartData,
            lineStyle: { color: '#2962ff', width: 1 },
            symbol: 'none',
            markLine: {
              data: [{ yAxis: 1, lineStyle: { color: '#089981', type: 'dashed' } }],
              symbol: 'none',
            },
          }],
        }
      }

      default:
        return {}
    }
  }, [data, type, isDark])

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <ReactECharts
        option={option}
        style={{ width: '100%', height: '100%' }}
        opts={{ renderer: 'canvas' }}
      />
    </div>
  )
}
