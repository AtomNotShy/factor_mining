import { useEffect, useRef } from 'react'
import { useThemeStore } from '../../stores/themeStore'
import { formatTimestamp } from '../../utils/formatTimestamp'

interface DrawdownChartProps {
  drawdownSeries: number[]
  timestamps: string[]
  maxDrawdownWindow?: {
    start_date: string
    end_date: string
    drawdown_pct: number
    duration_days: number
    peak_date: string
    trough_date: string
    peak_value: number
    trough_value: number
  }
  height?: number
}

export function DrawdownChart({
  drawdownSeries,
  timestamps,
  maxDrawdownWindow,
  height = 200
}: DrawdownChartProps) {
  const { theme } = useThemeStore()
  const isDark = theme === 'dark'
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!drawdownSeries || drawdownSeries.length === 0) return

    const chartRef = containerRef.current
    if (!chartRef) return

    import('echarts').then((echarts) => {
      const chart = echarts.init(chartRef, isDark ? 'dark' : 'light')

      const xAxisData = timestamps.map(formatTimestamp)
      const yValues = drawdownSeries.map(v => (v * 100))

      const option: any = {
        title: {
          text: 'Drawdown',
          left: 'left',
          textStyle: {
            fontSize: 14,
            color: isDark ? '#E5E7EB' : '#333'
          }
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'line' },
          backgroundColor: isDark ? '#1f2937' : '#fff',
          borderColor: isDark ? '#374151' : '#e5e7eb',
          textStyle: { color: isDark ? '#e5e7eb' : '#333' },
          formatter: (params: any) => {
            const value = params[0]?.value?.toFixed(2) || '0.00'
            return `${params[0]?.axisValue}<br/>Drawdown: <strong>${value}%</strong>`
          }
        },
        grid: {
          left: 50,
          right: 20,
          top: 40,
          bottom: 40
        },
        xAxis: {
          type: 'category',
          data: xAxisData,
          boundaryGap: false,
          axisLabel: {
            rotate: -45,
            fontSize: 9,
            color: isDark ? '#9CA3AF' : '#666'
          },
          axisLine: { lineStyle: { color: isDark ? '#374151' : '#e5e7eb' } }
        },
        yAxis: {
          type: 'value',
          name: 'Drawdown (%)',
          axisLabel: {
            formatter: (value: number) => `${value.toFixed(1)}%`,
            fontSize: 10,
            color: isDark ? '#E5E7EB' : '#333'
          },
          axisLine: { show: true, lineStyle: { color: isDark ? '#374151' : '#e5e7eb' } },
          splitLine: { lineStyle: { color: isDark ? '#243041' : '#eee' } },
          min: Math.min(...yValues) * 1.1,
          max: 5
        },
        series: [
          {
            name: 'Drawdown',
            type: 'line',
            data: yValues,
            smooth: true,
            symbol: 'none',
            areaStyle: {
              color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: 'rgba(244, 67, 54, 0.3)' },
                { offset: 1, color: 'rgba(244, 67, 54, 0.05)' }
              ])
            },
            lineStyle: { color: '#f44336', width: 1.5 },
            itemStyle: { color: '#f44336' }
          }
        ],
        animationDuration: 800
      }

      // Mark max drawdown period if available
      if (maxDrawdownWindow && maxDrawdownWindow.drawdown_pct < 0) {
        const startIdx = timestamps.findIndex(ts => 
          formatTimestamp(ts) === formatTimestamp(maxDrawdownWindow.start_date)
        )
        const endIdx = timestamps.findIndex(ts => 
          formatTimestamp(ts) === formatTimestamp(maxDrawdownWindow.end_date)
        )

        if (startIdx >= 0 && endIdx >= 0) {
          option.markArea = {
            silent: true,
            itemStyle: { color: 'rgba(255, 0, 0, 0.08)' },
            data: [[
              { xAxis: xAxisData[startIdx] },
              { xAxis: xAxisData[endIdx] }
            ]]
          }
        }
      }

      chart.setOption(option, true)

      const handleResize = () => chart.resize()
      window.addEventListener('resize', handleResize)

      return () => {
        window.removeEventListener('resize', handleResize)
        chart.dispose()
      }
    }).catch((err) => {
      console.error('[DrawdownChart] Failed to load ECharts:', err)
    })

  }, [drawdownSeries, timestamps, maxDrawdownWindow, isDark, height])

  if (!drawdownSeries || drawdownSeries.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <div className="text-gray-400">No drawdown data</div>
      </div>
    )
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-100 p-4">
      <div ref={containerRef} className="w-full" style={{ height }} />
    </div>
  )
}

export default DrawdownChart
