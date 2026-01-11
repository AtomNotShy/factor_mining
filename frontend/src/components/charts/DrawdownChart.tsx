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

    const xValues = timestamps.map(formatTimestamp)
    const yValues = drawdownSeries.map(v => (v * 100))

    const traces: any[] = [{
      x: xValues,
      y: yValues,
      type: 'scatter',
      fill: 'tozeroy',
      mode: 'lines',
      name: 'Drawdown',
      line: { color: '#f44336', width: 1.5 },
      fillcolor: 'rgba(244, 67, 54, 0.2)'
    }]

    const layout: any = {
      title: {
        text: 'Drawdown',
        font: { size: 14, color: isDark ? '#E5E7EB' : '#333' }
      },
      height,
      showlegend: false,
      margin: { l: 50, r: 20, t: 40, b: 40 },
      xaxis: {
        title: '',
        tickangle: -45,
        tickfont: { size: 9, color: isDark ? '#9CA3AF' : '#666' },
        gridcolor: isDark ? '#243041' : '#eee',
        showgrid: true
      },
      yaxis: {
        title: 'Drawdown (%)',
        tickformat: '.1f',
        tickfont: { size: 10, color: isDark ? '#E5E7EB' : '#333' },
        gridcolor: isDark ? '#243041' : '#eee',
        range: [Math.min(...yValues) * 1.1, 5]
      },
      plot_bgcolor: isDark ? '#0B1220' : '#fafafa',
      paper_bgcolor: isDark ? '#111827' : 'white',
      shapes: [] as any[]
    }

    if (maxDrawdownWindow && maxDrawdownWindow.drawdown_pct < 0) {
      const startIdx = timestamps.findIndex(ts => 
        formatTimestamp(ts) === formatTimestamp(maxDrawdownWindow.start_date)
      )
      const endIdx = timestamps.findIndex(ts => 
        formatTimestamp(ts) === formatTimestamp(maxDrawdownWindow.end_date)
      )

      if (startIdx >= 0 && endIdx >= 0) {
        layout.shapes = [
          {
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: xValues[startIdx],
            x1: xValues[endIdx],
            y0: 0,
            y1: 1,
            fillcolor: 'rgba(255, 0, 0, 0.08)',
            line: { width: 0 }
          }
        ]
      }
    }

    if (typeof window !== 'undefined' && (window as any).Plotly) {
      const Plotly = (window as any).Plotly
      if (containerRef.current) {
        Plotly.newPlot(containerRef.current, traces, layout, { responsive: true, displayModeBar: false })
      }
    }

  }, [drawdownSeries, timestamps, maxDrawdownWindow, isDark, height])

  if (!drawdownSeries || drawdownSeries.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 bg-gray-50 rounded-lg">
        <div className="text-gray-400">No drawdown data</div>
      </div>
    )
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-100 p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-200">Drawdown</h3>
        {maxDrawdownWindow && (
          <div className="text-xs text-gray-500 dark:text-gray-400">
            Max: {(maxDrawdownWindow.drawdown_pct * 100).toFixed(2)}% 
            <span className="mx-2">|</span>
            Duration: {maxDrawdownWindow.duration_days} days
          </div>
        )}
      </div>
      <div ref={containerRef} className="w-full" />
    </div>
  )
}

export default DrawdownChart
