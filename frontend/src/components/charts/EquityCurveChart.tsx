import { useEffect, useRef, useState } from 'react'
import { useThemeStore } from '../../stores/themeStore'
import { formatTimestamp } from '../../utils/formatTimestamp'

interface EquityCurveChartProps {
  strategyEquity: number[]
  benchmarkEquity?: number[]
  excessReturns?: number[]
  timestamps: string[]
  showBenchmark: boolean
  showExcess: boolean
  useLogScale: boolean
  height?: number
  benchmarkSymbol?: string
}

export function EquityCurveChart({
  strategyEquity,
  benchmarkEquity,
  excessReturns,
  timestamps,
  showBenchmark,
  showExcess,
  useLogScale,
  height = 400,
  benchmarkSymbol = 'Benchmark'
}: EquityCurveChartProps) {
  const { theme } = useThemeStore()
  const isDark = theme === 'dark'
  const containerRef = useRef<HTMLDivElement>(null)
  const [chartData, setChartData] = useState<any>(null)

  useEffect(() => {
    if (!strategyEquity || strategyEquity.length === 0) return

    const xValues = timestamps.map(formatTimestamp)
    const strategyY = strategyEquity.map((v, i) => ({
      x: xValues[i],
      y: v,
      date: timestamps[i]
    }))

    let traces: any[] = [{
      x: xValues,
      y: strategyEquity,
      type: 'scatter',
      mode: 'lines',
      name: 'Strategy',
      line: { color: '#4CAF50', width: 2 }
    }]

    if (showBenchmark && benchmarkEquity && benchmarkEquity.length > 0) {
      traces.push({
        x: timestamps.slice(0, benchmarkEquity.length).map(formatTimestamp),
        y: benchmarkEquity,
        type: 'scatter',
        mode: 'lines',
        name: `${benchmarkSymbol} (Benchmark)`,
        line: { color: '#666666', width: 2, dash: 'dash' }
      })
    }

    if (showExcess && excessReturns && excessReturns.length > 0) {
      traces.push({
        x: timestamps.slice(0, excessReturns.length).map(formatTimestamp),
        y: excessReturns,
        type: 'bar',
        name: 'Excess Return %',
        yaxis: 'y2',
        marker: {
          color: excessReturns.map((v: number) => v >= 0 ? '#4CAF50' : '#f44336'),
          opacity: 0.6
        }
      })
    }

    const layout: any = {
      title: {
        text: 'Equity Curve Comparison',
        font: { size: 18, color: isDark ? '#E5E7EB' : '#333' }
      },
      height,
      showlegend: true,
      legend: {
        orientation: 'h',
        y: 1.1,
        x: 0.5,
        xanchor: 'center'
      },
      margin: { l: 60, r: showExcess ? 60 : 20, t: 60, b: 60 },
      xaxis: {
        title: '',
        tickangle: -45,
        tickfont: { size: 10, color: isDark ? '#9CA3AF' : '#666' },
        gridcolor: isDark ? '#243041' : '#eee'
      },
      yaxis: {
        title: 'Portfolio Value ($)',
        tickformat: '$,.0f',
        tickfont: { size: 11, color: isDark ? '#E5E7EB' : '#333' },
        gridcolor: isDark ? '#243041' : '#eee',
        type: useLogScale ? 'log' : 'linear'
      },
      plot_bgcolor: isDark ? '#0B1220' : '#fafafa',
      paper_bgcolor: isDark ? '#111827' : 'white'
    }

    if (showExcess) {
      layout.yaxis2 = {
        title: 'Excess Return (%)',
        overlaying: 'y',
        side: 'right',
        tickformat: '.1f',
        tickfont: { size: 10, color: '#666' },
        gridcolor: 'rgba(0,0,0,0.05)'
      }
    }

    if (typeof window !== 'undefined' && (window as any).Plotly) {
      const Plotly = (window as any).Plotly
      if (containerRef.current) {
        Plotly.newPlot(containerRef.current, traces, layout, { responsive: true, displayModeBar: true })
      }
    }

    setChartData({ traces, layout })

  }, [strategyEquity, benchmarkEquity, excessReturns, timestamps, showBenchmark, showExcess, useLogScale, isDark, height])

  if (!strategyEquity || strategyEquity.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg">
        <div className="text-gray-400">No data available</div>
      </div>
    )
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-100 p-4">
      <div ref={containerRef} className="w-full" style={{ height }} />
    </div>
  )
}

export default EquityCurveChart
