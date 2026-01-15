import { useEffect, useRef } from 'react'
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

  useEffect(() => {
    console.log('[EquityCurveChart] Rendering:', {
      strategyLength: strategyEquity?.length,
      benchmarkLength: benchmarkEquity?.length,
      showBenchmark,
      timestampsLength: timestamps?.length
    })

    if (!strategyEquity || strategyEquity.length === 0) return

    const chartRef = containerRef.current
    if (!chartRef) return

    // 检查基准数据是否有效
    const hasValidBenchmark = showBenchmark && benchmarkEquity && 
                             benchmarkEquity.length > 0 && 
                             benchmarkEquity.length === strategyEquity.length &&
                             benchmarkEquity.some((v: number) => v !== 0 && v !== 100000) // 检查是否不是初始值

    console.log('[EquityCurveChart] Has valid benchmark:', hasValidBenchmark, 
                'benchmarkEquity length:', benchmarkEquity?.length,
                'benchmarkEquity sample:', benchmarkEquity?.slice(0, 5))

    // 数据对齐：确保所有数组长度一致
    const minLength = Math.min(
      strategyEquity.length,
      timestamps.length,
      hasValidBenchmark ? benchmarkEquity!.length : strategyEquity.length
    )

    if (minLength === 0) return

    // 准备数据
    const xAxisData = timestamps.slice(0, minLength).map(formatTimestamp)
    const strategyData = strategyEquity.slice(0, minLength)
    const benchmarkData = hasValidBenchmark ? benchmarkEquity!.slice(0, minLength) : null

    console.log('[EquityCurveChart] Prepared data:', {
      xAxisLength: xAxisData.length,
      strategyLength: strategyData.length,
      benchmarkLength: benchmarkData?.length,
      strategySample: strategyData.slice(0, 5),
      benchmarkSample: benchmarkData?.slice(0, 5)
    })

    // Calculate Y-axis ranges for strategy and benchmark separately
    const strategyValidValues = strategyData.filter((v: number) => !isNaN(v) && isFinite(v))
    if (strategyValidValues.length === 0) return

    const strategyMinVal = Math.min(...strategyValidValues)
    const strategyMaxVal = Math.max(...strategyValidValues)
    const strategyRange = strategyMaxVal - strategyMinVal || 1
    const strategyPadding = strategyRange * 0.1

    let benchmarkMinVal = 0
    let benchmarkMaxVal = 0
    let benchmarkPadding = 0
    
    if (benchmarkData) {
      const benchmarkValidValues = benchmarkData.filter((v: number) => !isNaN(v) && isFinite(v))
      if (benchmarkValidValues.length > 0) {
        benchmarkMinVal = Math.min(...benchmarkValidValues)
        benchmarkMaxVal = Math.max(...benchmarkValidValues)
        const benchmarkRange = benchmarkMaxVal - benchmarkMinVal || 1
        benchmarkPadding = benchmarkRange * 0.1
      }
    }

    console.log('[EquityCurveChart] Strategy range:', strategyMinVal, '-', strategyMaxVal)
    if (benchmarkData) {
      console.log('[EquityCurveChart] Benchmark range:', benchmarkMinVal, '-', benchmarkMaxVal)
    }

    // Dynamically import ECharts
    import('echarts').then((echarts) => {
      const chart = echarts.init(chartRef, isDark ? 'dark' : 'light')

      const series: any[] = [
        {
          name: 'Strategy',
          type: 'line',
          data: strategyData,
          smooth: true,
          symbol: 'none',
          lineStyle: { color: '#4CAF50', width: 2 },
          emphasis: { focus: 'series' }
        }
      ]

      if (benchmarkData) {
        series.push({
          name: `${benchmarkSymbol} (Benchmark)`,
          type: 'line',
          data: benchmarkData,
          smooth: true,
          symbol: 'none',
          lineStyle: { color: '#FF6B6B', width: 2, type: 'dashed' },
          emphasis: { focus: 'series' },
          yAxisIndex: 1 // Use the second Y-axis for benchmark
        })
      }

      if (showExcess && excessReturns && excessReturns.length > 0) {
        series.push({
          name: 'Excess Return',
          type: 'bar',
          data: excessReturns.slice(0, minLength),
          itemStyle: {
            color: (params: any) => params.value >= 0 ? '#4CAF50' : '#f44336',
            opacity: 0.6
          },
          yAxisIndex: 1,
          emphasis: { focus: 'series' }
        })
      }

      const option: any = {
        title: {
          text: 'Equity Curve Comparison',
          left: 'center',
          textStyle: {
            fontSize: 18,
            color: isDark ? '#E5E7EB' : '#333'
          }
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'cross' },
          backgroundColor: isDark ? '#1f2937' : '#fff',
          borderColor: isDark ? '#374151' : '#e5e7eb',
          textStyle: { color: isDark ? '#e5e7eb' : '#333' },
          formatter: (params: any) => {
            if (!Array.isArray(params)) return ''
            let result = `<div style="font-weight: bold; margin-bottom: 8px;">${params[0]?.axisValue || ''}</div>`
            params.forEach((param: any) => {
              const value = typeof param.value === 'number' 
                ? param.value.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })
                : param.value
              const marker = `<span style="display:inline-block;margin-right:4px;border-radius:10px;width:10px;height:10px;background-color:${param.color};"></span>`
              result += `<div style="display:flex;justify-content:space-between;gap:16px;">${marker}${param.seriesName}: <span style="font-weight:bold;">${value}</span></div>`
            })
            return result
          }
        },
        legend: {
          data: series.map(s => s.name),
          bottom: 10,
          textStyle: { color: isDark ? '#9CA3AF' : '#666' }
        },
        grid: {
          left: 60,
          right: (benchmarkData || showExcess) ? 80 : 20,
          top: 60,
          bottom: 80
        },
        xAxis: {
          type: 'category',
          data: xAxisData,
          boundaryGap: false,
          axisLabel: {
            rotate: -45,
            fontSize: 10,
            color: isDark ? '#9CA3AF' : '#666'
          },
          axisLine: { lineStyle: { color: isDark ? '#374151' : '#e5e7eb' } }
        },
        yAxis: [
          {
            type: 'value',
            name: 'Strategy Value ($)',
            position: 'left',
            min: Math.floor((strategyMinVal - strategyPadding) / 1000) * 1000,
            max: Math.ceil((strategyMaxVal + strategyPadding) / 1000) * 1000,
            axisLabel: {
              formatter: (value: number) => `$${(value / 1000).toFixed(0)}k`,
              fontSize: 11,
              color: isDark ? '#E5E7EB' : '#333'
            },
            axisLine: { show: true, lineStyle: { color: isDark ? '#374151' : '#e5e7eb' } },
            splitLine: { lineStyle: { color: isDark ? '#243041' : '#eee' } }
          },
          benchmarkData ? {
            type: 'value',
            name: `${benchmarkSymbol} Value`,
            position: 'right',
            min: benchmarkMinVal - benchmarkPadding,
            max: benchmarkMaxVal + benchmarkPadding,
            axisLabel: {
              formatter: (value: number) => {
                // Format benchmark value appropriately based on its magnitude
                if (value >= 1000) {
                  return `$${(value / 1000).toFixed(0)}k`
                } else if (value >= 1) {
                  return `$${value.toFixed(0)}`
                } else {
                  return `$${value.toFixed(2)}`
                }
              },
              fontSize: 11,
              color: '#FF6B6B'
            },
            axisLine: { 
              show: true, 
              lineStyle: { color: '#FF6B6B' } 
            },
            splitLine: { 
              show: false 
            }
          } : null,
          showExcess ? {
            type: 'value',
            name: 'Excess Return (%)',
            position: 'right',
            offset: benchmarkData ? 80 : 0,
            axisLabel: {
              formatter: (value: number) => `${(value * 100).toFixed(1)}%`,
              fontSize: 10,
              color: '#666'
            },
            axisLine: { show: true, lineStyle: { color: '#666' } },
            splitLine: { show: false }
          } : null
        ].filter(Boolean) as any[],
        series,
        dataZoom: [
          {
            type: 'slider',
            show: true,
            xAxisIndex: 0,
            start: 0,
            end: 100,
            bottom: 10,
            height: 20,
            borderColor: isDark ? '#374151' : '#e5e7eb',
            fillerColor: isDark ? 'rgba(96, 165, 250, 0.2)' : 'rgba(96, 165, 250, 0.2)',
            textStyle: { color: isDark ? '#9CA3AF' : '#666' }
          },
          {
            type: 'inside',
            xAxisIndex: 0,
            start: 0,
            end: 100
          }
        ],
        animationDuration: 1000
      }

      // Set log scale if enabled
      if (useLogScale) {
        option.yAxis[0].type = 'log'
        delete option.yAxis[0].axisLabel.formatter
      }

      chart.setOption(option, true)

      // Handle resize
      const handleResize = () => chart.resize()
      window.addEventListener('resize', handleResize)

      return () => {
        window.removeEventListener('resize', handleResize)
        chart.dispose()
      }
    }).catch((err) => {
      console.error('[EquityCurveChart] ECharts error:', err)
    })

  }, [strategyEquity, benchmarkEquity, excessReturns, timestamps, showBenchmark, showExcess, useLogScale, isDark, height])

  if (!strategyEquity || strategyEquity.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-50 dark:bg-gray-800 rounded-lg">
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
