import { useMemo } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from 'recharts'
import { useThemeStore } from '../../stores/themeStore'

interface ReturnsDistributionChartProps {
  distribution: {
    daily_returns?: number[]
    mean?: number
    std?: number
    skewness?: number
    kurtosis?: number
    min?: number
    max?: number
    median?: number
    positive_days?: number
    negative_days?: number
    neutral_days?: number
  }
}

export default function ReturnsDistributionChart({ distribution }: ReturnsDistributionChartProps) {
  const { theme } = useThemeStore()
  const isDark = theme === 'dark'

  const palette = useMemo(() => {
    return isDark
      ? {
          bg: '#0B1220',
          card: '#111827',
          text: '#E5E7EB',
          muted: '#9CA3AF',
          grid: '#243041',
          positive: '#34D399',
          negative: '#F87171',
        }
      : {
          bg: '#FFFFFF',
          card: '#FFFFFF',
          text: '#111827',
          muted: '#6B7280',
          grid: '#E5E7EB',
          positive: '#16A34A',
          negative: '#DC2626',
        }
  }, [isDark])

  const chartData = useMemo(() => {
    const dailyReturns = distribution.daily_returns || []
    if (dailyReturns.length === 0) return []

    const returns = dailyReturns.filter((r): r is number => typeof r === 'number' && !isNaN(r))
    if (returns.length === 0) return []

    const min = Math.min(...returns)
    const max = Math.max(...returns)
    const binCount = 30
    const binWidth = (max - min) / binCount

    if (binWidth === 0 || !isFinite(binWidth)) {
      return [{ range: '0%', count: 1, mid: 0, isPositive: false }]
    }

    const bins: { range: string; count: number; mid: number; isPositive: boolean }[] = []
    for (let i = 0; i < binCount; i++) {
      const binStart = min + i * binWidth
      const binEnd = min + (i + 1) * binWidth
      const mid = (binStart + binEnd) / 2
      bins.push({
        range: `${(binStart * 100).toFixed(2)}% ~ ${(binEnd * 100).toFixed(2)}%`,
        count: 0,
        mid,
        isPositive: mid > 0,
      })
    }

    for (const r of returns) {
      const binIndex = Math.min(Math.floor((r - min) / binWidth), binCount - 1)
      if (binIndex >= 0 && binIndex < bins.length) {
        bins[binIndex].count++
      }
    }

    return bins
  }, [distribution])

  const stats = useMemo(() => {
    return [
      { label: 'Mean', value: distribution.mean ? `${(distribution.mean * 100).toFixed(3)}%` : 'N/A' },
      { label: 'Std Dev', value: distribution.std ? `${(distribution.std * 100).toFixed(3)}%` : 'N/A' },
      { label: 'Skewness', value: distribution.skewness?.toFixed(3) || 'N/A' },
      { label: 'Kurtosis', value: distribution.kurtosis?.toFixed(3) || 'N/A' },
      { label: 'Positive Days', value: distribution.positive_days?.toString() || '0' },
      { label: 'Negative Days', value: distribution.negative_days?.toString() || '0' },
    ]
  }, [distribution])

  if (chartData.length === 0) {
    return (
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Returns Distribution</h3>
        <p className="text-gray-500 dark:text-gray-400">No distribution data available</p>
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white">Returns Distribution</h3>
        <div className="flex gap-4 text-sm">
          {stats.map((stat, idx) => (
            <div key={idx} className="text-center">
              <p className="text-gray-500 dark:text-gray-400">{stat.label}</p>
              <p className="font-semibold text-gray-900 dark:text-white">{stat.value}</p>
            </div>
          ))}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={chartData} margin={{ top: 8, right: 16, left: 8, bottom: 24 }}>
          <CartesianGrid stroke={palette.grid} strokeDasharray="3 3" opacity={0.7} />
          <XAxis
            dataKey="mid"
            tick={{ fill: palette.muted, fontSize: 11 }}
            tickFormatter={(v) => `${(v * 100).toFixed(1)}%`}
            label={{ value: 'Daily Return', position: 'bottom', fill: palette.muted, fontSize: 12 }}
          />
          <YAxis tick={{ fill: palette.muted, fontSize: 12 }} />
          <Tooltip
            contentStyle={{
              backgroundColor: isDark ? '#0B1220' : '#FFFFFF',
              border: `1px solid ${isDark ? '#243041' : '#E5E7EB'}`,
              borderRadius: '8px',
              color: palette.text,
            }}
            formatter={(value: number) => [value, 'Days']}
            labelFormatter={() => ''}
          />
          <ReferenceLine x={0} stroke={palette.muted} strokeDasharray="4 4" />
          <Bar dataKey="count" radius={[4, 4, 0, 0]}>
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.isPositive ? palette.positive : palette.negative} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
