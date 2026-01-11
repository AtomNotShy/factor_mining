import { useMemo } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from 'recharts'
import { useThemeStore } from '../../stores/themeStore'

interface RollingReturnsChartProps {
  rollingMetrics: {
    rolling_21d?: {
      returns?: number[]
      sharpe?: number[]
      volatility?: number[]
      max_drawdown?: number[]
    }
    rolling_63d?: {
      returns?: number[]
      sharpe?: number[]
      volatility?: number[]
      max_drawdown?: number[]
    }
    rolling_126d?: {
      returns?: number[]
      sharpe?: number[]
      volatility?: number[]
      max_drawdown?: number[]
    }
    cumulative?: {
      values?: number[]
      timestamps?: string[]
    }
  }
}

export default function RollingReturnsChart({ rollingMetrics }: RollingReturnsChartProps) {
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
          line21d: '#60A5FA',
          line63d: '#22D3EE',
          line126d: '#A78BFA',
        }
      : {
          bg: '#FFFFFF',
          card: '#FFFFFF',
          text: '#111827',
          muted: '#6B7280',
          grid: '#E5E7EB',
          line21d: '#2563EB',
          line63d: '#0284C7',
          line126d: '#7C3AED',
        }
  }, [isDark])

  const chartData = useMemo(() => {
    const data: any[] = []
    const windows = ['21d', '63d', '126d']

    for (const window of windows) {
      const rolling = (rollingMetrics as any)[`rolling_${window}`]
      if (!rolling?.returns) continue

      const returns = rolling.returns
      const length = returns.length

      for (let i = 0; i < length; i++) {
        if (!data[i]) {
          data[i] = { index: i }
        }
        data[i][`return_${window}`] = returns[i]
      }
    }

    return data.filter((d) => Object.keys(d).length > 1)
  }, [rollingMetrics])

  if (chartData.length === 0) {
    return (
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Rolling Returns</h3>
        <p className="text-gray-500 dark:text-gray-400">No rolling returns data available</p>
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
      <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Rolling Returns</h3>

      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={chartData} margin={{ top: 8, right: 16, left: 8, bottom: 24 }}>
          <CartesianGrid stroke={palette.grid} strokeDasharray="3 3" opacity={0.7} />
          <XAxis
            dataKey="index"
            tick={{ fill: palette.muted, fontSize: 12 }}
            tickFormatter={(v) => `Day ${v}`}
          />
          <YAxis
            tick={{ fill: palette.muted, fontSize: 12 }}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: isDark ? '#0B1220' : '#FFFFFF',
              border: `1px solid ${isDark ? '#243041' : '#E5E7EB'}`,
              borderRadius: '8px',
              color: palette.text,
            }}
            formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, 'Return']}
            labelFormatter={(label) => `Day ${label}`}
          />
          <Legend />
          <ReferenceLine y={0} stroke={palette.muted} strokeDasharray="4 4" />

          {rollingMetrics.rolling_21d?.returns && (
            <Line
              type="monotone"
              dataKey="return_21d"
              name="21-day"
              stroke={palette.line21d}
              strokeWidth={2}
              dot={false}
            />
          )}
          {rollingMetrics.rolling_63d?.returns && (
            <Line
              type="monotone"
              dataKey="return_63d"
              name="63-day"
              stroke={palette.line63d}
              strokeWidth={2}
              dot={false}
            />
          )}
          {rollingMetrics.rolling_126d?.returns && (
            <Line
              type="monotone"
              dataKey="return_126d"
              name="126-day"
              stroke={palette.line126d}
              strokeWidth={2}
              dot={false}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
