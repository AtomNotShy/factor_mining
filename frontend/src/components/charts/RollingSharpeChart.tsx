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
  ComposedChart,
  Area,
} from 'recharts'
import { useThemeStore } from '../../stores/themeStore'

interface RollingSharpeChartProps {
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

export default function RollingSharpeChart({ rollingMetrics }: RollingSharpeChartProps) {
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
          sharpe: '#F472B6',
          drawdown: '#FBBF24',
          vol: '#94A3B8',
        }
      : {
          bg: '#FFFFFF',
          card: '#FFFFFF',
          text: '#111827',
          muted: '#6B7280',
          grid: '#E5E7EB',
          sharpe: '#DB2777',
          drawdown: '#D97706',
          vol: '#64748B',
        }
  }, [isDark])

  const sharpeData = useMemo(() => {
    const data: any[] = []
    const windows = ['21d', '63d', '126d']

    for (const window of windows) {
      const rolling = (rollingMetrics as any)[`rolling_${window}`]
      if (!rolling?.sharpe) continue

      const values = rolling.sharpe
      const length = values.length

      for (let i = 0; i < length; i++) {
        if (!data[i]) {
          data[i] = { index: i }
        }
        data[i][`sharpe_${window}`] = values[i]
      }
    }

    return data.filter((d) => Object.keys(d).length > 1)
  }, [rollingMetrics])

  const drawdownData = useMemo(() => {
    const data: any[] = []
    const windows = ['21d', '63d', '126d']

    for (const window of windows) {
      const rolling = (rollingMetrics as any)[`rolling_${window}`]
      if (!rolling?.max_drawdown) continue

      const values = rolling.max_drawdown
      const length = values.length

      for (let i = 0; i < length; i++) {
        if (!data[i]) {
          data[i] = { index: i }
        }
        data[i][`drawdown_${window}`] = values[i]
      }
    }

    return data.filter((d) => Object.keys(d).length > 1)
  }, [rollingMetrics])

  if (sharpeData.length === 0 && drawdownData.length === 0) {
    return (
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Rolling Sharpe Ratio & Drawdown</h3>
        <p className="text-gray-500 dark:text-gray-400">No rolling data available</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Rolling Sharpe Ratio</h3>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={sharpeData} margin={{ top: 8, right: 16, left: 8, bottom: 24 }}>
            <CartesianGrid stroke={palette.grid} strokeDasharray="3 3" opacity={0.7} />
            <XAxis dataKey="index" tick={{ fill: palette.muted, fontSize: 12 }} tickFormatter={(v) => `Day ${v}`} />
            <YAxis tick={{ fill: palette.muted, fontSize: 12 }} />
            <Tooltip
              contentStyle={{
                backgroundColor: isDark ? '#0B1220' : '#FFFFFF',
                border: `1px solid ${isDark ? '#243041' : '#E5E7EB'}`,
                borderRadius: '8px',
                color: palette.text,
              }}
              formatter={(value: number) => [value.toFixed(2), 'Sharpe']}
              labelFormatter={(label) => `Day ${label}`}
            />
            <Legend />
            <ReferenceLine y={0} stroke={palette.muted} strokeDasharray="4 4" />
            <ReferenceLine y={1} stroke={palette.sharpe} strokeDasharray="2 2" label={{ value: 'Sharpe=1', fill: palette.muted, fontSize: 10 }} />
            {rollingMetrics.rolling_21d?.sharpe && (
              <Line type="monotone" dataKey="sharpe_21d" name="21-day" stroke={palette.sharpe} strokeWidth={2} dot={false} />
            )}
            {rollingMetrics.rolling_63d?.sharpe && (
              <Line type="monotone" dataKey="sharpe_63d" name="63-day" stroke={palette.drawdown} strokeWidth={2} dot={false} />
            )}
            {rollingMetrics.rolling_126d?.sharpe && (
              <Line type="monotone" dataKey="sharpe_126d" name="126-day" stroke={palette.vol} strokeWidth={2} dot={false} />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Rolling Maximum Drawdown</h3>
        <ResponsiveContainer width="100%" height={240}>
          <ComposedChart data={drawdownData} margin={{ top: 8, right: 16, left: 8, bottom: 24 }}>
            <CartesianGrid stroke={palette.grid} strokeDasharray="3 3" opacity={0.7} />
            <XAxis dataKey="index" tick={{ fill: palette.muted, fontSize: 12 }} tickFormatter={(v) => `Day ${v}`} />
            <YAxis tick={{ fill: palette.muted, fontSize: 12 }} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} domain={['dataMin', 0]} />
            <Tooltip
              contentStyle={{
                backgroundColor: isDark ? '#0B1220' : '#FFFFFF',
                border: `1px solid ${isDark ? '#243041' : '#E5E7EB'}`,
                borderRadius: '8px',
                color: palette.text,
              }}
              formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, 'Drawdown']}
              labelFormatter={(label) => `Day ${label}`}
            />
            <Legend />
            <ReferenceLine y={0} stroke={palette.muted} />
            {rollingMetrics.rolling_21d?.max_drawdown && (
              <Area type="monotone" dataKey="drawdown_21d" name="21-day" stroke={palette.sharpe} fill={palette.sharpe} fillOpacity={0.2} strokeWidth={2} dot={false} />
            )}
            {rollingMetrics.rolling_63d?.max_drawdown && (
              <Area type="monotone" dataKey="drawdown_63d" name="63-day" stroke={palette.drawdown} fill={palette.drawdown} fillOpacity={0.2} strokeWidth={2} dot={false} />
            )}
            {rollingMetrics.rolling_126d?.max_drawdown && (
              <Area type="monotone" dataKey="drawdown_126d" name="126-day" stroke={palette.vol} fill={palette.vol} fillOpacity={0.2} strokeWidth={2} dot={false} />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
