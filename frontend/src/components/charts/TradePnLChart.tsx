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

interface TradePnLChartProps {
  trades: Array<{
    timestamp?: string
    price?: number
    size?: number
    order_type?: string
    commission?: number
    symbol?: string
  }>
  priceData?: never // unused, kept for API compatibility
}

export default function TradePnLChart({ trades, priceData }: TradePnLChartProps) {
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

  const { chartData, stats } = useMemo(() => {
    if (!trades || trades.length < 2) {
      return { chartData: [], stats: null }
    }

    const sellTrades = trades.filter((t) => t.order_type?.toLowerCase() === 'sell')
    if (sellTrades.length === 0) {
      return { chartData: [], stats: null }
    }

    const pnlData = sellTrades.map((trade, index) => {
      const prevBuy = index > 0 ? trades.filter((t) => t.order_type?.toLowerCase() === 'buy')[index - 1] : null
      const buyPrice = prevBuy?.price || trade.price || 0
      const sellPrice = trade.price || 0
      const size = trade.size || 1
      const pnl = (sellPrice - buyPrice) * size - (trade.commission || 0)
      const pnlPct = buyPrice > 0 ? ((sellPrice - buyPrice) / buyPrice) * 100 : 0

      return {
        index: index + 1,
        tradeNum: `Trade ${index + 1}`,
        pnl,
        pnlPct,
        buyPrice,
        sellPrice,
        size,
      }
    })

    const winning = pnlData.filter((d) => d.pnl > 0)
    const losing = pnlData.filter((d) => d.pnl < 0)

    const stats = {
      totalTrades: pnlData.length,
      winningTrades: winning.length,
      losingTrades: losing.length,
      totalPnL: pnlData.reduce((sum, d) => sum + d.pnl, 0),
      avgWinning: winning.length > 0 ? winning.reduce((sum, d) => sum + d.pnl, 0) / winning.length : 0,
      avgLosing: losing.length > 0 ? losing.reduce((sum, d) => sum + d.pnl, 0) / losing.length : 0,
      maxWin: winning.length > 0 ? Math.max(...winning.map((d) => d.pnl)) : 0,
      maxLoss: losing.length > 0 ? Math.min(...losing.map((d) => d.pnl)) : 0,
    }

    return { chartData: pnlData, stats }
  }, [trades])

  if (chartData.length === 0) {
    return (
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Trade P&L Distribution</h3>
        <p className="text-gray-500 dark:text-gray-400">No sell trades available for P&L analysis</p>
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white">Trade P&L Distribution</h3>
        {stats && (
          <div className="flex gap-4 text-xs">
            <span className="text-green-600 dark:text-green-400">Win: {stats.winningTrades}</span>
            <span className="text-red-600 dark:text-red-400">Loss: {stats.losingTrades}</span>
            <span className="text-gray-500 dark:text-gray-400">Total: ${stats.totalPnL.toFixed(2)}</span>
          </div>
        )}
      </div>

      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={chartData} margin={{ top: 8, right: 16, left: 8, bottom: 24 }}>
          <CartesianGrid stroke={palette.grid} strokeDasharray="3 3" opacity={0.7} />
          <XAxis dataKey="tradeNum" tick={{ fill: palette.muted, fontSize: 10 }} interval={Math.floor(chartData.length / 10)} />
          <YAxis tick={{ fill: palette.muted, fontSize: 12 }} tickFormatter={(v) => `$${v.toFixed(0)}`} />
          <Tooltip
            contentStyle={{
              backgroundColor: isDark ? '#0B1220' : '#FFFFFF',
              border: `1px solid ${isDark ? '#243041' : '#E5E7EB'}`,
              borderRadius: '8px',
              color: palette.text,
            }}
            formatter={(value: number, name: string) => [
              name === 'pnl' ? `$${value.toFixed(2)}` : `${value.toFixed(2)}%`,
              name === 'pnl' ? 'P&L' : 'Return %',
            ]}
            labelFormatter={(label) => `Trade #${label}`}
          />
          <ReferenceLine y={0} stroke={palette.muted} strokeDasharray="4 4" />
          <Bar dataKey="pnl" radius={[4, 4, 0, 0]}>
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.pnl >= 0 ? palette.positive : palette.negative} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="text-center">
            <p className="text-xs text-gray-500 dark:text-gray-400">Avg Win</p>
            <p className="text-sm font-medium text-green-600 dark:text-green-400">
              ${stats.avgWinning > 0 ? stats.avgWinning.toFixed(2) : '0.00'}
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs text-gray-500 dark:text-gray-400">Avg Loss</p>
            <p className="text-sm font-medium text-red-600 dark:text-red-400">
              ${stats.avgLosing < 0 ? Math.abs(stats.avgLosing).toFixed(2) : '0.00'}
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs text-gray-500 dark:text-gray-400">Best Trade</p>
            <p className="text-sm font-medium text-green-600 dark:text-green-400">
              ${stats.maxWin > 0 ? stats.maxWin.toFixed(2) : '0.00'}
            </p>
          </div>
          <div className="text-center">
            <p className="text-xs text-gray-500 dark:text-gray-400">Worst Trade</p>
            <p className="text-sm font-medium text-red-600 dark:text-red-400">
              ${stats.maxLoss < 0 ? Math.abs(stats.maxLoss).toFixed(2) : '0.00'}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
