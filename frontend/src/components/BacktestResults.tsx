import { useTranslation } from 'react-i18next'
import { useMemo, useState } from 'react'
import { format } from 'date-fns'
import { useThemeStore } from '../stores/themeStore'
import {
  ReturnsDistributionChart,
  RollingReturnsChart,
  MonthlyReturnsHeatmap,
  RollingSharpeChart,
  TradePnLChart,
  EquityCurveChart,
  DrawdownChart,
  MetricCard,
  MetricsGrid,
} from './charts'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
  Scatter,
  ReferenceLine,
  Area,
  Brush,
  Bar,
} from 'recharts'

interface BacktestResultsProps {
  results: any
}

type PricePoint = {
  ts: number
  close: number
  open: number
  high: number
  low: number
  volume: number
}

type EquityPoint = {
  ts: number
  equity: number
  drawdown: number
}

type TradePoint = {
  ts: number
  price: number
  size: number
  amount: number
  commission: number
  side: 'buy' | 'sell'
  rawTs?: string
  symbol?: string
}

function formatMoney(v: number) {
  if (!Number.isFinite(v)) return 'N/A'
  return v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

function formatCompact(v: number) {
  if (!Number.isFinite(v)) return 'N/A'
  const abs = Math.abs(v)
  if (abs >= 1_000_000_000) return `${(v / 1_000_000_000).toFixed(2)}B`
  if (abs >= 1_000_000) return `${(v / 1_000_000).toFixed(2)}M`
  if (abs >= 1_000) return `${(v / 1_000).toFixed(1)}K`
  return v.toFixed(0)
}

export default function BacktestResults({ results }: BacktestResultsProps) {
  const { t } = useTranslation()
  const { theme } = useThemeStore()
  const isDark = theme === 'dark'
  
  // 处理数据，确保兼容性
  const resultsData = results.results || results
  const perfStats = resultsData.performance_stats || {}
  const tradeStats = resultsData.trade_stats || {}

  const portfolioData = resultsData.portfolio_value || {}
  const priceData = results.price_data || {}
  const tradesRaw = results.trades || []
  
  // 增强指标数据
  const enhancedMetrics = results.enhanced_metrics || {}
  const equityComparison = results.equity_comparison || {}
  const benchmarkData = results.benchmark_data || {}

  const benchmarkSymbol = benchmarkData.symbol || results.config?.benchmark_symbol || results.results?.config?.benchmark_symbol || 'Benchmark'

  const palette = useMemo(() => {
    return isDark
      ? {
          bg: '#0B1220',
          card: '#111827',
          text: '#E5E7EB',
          muted: '#9CA3AF',
          grid: '#243041',
          blue: '#60A5FA',
          cyan: '#22D3EE',
          green: '#34D399',
          greenStroke: '#10B981',
          red: '#F87171',
          redStroke: '#EF4444',
          amber: '#FBBF24',
        }
      : {
          bg: '#FFFFFF',
          card: '#FFFFFF',
          text: '#111827',
          muted: '#6B7280',
          grid: '#E5E7EB',
          blue: '#2563EB',
          cyan: '#0284C7',
          green: '#16A34A',
          greenStroke: '#16A34A',
          red: '#DC2626',
          redStroke: '#DC2626',
          amber: '#D97706',
        }
  }, [isDark])

  const priceSeries: PricePoint[] = useMemo(() => {
    const tsArr: string[] = priceData.timestamps || []
    return tsArr.map((ts: string, i: number) => {
      const d = new Date(ts)
      return {
        ts: d.getTime(),
        close: Number(priceData.close?.[i] ?? 0) || 0,
        open: Number(priceData.open?.[i] ?? 0) || 0,
        high: Number(priceData.high?.[i] ?? 0) || 0,
        low: Number(priceData.low?.[i] ?? 0) || 0,
        volume: Number(priceData.volume?.[i] ?? 0) || 0,
      }
    })
  }, [priceData])

  const equitySeries: EquityPoint[] = useMemo(() => {
    const tsArr: string[] = portfolioData.timestamps || []
    const values: number[] = portfolioData.values || []
    let peak = -Infinity
    return tsArr.map((ts: string, i: number) => {
      const d = new Date(ts)
      const equity = Number(values[i] ?? 0) || 0
      peak = Math.max(peak, equity || 0)
      const dd = peak > 0 ? (equity - peak) / peak : 0
      return { ts: d.getTime(), equity, drawdown: dd }
    })
  }, [portfolioData])

  const priceDomain = useMemo(() => {
    if (!priceSeries.length) return [0, 1] as [number, number]
    // 用价格自身的 low/high 计算域，避免交易点异常值把坐标轴"拉爆"
    let min = Infinity
    let max = -Infinity
    for (const p of priceSeries) {
      if (Number.isFinite(p.low)) min = Math.min(min, p.low)
      if (Number.isFinite(p.high)) max = Math.max(max, p.high)
    }
    if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return [0, 1] as [number, number]
    const pad = (max - min) * 0.03
    return [min - pad, max + pad] as [number, number]
  }, [priceSeries])

  const lastClose = useMemo(() => {
    const last = priceSeries[priceSeries.length - 1]
    return last ? last.close : 0
  }, [priceSeries])

  const priceSymbol = results.price_symbol || results.symbol?.split(',')?.[0] || ''

  const trades: TradePoint[] = useMemo(() => {
    return (tradesRaw || [])
      .map((tr: any) => {
        const sideRaw = String(tr.order_type || '').toLowerCase()
        const side: 'buy' | 'sell' | null = sideRaw === 'buy' ? 'buy' : sideRaw === 'sell' ? 'sell' : null
        const ts = new Date(tr.timestamp).getTime()
        if (!side || !Number.isFinite(ts)) return null
        const price = Number(tr.price ?? 0) || 0
        const size = Number(tr.size ?? 0) || 0
        const amount = Number(tr.amount ?? price * size) || 0
        const commission = Number(tr.commission ?? 0) || 0
        return {
          ts,
          price,
          size,
          amount,
          commission,
          side,
          rawTs: tr.timestamp,
          symbol: tr.symbol,
        }
      })
      .filter((trade: TradePoint | null): trade is TradePoint => {
        if (!trade) return false
        if (!priceSymbol) return true
        return !trade.symbol || trade.symbol === priceSymbol
      }) as TradePoint[]
  }, [tradesRaw, priceSymbol])

  const buySignals = useMemo(() => trades.filter((t) => t.side === 'buy'), [trades])
  const sellSignals = useMemo(() => trades.filter((t) => t.side === 'sell'), [trades])

  // 安全地获取数值
  const finalValue = Number(resultsData.final_value) || Number(enhancedMetrics.total_return * results?.config?.initial_capital + results?.config?.initial_capital) || 0
  const totalReturn = Number(resultsData.total_return) || Number(enhancedMetrics.total_return) || 0
  const benchmarkReturn = Number(enhancedMetrics.benchmark_return) || 0
  const sharpeRatio = Number(perfStats.sharpe_ratio) || Number(enhancedMetrics.sharpe_ratio) || 0
  const maxDrawdown = Number(perfStats.max_drawdown) || Number(enhancedMetrics.max_drawdown) || 0
  const winRate = Number(perfStats.win_rate) || Number(enhancedMetrics.win_rate) || 0
  const dailyWinRate = Number(enhancedMetrics.daily_win_rate) || 0
  const totalTrades = Number(tradeStats.total_trades) || Number(enhancedMetrics.total_trades) || 0
  
  // 增强指标
  const annualReturn = Number(enhancedMetrics.annual_return) || 0
  const annualVolatility = Number(enhancedMetrics.annual_volatility) || 0
  const benchmarkVolatility = Number(enhancedMetrics.benchmark_volatility) || 0
  const sortinoRatio = Number(enhancedMetrics.sortino_ratio) || 0
  const calmarRatio = Number(enhancedMetrics.calmar_ratio) || 0
  
  // 基准对比指标
  const excessReturn = Number(enhancedMetrics.excess_return) || 0
  const alpha = Number(enhancedMetrics.alpha) || 0
  const beta = Number(enhancedMetrics.beta) || 0
  const informationRatio = Number(enhancedMetrics.information_ratio) || 0
  const rSquared = Number(enhancedMetrics.r_squared) || 0
  
  // 交易统计
  const winningTrades = Number(enhancedMetrics.winning_trades) || 0
  const losingTrades = Number(enhancedMetrics.losing_trades) || 0
  const profitFactor = Number(enhancedMetrics.profit_factor) || 0
  const profitLossRatio = Number(enhancedMetrics.profit_loss_ratio) || 0
  const expectancy = Number(enhancedMetrics.expectancy) || 0
  const largestWin = Number(enhancedMetrics.largest_win) || 0
  const largestLoss = Number(enhancedMetrics.largest_loss) || 0
  const consecutiveWins = Number(enhancedMetrics.consecutive_wins) || 0
  const consecutiveLosses = Number(enhancedMetrics.consecutive_losses) || 0

  // 增强分析数据
  const returnsDistribution = resultsData.returns_distribution || {}
  const rollingMetrics = resultsData.rolling_metrics || {}
  const monthlyReturns = resultsData.monthly_returns || {}

  const stats = [
    { label: t('backtest.finalValue'), value: `$${formatMoney(finalValue)}` },
    { label: t('backtest.totalReturn'), value: `${(totalReturn * 100).toFixed(2)}%` },
    { label: t('backtest.sharpeRatio'), value: isFinite(sharpeRatio) ? sharpeRatio.toFixed(2) : 'N/A' },
    { label: t('backtest.maxDrawdown'), value: `${(maxDrawdown * 100).toFixed(2)}%` },
    { label: t('backtest.winRate'), value: `${(winRate * 100).toFixed(2)}%` },
    { label: t('backtest.totalTrades'), value: totalTrades },
  ]

  const [activeTab, setActiveTab] = useState<string>('performance')
  const [showBenchmark, setShowBenchmark] = useState(true)
  const [showExcess, setShowExcess] = useState(true)
  const [useLogScale, setUseLogScale] = useState(false)

  const tabs = [
    { id: 'performance', label: 'Performance' },
    { id: 'benchmark', label: 'Benchmark' },
    { id: 'analysis', label: 'Analysis' },
    { id: 'trades', label: 'Trades' },
  ]

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 border border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
          {t('backtest.results')}
        </h2>
        <div className="flex gap-1 bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-1.5 text-sm font-medium rounded-md transition-colors ${
                activeTab === tab.id
                  ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {activeTab === 'performance' && (
        <>
          {/* Performance Summary - All Key Metrics */}
          <div className="mb-6">
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-3">Performance Summary</h3>
            
            {/* Primary Return Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3 mb-3">
              <MetricCard
                title="Strategy Return"
                value={totalReturn}
                format="percent"
                color={totalReturn >= 0 ? 'green' : 'red'}
              />
              <MetricCard
                title={benchmarkSymbol}
                value={benchmarkReturn}
                format="percent"
                color={benchmarkReturn >= 0 ? 'green' : 'red'}
              />
              <MetricCard
                title="Annual Return"
                value={annualReturn}
                format="percent"
                color={annualReturn >= 0 ? 'green' : 'red'}
              />
              <MetricCard
                title="Alpha"
                value={alpha}
                color={alpha >= 0 ? 'green' : 'red'}
              />
              <MetricCard
                title="Beta"
                value={beta}
                color={beta >= 0.8 && beta <= 1.2 ? 'green' : 'gray'}
              />
              <MetricCard
                title="Sharpe"
                value={sharpeRatio}
                color={sharpeRatio >= 1 ? 'green' : sharpeRatio >= 0.5 ? 'blue' : 'red'}
              />
              <MetricCard
                title="Sortino"
                value={sortinoRatio}
                color={sortinoRatio >= 1 ? 'green' : 'gray'}
              />
              <MetricCard
                title="Info Ratio"
                value={informationRatio}
                color={informationRatio >= 0.5 ? 'green' : 'gray'}
              />
            </div>

            {/* Risk & Trade Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3">
              <MetricCard
                title="Volatility"
                value={annualVolatility}
                format="percent"
                color="blue"
              />
              <MetricCard
                title={`${benchmarkSymbol} Vol`}
                value={benchmarkVolatility}
                format="percent"
                color="gray"
              />
              <MetricCard
                title="Max Drawdown"
                value={maxDrawdown}
                format="percent"
                color="red"
              />
              <MetricCard
                title="Win Rate"
                value={winRate}
                format="percent"
                color={winRate >= 0.5 ? 'green' : 'red'}
              />
              <MetricCard
                title="Daily Win"
                value={dailyWinRate}
                format="percent"
                color={dailyWinRate >= 0.5 ? 'green' : 'gray'}
              />
              <MetricCard
                title="P/L Ratio"
                value={profitLossRatio}
                color={profitLossRatio >= 2 ? 'green' : profitLossRatio >= 1 ? 'blue' : 'red'}
              />
              <MetricCard
                title="Wins"
                value={winningTrades}
                format="integer"
                color="green"
              />
              <MetricCard
                title="Losses"
                value={losingTrades}
                format="integer"
                color="red"
              />
            </div>
          </div>

          {/* Secondary Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <MetricCard
              title="Calmar Ratio"
              value={calmarRatio}
              color={calmarRatio >= 3 ? 'green' : 'gray'}
            />
            <MetricCard
              title="R-Squared"
              value={rSquared}
              color={rSquared >= 0.8 ? 'green' : 'gray'}
            />
            <MetricCard
              title="Total Trades"
              value={totalTrades}
              format="integer"
              color="blue"
            />
            <MetricCard
              title="Final Value"
              value={finalValue}
              format="currency"
              color="blue"
            />
          </div>

          {/* Main Charts */}
          {equityComparison.strategy_equity && equityComparison.strategy_equity.length > 0 && (
            <div className="space-y-4">
              {/* Chart Controls */}
              <div className="flex items-center gap-4 flex-wrap">
                <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                  <input
                    type="checkbox"
                    checked={showBenchmark}
                    onChange={(e) => setShowBenchmark(e.target.checked)}
                    className="rounded"
                  />
                  Show {benchmarkSymbol}
                </label>
                <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                  <input
                    type="checkbox"
                    checked={showExcess}
                    onChange={(e) => setShowExcess(e.target.checked)}
                    className="rounded"
                  />
                  Show Excess
                </label>
                <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                  <input
                    type="checkbox"
                    checked={useLogScale}
                    onChange={(e) => setUseLogScale(e.target.checked)}
                    className="rounded"
                  />
                  Log Scale
                </label>
              </div>

              {/* Equity Curve Chart */}
              <EquityCurveChart
                strategyEquity={equityComparison.strategy_equity}
                benchmarkEquity={showBenchmark ? equityComparison.benchmark_equity : undefined}
                excessReturns={showExcess ? equityComparison.excess_returns : undefined}
                timestamps={equityComparison.timestamps || []}
                showBenchmark={showBenchmark}
                showExcess={showExcess}
                useLogScale={useLogScale}
                height={420}
                benchmarkSymbol={benchmarkSymbol}
              />

              {/* Drawdown Chart */}
              {enhancedMetrics.drawdown_series && (
                <DrawdownChart
                  drawdownSeries={enhancedMetrics.drawdown_series}
                  timestamps={equityComparison.timestamps || []}
                  maxDrawdownWindow={enhancedMetrics.max_drawdown_window}
                  height={200}
                />
              )}
            </div>
          )}
        </>
      )}

      {/* Benchmark Comparison Tab */}
      {activeTab === 'benchmark' && (
        <div className="space-y-6">
          {/* Benchmark Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
            <MetricCard
              title="Strategy Return"
              value={totalReturn}
              format="percent"
              color={totalReturn >= 0 ? 'green' : 'red'}
            />
            <MetricCard
              title={`${benchmarkSymbol} Return`}
              value={benchmarkReturn}
              format="percent"
              color={benchmarkReturn >= 0 ? 'green' : 'red'}
            />
            <MetricCard
              title="Excess Return"
              value={excessReturn}
              format="percent"
              color={excessReturn >= 0 ? 'green' : 'red'}
            />
            <MetricCard
              title="Alpha"
              value={alpha}
              color={alpha >= 0 ? 'green' : 'red'}
              benchmark={0}
            />
            <MetricCard
              title="Beta"
              value={beta}
              color={beta >= 0.8 && beta <= 1.2 ? 'green' : beta > 1.2 ? 'blue' : 'gray'}
              benchmark={1}
            />
          </div>

          {/* Risk Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <MetricCard
              title="Information Ratio"
              value={informationRatio}
              color={informationRatio >= 0.5 ? 'green' : 'gray'}
            />
            <MetricCard
              title="R-Squared"
              value={rSquared}
              color={rSquared >= 0.8 ? 'green' : 'gray'}
            />
            <MetricCard
              title="Strategy Volatility"
              value={annualVolatility}
              format="percent"
              color="blue"
            />
            <MetricCard
              title={`${benchmarkSymbol} Volatility`}
              value={benchmarkVolatility}
              format="percent"
              color="gray"
            />
          </div>

          {/* Max Drawdown Window */}
          {enhancedMetrics.max_drawdown_window && (
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-3">Maximum Drawdown Period</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Period</p>
                  <p className="font-semibold text-gray-900 dark:text-white">
                    {enhancedMetrics.max_drawdown_window.start_date?.split('T')[0]} → {enhancedMetrics.max_drawdown_window.end_date?.split('T')[0]}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Drawdown</p>
                  <p className="font-semibold text-red-600">
                    {(enhancedMetrics.max_drawdown_window.drawdown_pct * 100).toFixed(2)}%
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Duration</p>
                  <p className="font-semibold text-gray-900 dark:text-white">
                    {enhancedMetrics.max_drawdown_window.duration_days} days
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Recovery</p>
                  <p className="font-semibold text-gray-900 dark:text-white">
                    {enhancedMetrics.max_drawdown_window.recovery_days 
                      ? `${enhancedMetrics.max_drawdown_window.recovery_days} days` 
                      : 'Not recovered'}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* All Drawdown Windows */}
          {enhancedMetrics.drawdown_windows && enhancedMetrics.drawdown_windows.length > 0 && (
            <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Drawdown Periods</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <th className="text-left py-2 px-3 text-gray-500 dark:text-gray-400">Period</th>
                      <th className="text-right py-2 px-3 text-gray-500 dark:text-gray-400">Drawdown</th>
                      <th className="text-right py-2 px-3 text-gray-500 dark:text-gray-400">Duration</th>
                      <th className="text-right py-2 px-3 text-gray-500 dark:text-gray-400">Recovery</th>
                    </tr>
                  </thead>
                  <tbody>
                    {enhancedMetrics.drawdown_windows.slice(0, 10).map((window: any, idx: number) => (
                      <tr key={idx} className="border-b border-gray-100 dark:border-gray-700/50">
                        <td className="py-2 px-3 text-gray-700 dark:text-gray-300">
                          {window.start_date?.split('T')[0]} → {window.end_date?.split('T')[0]}
                        </td>
                        <td className="py-2 px-3 text-right text-red-600 font-medium">
                          {(window.drawdown_pct * 100).toFixed(2)}%
                        </td>
                        <td className="py-2 px-3 text-right text-gray-700 dark:text-gray-300">
                          {window.duration_days} days
                        </td>
                        <td className="py-2 px-3 text-right text-gray-700 dark:text-gray-300">
                          {window.recovery_days ? `${window.recovery_days} days` : '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
      
      {activeTab === 'analysis' && (
        <div className="space-y-6">
          {/* Returns Distribution */}
          {Object.keys(returnsDistribution).length > 0 && (
            <ReturnsDistributionChart distribution={returnsDistribution} />
          )}

          {/* Rolling Returns */}
          {Object.keys(rollingMetrics).length > 0 && (
            <RollingReturnsChart rollingMetrics={rollingMetrics} />
          )}

          {/* Rolling Sharpe & Drawdown */}
          {Object.keys(rollingMetrics).length > 0 && (
            <RollingSharpeChart rollingMetrics={rollingMetrics} />
          )}

          {/* Monthly Returns Heatmap */}
          {Object.keys(monthlyReturns).length > 0 && (
            <MonthlyReturnsHeatmap monthlyReturns={monthlyReturns} />
          )}
        </div>
      )}

      {/* Trade Analysis */}
      {activeTab === 'trades' && (
        <div className="space-y-6">
          {/* Trade Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            <MetricCard
              title="Total Trades"
              value={totalTrades}
              format="integer"
              color="blue"
            />
            <MetricCard
              title="Winning Trades"
              value={winningTrades}
              format="integer"
              color="green"
            />
            <MetricCard
              title="Losing Trades"
              value={losingTrades}
              format="integer"
              color="red"
            />
            <MetricCard
              title="Win Rate"
              value={winRate}
              format="percent"
              color={winRate >= 0.5 ? 'green' : 'red'}
            />
            <MetricCard
              title="Daily Win Rate"
              value={dailyWinRate}
              format="percent"
              color={dailyWinRate >= 0.5 ? 'green' : dailyWinRate >= 0.4 ? 'blue' : 'gray'}
            />
            <MetricCard
              title="Profit/Loss Ratio"
              value={profitLossRatio}
              color={profitLossRatio >= 2 ? 'green' : profitLossRatio >= 1 ? 'blue' : 'red'}
            />
          </div>
          
          {/* Extended Trade Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard
              title="Largest Win"
              value={largestWin}
              format="currency"
              color="green"
            />
            <MetricCard
              title="Largest Loss"
              value={largestLoss}
              format="currency"
              color="red"
            />
            <MetricCard
              title="Consecutive Wins"
              value={consecutiveWins}
              format="integer"
              color="green"
            />
            <MetricCard
              title="Consecutive Losses"
              value={consecutiveLosses}
              format="integer"
              color="red"
            />
          </div>

          {/* Trade P&L Distribution */}
          {trades.length > 0 && (
            <TradePnLChart trades={trades} priceData={priceData} />
          )}

          {/* Trade List */}
          {trades.length > 0 && (
            <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Trade List</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200 dark:border-gray-700">
                      <th className="text-left py-2 px-3 text-gray-500 dark:text-gray-400">Time</th>
                      <th className="text-left py-2 px-3 text-gray-500 dark:text-gray-400">Type</th>
                      <th className="text-right py-2 px-3 text-gray-500 dark:text-gray-400">Price</th>
                      <th className="text-right py-2 px-3 text-gray-500 dark:text-gray-400">Shares</th>
                      <th className="text-right py-2 px-3 text-gray-500 dark:text-gray-400">Amount</th>
                      <th className="text-right py-2 px-3 text-gray-500 dark:text-gray-400">Commission</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trades.slice(0, 50).map((trade, idx) => (
                      <tr key={idx} className="border-b border-gray-100 dark:border-gray-700/50">
                        <td className="py-2 px-3 text-gray-700 dark:text-gray-300">
                          {trade.rawTs ? format(new Date(trade.rawTs), 'yyyy-MM-dd HH:mm') : '-'}
                        </td>
                        <td className="py-2 px-3">
                          <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                            trade.side === 'buy'
                              ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                              : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                          }`}>
                            {trade.side?.toUpperCase()}
                          </span>
                        </td>
                        <td className="py-2 px-3 text-right text-gray-700 dark:text-gray-300">
                          ${trade.price?.toFixed(2) || '0.00'}
                        </td>
                        <td className="py-2 px-3 text-right text-gray-700 dark:text-gray-300">
                          {Math.round(trade.size) || '0'}
                        </td>
                        <td className="py-2 px-3 text-right text-gray-700 dark:text-gray-300">
                          ${formatMoney(trade.amount || 0)}
                        </td>
                        <td className="py-2 px-3 text-right text-gray-500 dark:text-gray-400">
                          ${trade.commission?.toFixed(2) || '0.00'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {trades.length > 50 && (
                  <p className="text-center text-gray-500 dark:text-gray-400 mt-2">
                    Showing 50 of {trades.length} trades
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {equitySeries.length > 0 && activeTab === 'performance' && (
        <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900/20 p-3">
            <div className="flex items-baseline justify-between gap-3 mb-2">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                {t('backtest.equity')}
              </h3>
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {equitySeries.length > 0
                  ? `${format(new Date(equitySeries[0].ts), 'yyyy-MM-dd')} → ${format(
                      new Date(equitySeries[equitySeries.length - 1].ts),
                      'yyyy-MM-dd'
                    )}`
                  : null}
              </div>
            </div>

            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={equitySeries} margin={{ top: 8, right: 16, left: 8, bottom: 24 }}>
                <CartesianGrid stroke={palette.grid} strokeDasharray="3 3" opacity={0.9} />
                <XAxis
                  dataKey="ts"
                  type="number"
                  scale="time"
                  domain={['dataMin', 'dataMax']}
                  tick={{ fill: palette.muted, fontSize: 12 }}
                  tickFormatter={(v) => format(new Date(v), 'MM-dd')}
                  minTickGap={24}
                />
                <YAxis
                  tick={{ fill: palette.muted, fontSize: 12 }}
                  width={72}
                  tickFormatter={(v) => `$${formatMoney(Number(v))}`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: isDark ? '#0B1220' : '#FFFFFF',
                    border: `1px solid ${isDark ? '#243041' : '#E5E7EB'}`,
                    borderRadius: '10px',
                    color: palette.text,
                    boxShadow: isDark ? '0 10px 30px rgba(0,0,0,0.35)' : '0 10px 30px rgba(0,0,0,0.10)',
                  }}
                  labelFormatter={(label) => format(new Date(Number(label)), 'yyyy-MM-dd')}
                  formatter={(value: any) => [`$${formatMoney(Number(value))}`, t('backtest.equity')]}
                />
                <Legend wrapperStyle={{ color: palette.muted, fontSize: 12 }} />
                {/* initial capital reference */}
                <ReferenceLine
                  y={Number(results?.config?.initial_capital ?? 0) || 0}
                  stroke={palette.grid}
                  strokeDasharray="4 4"
                  label={{
                    position: 'right',
                    value: 'Init',
                    fill: palette.muted,
                    fontSize: 11,
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="equity"
                  name={t('backtest.equity')}
                  stroke={palette.cyan}
                  fill={palette.cyan}
                  fillOpacity={isDark ? 0.10 : 0.12}
                  strokeWidth={2}
                  dot={false}
                />
                <Line type="monotone" dataKey="equity" stroke={palette.cyan} strokeWidth={2} dot={false} />
                <Brush
                  dataKey="ts"
                  height={24}
                  stroke={palette.grid}
                  travellerWidth={10}
                  tickFormatter={(v) => format(new Date(v), 'MM-dd')}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900/20 p-3">
            <div className="flex items-baseline justify-between gap-3 mb-2">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                {t('backtest.drawdown')}
              </h3>
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {t('backtest.maxDrawdown')}: {(maxDrawdown * 100).toFixed(2)}%
              </div>
            </div>

            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={equitySeries} margin={{ top: 8, right: 16, left: 8, bottom: 24 }}>
                <CartesianGrid stroke={palette.grid} strokeDasharray="3 3" opacity={0.9} />
                <XAxis
                  dataKey="ts"
                  type="number"
                  scale="time"
                  domain={['dataMin', 'dataMax']}
                  tick={{ fill: palette.muted, fontSize: 12 }}
                  tickFormatter={(v) => format(new Date(v), 'MM-dd')}
                  minTickGap={24}
                />
                <YAxis
                  tick={{ fill: palette.muted, fontSize: 12 }}
                  width={56}
                  tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`}
                  domain={['dataMin', 0]}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: isDark ? '#0B1220' : '#FFFFFF',
                    border: `1px solid ${isDark ? '#243041' : '#E5E7EB'}`,
                    borderRadius: '10px',
                    color: palette.text,
                    boxShadow: isDark ? '0 10px 30px rgba(0,0,0,0.35)' : '0 10px 30px rgba(0,0,0,0.10)',
                  }}
                  labelFormatter={(label) => format(new Date(Number(label)), 'yyyy-MM-dd')}
                  formatter={(value: any) => [`${(Number(value) * 100).toFixed(2)}%`, t('backtest.drawdown')]}
                />
                <ReferenceLine y={0} stroke={palette.grid} />
                <Area
                  type="monotone"
                  dataKey="drawdown"
                  name={t('backtest.drawdown')}
                  stroke={palette.amber}
                  fill={palette.amber}
                  fillOpacity={isDark ? 0.12 : 0.10}
                  strokeWidth={2}
                  dot={false}
                />
                <Line type="monotone" dataKey="drawdown" stroke={palette.amber} strokeWidth={2} dot={false} />
                <Brush
                  dataKey="ts"
                  height={24}
                  stroke={palette.grid}
                  travellerWidth={10}
                  tickFormatter={(v) => format(new Date(v), 'MM-dd')}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  )
}
