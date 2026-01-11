import { useTranslation } from 'react-i18next'
import { useEffect, useState } from 'react'
import { api } from '../services/api'
import { Link } from 'react-router-dom'

interface Stats {
  totalBacktests: number
  activeStrategies: number
  totalReturn: number
  sharpeRatio: number
}

export default function Dashboard() {
  const { t } = useTranslation()
  const [stats, setStats] = useState<Stats>({
    totalBacktests: 0,
    activeStrategies: 0,
    totalReturn: 0,
    sharpeRatio: 0,
  })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadStats()
  }, [])

  const loadStats = async () => {
    try {
      const response = await api.get('/strategy-backtest/history?limit=1000')
      const backtests = response.data.backtests || []

      setStats({
        totalBacktests: backtests.length,
        activeStrategies: new Set(backtests.map((b: any) => b.strategy_name)).size,
        totalReturn: backtests.reduce((sum: number, b: any) => sum + (b.total_return || 0), 0),
        sharpeRatio: backtests.length > 0
          ? backtests.reduce((sum: number, b: any) => sum + (b.sharpe_ratio || 0), 0) / backtests.length
          : 0,
      })
    } catch (error) {
      console.error('Failed to load stats:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <span className="text-sm text-gray-500 dark:text-gray-400">Loading...</span>
      </div>
    )
  }

  const statItems = [
    { label: 'Total Backtests', value: stats.totalBacktests.toString() },
    { label: 'Active Strategies', value: stats.activeStrategies.toString() },
    { label: 'Total Return', value: `${(stats.totalReturn * 100).toFixed(2)}%` },
    { label: 'Avg Sharpe Ratio', value: stats.sharpeRatio.toFixed(2) },
  ]

  const quickActions = [
    { title: 'New Backtest', description: 'Run a strategy backtest', href: '/backtest' },
    { title: 'View History', description: 'Browse past results', href: '/history' },
    { title: 'Monitoring', description: 'System status', href: '/monitoring' },
  ]

  return (
    <div className="space-y-10">
      <div>
        <h1 className="text-2xl font-semibold text-gray-900 dark:text-gray-100 tracking-tight">
          {t('dashboard.title')}
        </h1>
        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
          Overview of your backtesting activity and performance metrics.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {statItems.map((item) => (
          <div
            key={item.label}
            className="p-5 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg"
          >
            <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
              {item.label}
            </p>
            <p className="mt-2 text-2xl font-semibold text-gray-900 dark:text-gray-100">
              {item.value}
            </p>
          </div>
        ))}
      </div>

      <div>
        <h2 className="text-base font-semibold text-gray-900 dark:text-gray-100 mb-4">
          Quick Actions
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {quickActions.map((action) => (
            <Link
              key={action.title}
              to={action.href}
              className="group p-4 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg hover:border-gray-300 dark:hover:border-gray-700 transition-colors duration-150"
            >
              <p className="text-sm font-medium text-gray-900 dark:text-gray-100 group-hover:text-gray-700 dark:group-hover:text-gray-300 transition-colors">
                {action.title}
              </p>
              <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                {action.description}
              </p>
            </Link>
          ))}
        </div>
      </div>
    </div>
  )
}
