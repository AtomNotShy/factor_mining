import { useTranslation } from 'react-i18next'
import { useState, useEffect } from 'react'
import { api } from '../services/api'
import { format } from 'date-fns'
import { BacktestDetails } from '../components/tradingview'

export default function History() {
  const { t } = useTranslation()
  const [backtests, setBacktests] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedBacktest, setSelectedBacktest] = useState<any>(null)
  const [selectedBacktestId, setSelectedBacktestId] = useState<string | null>(null)
  const [loadingDetail, setLoadingDetail] = useState(false)
  const [filters, setFilters] = useState({
    strategy_name: '',
    symbol: '',
  })

  const formatSymbols = (symbol?: string, universe?: string[]) => {
    const list = Array.isArray(universe) && universe.length
      ? universe
      : (symbol || '').split(',').map((item) => item.trim()).filter(Boolean)
    if (list.length === 0) return symbol || '-'
    if (list.length <= 3) return list.join(', ')
    return `${list.slice(0, 3).join(', ')} +${list.length - 3}`
  }

  useEffect(() => {
    loadHistory()
  }, [filters])

  const loadHistory = async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams()
      if (filters.strategy_name) params.append('strategy_name', filters.strategy_name)
      if (filters.symbol) params.append('symbol', filters.symbol)

      const response = await api.get(`/strategy-backtest/history?${params.toString()}`)
      setBacktests(response.data.backtests || [])
    } catch (error) {
      console.error('Failed to load history:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleViewDetail = async (id: string) => {
    setLoadingDetail(true)
    try {
      const response = await api.get(`/strategy-backtest/history/${id}`)
      console.log('Loaded backtest detail:', response.data)

      setSelectedBacktestId(id)

      let backtestData = response.data

      if (!backtestData.results && backtestData.final_value !== undefined) {
        console.warn('Old format detected, adapting...')
        backtestData = {
          ...backtestData,
          results: {
            final_value: backtestData.final_value,
            total_return: backtestData.total_return,
            performance_stats: backtestData.performance_stats || {},
            trade_stats: backtestData.trade_stats || {},
            portfolio_value: backtestData.portfolio_value,
            returns: backtestData.returns
          }
        }
      }

      if (!backtestData.id) {
        backtestData.id = id
      }

      setSelectedBacktest(backtestData)
    } catch (error) {
      console.error('Failed to load backtest detail:', error)
      alert(t('history.loadError'))
    } finally {
      setLoadingDetail(false)
    }
  }

  const handleDelete = async (id: string) => {
    if (!confirm(t('history.confirmDelete'))) return

    try {
      await api.delete(`/strategy-backtest/history/${id}`)
      if (selectedBacktest?.id === id) {
        setSelectedBacktest(null)
      }
      loadHistory()
    } catch (error) {
      console.error('Failed to delete:', error)
      alert(t('history.deleteError'))
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <span className="text-sm text-gray-500 dark:text-gray-400">Loading...</span>
      </div>
    )
  }

  if (selectedBacktest) {
    return (
      <div className="h-screen">
        <BacktestDetails
          data={selectedBacktest}
          onBack={() => {
            setSelectedBacktest(null)
            setSelectedBacktestId(null)
          }}
          onRerun={() => {
            if (selectedBacktestId) {
              handleViewDetail(selectedBacktestId)
            }
          }}
        />
      </div>
    )
  }

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold text-gray-900 dark:text-gray-100 tracking-tight">
          {t('history.title')}
        </h1>
        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
          View and manage your backtest history.
        </p>
      </div>

      <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg p-5">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="label">Strategy</label>
            <input
              type="text"
              value={filters.strategy_name}
              onChange={(e) => setFilters({ ...filters, strategy_name: e.target.value })}
              className="input"
              placeholder="Search..."
            />
          </div>
          <div>
            <label className="label">Symbol</label>
            <input
              type="text"
              value={filters.symbol}
              onChange={(e) => setFilters({ ...filters, symbol: e.target.value.toUpperCase() })}
              className="input"
              placeholder="Search..."
            />
          </div>
        </div>
      </div>

      {loadingDetail && (
        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg p-12 flex items-center justify-center">
          <span className="text-sm text-gray-500 dark:text-gray-400">Loading details...</span>
        </div>
      )}

      <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-800">
            <thead className="bg-gray-50 dark:bg-gray-900">
              <tr>
                <th className="px-5 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Date
                </th>
                <th className="px-5 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Strategy
                </th>
                <th className="px-5 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Symbol
                </th>
                <th className="px-5 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Return
                </th>
                <th className="px-5 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Sharpe
                </th>
                <th className="px-5 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-800">
              {backtests.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-5 py-8 text-center text-sm text-gray-500 dark:text-gray-400">
                    No results found
                  </td>
                </tr>
              ) : (
                backtests.map((backtest) => (
                  <tr
                    key={backtest.id}
                    className="hover:bg-gray-50 dark:hover:bg-gray-800 cursor-pointer transition-colors"
                    onClick={() => handleViewDetail(backtest.id)}
                  >
                    <td className="px-5 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                      {format(new Date(backtest.created_at), 'yyyy-MM-dd HH:mm')}
                    </td>
                    <td className="px-5 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                      {backtest.strategy_name}
                    </td>
                    <td className="px-5 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                      {formatSymbols(backtest.symbol, backtest.universe)}
                    </td>
                    <td className={`px-5 py-4 whitespace-nowrap text-sm font-medium ${
                      (backtest.total_return || 0) >= 0
                        ? 'text-green-700 dark:text-green-400'
                        : 'text-red-700 dark:text-red-400'
                    }`}>
                      {((backtest.total_return || 0) * 100).toFixed(2)}%
                    </td>
                    <td className="px-5 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                      {backtest.sharpe_ratio?.toFixed(2) || 'N/A'}
                    </td>
                    <td className="px-5 py-4 whitespace-nowrap text-sm" onClick={(e) => e.stopPropagation()}>
                      <button
                        onClick={() => handleViewDetail(backtest.id)}
                        className="text-gray-900 dark:text-gray-100 hover:underline mr-4"
                      >
                        View
                      </button>
                      <button
                        onClick={() => handleDelete(backtest.id)}
                        className="text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
