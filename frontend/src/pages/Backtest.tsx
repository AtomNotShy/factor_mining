import { useTranslation } from 'react-i18next'
import { useState, useEffect } from 'react'
import { api } from '../services/api'
import { useThemeStore } from '../stores/themeStore'
import { BacktestDetails } from '../components/tradingview'

interface Strategy {
  name: string
  description?: string
  params?: Record<string, any>
}

export default function Backtest() {
  const { t } = useTranslation()
  const { theme } = useThemeStore()
  const isDark = theme === 'dark'

  const [strategies, setStrategies] = useState<Strategy[]>([])
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const [formData, setFormData] = useState({
    strategy_name: 'vwap_pullback',
    symbol: 'SPY',
    timeframe: '1d',
    days: 365,
    start_date: '',
    end_date: '',
    initial_capital: 100000,
    commission_rate: 0.0005,
    slippage_rate: 0.0002,
    benchmark_symbol: '',
  })

  const [dateMode, setDateMode] = useState<'days' | 'range'>('days')

  useEffect(() => {
    loadStrategies()
  }, [])

  const loadStrategies = async () => {
    try {
      const response = await api.get('/strategy-backtest/strategies')
      const loaded = response.data.strategies || []
      setStrategies(loaded)
      const current = loaded.find((strategy: Strategy) => strategy.name === formData.strategy_name)
      if (current?.params?.benchmark_symbol && !formData.benchmark_symbol) {
        setFormData((prev) => ({
          ...prev,
          benchmark_symbol: String(current.params.benchmark_symbol).toUpperCase(),
        }))
      }
    } catch (err) {
      console.error('Failed to load strategies:', err)
    }
  }

  const handleRunBacktest = async () => {
    setLoading(true)
    setError(null)
    setResults(null)

    try {
      const response = await api.post('/strategy-backtest/run', formData)
      setResults(response.data)
    } catch (err: any) {
      const data = err.response?.data
      let message: string
      
      if (typeof data === 'string') {
        message = data
      } else if (Array.isArray(data?.detail)) {
        // 422 validation errors - format as readable string
        message = data.detail
          .map((e: any) => {
            const loc = e.loc?.join('.') || e.field || 'field'
            const msg = e.msg || e.message || 'invalid'
            return `${loc}: ${msg}`
          })
          .join('; ')
      } else if (data?.detail && typeof data.detail === 'object') {
        // Single error object
        const e = data.detail
        const loc = e.loc?.join('.') || e.field || 'field'
        const msg = e.msg || e.message || 'invalid'
        message = `${loc}: ${msg}`
      } else if (data?.error) {
        message = data.error
      } else if (data?.message) {
        message = data.message
      } else if (err.message) {
        message = err.message
      } else {
        message = 'Backtest failed'
      }
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  const handleFormUpdate = (key: string, value: any) => {
    setFormData((prev) => {
      const newData = { ...prev, [key]: value }
      if (key === 'strategy_name') {
        const selected = strategies.find((strategy) => strategy.name === value)
        newData.benchmark_symbol = selected?.params?.benchmark_symbol || ''
      }
      if (key === 'days' && value > 0) {
        newData.start_date = ''
        newData.end_date = ''
      }
      if (key === 'start_date' || key === 'end_date') {
        newData.days = 365
      }
      return newData
    })
  }

  const getPresetDateRange = (preset: string) => {
    const endDate = new Date()
    const startDate = new Date()

    switch (preset) {
      case '1w':
        startDate.setDate(startDate.getDate() - 7)
        break
      case '1m':
        startDate.setMonth(startDate.getMonth() - 1)
        break
      case '3m':
        startDate.setMonth(startDate.getMonth() - 3)
        break
      case '6m':
        startDate.setMonth(startDate.getMonth() - 6)
        break
      case '1y':
        startDate.setFullYear(startDate.getFullYear() - 1)
        break
      case '2y':
        startDate.setFullYear(startDate.getFullYear() - 2)
        break
      case '3y':
        startDate.setFullYear(startDate.getFullYear() - 3)
        break
      case '5y':
        startDate.setFullYear(startDate.getFullYear() - 5)
        break
      case 'ytd':
        startDate.setMonth(0)
        startDate.setDate(1)
        break
      default:
        startDate.setFullYear(startDate.getFullYear() - 1)
    }

    return {
      start_date: startDate.toISOString().split('T')[0],
      end_date: endDate.toISOString().split('T')[0],
    }
  }

  const applyDatePreset = (preset: string) => {
    const range = getPresetDateRange(preset)
    setDateMode('range')
    setFormData((prev) => ({
      ...prev,
      start_date: range.start_date,
      end_date: range.end_date,
    }))
  }

  return (
    <div className="min-h-[calc(100vh-8rem)]">
      {loading && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-900 px-8 py-6 rounded-lg flex items-center gap-4">
            <div className="w-5 h-5 border-2 border-gray-300 dark:border-gray-700 border-t-gray-900 dark:border-t-gray-100 rounded-full animate-spin" />
            <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
              Running Backtest...
            </span>
          </div>
        </div>
      )}

      {error && (
        <div className="fixed top-5 right-5 max-w-md bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 px-4 py-3 rounded-md z-50">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-semibold text-red-800 dark:text-red-200">Error</span>
            <button
              onClick={() => setError(null)}
              className="text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-200"
            >
              Close
            </button>
          </div>
          <span className="text-sm text-red-700 dark:text-red-300">{error}</span>
        </div>
      )}

      {!results && (
        <div className="max-w-2xl mx-auto">
          <div className="mb-8">
            <h1 className="text-2xl font-semibold text-gray-900 dark:text-gray-100 tracking-tight">
              Strategy Backtest
            </h1>
            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
              Configure and run a backtest for your strategy.
            </p>
          </div>

          <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg p-6">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
              <div>
                <label className="label">Strategy</label>
                <select
                  value={formData.strategy_name}
                  onChange={(e) => handleFormUpdate('strategy_name', e.target.value)}
                  className="input"
                >
                  {strategies.map((s) => (
                    <option key={s.name} value={s.name}>
                      {s.name}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="label">Symbols</label>
                <input
                  type="text"
                  value={formData.symbol}
                  onChange={(e) => handleFormUpdate('symbol', e.target.value.toUpperCase())}
                  placeholder="SPY, QQQ, IWM"
                  className="input"
                />
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  Separate multiple symbols with commas
                </p>
              </div>

              <div>
                <label className="label">Benchmark</label>
                <input
                  type="text"
                  value={formData.benchmark_symbol}
                  onChange={(e) => handleFormUpdate('benchmark_symbol', e.target.value.toUpperCase())}
                  placeholder="SPY"
                  className="input"
                />
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  Optional. Default uses strategy benchmark.
                </p>
              </div>

              <div>
                <label className="label">Timeframe</label>
                <select
                  value={formData.timeframe}
                  onChange={(e) => handleFormUpdate('timeframe', e.target.value)}
                  className="input"
                >
                  <option value="1m">1 Minute</option>
                  <option value="5m">5 Minutes</option>
                  <option value="15m">15 Minutes</option>
                  <option value="1h">1 Hour</option>
                  <option value="1d">1 Day</option>
                </select>
              </div>

              <div>
                <label className="label">Date Range</label>
                <div className="flex flex-wrap gap-2 mb-2">
                  {[
                    { value: '1w', label: '1W' },
                    { value: '1m', label: '1M' },
                    { value: '3m', label: '3M' },
                    { value: '6m', label: '6M' },
                    { value: '1y', label: '1Y' },
                    { value: '2y', label: '2Y' },
                    { value: '3y', label: '3Y' },
                    { value: '5y', label: '5Y' },
                    { value: 'ytd', label: 'YTD' },
                  ].map((preset) => (
                    <button
                      key={preset.value}
                      type="button"
                      onClick={() => applyDatePreset(preset.value)}
                      className="px-2.5 py-1 text-xs rounded bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>
                <div className="flex gap-2 mb-2">
                  <button
                    type="button"
                    onClick={() => setDateMode('days')}
                    className={`px-3 py-1 text-xs rounded ${
                      dateMode === 'days'
                        ? 'bg-gray-900 text-white dark:bg-gray-100 dark:text-gray-900'
                        : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'
                    }`}
                  >
                    Last N Days
                  </button>
                  <button
                    type="button"
                    onClick={() => setDateMode('range')}
                    className={`px-3 py-1 text-xs rounded ${
                      dateMode === 'range'
                        ? 'bg-gray-900 text-white dark:bg-gray-100 dark:text-gray-900'
                        : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'
                    }`}
                  >
                    Date Range
                  </button>
                </div>
                {dateMode === 'days' ? (
                  <input
                    type="number"
                    value={formData.days}
                    onChange={(e) => handleFormUpdate('days', parseInt(e.target.value) || 0)}
                    className="input"
                  />
                ) : (
                  <div className="grid grid-cols-2 gap-2">
                    <input
                      type="date"
                      value={formData.start_date}
                      onChange={(e) => handleFormUpdate('start_date', e.target.value)}
                      className="input"
                    />
                    <input
                      type="date"
                      value={formData.end_date}
                      onChange={(e) => handleFormUpdate('end_date', e.target.value)}
                      className="input"
                    />
                  </div>
                )}
              </div>

              <div>
                <label className="label">Initial Capital</label>
                <input
                  type="number"
                  value={formData.initial_capital}
                  onChange={(e) => handleFormUpdate('initial_capital', parseFloat(e.target.value))}
                  className="input"
                />
              </div>

              <div>
                <label className="label">Commission Rate</label>
                <input
                  type="number"
                  step="0.0001"
                  value={formData.commission_rate}
                  onChange={(e) => handleFormUpdate('commission_rate', parseFloat(e.target.value))}
                  className="input"
                />
              </div>
            </div>

            <div className="mt-6">
              <button
                onClick={handleRunBacktest}
                disabled={loading}
                className="btn btn-primary w-full"
              >
                {loading ? 'Running...' : 'Run Backtest'}
              </button>
            </div>
          </div>
        </div>
      )}

      {results && (
        <BacktestDetails
          data={results}
          onBack={() => setResults(null)}
          onRerun={handleRunBacktest}
        />
      )}
    </div>
  )
}
