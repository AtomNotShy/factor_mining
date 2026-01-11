import { useTranslation } from 'react-i18next'
import { useState, useEffect, useCallback } from 'react'
import { api } from '../services/api'

interface BacktestFormProps {
  strategies: any[]
  onSubmit: (data: any) => void
  loading: boolean
}

// 默认美股ETF池
const DEFAULT_ETF_POOL = [
  "SPY", "QQQ", "IWM", "VTI", "VOO",
  "VEA", "VWO", "EFA", "EEM",
  "TLT", "IEF", "AGG", "LQD", "HYG",
  "XLF", "XLE", "XLI", "XLK", "XLV", "XLP"
]

export default function BacktestForm({ strategies, onSubmit, loading }: BacktestFormProps) {
  const { t } = useTranslation()
  const [formData, setFormData] = useState({
    strategy_name: '',
    symbol: '',
    timeframe: '1d',
    dateMode: 'days',
    datePreset: '1y',  // 日期快捷选项: 1m, 3m, 6m, 1y, 2y, 3y, 5y, custom
    days: 365,
    start_date: '',
    end_date: '',
    initial_capital: 100000,
    commission_rate: 0.001,
    slippage_rate: 0.0005,
    etf_pool: DEFAULT_ETF_POOL.join(','),  // ETF池（逗号分隔的字符串）
  })
  const [dateError, setDateError] = useState<string | null>(null)
  const [availableSymbols, setAvailableSymbols] = useState<string[]>([])
  const [loadingSymbols, setLoadingSymbols] = useState(true)
  const [selectedStrategy, setSelectedStrategy] = useState<any>(null)

  // 获取默认日期范围
  const getDefaultDateRange = useCallback((preset: string) => {
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
      end_date: endDate.toISOString().split('T')[0]
    }
  }, [])

  // 验证日期范围
  const validateDateRange = useCallback((start: string, end: string): string | null => {
    if (!start || !end) {
      return t('backtest.dateRequired') || '请选择开始和结束日期'
    }
    
    const startDate = new Date(start)
    const endDate = new Date(end)
    
    if (startDate >= endDate) {
      return t('backtest.dateInvalid') || '开始日期必须早于结束日期'
    }
    
    // 验证日期范围不能超过5年
    const maxDate = new Date()
    maxDate.setFullYear(maxDate.getFullYear() - 5)
    if (startDate < maxDate) {
      return t('backtest.dateRangeTooLong') || '日期范围不能超过5年'
    }
    
    return null
  }, [t])

  // 当日期模式改变时，自动填充默认日期
  useEffect(() => {
    if (formData.dateMode === 'range' && (!formData.start_date || !formData.end_date)) {
      const defaultRange = getDefaultDateRange(formData.datePreset)
      setFormData(prev => ({
        ...prev,
        start_date: defaultRange.start_date,
        end_date: defaultRange.end_date
      }))
    }
  }, [formData.dateMode, formData.datePreset, getDefaultDateRange])

  // 检查选中的策略是否为ETF策略
  useEffect(() => {
    const strategy = strategies.find(s => s.name === formData.strategy_name)
    setSelectedStrategy(strategy)
    
    // 如果选择了ETF策略，自动切换到日线
    if (strategy?.name === 'etf_momentum_rotation' || strategy?.name === 'etf_momentum_rotation_fixed') {
      setFormData(prev => ({ ...prev, timeframe: '1d', days: 365 }))
    }
  }, [formData.strategy_name, strategies])

  // 加载本地存在的标的列表
  useEffect(() => {
    const loadLocalSymbols = async () => {
      try {
        setLoadingSymbols(true)
        const response = await api.get('/data/local-symbols', {
          params: { timeframe: formData.timeframe }
        })
        const symbols = response.data.symbols || []
        setAvailableSymbols(symbols)
        
        // 如果当前选择的标的不在列表中，且列表不为空，则选择第一个
        if (symbols.length > 0) {
          setFormData(prev => {
            if (!prev.symbol || !symbols.includes(prev.symbol)) {
              return { ...prev, symbol: symbols[0] }
            }
            return prev
          })
        } else {
          setFormData(prev => ({ ...prev, symbol: '' }))
        }
      } catch (error) {
        console.error('Failed to load local symbols:', error)
        setAvailableSymbols([])
        setFormData(prev => ({ ...prev, symbol: '' }))
      } finally {
        setLoadingSymbols(false)
      }
    }
    
    loadLocalSymbols()
  }, [formData.timeframe])

  // 当时间周期改变时，重新加载标的列表
  const handleTimeframeChange = (timeframe: string) => {
    setFormData(prev => ({ ...prev, timeframe, symbol: '' }))
  }

  // 解析ETF池
  const parseEtfPool = useCallback((poolStr: string): string[] => {
    if (!poolStr.trim()) return []
    return poolStr
      .split(',')
      .map(s => s.trim().toUpperCase())
      .filter(s => s.length > 0)
  }, [])

  // 处理日期模式切换
  const handleDateModeChange = (mode: string) => {
    if (mode === 'range') {
      // 切换到日期范围模式时，填充默认值
      const defaultRange = getDefaultDateRange(formData.datePreset)
      setFormData(prev => ({
        ...prev,
        dateMode: mode,
        start_date: defaultRange.start_date,
        end_date: defaultRange.end_date
      }))
    } else {
      setFormData(prev => ({ ...prev, dateMode: mode }))
    }
    setDateError(null)
  }

  // 处理快捷日期选项变化
  const handleDatePresetChange = (preset: string) => {
    const defaultRange = getDefaultDateRange(preset)
    setFormData(prev => ({
      ...prev,
      dateMode: 'range',
      datePreset: preset,
      start_date: defaultRange.start_date,
      end_date: defaultRange.end_date
    }))
    setDateError(null)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    
    // ETF策略需要验证ETF池
    if (formData.strategy_name === 'etf_momentum_rotation' || formData.strategy_name === 'etf_momentum_rotation_fixed') {
      const etfPool = parseEtfPool(formData.etf_pool)
      if (etfPool.length < 2) {
        alert(t('backtest.etfPoolRequired') || 'ETF策略需要至少2个ETF')
        return
      }
    }
    
    // 验证日期范围
    if (formData.dateMode === 'range') {
      const error = validateDateRange(formData.start_date, formData.end_date)
      if (error) {
        setDateError(error)
        return
      }
    }
    
    const submitData: any = {
      strategy_name: formData.strategy_name,
      symbol: formData.symbol,
      timeframe: formData.timeframe,
      initial_capital: formData.initial_capital,
      commission_rate: formData.commission_rate,
      slippage_rate: formData.slippage_rate,
    }

    // ETF策略：传递ETF池
    if (formData.strategy_name === 'etf_momentum_rotation' || formData.strategy_name === 'etf_momentum_rotation_fixed') {
      submitData.etf_pool = parseEtfPool(formData.etf_pool)
      // ETF策略不需要单标的
      delete submitData.symbol
    }

    if (formData.dateMode === 'days') {
      submitData.days = formData.days
    } else {
      submitData.start_date = formData.start_date
      submitData.end_date = formData.end_date
    }

    setDateError(null)
    onSubmit(submitData)
  }

  return (
    <form onSubmit={handleSubmit} className="bg-white dark:bg-gray-800 rounded-lg shadow p-6 border border-gray-200 dark:border-gray-700">
      <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">
        {t('backtest.title')}
      </h2>

      <div className="space-y-4">
        {/* Strategy */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('backtest.strategy')}
          </label>
          <select
            required
            value={formData.strategy_name}
            onChange={(e) => setFormData({ ...formData, strategy_name: e.target.value })}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          >
            <option value="">{t('common.select')}</option>
            {strategies.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name} - {s.description}
              </option>
            ))}
          </select>
        </div>

        {/* Symbol */}
        <div className={(selectedStrategy?.name === 'etf_momentum_rotation' || selectedStrategy?.name === 'etf_momentum_rotation_fixed') ? 'opacity-50' : ''}>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('backtest.symbol')}
          </label>
          <select
            required={selectedStrategy?.name !== 'etf_momentum_rotation' && selectedStrategy?.name !== 'etf_momentum_rotation_fixed'}
            value={formData.symbol}
            onChange={(e) => setFormData({ ...formData, symbol: e.target.value })}
            disabled={loadingSymbols || availableSymbols.length === 0 || selectedStrategy?.name === 'etf_momentum_rotation' || selectedStrategy?.name === 'etf_momentum_rotation_fixed'}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <option value="">
              {loadingSymbols 
                ? (t('common.loading') || 'Loading...')
                : availableSymbols.length === 0
                ? (t('backtest.noSymbols') || 'No local symbols available')
                : (t('backtest.selectSymbol') || 'Select a symbol')}
            </option>
            {availableSymbols.map((symbol) => (
              <option key={symbol} value={symbol}>
                {symbol}
              </option>
            ))}
          </select>
          {availableSymbols.length === 0 && !loadingSymbols && (
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              {t('backtest.noLocalData') || 'No local data found. Please download data first.'}
            </p>
          )}
        </div>

        {/* ETF Pool Configuration */}
        {(selectedStrategy?.name === 'etf_momentum_rotation' || selectedStrategy?.name === 'etf_momentum_rotation_fixed') && (
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              ETF Pool (comma-separated)
            </label>
            <textarea
              value={formData.etf_pool}
              onChange={(e) => setFormData({ ...formData, etf_pool: e.target.value })}
              rows={4}
              placeholder="SPY, QQQ, IWM, VTI, VOO..."
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white font-mono text-sm"
            />
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              Enter ETF symbols separated by commas (e.g., SPY, QQQ, IWM, VTI, VOO)
            </p>
          </div>
        )}

        {/* Timeframe */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('backtest.timeframe')}
          </label>
          <select
            value={formData.timeframe}
            onChange={(e) => handleTimeframeChange(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          >
            <option value="1m">1 Minute</option>
            <option value="5m">5 Minutes</option>
            <option value="15m">15 Minutes</option>
            <option value="1h">1 Hour</option>
            <option value="1d">1 Day</option>
          </select>
        </div>

        {/* Quick Date Presets */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('backtest.datePreset') || '快捷选项'}
          </label>
          <div className="flex flex-wrap gap-2">
            {[
              { value: '1w', label: t('backtest.preset_1w') || '1周' },
              { value: '1m', label: t('backtest.preset_1m') || '1个月' },
              { value: '3m', label: t('backtest.preset_3m') || '3个月' },
              { value: '6m', label: t('backtest.preset_6m') || '6个月' },
              { value: '1y', label: t('backtest.preset_1y') || '1年' },
              { value: '2y', label: t('backtest.preset_2y') || '2年' },
              { value: '3y', label: t('backtest.preset_3y') || '3年' },
              { value: '5y', label: t('backtest.preset_5y') || '5年' },
              { value: 'ytd', label: t('backtest.preset_ytd') || '今年以来' },
              { value: 'custom', label: t('backtest.preset_custom') || '自定义' },
            ].map((preset) => (
              <button
                key={preset.value}
                type="button"
                onClick={() => preset.value !== 'custom' && handleDatePresetChange(preset.value)}
                disabled={preset.value === 'custom'}
                className={`px-3 py-1 text-sm rounded-md border transition-colors ${
                  formData.datePreset === preset.value
                    ? 'bg-primary-600 text-white border-primary-600'
                    : 'bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300 border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-600'
                }`}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>

        {/* Date Range Mode */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('backtest.dateRange')}
          </label>
          <select
            value={formData.dateMode}
            onChange={(e) => handleDateModeChange(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          >
            <option value="days">{t('backtest.days')}</option>
            <option value="range">{t('backtest.startDate')} - {t('backtest.endDate')}</option>
          </select>
        </div>

        {/* Days or Date Range */}
        {formData.dateMode === 'days' ? (
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              {t('backtest.days')}
            </label>
            <input
              type="number"
              required
              min="1"
              max="365"
              value={formData.days}
              onChange={(e) => setFormData({ ...formData, days: parseInt(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
          </div>
        ) : (
          <>
            {/* 开始日期和结束日期 */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  {t('backtest.startDate')}
                </label>
                <input
                  type="date"
                  required
                  value={formData.start_date}
                  onChange={(e) => {
                    setFormData({ ...formData, start_date: e.target.value, datePreset: 'custom' })
                    setDateError(null)
                  }}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  {t('backtest.endDate')}
                </label>
                <input
                  type="date"
                  required
                  value={formData.end_date}
                  onChange={(e) => {
                    setFormData({ ...formData, end_date: e.target.value, datePreset: 'custom' })
                    setDateError(null)
                  }}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                />
              </div>
            </div>

            {/* 日期验证错误提示 */}
            {dateError && (
              <div className="text-sm text-red-600 dark:text-red-400 mt-1">
                {dateError}
              </div>
            )}
          </>
        )}

        {/* Initial Capital */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('backtest.initialCapital')} ($)
          </label>
          <input
            type="number"
            required
            min="1000"
            step="1000"
            value={formData.initial_capital}
            onChange={(e) => setFormData({ ...formData, initial_capital: parseFloat(e.target.value) })}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          />
        </div>

        {/* Commission Rate */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('backtest.commissionRate')}
          </label>
          <input
            type="number"
            required
            min="0"
            max="0.01"
            step="0.0001"
            value={formData.commission_rate}
            onChange={(e) => setFormData({ ...formData, commission_rate: parseFloat(e.target.value) })}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          />
        </div>

        {/* Slippage Rate */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            {t('backtest.slippageRate')}
          </label>
          <input
            type="number"
            required
            min="0"
            max="0.01"
            step="0.0001"
            value={formData.slippage_rate}
            onChange={(e) => setFormData({ ...formData, slippage_rate: parseFloat(e.target.value) })}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          />
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading}
          className="w-full bg-primary-600 hover:bg-primary-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-md transition-colors"
        >
          {loading ? t('backtest.running') : t('backtest.run')}
        </button>
      </div>
    </form>
  )
}
