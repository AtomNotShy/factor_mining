import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import { useState, useEffect } from 'react'

const resources = {
  en: {
    translation: {
      // Navigation
      nav: {
        dashboard: 'Dashboard',
        backtest: 'Backtest',
        history: 'History',
        monitoring: 'Monitoring',
        settings: 'Settings',
      },
      // Dashboard
      dashboard: {
        title: 'Dashboard',
        totalBacktests: 'Total Backtests',
        activeStrategies: 'Active Strategies',
        totalReturn: 'Total Return',
        sharpeRatio: 'Sharpe Ratio',
      },
      // Backtest
      backtest: {
        title: 'Run Backtest',
        strategy: 'Strategy',
        symbol: 'Symbol',
        timeframe: 'Timeframe',
        dateRange: 'Date Range',
        startDate: 'Start Date',
        endDate: 'End Date',
        days: 'Days',
        datePreset: 'Quick Select',
        preset_1m: '1M',
        preset_3m: '3M',
        preset_6m: '6M',
        preset_1y: '1Y',
        preset_2y: '2Y',
        preset_3y: '3Y',
        preset_5y: '5Y',
        preset_custom: 'Custom',
        dateRequired: 'Please select start and end dates',
        dateInvalid: 'Start date must be before end date',
        dateRangeTooLong: 'Date range cannot exceed 5 years',
        initialCapital: 'Initial Capital',
        commissionRate: 'Commission Rate',
        slippageRate: 'Slippage Rate',
        run: 'Run Backtest',
        running: 'Running...',
        results: 'Results',
        finalValue: 'Final Value',
        totalReturn: 'Total Return',
        sharpeRatio: 'Sharpe Ratio',
        maxDrawdown: 'Max Drawdown',
        winRate: 'Win Rate',
        totalTrades: 'Total Trades',
        priceChart: 'Price Chart with Trading Signals',
        equity: 'Equity Curve',
        drawdown: 'Drawdown',
        buySignals: 'Buy',
        sellSignals: 'Sell',
        close: 'Close',
        selectSymbol: 'Select a symbol',
        noSymbols: 'No local symbols available',
        noLocalData: 'No local data found. Please download data first.',
        symbolRequired: 'Please select a symbol',
      },
      // History
      history: {
        title: 'Backtest History',
        search: 'Search',
        filter: 'Filter',
        strategy: 'Strategy',
        symbol: 'Symbol',
        date: 'Date',
        return: 'Return',
        actions: 'Actions',
        view: 'View',
        delete: 'Delete',
        noResults: 'No results found',
        backToList: 'Back to List',
        detailTitle: 'Backtest Details',
        confirmDelete: 'Are you sure you want to delete this backtest?',
        loadError: 'Failed to load backtest details',
        deleteError: 'Failed to delete backtest',
      },
      // Common
      common: {
        loading: 'Loading...',
        select: 'Please select',
        error: 'Error',
        success: 'Success',
        cancel: 'Cancel',
        confirm: 'Confirm',
        save: 'Save',
        delete: 'Delete',
        edit: 'Edit',
        close: 'Close',
      },
    },
  },
  zh: {
    translation: {
      // Navigation
      nav: {
        dashboard: '仪表盘',
        backtest: '回测',
        history: '历史',
        monitoring: '监控',
        settings: '设置',
      },
      // Dashboard
      dashboard: {
        title: '仪表盘',
        totalBacktests: '总回测数',
        activeStrategies: '活跃策略',
        totalReturn: '总收益',
        sharpeRatio: '夏普比率',
      },
      // Backtest
      backtest: {
        title: '运行回测',
        strategy: '策略',
        symbol: '标的',
        timeframe: '时间周期',
        dateRange: '日期范围',
        startDate: '开始日期',
        endDate: '结束日期',
        days: '天数',
        datePreset: '快捷选项',
        preset_1m: '1个月',
        preset_3m: '3个月',
        preset_6m: '6个月',
        preset_1y: '1年',
        preset_2y: '2年',
        preset_3y: '3年',
        preset_5y: '5年',
        preset_custom: '自定义',
        dateRequired: '请选择开始和结束日期',
        dateInvalid: '开始日期必须早于结束日期',
        dateRangeTooLong: '日期范围不能超过5年',
        initialCapital: '初始资金',
        commissionRate: '手续费率',
        slippageRate: '滑点率',
        run: '运行回测',
        running: '运行中...',
        results: '结果',
        finalValue: '最终价值',
        totalReturn: '总收益',
        sharpeRatio: '夏普比率',
        maxDrawdown: '最大回撤',
        winRate: '胜率',
        totalTrades: '总交易数',
        priceChart: '价格走势与交易信号',
        equity: '净值曲线',
        drawdown: '回撤曲线',
        buySignals: '买入',
        sellSignals: '卖出',
        close: '收盘价',
        selectSymbol: '请选择标的',
        noSymbols: '没有可用的本地标的',
        noLocalData: '未找到本地数据，请先下载数据。',
        symbolRequired: '请选择标的',
        // Performance metrics
        annualReturn: '年化收益',
        annualVolatility: '年化波动率',
        benchmarkReturn: '基准收益',
        benchmarkVolatility: '基准波动率',
        sortinoRatio: '索提诺比率',
        calmarRatio: '卡玛比率',
        excessReturn: '超额收益',
        alpha: 'Alpha',
        beta: 'Beta',
        informationRatio: '信息比率',
        rSquared: 'R平方',
        dailyWinRate: '日胜率',
        profitLossRatio: '盈亏比',
        winningTrades: '盈利次数',
        losingTrades: '亏损次数',
        profitFactor: '盈利因子',
        expectancy: '期望收益',
        largestWin: '最大盈利',
        largestLoss: '最大亏损',
        consecutiveWins: '最大连胜',
        consecutiveLosses: '最大连亏',
      },
      // History
      history: {
        title: '回测历史',
        search: '搜索',
        filter: '筛选',
        strategy: '策略',
        symbol: '标的',
        date: '日期',
        return: '收益',
        actions: '操作',
        view: '查看',
        delete: '删除',
        noResults: '未找到结果',
        backToList: '返回列表',
        detailTitle: '回测详情',
        confirmDelete: '确定要删除这个回测吗？',
        loadError: '加载回测详情失败',
        deleteError: '删除回测失败',
      },
      // Common
      common: {
        loading: '加载中...',
        select: '请选择',
        error: '错误',
        success: '成功',
        cancel: '取消',
        confirm: '确认',
        save: '保存',
        delete: '删除',
        edit: '编辑',
        close: '关闭',
      },
    },
  },
}

i18n.use(initReactI18next).init({
  resources,
  lng: localStorage.getItem('language') || 'zh',
  fallbackLng: 'en',
  interpolation: {
    escapeValue: false,
  },
})

export const useI18n = () => {
  const [language, setLanguage] = useState(i18n.language)

  useEffect(() => {
    const handleLanguageChange = (lng: string) => {
      setLanguage(lng)
    }
    i18n.on('languageChanged', handleLanguageChange)
    return () => {
      i18n.off('languageChanged', handleLanguageChange)
    }
  }, [])

  const changeLanguage = (lng: string) => {
    i18n.changeLanguage(lng)
    localStorage.setItem('language', lng)
  }

  return { i18n, language, changeLanguage }
}

export default i18n
