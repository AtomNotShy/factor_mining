import React from 'react'
import { format } from 'date-fns'

interface BacktestData {
  strategy_name: string
  symbol?: string
  universe?: string[]
  backtest_period: {
    start_date: string
    end_date: string
    days: number
  }
  enhanced_metrics: any
  benchmark_data: any
}

interface LeftPanelProps {
  data: BacktestData
  collapsed: boolean
  width: number
  isDark: boolean
  onToggle: () => void
}

const formatValue = (value: number, format: 'percent' | 'currency' | 'number' = 'number'): string => {
  if (!Number.isFinite(value)) return '-'
  
  if (format === 'percent') {
    return `${(value * 100).toFixed(2)}%`
  }
  if (format === 'currency') {
    return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  }
  return value.toFixed(2)
}

const MetricRow = ({ 
  label, 
  value, 
  format = 'number', 
  color,
  subValue 
}: { 
  label: string
  value: number
  format?: 'percent' | 'currency' | 'number'
  color?: string
  subValue?: string
}) => (
  <div style={{ 
    display: 'flex', 
    justifyContent: 'space-between', 
    alignItems: 'center',
    padding: '8px 0',
    borderBottom: '1px solid',
    borderColor: 'inherit',
  }}>
    <span style={{ fontSize: 12, color: 'inherit', opacity: 0.7 }}>{label}</span>
    <div style={{ textAlign: 'right' }}>
      <span style={{ 
        fontSize: 13, 
        fontWeight: 600,
        color: color || 'inherit',
      }}>
        {formatValue(value, format)}
      </span>
      {subValue && (
        <div style={{ fontSize: 10, color: 'inherit', opacity: 0.6 }}>
          {subValue}
        </div>
      )}
    </div>
  </div>
)

const Section = ({ title, children }: { title: string; children: React.ReactNode }) => (
  <div style={{ marginBottom: 16 }}>
    <div style={{ 
      fontSize: 11, 
      fontWeight: 600, 
      color: 'inherit', 
      opacity: 0.6,
      textTransform: 'uppercase',
      letterSpacing: 0.5,
      marginBottom: 8,
    }}>
      {title}
    </div>
    {children}
  </div>
)

export default function LeftPanel({ data, collapsed, width, isDark, onToggle }: LeftPanelProps) {
  const m = data.enhanced_metrics || {}
  const period = data.backtest_period

  const performanceMetrics = [
    { label: 'Total Return', value: m.total_return, format: 'percent' as const, color: m.total_return >= 0 ? '#089981' : '#f23645' },
    { label: 'Annual Return', value: m.annual_return, format: 'percent' as const, color: m.annual_return >= 0 ? '#089981' : '#f23645' },
    { label: 'Sharpe Ratio', value: m.sharpe_ratio, color: m.sharpe_ratio >= 1 ? '#089981' : m.sharpe_ratio >= 0.5 ? '#2962ff' : '#f23645' },
    { label: 'Sortino Ratio', value: m.sortino_ratio, color: m.sortino_ratio >= 1 ? '#089981' : '#f23645' },
    { label: 'Calmar Ratio', value: m.calmar_ratio, color: m.calmar_ratio >= 3 ? '#089981' : '#f23645' },
    { label: 'Max Drawdown', value: m.max_drawdown, format: 'percent' as const, color: '#f23645' },
  ]

  const benchmarkMetrics = [
    { label: 'Excess Return', value: m.excess_return, format: 'percent' as const, color: m.excess_return >= 0 ? '#089981' : '#f23645' },
    { label: 'Alpha', value: m.alpha, color: m.alpha >= 0 ? '#089981' : '#f23645' },
    { label: 'Beta', value: m.beta, color: m.beta >= 0.8 && m.beta <= 1.2 ? '#089981' : '#2962ff' },
    { label: 'Information Ratio', value: m.information_ratio, color: m.information_ratio >= 0.5 ? '#089981' : '#787b86' },
    { label: 'R-Squared', value: m.r_squared, color: m.r_squared >= 0.8 ? '#089981' : '#787b86' },
  ]

  const tradeMetrics = [
    { label: 'Total Trades', value: m.total_trades },
    { label: 'Win Rate', value: m.win_rate, format: 'percent' as const, color: m.win_rate >= 0.5 ? '#089981' : '#f23645' },
    { label: 'Profit Factor', value: m.profit_factor, color: m.profit_factor >= 2 ? '#089981' : m.profit_factor >= 1 ? '#2962ff' : '#f23645' },
    { label: 'Expectancy', value: m.expectancy, color: m.expectancy >= 0 ? '#089981' : '#f23645' },
    { label: 'Largest Win', value: m.largest_win, format: 'currency' as const, color: '#089981' },
    { label: 'Largest Loss', value: m.largest_loss, format: 'currency' as const, color: '#f23645' },
    { label: 'Consecutive Wins', value: m.consecutive_wins },
    { label: 'Consecutive Losses', value: m.consecutive_losses },
  ]

  if (collapsed) {
    return (
      <div
        style={{
          width: 48,
          height: '100%',
          backgroundColor: isDark ? '#1e222d' : '#f5f5f5',
          borderRight: '1px solid',
          borderColor: 'inherit',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: '12px 0',
          gap: 8,
        }}
      >
        <button
          onClick={onToggle}
          style={{
            width: 36,
            height: 36,
            borderRadius: 4,
            border: 'none',
            backgroundColor: isDark ? '#2a2e39' : '#e5e5e5',
            color: 'inherit',
            cursor: 'pointer',
            fontSize: 16,
          }}
          title="Expand"
        >
          →
        </button>
      </div>
    )
  }

  return (
    <div
      style={{
        width,
        height: '100%',
        backgroundColor: isDark ? '#1e222d' : '#f5f5f5',
        borderRight: '1px solid',
        borderColor: 'inherit',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: 12,
          borderBottom: '1px solid',
          borderColor: 'inherit',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <span style={{ fontSize: 11, fontWeight: 600, color: 'inherit', opacity: 0.6 }}>
          PROPERTIES
        </span>
        <button
          onClick={onToggle}
          style={{
            width: 24,
            height: 24,
            borderRadius: 4,
            border: 'none',
            backgroundColor: 'transparent',
            color: 'inherit',
            cursor: 'pointer',
            fontSize: 12,
            opacity: 0.6,
          }}
          title="Collapse"
        >
          ←
        </button>
      </div>

      {/* Period Info */}
      <div
        style={{
          padding: 12,
          borderBottom: '1px solid',
          borderColor: 'inherit',
        }}
      >
        <div style={{ fontSize: 12, color: 'inherit', marginBottom: 4 }}>
          {format(new Date(period.start_date), 'MMM d, yyyy')} - {format(new Date(period.end_date), 'MMM d, yyyy')}
        </div>
        <div style={{ fontSize: 11, color: 'inherit', opacity: 0.6 }}>
          {period.days} days
        </div>
      </div>

      {/* Metrics */}
      <div style={{ flex: 1, overflow: 'auto', padding: 12 }}>
        <Section title="Performance">
          {performanceMetrics.map((metric, i) => (
            <MetricRow key={i} {...metric} />
          ))}
        </Section>

        <Section title="Benchmark">
          {benchmarkMetrics.map((metric, i) => (
            <MetricRow key={i} {...metric} />
          ))}
        </Section>

        <Section title="Trades">
          {tradeMetrics.map((metric, i) => (
            <MetricRow key={i} {...metric} />
          ))}
        </Section>
      </div>
    </div>
  )
}
