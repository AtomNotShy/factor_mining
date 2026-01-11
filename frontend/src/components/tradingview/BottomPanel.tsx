import React, { useState } from 'react'
import { format } from 'date-fns'
import BacktestSubChart from './SubChart'

interface BacktestData {
  trades: any[]
  enhanced_metrics: any
  results: any
  equity_comparison: {
    strategy_equity: number[]
    timestamps: string[]
  }
}

type BottomTab = 'trades' | 'summary' | 'chart'

interface BottomPanelProps {
  data: BacktestData
  activeTab: BottomTab
  height: number
  isDark: boolean
  onTabChange: (tab: BottomTab) => void
  onClose: () => void
}

const TAB_ITEMS = [
  { id: 'trades' as BottomTab, label: 'Trades', icon: '📋' },
  { id: 'summary' as BottomTab, label: 'Summary', icon: '📊' },
  { id: 'chart' as BottomTab, label: 'Chart', icon: '📈' },
]

const SUB_CHART_TYPES = [
  { value: 'drawdown', label: 'Drawdown' },
  { value: 'returns', label: 'Returns' },
  { value: 'distribution', label: 'Distribution' },
  { value: 'monthly', label: 'Monthly' },
  { value: 'rolling', label: 'Rolling' },
]

export default function BottomPanel({ data, activeTab, height, isDark, onTabChange, onClose }: BottomPanelProps) {
  const [subChartType, setSubChartType] = useState('drawdown')
  const [tradePage, setTradePage] = useState(1)
  const pageSize = 20

  const totalPages = Math.ceil(data.trades.length / pageSize)

  const paginatedTrades = data.trades.slice(
    (tradePage - 1) * pageSize,
    tradePage * pageSize
  )

  return (
    <div
      style={{
        height,
        backgroundColor: isDark ? '#1e222d' : '#ffffff',
        borderTop: '1px solid',
        borderColor: 'inherit',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Tabs */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: '0 12px',
          borderBottom: '1px solid',
          borderColor: 'inherit',
          height: 36,
        }}
      >
        <div style={{ display: 'flex', gap: 4 }}>
          {TAB_ITEMS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              style={{
                padding: '4px 12px',
                fontSize: 12,
                fontWeight: 500,
                backgroundColor: activeTab === tab.id ? 'rgba(41, 98, 255, 0.1)' : 'transparent',
                color: activeTab === tab.id ? '#2962ff' : 'inherit',
                border: 'none',
                borderRadius: 4,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 4,
              }}
            >
              <span>{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </div>

        <div style={{ flex: 1 }} />

        <button
          onClick={onClose}
          style={{
            padding: '4px 8px',
            fontSize: 14,
            backgroundColor: 'transparent',
            border: 'none',
            color: 'inherit',
            cursor: 'pointer',
            borderRadius: 4,
          }}
          title="Close"
        >
          ×
        </button>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: 'hidden' }}>
        {activeTab === 'trades' && (
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Table Header */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: '140px 80px 100px 100px 120px 100px',
                padding: '8px 12px',
                backgroundColor: isDark ? '#2a2e39' : '#f5f5f5',
                fontSize: 11,
                fontWeight: 600,
                color: 'inherit',
                opacity: 0.7,
                textTransform: 'uppercase',
              }}
            >
              <span>Time</span>
              <span>Symbol</span>
              <span>Type</span>
              <span style={{ textAlign: 'right' }}>Price</span>
              <span style={{ textAlign: 'right' }}>Size</span>
              <span style={{ textAlign: 'right' }}>Amount</span>
            </div>

            {/* Table Body */}
            <div style={{ flex: 1, overflow: 'auto' }}>
              {paginatedTrades.map((trade: any, i: number) => (
                <div
                  key={i}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '140px 80px 100px 100px 120px 100px',
                    padding: '8px 12px',
                    borderBottom: '1px solid',
                    borderColor: 'inherit',
                    opacity: 0.6,
                    fontSize: 12,
                    alignItems: 'center',
                  }}
                >
                  <span style={{ color: 'inherit' }}>
                    {trade.timestamp ? format(new Date(trade.timestamp), 'yyyy-MM-dd HH:mm') : '-'}
                  </span>
                  <span style={{ color: 'inherit' }}>{trade.symbol || '-'}</span>
                  <span>
                    <span
                      style={{
                        padding: '2px 6px',
                        borderRadius: 2,
                        fontSize: 10,
                        fontWeight: 600,
                        backgroundColor: trade.order_type === 'buy' 
                          ? 'rgba(8, 153, 129, 0.2)' 
                          : 'rgba(242, 54, 69, 0.2)',
                        color: trade.order_type === 'buy' ? '#089981' : '#f23645',
                      }}
                    >
                      {trade.order_type?.toUpperCase() || '-'}
                    </span>
                  </span>
                  <span style={{ textAlign: 'right', color: 'inherit' }}>
                    ${trade.price?.toFixed(2) || '0.00'}
                  </span>
                  <span style={{ textAlign: 'right', color: 'inherit' }}>
                    {trade.size?.toFixed(4) || '0'}
                  </span>
                  <span style={{ textAlign: 'right', color: 'inherit' }}>
                    ${(trade.price * trade.size).toFixed(2)}
                  </span>
                </div>
              ))}
            </div>

            {/* Pagination */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '8px 12px',
                borderTop: '1px solid',
                borderColor: 'inherit',
                fontSize: 12,
              }}
            >
              <span style={{ color: 'inherit', opacity: 0.6 }}>
                {data.trades.length} trades
              </span>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <button
                  onClick={() => setTradePage(Math.max(1, tradePage - 1))}
                  disabled={tradePage === 1}
                  style={{
                    padding: '4px 8px',
                    backgroundColor: 'transparent',
                    border: '1px solid',
                    borderColor: 'inherit',
                    borderRadius: 4,
                    color: 'inherit',
                    cursor: tradePage === 1 ? 'not-allowed' : 'pointer',
                    opacity: tradePage === 1 ? 0.5 : 1,
                  }}
                >
                  Prev
                </button>
                <span style={{ color: 'inherit' }}>
                  {tradePage} / {totalPages || 1}
                </span>
                <button
                  onClick={() => setTradePage(Math.min(totalPages, tradePage + 1))}
                  disabled={tradePage === totalPages}
                  style={{
                    padding: '4px 8px',
                    backgroundColor: 'transparent',
                    border: '1px solid',
                    borderColor: 'inherit',
                    borderRadius: 4,
                    color: 'inherit',
                    cursor: tradePage === totalPages ? 'not-allowed' : 'pointer',
                    opacity: tradePage === totalPages ? 0.5 : 1,
                  }}
                >
                  Next
                </button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'summary' && (
          <div style={{ height: '100%', overflow: 'auto', padding: 12 }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
              {/* Key Metrics */}
              {[
                { label: 'Total Trades', value: data.enhanced_metrics?.total_trades || 0 },
                { label: 'Winning Trades', value: data.enhanced_metrics?.winning_trades || 0 },
                { label: 'Losing Trades', value: data.enhanced_metrics?.losing_trades || 0 },
                { label: 'Win Rate', value: `${((data.enhanced_metrics?.win_rate || 0) * 100).toFixed(1)}%` },
                { label: 'Profit Factor', value: (data.enhanced_metrics?.profit_factor || 0).toFixed(2) },
                { label: 'Expectancy', value: (data.enhanced_metrics?.expectancy || 0).toFixed(4) },
                { label: 'Largest Win', value: `$${(data.enhanced_metrics?.largest_win || 0).toFixed(0)}` },
                { label: 'Largest Loss', value: `$${(data.enhanced_metrics?.largest_loss || 0).toFixed(0)}` },
                { label: 'Avg Win', value: `$${(data.enhanced_metrics?.avg_win || 0).toFixed(0)}` },
                { label: 'Avg Loss', value: `$${(data.enhanced_metrics?.avg_loss || 0).toFixed(0)}` },
                { label: 'Consecutive Wins', value: data.enhanced_metrics?.consecutive_wins || 0 },
                { label: 'Consecutive Losses', value: data.enhanced_metrics?.consecutive_losses || 0 },
              ].map((item, i) => (
                <div
                  key={i}
                  style={{
                    padding: 12,
                    backgroundColor: isDark ? '#2a2e39' : '#f5f5f5',
                    borderRadius: 4,
                  }}
                >
                  <div style={{ fontSize: 11, color: 'inherit', opacity: 0.6, marginBottom: 4 }}>
                    {item.label}
                  </div>
                  <div style={{ fontSize: 16, fontWeight: 600, color: 'inherit' }}>
                    {item.value}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'chart' && (
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Chart Type Selector */}
            <div
              style={{
                display: 'flex',
                gap: 8,
                padding: '8px 12px',
                borderBottom: '1px solid',
                borderColor: 'inherit',
              }}
            >
              {SUB_CHART_TYPES.map((type) => (
                <button
                  key={type.value}
                  onClick={() => setSubChartType(type.value)}
                  style={{
                    padding: '4px 12px',
                    fontSize: 12,
                    backgroundColor: subChartType === type.value ? '#2962ff' : 'transparent',
                    color: subChartType === type.value ? '#ffffff' : 'inherit',
                    border: 'none',
                    borderRadius: 4,
                    cursor: 'pointer',
                  }}
                >
                  {type.label}
                </button>
              ))}
            </div>

            {/* Chart */}
            <div style={{ flex: 1 }}>
              <BacktestSubChart data={data} type={subChartType as any} isDark={isDark} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
