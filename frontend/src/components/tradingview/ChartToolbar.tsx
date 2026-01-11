import React from 'react'

interface ChartToolbarProps {
  strategyName: string
  symbol: string
  timeRange: string
  chartType: string
  showBenchmark: boolean
  showExcess: boolean
  useLogScale: boolean
  onTimeRangeChange: (range: string) => void
  onChartTypeChange: (type: string) => void
  onShowBenchmarkChange: (show: boolean) => void
  onShowExcessChange: (show: boolean) => void
  onUseLogScaleChange: (use: boolean) => void
  onFullScreen: () => void
  onSettings: () => void
}

const TIME_RANGES = [
  { value: '1D', label: '1D' },
  { value: '1W', label: '1W' },
  { value: '1M', label: '1M' },
  { value: '3M', label: '3M' },
  { value: '6M', label: '6M' },
  { value: 'YTD', label: 'YTD' },
  { value: '1Y', label: '1Y' },
  { value: '2Y', label: '2Y' },
  { value: '5Y', label: '5Y' },
  { value: 'ALL', label: 'ALL' },
]

const CHART_TYPES = [
  { value: 'area', label: 'Area' },
  { value: 'line', label: 'Line' },
]

export default function ChartToolbar({
  strategyName,
  symbol,
  timeRange,
  chartType,
  showBenchmark,
  showExcess,
  useLogScale,
  onTimeRangeChange,
  onChartTypeChange,
  onShowBenchmarkChange,
  onShowExcessChange,
  onUseLogScaleChange,
  onFullScreen,
  onSettings,
}: ChartToolbarProps) {
  return (
    <div
      style={{
        height: 48,
        backgroundColor: 'inherit',
        borderBottom: '1px solid',
        borderColor: 'inherit',
        display: 'flex',
        alignItems: 'center',
        padding: '0 12px',
        gap: 8,
      }}
    >
      {/* Strategy Name */}
      <div
        style={{
          fontWeight: 600,
          fontSize: 14,
          padding: '6px 12px',
          backgroundColor: 'rgba(41, 98, 255, 0.1)',
          color: '#2962ff',
          borderRadius: 4,
        }}
      >
        {strategyName || 'Strategy'}
      </div>

      {/* Symbol */}
      <div
        style={{
          fontWeight: 600,
          fontSize: 14,
          padding: '6px 12px',
          backgroundColor: 'inherit',
          border: '1px solid',
          borderColor: 'inherit',
          borderRadius: 4,
          opacity: 0.8,
        }}
      >
        {symbol || 'QQQ'}
      </div>

      {/* Time Range */}
      <div style={{ display: 'flex', gap: 2, marginLeft: 8 }}>
        {TIME_RANGES.map((tr) => (
          <button
            key={tr.value}
            onClick={() => onTimeRangeChange(tr.value)}
            style={{
              padding: '6px 10px',
              fontSize: 12,
              fontWeight: 500,
              backgroundColor: timeRange === tr.value ? '#2962ff' : 'transparent',
              color: timeRange === tr.value ? '#ffffff' : 'inherit',
              border: 'none',
              borderRadius: 4,
              cursor: 'pointer',
              transition: 'all 0.2s',
            }}
          >
            {tr.label}
          </button>
        ))}
      </div>

      {/* Chart Type */}
      <select
        value={chartType}
        onChange={(e) => onChartTypeChange(e.target.value)}
        style={{
          padding: '6px 12px',
          fontSize: 12,
          backgroundColor: 'transparent',
          border: '1px solid',
          borderColor: 'inherit',
          borderRadius: 4,
          color: 'inherit',
          cursor: 'pointer',
          marginLeft: 'auto',
        }}
      >
        {CHART_TYPES.map((ct) => (
          <option key={ct.value} value={ct.value}>
            {ct.label}
          </option>
        ))}
      </select>

      {/* Toggles */}
      <label
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          fontSize: 12,
          cursor: 'pointer',
        }}
      >
        <input
          type="checkbox"
          checked={showBenchmark}
          onChange={(e) => onShowBenchmarkChange(e.target.checked)}
          style={{ accentColor: '#2962ff' }}
        />
        QQQ
      </label>

      <label
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          fontSize: 12,
          cursor: 'pointer',
        }}
      >
        <input
          type="checkbox"
          checked={showExcess}
          onChange={(e) => onShowExcessChange(e.target.checked)}
          style={{ accentColor: '#2962ff' }}
        />
        Excess
      </label>

      <label
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          fontSize: 12,
          cursor: 'pointer',
        }}
      >
        <input
          type="checkbox"
          checked={useLogScale}
          onChange={(e) => onUseLogScaleChange(e.target.checked)}
          style={{ accentColor: '#2962ff' }}
        />
        Log
      </label>

      {/* Fullscreen */}
      <button
        onClick={onFullScreen}
        style={{
          padding: '6px 10px',
          fontSize: 14,
          backgroundColor: 'transparent',
          border: 'none',
          color: 'inherit',
          cursor: 'pointer',
          borderRadius: 4,
        }}
        title="Fullscreen"
      >
        ⛶
      </button>

      {/* Settings */}
      <button
        onClick={onSettings}
        style={{
          padding: '6px 10px',
          fontSize: 14,
          backgroundColor: 'transparent',
          border: 'none',
          color: 'inherit',
          cursor: 'pointer',
          borderRadius: 4,
        }}
        title="Settings"
      >
        ⚙
      </button>
    </div>
  )
}
