import { ReactNode } from 'react'

interface MetricCardProps {
  title: string
  value: string | number
  subtitle?: string
  change?: number
  format?: 'percent' | 'currency' | 'number' | 'integer' | 'none'
  benchmark?: string | number
  color?: 'green' | 'red' | 'blue' | 'gray'
  icon?: ReactNode
  tooltip?: string
  onClick?: () => void
}

export function MetricCard({
  title,
  value,
  subtitle,
  change,
  format = 'number',
  benchmark,
  color = 'gray',
  icon,
  tooltip,
  onClick
}: MetricCardProps) {
  const formatValue = (val: string | number): string => {
    if (typeof val === 'string') return val
    if (format === 'percent') {
      // 如果值已经 >= 1，直接显示；否则乘以100
      const percentVal = val >= 1 ? val : val * 100
      return `${percentVal.toFixed(0)}%`
    }
    if (format === 'currency') return `$${val.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
    if (typeof val === 'number' && Math.abs(val) >= 1000) {
      return val.toLocaleString('en-US', { maximumFractionDigits: 2 })
    }
    if (format === 'integer') return val.toFixed(0)
    if (typeof val === 'number') return val.toFixed(2)
    return String(val)
  }

  const colorClasses = {
    green: 'border-l-4 border-green-500',
    red: 'border-l-4 border-red-500',
    blue: 'border-l-4 border-blue-500',
    gray: 'border-l-4 border-gray-300'
  }

  const valueColorClass = {
    green: 'text-green-600',
    red: 'text-red-600',
    blue: 'text-blue-600',
    gray: 'text-gray-700'
  }

  return (
    <div
      className={`bg-white rounded-lg shadow-sm p-4 ${colorClasses[color]} ${onClick ? 'cursor-pointer hover:shadow-md transition-shadow' : ''}`}
      onClick={onClick}
      title={tooltip}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">{title}</p>
          <p className={`mt-1 text-2xl font-bold ${valueColorClass[color]}`}>
            {formatValue(value)}
          </p>
          {subtitle && (
            <p className="mt-1 text-sm text-gray-400">{subtitle}</p>
          )}
          {change !== undefined && (
            <div className="mt-2 flex items-center gap-1">
              <span className={`text-sm font-medium ${change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {change >= 0 ? '↑' : '↓'} {Math.abs(change * 100).toFixed(2)}%
              </span>
              <span className="text-xs text-gray-400">vs benchmark</span>
            </div>
          )}
          {benchmark !== undefined && benchmark !== null && (
            <p className="mt-1 text-sm text-gray-400">
              Benchmark: {formatValue(benchmark)}
            </p>
          )}
        </div>
        {icon && (
          <div className="text-gray-300">
            {icon}
          </div>
        )}
      </div>
    </div>
  )
}

interface MetricsGridProps {
  metrics: MetricCardProps[]
  columns?: 2 | 3 | 4 | 6
}

export function MetricsGrid({ metrics, columns = 4 }: MetricsGridProps) {
  const gridCols = {
    2: 'grid-cols-1 sm:grid-cols-2',
    3: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4',
    6: 'grid-cols-2 sm:grid-cols-3 lg:grid-cols-6'
  }

  return (
    <div className={`grid ${gridCols[columns]} gap-4`}>
      {metrics.map((metric, index) => (
        <MetricCard key={index} {...metric} />
      ))}
    </div>
  )
}

export default MetricCard
