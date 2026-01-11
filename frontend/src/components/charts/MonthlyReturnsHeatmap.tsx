import { useMemo } from 'react'
import { useThemeStore } from '../../stores/themeStore'

interface MonthlyReturnsHeatmapProps {
  monthlyReturns: {
    year_month_matrix?: Record<string, Record<string, number | null>>
    monthly_stats?: Record<string, {
      mean?: number
      std?: number
      positive_pct?: number
      count?: number
    }>
  }
}

const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

export default function MonthlyReturnsHeatmap({ monthlyReturns }: MonthlyReturnsHeatmapProps) {
  const { theme } = useThemeStore()
  const isDark = theme === 'dark'

  const { gridData, years, stats } = useMemo(() => {
    const matrix = monthlyReturns.year_month_matrix || {}
    const years = Object.keys(matrix).sort()
    const monthlyStats = monthlyReturns.monthly_stats || {}

    const gridData: Record<string, (number | null)[]> = {}
    for (const year of years) {
      gridData[year] = []
      for (let m = 1; m <= 12; m++) {
        gridData[year].push(matrix[year]?.[String(m)] ?? null)
      }
    }

    return { gridData, years, stats: monthlyStats }
  }, [monthlyReturns])

  const getColor = (value: number | null): string => {
    if (value === null || value === undefined) return 'bg-gray-100 dark:bg-gray-700'

    const absValue = Math.abs(value)
    const intensity = Math.min(absValue * 3, 1)

    if (value > 0) {
      return isDark
        ? `rgba(52, 211, 153, ${intensity})`
        : `rgba(22, 163, 74, ${intensity})`
    } else {
      return isDark
        ? `rgba(248, 113, 113, ${intensity})`
        : `rgba(220, 38, 38, ${intensity})`
    }
  }

  const formatValue = (value: number | null): string => {
    if (value === null || value === undefined) return ''
    return `${(value * 100).toFixed(1)}%`
  }

  if (years.length === 0) {
    return (
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Monthly Returns</h3>
        <p className="text-gray-500 dark:text-gray-400">No monthly data available</p>
      </div>
    )
  }

  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white">Monthly Returns</h3>
        <div className="flex gap-2 text-xs">
          <span className="px-2 py-1 bg-red-500/20 text-red-600 dark:text-red-400 rounded">Loss</span>
          <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded">Neutral</span>
          <span className="px-2 py-1 bg-green-500/20 text-green-600 dark:text-green-400 rounded">Gain</span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className="p-1 text-left text-gray-500 dark:text-gray-400 font-medium">Year</th>
              {MONTH_NAMES.map((month) => (
                <th key={month} className="p-1 text-center text-gray-500 dark:text-gray-400 font-medium w-16">
                  {month}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {years.map((year) => (
              <tr key={year}>
                <td className="p-1 text-gray-700 dark:text-gray-300 font-medium">{year}</td>
                {gridData[year].map((value, idx) => (
                  <td key={idx} className="p-0.5">
                    <div
                      className={`w-full h-8 flex items-center justify-center rounded text-xs font-medium ${
                        isDark ? 'text-white' : 'text-gray-800'
                      }`}
                      style={{ backgroundColor: getColor(value) }}
                      title={value !== null ? `${year}-${MONTH_NAMES[idx]}: ${(value * 100).toFixed(2)}%` : 'No data'}
                    >
                      {formatValue(value)}
                    </div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {Object.keys(stats).length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Monthly Statistics</h4>
          <div className="grid grid-cols-3 md:grid-cols-4 gap-2">
            {MONTH_NAMES.filter((m) => stats[m]).map((month) => {
              const stat = stats[month]
              return (
                <div key={month} className="bg-gray-50 dark:bg-gray-700/50 rounded p-2">
                  <p className="text-xs text-gray-500 dark:text-gray-400">{month}</p>
                  <p className="text-sm font-medium text-gray-900 dark:text-white">
                    Mean: {(stat.mean! * 100).toFixed(2)}%
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Win: {(stat.positive_pct! * 100).toFixed(0)}%
                  </p>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
