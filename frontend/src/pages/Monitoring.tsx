import { useTranslation } from 'react-i18next'
import { useState, useEffect } from 'react'
import { api } from '../services/api'

export default function Monitoring() {
  const { t } = useTranslation()
  const [status, setStatus] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadStatus()
    const interval = setInterval(loadStatus, 30000)
    return () => clearInterval(interval)
  }, [])

  const loadStatus = async () => {
    try {
      const response = await api.get('/health')
      setStatus(response.data)
    } catch (error) {
      console.error('Failed to load status:', error)
      setStatus({ status: 'error' })
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

  const isHealthy = status?.status === 'healthy'

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold text-gray-900 dark:text-gray-100 tracking-tight">
          {t('nav.monitoring')}
        </h1>
        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
          System status and health metrics.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg p-5">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
            System Status
          </p>
          <div className="flex items-center gap-3">
            <span className={`inline-block w-2.5 h-2.5 rounded-full ${isHealthy ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              {status?.status || 'Unknown'}
            </span>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg p-5">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
            Version
          </p>
          <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {status?.version || 'N/A'}
          </p>
        </div>

        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg p-5">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
            Data Quality
          </p>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Monitoring data quality metrics...
          </p>
        </div>
      </div>
    </div>
  )
}
