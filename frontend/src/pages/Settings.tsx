import { useTranslation } from 'react-i18next'
import { useThemeStore } from '../stores/themeStore'
import { useI18n } from '../i18n/config'

export default function Settings() {
  const { t } = useTranslation()
  const { theme, setTheme } = useThemeStore()
  const { language, changeLanguage } = useI18n()

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold text-gray-900 dark:text-gray-100 tracking-tight">
          {t('nav.settings')}
        </h1>
        <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
          Manage your preferences and settings.
        </p>
      </div>

      <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg p-6 max-w-xl">
        <h2 className="text-base font-semibold text-gray-900 dark:text-gray-100 mb-6">
          Preferences
        </h2>

        <div className="space-y-6">
          <div>
            <label className="label">Theme</label>
            <select
              value={theme}
              onChange={(e) => setTheme(e.target.value as 'light' | 'dark')}
              className="input"
            >
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
          </div>

          <div>
            <label className="label">Language</label>
            <select
              value={language}
              onChange={(e) => changeLanguage(e.target.value)}
              className="input"
            >
              <option value="zh">中文</option>
              <option value="en">English</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  )
}
