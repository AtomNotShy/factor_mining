const padNumber = (value: number): string => {
  return String(value).padStart(2, '0')
}

const formatDate = (date: Date): string => {
  const year = date.getFullYear()
  const month = padNumber(date.getMonth() + 1)
  const day = padNumber(date.getDate())
  return `${year}/${month}/${day}`
}

const formatNumericTimestamp = (value: number): string => {
  if (!Number.isFinite(value)) {
    return String(value)
  }

  const absValue = Math.abs(value)

  if (absValue >= 1_000_000_000_000) {
    return formatDate(new Date(value))
  }

  if (absValue >= 1_000_000_000) {
    return formatDate(new Date(value * 1000))
  }

  return `${value}`
}

const isNumericString = (value: string): boolean => {
  return /^-?\d+(\.\d+)?$/.test(value.trim())
}

const isLikelyDate = (ts: string | number): boolean => {
  if (typeof ts === 'number') {
    const absValue = Math.abs(ts)
    return absValue >= 1_000_000_000
  }

  const trimmed = ts.trim()
  if (trimmed.length === 0) {
    return false
  }

  if (isNumericString(trimmed)) {
    const numeric = Number(trimmed)
    return Math.abs(numeric) >= 1_000_000_000
  }

  const parsed = new Date(ts)
  return !Number.isNaN(parsed.getTime())
}

export const resolveTimestampSeries = (
  timestamps: Array<string | number>,
  fallback?: Array<string | number>
): string[] => {
  if (!timestamps || timestamps.length === 0) {
    return []
  }

  const hasDate = timestamps.some((ts) => isLikelyDate(ts))
  if (!hasDate && fallback && fallback.length === timestamps.length) {
    return fallback.map((ts) => String(ts))
  }

  return timestamps.map((ts) => String(ts))
}

export const formatTimestamp = (ts: string | number): string => {
  if (typeof ts === 'number') {
    return formatNumericTimestamp(ts)
  }

  const trimmed = ts.trim()

  if (trimmed.length > 0 && isNumericString(trimmed)) {
    return formatNumericTimestamp(Number(trimmed))
  }

  const date = new Date(ts)
  if (!Number.isNaN(date.getTime())) {
    return formatDate(date)
  }

  return ts
}
