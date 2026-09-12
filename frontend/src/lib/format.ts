export function formatSigned(value: number, unit = ''): string {
  const sign = value > 0 ? '+' : ''
  return `${sign}${value.toFixed(2)}${unit}`
}

export function formatNumber(value: number | null, decimals = 2, unit = ''): string {
  if (value === null || Number.isNaN(value)) return '—'
  return `${value.toFixed(decimals)}${unit}`
}

export function formatOrdinalPosition(value: number | null): string {
  if (value === null || Number.isNaN(value)) return '—'
  return value.toFixed(1)
}

export function formatRelativeTime(iso: string): string {
  const then = new Date(iso).getTime()
  const diffMs = Date.now() - then
  const minutes = Math.round(diffMs / 60_000)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes} min ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours} hr${hours === 1 ? '' : 's'} ago`
  const days = Math.round(hours / 24)
  return `${days} day${days === 1 ? '' : 's'} ago`
}

export function formatFeatureName(feature: string): string {
  return feature
    .split('_')
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(' ')
}

export function formatRaceDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
}
