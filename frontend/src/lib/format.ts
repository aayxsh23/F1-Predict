export function formatSigned(value: number, decimals = 2, unit = ''): string {
  const sign = value > 0 ? '+' : ''
  return `${sign}${value.toFixed(decimals)}${unit}`
}

export function formatNumber(value: number | null, decimals = 2, unit = ''): string {
  if (value === null || Number.isNaN(value)) return '—'
  return `${value.toFixed(decimals)}${unit}`
}

/** 0.142 -> "14.2%"; a real but tiny chance stays visibly non-zero. */
export function formatPct(p: number | null): string {
  if (p === null || Number.isNaN(p)) return '—'
  if (p > 0 && p < 0.001) return '<0.1%'
  return `${(p * 100).toFixed(1)}%`
}

export function formatRelativeTime(iso: string): string {
  const minutes = Math.round((Date.now() - new Date(iso).getTime()) / 60_000)
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
