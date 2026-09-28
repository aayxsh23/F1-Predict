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

/** 101.234 -> "1:41.234" */
export function formatLapTime(seconds: number | null): string {
  if (seconds === null || Number.isNaN(seconds)) return '—'
  const m = Math.floor(seconds / 60)
  return `${m}:${(seconds - m * 60).toFixed(3).padStart(6, '0')}`
}

/** 5520 -> "1h 32m" */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return '—'
  const minutes = Math.round(seconds / 60)
  return `${Math.floor(minutes / 60)}h ${String(minutes % 60).padStart(2, '0')}m`
}

/** A chance in words first, number second: "likely (74%)". */
export function describeChance(p: number | null | undefined): string {
  if (p === null || p === undefined) return '—'
  const words = p >= 0.9 ? 'almost certain' : p >= 0.65 ? 'likely' : p >= 0.35 ? 'about even' : p >= 0.1 ? 'unlikely' : 'a long shot'
  return `${words} (${Math.round(p * 100)}%)`
}
