import { Check, RefreshCw } from 'lucide-react'

import { cn } from '@/lib/cn'
import { useRefresh } from '@/lib/queries'

/** Top-right Refresh: pulls new races and history, retrains, and republishes the forecast (scripts/refresh_all.py).
 *  Only the local API can run it; the deployed app reports `available: false` and shows nothing. */
export function RefreshButton({ compact = false }: { compact?: boolean }) {
  const { status, start, starting, startError } = useRefresh()
  if (!status?.available) return null

  const running = starting || status.state === 'running'
  const failed = !running && (status.state === 'error' || status.state === 'interrupted' || !!startError)
  const done = !running && status.state === 'done'
  const progress = status.total ? ` ${status.done ?? 0}/${status.total}` : ''
  const label = running ? `${status.step ?? 'Starting'}${progress}` : failed ? 'Refresh failed' : done ? 'Up to date' : 'Refresh'
  const title = running
    ? 'Refreshing: new races, history, models and the forecast. This takes several minutes.'
    : failed
      ? (status.error ?? (startError as Error | null)?.message ?? 'The last refresh did not finish. Click to run it again.')
      : 'Refresh: pull new races and history, retrain the models and republish the forecast'
  const Icon = done ? Check : RefreshCw

  return (
    <button
      type="button"
      onClick={() => start()}
      disabled={running}
      title={title}
      aria-label={compact ? label : undefined}
      aria-busy={running}
      className={cn(
        'flex shrink-0 items-center gap-2 rounded-[4px] border leading-none transition-colors disabled:cursor-progress',
        compact ? 'size-11 justify-center border-transparent' : 'border-white/[.12] px-2.5 py-1.5 hover:border-white/30 hover:text-ink',
        failed ? 'text-pending' : done ? 'text-gain' : 'text-ink-2',
      )}
    >
      <Icon className={cn('size-3.5', compact && 'size-5', running && 'motion-safe:animate-spin')} aria-hidden />
      {!compact && <span className={cn('max-w-[240px] truncate', running && 'hidden min-[1040px]:inline')}>{label}</span>}
    </button>
  )
}
