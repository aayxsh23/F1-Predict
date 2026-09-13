import { useHealth, usePredictionsLatest } from '@/lib/queries'
import type { KnownSessions } from '@/lib/types'

function sessionPhaseLabel(sessions?: KnownSessions): string {
  if (!sessions) return 'No live race data'
  if (sessions.compound) return 'Race day'
  if (sessions.grid) return 'Grid set'
  if (sessions.qualifying) return 'Qualifying done'
  if (sessions.practice) return 'Practice'
  return 'Pre-weekend'
}

export function StatusStrip() {
  const { data: health, isPending: healthPending, isError: healthError } = useHealth()
  const { data: latest } = usePredictionsLatest()

  const dotClass = healthPending
    ? 'bg-telemetry-amber'
    : healthError || health?.status !== 'ok'
      ? 'bg-telemetry-crimson'
      : 'bg-telemetry-green'
  const connectionLabel = healthPending ? 'Connecting…' : healthError || health?.status !== 'ok' ? 'API down' : 'API connected'

  return (
    <div className="hidden shrink-0 items-center gap-4 border-t border-border-default bg-surface-sunken px-4 py-1.5 text-xs text-text-secondary md:flex">
      <span className="flex items-center gap-1.5">
        <span className={`h-1.5 w-1.5 rounded-full ${dotClass}`} />
        {connectionLabel}
      </span>
      <span className="text-border-default">·</span>
      <span>XGBoost · 4 targets</span>
      <span className="text-border-default">·</span>
      <span>{sessionPhaseLabel(latest?.known_sessions)}</span>
    </div>
  )
}
