import { useQueryClient } from '@tanstack/react-query'
import { Box, Layers } from 'lucide-react'
import { Suspense, lazy, useState } from 'react'

import { cn } from '@/lib/cn'
import { formatRelativeTime } from '@/lib/format'
import { useFxMode } from '@/lib/fx'
import { useRaceView, type RaceView } from '@/lib/raceView'
import { useSelection } from '@/lib/selection'
import type { KnownSessions } from '@/lib/types'

import { CarTelemetryBadge } from './CarTelemetryBadge'
import PodiumStageLite from './PodiumStageLite'
import type { Rank, StageCar, StageProps } from './slots'

// three.js is ~600 KB; it loads after first paint, behind the lite stage
const PodiumStage3D = lazy(() => import('./PodiumStage3D'))

const SESSIONS: Array<[keyof KnownSessions, string]> = [
  ['practice', 'Practice'],
  ['sprint_qualifying', 'Sprint quali'],
  ['sprint', 'Sprint'],
  ['qualifying', 'Quali'],
  ['grid', 'Grid'],
  ['compound', 'Compound'],
]

function SessionStrip({ known }: { known: KnownSessions }) {
  return (
    <ul className="flex flex-wrap gap-x-3 gap-y-1" aria-label="Sessions this forecast has seen">
      {SESSIONS.filter(([key]) => key in known).map(([key, label]) => (
        <li key={key} className={cn('flex items-center gap-1.5', known[key] ? 'text-silver-200' : 'text-silver-400')}>
          <span className={cn('h-1.5 w-1.5 rounded-full', known[key] ? 'bg-laser-400' : 'border border-silver-400')} aria-hidden />
          <span className="hud-label">{label}</span>
          <span className="sr-only">{known[key] ? 'seen' : 'pending'}</span>
        </li>
      ))}
    </ul>
  )
}

function SourceChip({ view, pending }: { view: RaceView | null; pending: boolean }) {
  let label = 'Loading forecast'
  let tone: 'live' | 'quiet' = 'quiet'
  let hint: string | undefined
  if (!pending && view) {
    if (view.mode === 'forecast' && view.isLatest && view.generatedAt) {
      const fresh = Date.now() - new Date(view.generatedAt).getTime() < 36 * 3_600_000
      label = `${fresh ? 'Live' : 'Latest'} forecast${view.sessionLabel ? ` · ${view.sessionLabel.toLowerCase()}` : ''} · updated ${formatRelativeTime(view.generatedAt)}`
      tone = fresh ? 'live' : 'quiet'
      if (!fresh) hint = "The scheduled refresh hasn't produced a newer forecast since then."
    } else if (view.mode === 'forecast') {
      label = view.hasActuals ? 'Pre-race forecast vs result' : `Archived forecast${view.generatedAt ? ` · ${formatRelativeTime(view.generatedAt)}` : ''}`
    } else if (view.mode === 'replay') {
      label = 'Past race · predicted vs result'
      hint = 'Predicted by a model trained only on earlier races, so this is how it would really have called it.'
    } else {
      label = 'No forecast yet'
    }
  }
  return (
    <p
      className={cn('hud-label inline-flex items-center gap-2', tone === 'live' ? 'text-laser-300' : 'text-silver-300')}
      title={hint}
    >
      <span
        className={cn(
          'h-1.5 w-1.5 rounded-full',
          tone === 'live' && 'bg-laser-400 motion-safe:animate-[pulse-dot_1.6s_ease-in-out_infinite]',
          tone === 'quiet' && 'border border-silver-400',
        )}
        aria-hidden
      />
      {label}
    </p>
  )
}

function StateOverlay({ status, mode, onRetry }: { status: 'pending' | 'error' | 'ready'; mode: RaceView['mode'] | null; onRetry: () => void }) {
  if (status === 'ready' && mode !== 'none') return null
  const copy =
    status === 'pending'
      ? { title: 'Waking the pit wall…', body: 'The first request after a quiet spell takes a few seconds.', delayed: true }
      : status === 'error'
        ? { title: "Predictions aren't reachable right now", body: 'The backend may still be starting up.', delayed: false }
        : {
            title: 'No forecast for this race yet',
            body: "Forecasts appear as the weekend's sessions arrive. Pick a decided round on the ribbon to review the model's call.",
            delayed: false,
          }
  return (
    <div
      className={cn(
        'pointer-events-none absolute inset-x-0 bottom-[14%] z-30 mx-auto flex max-w-sm flex-col items-center px-6 text-center',
        copy.delayed && 'motion-safe:animate-[fade-in_0.5s_ease-out_1.2s_both]',
      )}
      role="status"
    >
      <p className="font-display text-lg font-semibold uppercase tracking-[0.08em] text-silver-100">{copy.title}</p>
      <p className="mt-1 text-sm text-silver-400">{copy.body}</p>
      {status === 'error' && (
        <button type="button" onClick={onRetry} className="hud-label pointer-events-auto mt-3 border border-laser-500 px-3 py-1.5 text-laser-300 hover:bg-laser-900">
          Retry
        </button>
      )}
    </div>
  )
}

export function PodiumHero() {
  const sel = useSelection()
  const { status, view } = useRaceView(sel.selectedRace)
  const [fx, setFx] = useFxMode()
  const [hovered, setHovered] = useState<Rank | null>(null)
  const queryClient = useQueryClient()

  const race = sel.calendar.find((r) => r.round === sel.selectedRace?.round)
  const podium = view && view.mode !== 'none' ? view.drivers.slice(0, 3) : []
  const cars: StageCar[] = ([1, 2, 3] as Rank[]).map((rank) => ({ rank, driver: podium[rank - 1] ?? null }))
  // a different race, or a changed top three, replays the reveal; a routine data refresh does not
  const raceKey = `${sel.selectedRace?.season}-${sel.selectedRace?.round}-${podium.map((d) => d.code).join('')}`

  const stageProps: StageProps = {
    cars,
    raceKey,
    hovered,
    onHover: setHovered,
    onSelect: (driver) => sel.openDriver(driver.code, 'shap'),
    renderHud: (driver, rank) => (
      <CarTelemetryBadge driver={driver} rank={rank} hovered={hovered === rank} onHover={setHovered} onExplain={() => sel.openDriver(driver.code, 'shap')} />
    ),
  }

  const title = race?.eventName ?? (view?.location ? `${view.location} Grand Prix` : '')

  return (
    <section
      aria-label="Predicted podium"
      className="relative isolate min-h-[540px] flex-1 overflow-hidden sm:min-h-[440px] lg:min-h-0"
      style={{
        background:
          'radial-gradient(60% 55% at 50% 68%, color-mix(in srgb, var(--color-laser-400) 15%, transparent), transparent 70%), radial-gradient(90% 70% at 50% 0%, color-mix(in srgb, var(--color-silver-400) 10%, transparent), transparent 60%), linear-gradient(180deg, var(--color-obsidian-900), var(--color-obsidian-950))',
      }}
    >
      <div aria-hidden className="hud-grid absolute inset-0 -z-10" />

      <div className="absolute inset-x-0 bottom-0 top-[150px] sm:top-0">
        <Suspense fallback={<PodiumStageLite {...stageProps} />}>
          {fx === '3d' ? <PodiumStage3D {...stageProps} /> : <PodiumStageLite {...stageProps} />}
        </Suspense>
      </div>

      <div className={cn('pointer-events-none absolute left-4 right-4 top-4 z-40 sm:right-auto sm:max-w-[min(70%,600px)] lg:left-8', fx === '3d' ? 'lg:bottom-5 lg:top-auto' : 'lg:top-6')}>
        <h1 className="font-display text-2xl font-semibold uppercase leading-tight tracking-[0.04em] text-silver-100 sm:text-3xl">
          {title || ' '}
        </h1>
        {race && (
          <p className="mt-1 font-mono text-hud text-silver-400">
            R{String(race.round).padStart(2, '0')} · {race.location}
            {race.dateLabel && ` · ${race.dateLabel}`}
          </p>
        )}
        <div className="mt-2 flex flex-wrap items-center gap-x-5 gap-y-1.5">
          <SourceChip view={view} pending={status === 'pending'} />
          {view?.knownSessions && <SessionStrip known={view.knownSessions} />}
        </div>
      </div>

      <StateOverlay status={status} mode={view?.mode ?? null} onRetry={() => queryClient.invalidateQueries()} />

      <button
        type="button"
        aria-pressed={fx === '3d'}
        onClick={() => setFx(fx === '3d' ? 'lite' : '3d')}
        title="Switch between the 3D stage and the lightweight stage"
        className="hud-label absolute bottom-3 right-3 z-40 flex items-center gap-1.5 border border-line-strong bg-obsidian-950/80 px-2.5 py-1.5 text-silver-300 hover:border-laser-500 hover:text-laser-300 lg:bottom-4 lg:right-6"
      >
        {fx === '3d' ? <Box className="h-3.5 w-3.5" aria-hidden /> : <Layers className="h-3.5 w-3.5" aria-hidden />}
        {fx === '3d' ? '3D' : 'Lite'}
      </button>

      <ol className="sr-only">
        {podium.map((d) => (
          <li key={d.code}>
            Predicted P{d.rank}: {d.code}, {d.team}
            {d.win !== null && `, ${(d.win * 100).toFixed(1)} percent to win`}
          </li>
        ))}
      </ol>
    </section>
  )
}
