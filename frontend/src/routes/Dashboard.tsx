import { ArrowRight, ChevronRight } from 'lucide-react'
import { Link } from 'react-router-dom'

import { Card } from '@/components/ui/Card'
import { Skeleton } from '@/components/ui/Skeleton'
import { formatRelativeTime } from '@/lib/format'
import { usePredictionsLatest } from '@/lib/queries'
import type { DriverPrediction } from '@/lib/types'

import { KnownSessionsBadges } from '../components/predictions/KnownSessionsBadges'

export function Dashboard() {
  const { data, isPending, isError, error } = usePredictionsLatest()

  if (isPending) return <DashboardSkeleton />

  if (isError) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-16 text-center md:px-6">
        <p className="text-lg font-semibold text-text-primary">Predictions aren't reachable right now</p>
        <p className="mt-2 text-sm text-text-secondary">
          {error instanceof Error ? error.message : 'The backend may still be waking up.'} Try again in a moment.
        </p>
      </div>
    )
  }

  const podium = [...data.drivers]
    .filter((d) => d.predicted_finish_position !== null)
    .sort((a, b) => (a.predicted_finish_position ?? 0) - (b.predicted_finish_position ?? 0))
    .slice(0, 3)

  return (
    <div className="mx-auto max-w-3xl px-4 py-8 md:px-6 md:py-12">
      <h1 className="text-3xl font-semibold tracking-tight text-text-primary md:text-4xl">
        {data.location} Grand Prix
      </h1>
      <p className="mt-2 text-sm text-text-secondary">
        Round {data.round} · {data.season} — updated {formatRelativeTime(data.generated_at)}
      </p>

      <div className="mt-6">
        <KnownSessionsBadges sessions={data.known_sessions} />
      </div>

      <h2 className="mt-10 text-xs font-semibold uppercase tracking-wider text-text-muted">
        Predicted top 3
      </h2>
      <Card className="mt-3 p-0">
        {podium.map((driver, i) => (
          <PodiumRow
            key={driver.driver}
            driver={driver}
            rank={i + 1}
            last={i === podium.length - 1}
            season={data.season}
            round={data.round}
          />
        ))}
      </Card>

      <Link
        to={`/race/${data.season}/${data.round}`}
        className="mt-4 inline-flex items-center gap-1.5 text-sm font-medium text-accent-text hover:underline"
      >
        View full grid ({data.drivers.length} drivers)
        <ArrowRight className="h-4 w-4" strokeWidth={2} />
      </Link>
    </div>
  )
}

function PodiumRow({
  driver,
  rank,
  last,
  season,
  round,
}: {
  driver: DriverPrediction
  rank: number
  last: boolean
  season: number
  round: number
}) {
  return (
    <Link
      to={`/race/${season}/${round}/driver/${driver.driver}`}
      className={`flex items-center gap-4 px-5 py-4 transition-colors duration-150 hover:bg-surface-sunken ${last ? '' : 'border-b border-border-default'}`}
    >
      <span
        className={`w-6 shrink-0 font-mono text-2xl font-semibold tabular-nums ${rank === 1 ? 'text-accent-text' : 'text-text-muted'}`}
      >
        {rank}
      </span>
      <div className="min-w-0 flex-1">
        <p className="font-mono text-base font-semibold tracking-wide text-text-primary">{driver.driver}</p>
        <p className="truncate text-sm text-text-secondary">{driver.team}</p>
      </div>
      <div className="flex items-center gap-2 text-text-muted">
        <span className="text-sm">Why</span>
        <ChevronRight className="h-4 w-4" strokeWidth={2} />
      </div>
    </Link>
  )
}

function DashboardSkeleton() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-8 md:px-6 md:py-12">
      <Skeleton className="h-4 w-24" />
      <Skeleton className="mt-3 h-9 w-64" />
      <Skeleton className="mt-3 h-4 w-48" />
      <div className="mt-6 flex gap-1.5">
        <Skeleton className="h-6 w-20" />
        <Skeleton className="h-6 w-20" />
        <Skeleton className="h-6 w-16" />
        <Skeleton className="h-6 w-24" />
      </div>
      <Skeleton className="mt-10 h-3 w-32" />
      <Skeleton className="mt-3 h-48 w-full" />
    </div>
  )
}
