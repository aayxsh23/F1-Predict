import { ChevronRight } from 'lucide-react'
import { Link } from 'react-router-dom'

import { Card } from '@/components/ui/Card'
import { Skeleton } from '@/components/ui/Skeleton'
import { useBacktestRaces } from '@/lib/queries'

export function History() {
  const { data, isPending, isError } = useBacktestRaces()

  if (isPending) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-8 md:px-6 md:py-12">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="mt-6 h-96 w-full" />
      </div>
    )
  }

  if (isError || !data) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-16 text-center md:px-6">
        <p className="text-lg font-semibold text-text-primary">History isn't reachable right now</p>
        <p className="mt-2 text-sm text-text-secondary">The backend may still be waking up. Try again shortly.</p>
      </div>
    )
  }

  const bySeason = new Map<number, typeof data>()
  for (const race of data) {
    const list = bySeason.get(race.season) ?? []
    list.push(race)
    bySeason.set(race.season, list)
  }
  const seasons = [...bySeason.keys()].sort((a, b) => b - a)

  return (
    <div className="mx-auto max-w-3xl px-4 py-8 md:px-6 md:py-12">
      <h1 className="text-2xl font-semibold tracking-tight text-text-primary md:text-3xl">Race history</h1>
      <p className="mt-2 text-sm text-text-secondary">
        How the model's predictions compared to what actually happened, race by race.
      </p>

      {seasons.map((season) => {
        const races = [...(bySeason.get(season) ?? [])].sort((a, b) => a.round - b.round)
        return (
          <section key={season} className="mt-8">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-text-muted">{season}</h2>
            <Card className="mt-3 p-0">
              {races.map((race, i) => (
                <Link
                  key={`${race.season}-${race.round}`}
                  to={`/history/${race.season}/${race.round}`}
                  className={`flex items-center justify-between px-5 py-3 text-sm transition-colors duration-150 hover:bg-surface-sunken ${
                    i === races.length - 1 ? '' : 'border-b border-border-default'
                  }`}
                >
                  <span className="text-text-primary">{race.location}</span>
                  <span className="flex items-center gap-2 text-text-muted">
                    Round {race.round}
                    <ChevronRight className="h-4 w-4" strokeWidth={2} />
                  </span>
                </Link>
              ))}
            </Card>
          </section>
        )
      })}
    </div>
  )
}
