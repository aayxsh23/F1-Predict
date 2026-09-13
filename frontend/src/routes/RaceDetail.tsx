import { ChevronRight } from 'lucide-react'
import type { ReactNode } from 'react'
import { Link, useParams } from 'react-router-dom'

import { Card } from '@/components/ui/Card'
import { Skeleton } from '@/components/ui/Skeleton'
import { AskCopilotButton } from '@/components/workbench/AskCopilotButton'
import { useHighlightFlash } from '@/components/workbench/useHighlightFlash'
import { formatNumber, formatRelativeTime, formatSigned } from '@/lib/format'
import { usePredictionsForRace } from '@/lib/queries'
import type { DriverPrediction } from '@/lib/types'

import { KnownSessionsBadges } from '../components/predictions/KnownSessionsBadges'

export function RaceDetail() {
  const { season, round } = useParams()
  const seasonNum = Number(season)
  const roundNum = Number(round)
  const { data, isPending, isError } = usePredictionsForRace(seasonNum, roundNum)

  if (isPending) return <RaceDetailSkeleton />

  if (isError || !data) {
    return (
      <div className="mx-auto max-w-5xl px-4 py-16 text-center md:px-6">
        <p className="text-lg font-semibold text-text-primary">No prediction cached for this race</p>
        <p className="mt-2 text-sm text-text-secondary">
          Only the upcoming/current race has a live prediction —{' '}
          <Link to="/history" className="text-accent-text hover:underline">
            check History
          </Link>{' '}
          for past races.
        </p>
      </div>
    )
  }

  const rows = [...data.drivers].sort(
    (a, b) => (a.predicted_finish_position ?? 99) - (b.predicted_finish_position ?? 99),
  )

  return (
    <div className="mx-auto max-w-5xl px-4 py-8 md:px-6 md:py-12">
      <h1 className="text-2xl font-semibold tracking-tight text-text-primary md:text-3xl">
        {data.location} Grand Prix
      </h1>
      <p className="mt-2 text-sm text-text-secondary">
        Round {data.round} · {data.season} — updated {formatRelativeTime(data.generated_at)}
      </p>
      <div className="mt-5">
        <KnownSessionsBadges sessions={data.known_sessions} />
      </div>

      {/* Desktop: dense results table */}
      <Card className="mt-8 hidden overflow-x-auto p-0 md:block">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border-default text-left text-xs uppercase tracking-wide text-text-muted">
              <Th>Rank</Th>
              <Th>Driver</Th>
              <Th>Grid</Th>
              <Th>Quali gap</Th>
              <Th>Practice</Th>
              <Th align="right">Pred. quali</Th>
              <Th align="right">Pred. finish</Th>
              <Th align="right">Pred. delta</Th>
              <Th align="right">Pred. time gap</Th>
              <Th />
            </tr>
          </thead>
          <tbody>
            {rows.map((d, i) => (
              <DriverRow key={d.driver} d={d} i={i} season={data.season} round={data.round} />
            ))}
          </tbody>
        </table>
      </Card>

      {/* Mobile: stacked driver cards, not a squeezed table */}
      <div className="mt-8 space-y-2 md:hidden">
        {rows.map((d, i) => (
          <MobileDriverRow key={d.driver} driver={d} rank={i + 1} season={data.season} round={data.round} />
        ))}
      </div>
    </div>
  )
}

function DriverRow({ d, i, season, round }: { d: DriverPrediction; i: number; season: number; round: number }) {
  const { ref, flashing } = useHighlightFlash(d.driver)
  return (
    <tr
      ref={ref as React.RefObject<HTMLTableRowElement>}
      className={`group border-b border-border-default last:border-0 hover:bg-surface-sunken ${
        flashing ? 'animate-[flash-highlight_1.2s_ease-out]' : ''
      }`}
    >
      <Td className="font-mono tabular-nums text-text-muted">{i + 1}</Td>
      <Td>
        <p className="font-mono font-semibold text-text-primary">{d.driver}</p>
        <p className="text-xs text-text-secondary">{d.team}</p>
      </Td>
      <Td className="font-mono tabular-nums">{d.grid_position ?? '—'}</Td>
      <Td className="font-mono tabular-nums">{formatNumber(d.quali_gap_to_pole, 3, 's')}</Td>
      <Td className="font-mono tabular-nums">{formatNumber(d.practice_pace, 3, 's')}</Td>
      <Td align="right" className="font-mono tabular-nums">
        {formatNumber(d.predicted_qualifying_gap, 2, 's')}
      </Td>
      <Td align="right" className="font-mono font-semibold tabular-nums text-text-primary">
        {formatNumber(d.predicted_finish_position, 1)}
      </Td>
      <Td align="right" className="font-mono tabular-nums">
        {d.predicted_quali_to_race_delta === null ? '—' : formatSigned(d.predicted_quali_to_race_delta)}
      </Td>
      <Td align="right" className="font-mono tabular-nums">
        {formatNumber(d.predicted_race_time_gap, 1, 's')}
      </Td>
      <Td align="right">
        <div className="flex items-center justify-end gap-1">
          <AskCopilotButton
            prompt={`Why is ${d.driver} predicted to finish around P${Math.round(d.predicted_finish_position ?? 0)}?`}
            className="opacity-0 group-hover:opacity-100"
          />
          <Link to={`/race/${season}/${round}/driver/${d.driver}`} className="text-accent-text hover:underline">
            Why
          </Link>
        </div>
      </Td>
    </tr>
  )
}

function Th({ children, align = 'left' }: { children?: ReactNode; align?: 'left' | 'right' }) {
  return <th className={`px-4 py-3 font-medium ${align === 'right' ? 'text-right' : ''}`}>{children}</th>
}

function Td({
  children,
  align = 'left',
  className = '',
}: {
  children?: ReactNode
  align?: 'left' | 'right'
  className?: string
}) {
  return <td className={`px-4 py-3 ${align === 'right' ? 'text-right' : ''} ${className}`}>{children}</td>
}

function MobileDriverRow({
  driver,
  rank,
  season,
  round,
}: {
  driver: DriverPrediction
  rank: number
  season: number
  round: number
}) {
  const { ref, flashing } = useHighlightFlash(driver.driver)
  return (
    <div
      ref={ref as React.RefObject<HTMLDivElement>}
      className={`relative ${flashing ? 'animate-[flash-highlight_1.2s_ease-out] rounded-md' : ''}`}
    >
      <Link to={`/race/${season}/${round}/driver/${driver.driver}`} className="block">
        <Card className="flex items-center gap-3 p-4 pr-12">
          <span className="w-5 shrink-0 font-mono text-lg font-semibold tabular-nums text-text-muted">{rank}</span>
          <div className="min-w-0 flex-1">
            <p className="font-mono font-semibold text-text-primary">{driver.driver}</p>
            <p className="truncate text-xs text-text-secondary">{driver.team}</p>
            <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
              <div className="flex justify-between">
                <dt className="text-text-muted">Pred. finish</dt>
                <dd className="font-mono tabular-nums text-text-primary">
                  {formatNumber(driver.predicted_finish_position, 1)}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-text-muted">Pred. delta</dt>
                <dd className="font-mono tabular-nums text-text-primary">
                  {driver.predicted_quali_to_race_delta === null
                    ? '—'
                    : formatSigned(driver.predicted_quali_to_race_delta)}
                </dd>
              </div>
            </dl>
          </div>
          <ChevronRight className="h-4 w-4 shrink-0 text-text-muted" strokeWidth={2} />
        </Card>
      </Link>
      {/* sibling of the Link, not nested inside it -- a <button> inside an
      <a> is invalid HTML and would need stopPropagation hacks to avoid
      double-triggering navigation */}
      <AskCopilotButton
        prompt={`Why is ${driver.driver} predicted to finish there?`}
        className="absolute right-3 top-1/2 -translate-y-1/2 bg-surface"
      />
    </div>
  )
}

function RaceDetailSkeleton() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-8 md:px-6 md:py-12">
      <Skeleton className="h-4 w-24" />
      <Skeleton className="mt-3 h-8 w-56" />
      <Skeleton className="mt-3 h-4 w-40" />
      <Skeleton className="mt-8 h-96 w-full" />
    </div>
  )
}
