import { ArrowLeft } from 'lucide-react'
import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { Card } from '@/components/ui/Card'
import { Skeleton } from '@/components/ui/Skeleton'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/Tabs'
import { formatNumber } from '@/lib/format'
import { useBacktestForRace } from '@/lib/queries'
import { TARGET_LABELS, TARGET_UNITS, TARGETS, type Target } from '@/lib/types'

const DECIMALS: Record<Target, number> = {
  qualifying: 2,
  finish_position: 0,
  quali_delta: 0,
  race_time: 1,
}

export function HistoryDetail() {
  const { season, round } = useParams()
  const seasonNum = Number(season)
  const roundNum = Number(round)
  const [target, setTarget] = useState<Target>('finish_position')
  const { data, isPending, isError } = useBacktestForRace(seasonNum, roundNum)

  if (isPending) {
    return (
      <div className="mx-auto max-w-2xl px-4 py-8 md:px-6 md:py-12">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="mt-3 h-8 w-56" />
        <Skeleton className="mt-8 h-96 w-full" />
      </div>
    )
  }

  if (isError || !data) {
    return (
      <div className="mx-auto max-w-2xl px-4 py-16 text-center md:px-6">
        <p className="text-lg font-semibold text-text-primary">No backtest data for this race</p>
        <Link to="/history" className="mt-2 inline-block text-sm text-accent-text hover:underline">
          Back to history
        </Link>
      </div>
    )
  }

  const rows = [...data.drivers].sort(
    (a, b) => (a.finish_position.actual ?? 99) - (b.finish_position.actual ?? 99),
  )
  const unit = TARGET_UNITS[target]
  const decimals = DECIMALS[target]

  return (
    <div className="mx-auto max-w-2xl px-4 py-8 md:px-6 md:py-12">
      <Link
        to="/history"
        className="inline-flex items-center gap-1.5 text-sm font-medium text-text-secondary hover:text-text-primary"
      >
        <ArrowLeft className="h-4 w-4" strokeWidth={2} />
        Back to history
      </Link>
      <h1 className="mt-4 text-2xl font-semibold tracking-tight text-text-primary md:text-3xl">
        {data.location} Grand Prix
      </h1>
      <p className="mt-2 text-sm text-text-secondary">
        Round {data.round} · {data.season} — actual result vs. what the model predicted, in hindsight.
      </p>

      <div className="mt-6 overflow-x-auto">
        <Tabs value={target} onValueChange={(v) => setTarget(v as Target)}>
          <TabsList>
            {TARGETS.map((t) => (
              <TabsTrigger key={t} value={t}>
                {TARGET_LABELS[t]}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
      </div>

      <Card className="mt-4 p-0">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border-default text-xs uppercase tracking-wide text-text-muted">
              <th className="px-4 py-3 text-left font-medium">Driver</th>
              <th className="px-4 py-3 text-right font-medium">Actual</th>
              <th className="px-4 py-3 text-right font-medium">Predicted</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((d, i) => (
              <tr key={d.driver} className={i === rows.length - 1 ? '' : 'border-b border-border-default'}>
                <td className="px-4 py-3">
                  <p className="font-mono font-semibold text-text-primary">{d.driver}</p>
                  <p className="text-xs text-text-secondary">{d.team}</p>
                </td>
                <td className="px-4 py-3 text-right font-mono tabular-nums text-text-primary">
                  {formatNumber(d[target].actual, decimals, unit)}
                </td>
                <td className="px-4 py-3 text-right font-mono tabular-nums text-text-secondary">
                  {formatNumber(d[target].predicted, decimals, unit)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  )
}
