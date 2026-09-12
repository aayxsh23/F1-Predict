import { ArrowLeft } from 'lucide-react'
import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { ShapBarChart } from '@/components/charts/ShapBarChart'
import { Badge } from '@/components/ui/Badge'
import { Card } from '@/components/ui/Card'
import { Skeleton } from '@/components/ui/Skeleton'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/Tabs'
import { formatFeatureName, formatNumber } from '@/lib/format'
import { useDriverExplain, useNaturalExplanation } from '@/lib/queries'
import { TARGET_HIGHER_IS_BETTER, TARGET_LABELS, TARGET_UNITS, TARGETS, type Target } from '@/lib/types'

export function DriverExplain() {
  const { season, round, driver } = useParams()
  const seasonNum = Number(season)
  const roundNum = Number(round)
  const [target, setTarget] = useState<Target>('finish_position')

  if (!driver) return null

  return (
    <div className="mx-auto max-w-3xl px-4 py-8 md:px-6 md:py-12">
      <Link
        to={`/race/${seasonNum}/${roundNum}`}
        className="inline-flex items-center gap-1.5 text-sm font-medium text-text-secondary hover:text-text-primary"
      >
        <ArrowLeft className="h-4 w-4" strokeWidth={2} />
        Back to full grid
      </Link>

      <h1 className="mt-4 font-mono text-2xl font-semibold tracking-wide text-text-primary md:text-3xl">
        {driver}
      </h1>

      <Tabs value={target} onValueChange={(v) => setTarget(v as Target)}>
        <TabsList>
          {TARGETS.map((t) => (
            <TabsTrigger key={t} value={t}>
              {TARGET_LABELS[t]}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

      <div className="mt-6 space-y-6">
        <ShapPanel season={seasonNum} round={roundNum} driver={driver} target={target} />
        <ExplanationPanel season={seasonNum} round={roundNum} driver={driver} target={target} />
      </div>
    </div>
  )
}

function ShapPanel({
  season,
  round,
  driver,
  target,
}: {
  season: number
  round: number
  driver: string
  target: Target
}) {
  const { data, isPending, isError } = useDriverExplain(season, round, driver, target)

  if (isPending) {
    return (
      <Card>
        <Skeleton className="h-6 w-32" />
        <Skeleton className="mt-4 h-40 w-full" />
      </Card>
    )
  }

  if (isError || !data) {
    return (
      <Card>
        <p className="text-sm text-text-secondary">SHAP breakdown unavailable for this driver/target.</p>
      </Card>
    )
  }

  const higherIsBetter = TARGET_HIGHER_IS_BETTER[target]
  const unit = TARGET_UNITS[target]

  return (
    <Card>
      <div className="flex items-baseline justify-between">
        <p className="text-xs font-semibold uppercase tracking-wider text-text-muted">
          {TARGET_LABELS[target]}
        </p>
        <p className="font-mono text-2xl font-semibold tabular-nums text-text-primary">
          {formatNumber(data.predicted_value, target === 'finish_position' ? 1 : 2, unit)}
        </p>
      </div>
      <p className="mt-1 text-xs text-text-muted">
        Model average for this target: {formatNumber(data.base_value, 2, unit)} · {higherIsBetter ? 'higher is better' : 'lower is better'}
      </p>
      <div className="mt-6">
        <ShapBarChart contributions={data.top_contributions} target={target} />
      </div>
    </Card>
  )
}

function ExplanationPanel({
  season,
  round,
  driver,
  target,
}: {
  season: number
  round: number
  driver: string
  target: Target
}) {
  const { data, isPending, isError } = useNaturalExplanation(season, round, driver, target)

  return (
    <Card>
      <p className="text-xs font-semibold uppercase tracking-wider text-text-muted">Why</p>

      {isPending && (
        <div className="mt-3 space-y-2">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-2/3" />
          <p className="pt-1 text-xs text-text-muted">
            Generating a grounded explanation locally — this can take a little while.
          </p>
        </div>
      )}

      {isError && (
        <p className="mt-3 text-sm text-text-secondary">
          Couldn't generate an explanation for this driver right now. The SHAP breakdown above still holds.
        </p>
      )}

      {data && (
        <>
          <p className="mt-3 whitespace-pre-line text-sm leading-relaxed text-text-primary">
            {data.explanation}
          </p>
          {data.sources.length > 0 && (
            <div className="mt-5 flex flex-wrap items-center gap-1.5 border-t border-border-default pt-4">
              <span className="text-xs text-text-muted">Grounded in:</span>
              {data.sources.map((source, i) => (
                <Badge key={`${source}-${i}`} tone="neutral">
                  {formatSourceName(source)}
                </Badge>
              ))}
            </div>
          )}
        </>
      )}
    </Card>
  )
}

function formatSourceName(path: string): string {
  const file = path.split(/[/\\]/).pop() ?? path
  return formatFeatureName(file.replace(/\.(txt|pdf)$/i, ''))
}
