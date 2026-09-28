import { useMemo } from 'react'

import { ApiError } from './api'
import { usePredictionsForRace, usePredictionsLatest, useBacktestForRace } from './queries'
import { teamAccent } from './teams'
import type { BacktestRacePayload, KnownSessions, PredictionPayload } from './types'

export type ViewMode =
  | 'forecast' // the pre-race model output the pipeline cached at the time
  | 'replay' // no cached forecast; the archive re-scores a decided race (fitted on it, so flattering)
  | 'none'

export interface RaceDriver {
  code: string
  team: string
  accent: string
  number: number | null
  rank: number // predicted finishing order, 1-based; the model's raw float only orders it
  modelFinish: number | null
  grid: number | null
  delta: number | null // grid minus predicted rank; positive = gains places
  gapToLeader: number | null // seconds, monotone in rank
  qualiGap: number | null
  qualiGapIsEstimate: boolean
  win: number | null
  podium: number | null
  top10: number | null
  band: [number, number] | null
  expected: number | null
  compound: string | null
  actual: number | null // finishing position, once the race is decided
  featureRow: Record<string, number | string | null> | null
}

export interface RaceView {
  mode: ViewMode
  season: number
  round: number
  location: string
  isLatest: boolean
  generatedAt: string | null
  knownSessions: KnownSessions | null
  hasActuals: boolean
  drivers: RaceDriver[]
}

export type RaceViewState =
  | { status: 'pending'; view: null }
  | { status: 'error'; view: null }
  | { status: 'ready'; view: RaceView }

/** The four models predict independently, so their raw gaps can disagree with
 *  the finishing order. Sorting the gaps and re-anchoring on the smallest makes
 *  "gap to P1" monotone in rank and zero for the leader. */
function gapsByRank(raw: Array<number | null>): Array<number | null> {
  if (raw.length === 0 || raw.some((g) => g === null)) return raw.map(() => null)
  const sorted = [...(raw as number[])].sort((a, b) => a - b)
  return sorted.map((g) => g - sorted[0])
}

function byPredictedFinish<T>(items: T[], finish: (item: T) => number | null): T[] {
  return items
    .map((item, index) => ({ item, index, value: finish(item) }))
    .sort((a, b) => (a.value ?? Infinity) - (b.value ?? Infinity) || a.index - b.index)
    .map(({ item }) => item)
}

function fromForecast(p: PredictionPayload, isLatest: boolean, actuals: Map<string, number | null>): RaceView {
  const ordered = byPredictedFinish(p.drivers, (d) => d.predicted_finish_position)
  const gaps = gapsByRank(ordered.map((d) => d.predicted_race_time_gap))
  const drivers = ordered.map((d, i): RaceDriver => {
    const rank = i + 1
    const compound = d.feature_row.starting_tire_compound
    return {
      code: d.driver,
      team: d.team,
      accent: teamAccent(d.team),
      number: d.driver_number ?? null,
      rank,
      modelFinish: d.predicted_finish_position,
      grid: d.grid_position,
      delta: d.grid_position === null ? null : d.grid_position - rank,
      gapToLeader: gaps[i],
      qualiGap: d.quali_gap_to_pole ?? d.predicted_qualifying_gap,
      qualiGapIsEstimate: d.quali_gap_to_pole === null,
      win: d.probabilities?.win ?? null,
      podium: d.probabilities?.podium ?? null,
      top10: d.probabilities?.top10 ?? null,
      band: d.position_band ?? null,
      expected: d.expected_position ?? null,
      compound: typeof compound === 'string' ? compound : null,
      actual: actuals.get(d.driver) ?? null,
      featureRow: d.feature_row,
    }
  })
  return {
    mode: 'forecast',
    season: p.season,
    round: p.round,
    location: p.location,
    isLatest,
    generatedAt: p.generated_at,
    knownSessions: p.known_sessions,
    hasActuals: actuals.size > 0,
    drivers,
  }
}

function fromBacktest(b: BacktestRacePayload): RaceView {
  const ordered = byPredictedFinish(b.drivers, (d) => d.finish_position.predicted)
  const gaps = gapsByRank(ordered.map((d) => d.race_time.predicted))
  const drivers = ordered.map((d, i): RaceDriver => {
    const rank = i + 1
    const { actual: finish } = d.finish_position
    const { actual: change } = d.quali_delta
    const grid = finish !== null && change !== null ? Math.round(finish + change) : null // delta = grid - finish
    return {
      code: d.driver,
      team: d.team,
      accent: teamAccent(d.team),
      number: d.driver_number ?? null,
      rank,
      modelFinish: d.finish_position.predicted,
      grid,
      delta: grid === null ? null : grid - rank,
      gapToLeader: gaps[i],
      qualiGap: d.qualifying.actual,
      qualiGapIsEstimate: false,
      win: null,
      podium: null,
      top10: null,
      band: null,
      expected: null,
      compound: null,
      actual: finish,
      featureRow: null,
    }
  })
  return {
    mode: 'replay',
    season: b.season,
    round: b.round,
    location: b.location,
    isLatest: false,
    generatedAt: null,
    knownSessions: null,
    hasActuals: true,
    drivers,
  }
}

const isRealFailure = (q: { isError: boolean; error: unknown }) =>
  q.isError && !(q.error instanceof ApiError && q.error.status === 404)

/** One selected race, whichever source has it: the live payload if it is the
 *  latest, else a cached forecast for that round, else the archived backtest.
 *  Actual results come from the archive whenever the race is decided. */
export function useRaceView(sel: { season: number; round: number } | null): RaceViewState {
  const latest = usePredictionsLatest()
  const season = sel?.season ?? 0
  const round = sel?.round ?? 0
  const isLatest = !!sel && latest.data?.season === season && latest.data.round === round

  const cached = usePredictionsForRace(season, round, !!sel && !isLatest && !latest.isPending)
  const archive = useBacktestForRace(season, round, !!sel)

  return useMemo((): RaceViewState => {
    if (!sel || (latest.isPending && !latest.isError)) return { status: 'pending', view: null }

    const forecast = isLatest ? latest.data : cached.data
    if (forecast) {
      const actuals = new Map((archive.data?.drivers ?? []).map((d) => [d.driver, d.finish_position.actual] as const))
      return { status: 'ready', view: fromForecast(forecast, isLatest, actuals) }
    }
    if (archive.data) return { status: 'ready', view: fromBacktest(archive.data) }

    if (cached.isPending || archive.isPending) return { status: 'pending', view: null }
    if (isRealFailure(cached) || isRealFailure(archive) || (isLatest && isRealFailure(latest))) return { status: 'error', view: null }
    return {
      status: 'ready',
      view: {
        mode: 'none',
        season,
        round,
        location: '',
        isLatest,
        generatedAt: null,
        knownSessions: null,
        hasActuals: false,
        drivers: [],
      },
    }
  }, [sel, season, round, isLatest, latest, cached, archive])
}
