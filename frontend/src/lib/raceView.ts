import { useMemo } from 'react'

import { ApiError } from './api'
import { useBacktestForRace, usePredictionsForRace, usePredictionsLatest } from './queries'
import { teamAccent } from './teams'
import type { BacktestRacePayload, KnownSessions, PredictionPayload, RaceInfo, Stage } from './types'

export type ViewMode =
  | 'forecast' // the forecast the pipeline stored before the race
  | 'replay' // a past race without a stored forecast: the walk-forward prediction from a model that hadn't seen it
  | 'none'

export interface RaceDriver {
  code: string
  team: string
  accent: string
  number: number | null
  rank: number // predicted finishing order, 1-based; the model's raw number only orders it
  modelFinish: number | null
  grid: number | null
  delta: number | null // grid minus predicted rank; positive = gains places
  gapToLeader: number | null // seconds, monotone in rank
  // qualifying
  qualiRank: number // predicted qualifying order
  qualiGap: number | null // seconds to pole: the real one once qualifying has run, else the prediction
  qualiGapIsEstimate: boolean
  qualiGapPredicted: number | null // seconds to the predicted pole
  qualiLap: number | null // predicted lap, seconds
  pole: number | null
  q3: number | null
  q1Out: number | null
  // race odds
  win: number | null
  podium: number | null
  top10: number | null
  band: [number, number] | null
  expected: number | null
  retireRisk: number | null
  beatsTeammate: number | null
  compound: string | null
  actual: number | null // finishing position, once the race is decided
  featureRow: Record<string, number | string | null> | null
}

export interface RaceView {
  mode: ViewMode
  season: number
  round: number
  location: string
  eventName: string | null
  isLatest: boolean
  generatedAt: string | null
  stage: Stage | null
  sessionLabel: string | null
  knownSessions: KnownSessions | null
  race: RaceInfo | null
  hasActuals: boolean
  drivers: RaceDriver[]
}

export type RaceViewState =
  | { status: 'pending'; view: null }
  | { status: 'error'; view: null }
  | { status: 'ready'; view: RaceView }

/** The four models predict independently, so their raw gaps can disagree with
 *  the finishing order. Handing the sorted gaps out in rank order, re-anchored
 *  on the smallest, makes "gap to P1" monotone in rank and zero for the leader.
 *  Cars without a gap (retirements have none) keep null. `raw` is in rank order. */
function gapsByRank(raw: Array<number | null>): Array<number | null> {
  const sorted = raw.filter((g): g is number => g !== null).sort((a, b) => a - b)
  let next = 0
  return raw.map((g) => (g === null ? null : sorted[next++] - sorted[0]))
}

function ranksBy<T>(items: T[], value: (item: T) => number | null): number[] {
  const order = items.map((item, i) => ({ i, v: value(item) })).sort((a, b) => (a.v ?? Infinity) - (b.v ?? Infinity) || a.i - b.i)
  const ranks = new Array<number>(items.length)
  order.forEach(({ i }, r) => (ranks[i] = r + 1))
  return ranks
}

function byPredictedFinish<T>(items: T[], finish: (item: T) => number | null): T[] {
  return items
    .map((item, index) => ({ item, index, value: finish(item) }))
    .sort((a, b) => (a.value ?? Infinity) - (b.value ?? Infinity) || a.index - b.index)
    .map(({ item }) => item)
}

const pctToSeconds = (pct: number | null | undefined, total: number | null | undefined) =>
  pct === null || pct === undefined || !total ? null : (pct / 100) * total

function fromForecast(p: PredictionPayload, isLatest: boolean, actuals: Map<string, number | null>): RaceView {
  const ordered = byPredictedFinish(p.drivers, (d) => d.predicted_finish_position)
  const gaps = gapsByRank(ordered.map((d) => d.predicted_race_time_gap))
  const qualiRanks = ranksBy(ordered, (d) => d.predicted_qualifying_gap_pct ?? d.predicted_qualifying_gap)
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
      qualiRank: qualiRanks[i],
      qualiGap: d.quali_gap_to_pole ?? d.predicted_qualifying_gap,
      qualiGapIsEstimate: d.quali_gap_to_pole === null,
      qualiGapPredicted: d.predicted_qualifying_gap,
      qualiLap: d.predicted_quali_lap_s ?? null,
      pole: d.quali_odds?.pole ?? null,
      q3: d.quali_odds?.q3 ?? null,
      q1Out: d.quali_odds?.q1_out ?? null,
      win: d.probabilities?.win ?? null,
      podium: d.probabilities?.podium ?? null,
      top10: d.probabilities?.top10 ?? null,
      band: d.position_band ?? null,
      expected: d.expected_position ?? null,
      retireRisk: d.retire_risk ?? null,
      beatsTeammate: d.beats_teammate ?? null,
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
    eventName: p.event_name ?? null,
    isLatest,
    generatedAt: p.generated_at,
    stage: p.stage ?? null,
    sessionLabel: p.session_label ?? null,
    knownSessions: p.known_sessions,
    race: p.race ?? null,
    hasActuals: actuals.size > 0,
    drivers,
  }
}

function fromBacktest(b: BacktestRacePayload): RaceView {
  const ordered = byPredictedFinish(b.drivers, (d) => d.finish_position.predicted)
  const gaps = gapsByRank(ordered.map((d) => pctToSeconds(d.race_time.predicted, b.race_duration_s)))
  const qualiRanks = ranksBy(ordered, (d) => d.qualifying.predicted)
  const minPredQuali = Math.min(...ordered.map((d) => d.qualifying.predicted ?? Infinity))
  const drivers = ordered.map((d, i): RaceDriver => {
    const rank = i + 1
    const { actual: finish } = d.finish_position
    const { actual: change } = d.quali_delta
    const grid = finish !== null && change !== null ? Math.round(finish + change) : null // delta = grid - finish
    const predGap = d.qualifying.predicted === null ? null : pctToSeconds(d.qualifying.predicted - minPredQuali, b.pole_time_s)
    const realGap = pctToSeconds(d.qualifying.actual, b.pole_time_s)
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
      qualiRank: qualiRanks[i],
      qualiGap: realGap ?? predGap,
      qualiGapIsEstimate: realGap === null,
      qualiGapPredicted: predGap,
      qualiLap: null,
      pole: null,
      q3: null,
      q1Out: null,
      win: null,
      podium: null,
      top10: null,
      band: null,
      expected: null,
      retireRisk: null,
      beatsTeammate: null,
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
    eventName: null,
    isLatest: false,
    generatedAt: null,
    stage: null,
    sessionLabel: null,
    knownSessions: null,
    race: null,
    hasActuals: true,
    drivers,
  }
}

const isRealFailure = (q: { isError: boolean; error: unknown }) => q.isError && !(q.error instanceof ApiError && q.error.status === 404)

/** One selected race, whichever source has it: the live payload if it is the
 *  latest, else a stored forecast for that round, else the walk-forward
 *  archive. Actual results come from the archive whenever the race is decided. */
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
        mode: 'none', season, round, location: '', eventName: null, isLatest, generatedAt: null, stage: null,
        sessionLabel: null, knownSessions: null, race: null, hasActuals: false, drivers: [],
      },
    }
  }, [sel, season, round, isLatest, latest, cached, archive])
}
