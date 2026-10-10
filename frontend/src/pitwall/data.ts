import { useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'

import { useChampionship } from '@/lib/queries'
import type { RaceDriver } from '@/lib/raceView'
import { TARGET_HIGHER_IS_BETTER, TARGETS, type KnownSessions, type Target } from '@/lib/types'

/** "+2.53" / "−0.92": a real minus sign so signed columns line up. */
export const signed = (v: number, decimals: number) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(decimals)

export const orDash = (v: number | null | undefined, fmt: (v: number) => string) => (v === null || v === undefined ? '—' : fmt(v))

export interface Lens {
  id: Target
  label: string // the tab
  title: string // the column and field heading
  value: (d: RaceDriver) => number | null
  fmt: (v: number) => string
}

// The four targets, in the order a weekend answers them. The gaps are the
// model's own unit (% of the pole lap / the winner's race time): seconds only
// exist once the session has run, a percentage exists from Monday.
export const LENSES: Lens[] = [
  { id: 'qualifying', label: 'Quali gap', title: 'Quali gap to pole', value: (d) => d.qualiGapPct, fmt: (v) => `${v.toFixed(2)}%` },
  { id: 'finish_position', label: 'Finish', title: 'Finishing position', value: (d) => d.modelFinish, fmt: (v) => v.toFixed(1) },
  { id: 'quali_delta', label: 'Grid → finish', title: 'Grid-to-finish change', value: (d) => d.modelDelta, fmt: (v) => signed(v, 2) },
  { id: 'race_time', label: 'Time gap', title: 'Gap to the winner', value: (d) => d.raceGapPct, fmt: (v) => `${v.toFixed(2)}%` },
]

export const lensOf = (id: Target) => LENSES.find((l) => l.id === id)!

export const betterWord = (t: Target) => (TARGET_HIGHER_IS_BETTER[t] ? 'higher is better' : 'lower is better')

/** The lens is URL state (?lens=quali_delta), so a shared link opens the same framing. */
export function useLens(): [Target, (t: Target) => void] {
  const [params, setParams] = useSearchParams()
  const raw = params.get('lens')
  const lens = TARGETS.includes(raw as Target) ? (raw as Target) : 'finish_position'
  const set = (t: Target) =>
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev)
        next.set('lens', t)
        return next
      },
      { replace: true },
    )
  return [lens, set]
}

/** Drivers ordered best-first under a lens; anyone the lens has no number for goes last. */
export function rankBy(drivers: RaceDriver[], lens: Lens): RaceDriver[] {
  const hib = TARGET_HIGHER_IS_BETTER[lens.id]
  const key = (d: RaceDriver) => {
    const v = lens.value(d)
    return v === null ? Infinity : hib ? -v : v
  }
  return [...drivers].sort((a, b) => key(a) - key(b) || a.rank - b.rank)
}

/** 0 = best, 1 = worst, across the field's own range, so "left is better" holds for every lens. */
export function normaliser(drivers: RaceDriver[], lens: Lens): (v: number) => number {
  const vals = drivers.map(lens.value).filter((v): v is number => v !== null)
  const mn = Math.min(...vals)
  const mx = Math.max(...vals)
  const hib = TARGET_HIGHER_IS_BETTER[lens.id]
  return (v) => {
    const t = (v - mn) / (mx - mn || 1)
    return hib ? 1 - t : t
  }
}

export function lensRange(drivers: RaceDriver[], lens: Lens): [number, number] {
  const vals = drivers.map(lens.value).filter((v): v is number => v !== null)
  return [Math.min(...vals), Math.max(...vals)]
}

// --- the weekend's information state ---

type Node = { key: keyof KnownSessions; label: string; title: string; copy: string }

const PRACTICE: Node = { key: 'practice', label: 'Practice', title: 'Practice', copy: 'Practice pace is in; the compressed pack starts to separate.' }
const SPRINT: Node[] = [
  { key: 'sprint_qualifying', label: 'Sprint quali', title: 'Sprint quali done', copy: 'One-lap pace from Sprint Qualifying is in: the best read on Saturday so far.' },
  { key: 'sprint', label: 'Sprint', title: 'Sprint done', copy: 'A real race result is in: race pace and racecraft, not just practice laps.' },
]
const LATER: Node[] = [
  { key: 'qualifying', label: 'Qualifying', title: 'Qualifying done', copy: 'Qualifying gaps are known, the single biggest sharpening of the weekend. The qualifying order stands in for the grid until the official one is out.' },
  { key: 'grid', label: 'Official grid', title: 'Grid set', copy: "The FIA's official grid, penalties applied: as informed as this prediction gets." },
]

/** The weekend's information steps; a sprint weekend has two more (the payload only sends their keys then). */
export function weekendNodes(known: KnownSessions): Node[] {
  return [PRACTICE, ...('sprint' in known ? SPRINT : []), ...LATER]
}

export function weekendPhase(known: KnownSessions) {
  const nodes = weekendNodes(known)
  const count = nodes.filter((n) => known[n.key]).length
  const next = nodes.findIndex((n) => !known[n.key])
  const last = [...nodes].reverse().find((n) => known[n.key])
  const sprint = 'sprint' in known
  const { title, copy } = last ?? {
    title: 'Pre-weekend',
    copy: sprint
      ? 'Sprint weekend: one practice, then Sprint Qualifying and the Sprint. Built from rolling form and circuit history until they run.'
      : 'Less informed yet: built from rolling form and circuit history alone.',
  }
  return { nodes, count, next, title, copy }
}

// --- circuit facts, straight from the forecast's feature row ---

const num = (row: Record<string, number | string | null>, key: string) => (typeof row[key] === 'number' ? (row[key] as number) : null)

export function circuitSpecs(row: Record<string, number | string | null> | null) {
  if (!row) return []
  const specs: Array<{ k: string; v: string; u: string }> = []
  const add = (k: string, v: number | null, fmt: (v: number) => string, u: string) => v !== null && specs.push({ k, v: fmt(v), u })
  add('Track', num(row, 'track_length_km'), (v) => v.toFixed(3), 'km')
  add('Longest straight', num(row, 'longest_straight_m'), (v) => Math.round(v).toLocaleString('en'), 'm')
  add('Safety car', num(row, 'safety_car_frequency'), (v) => String(Math.round(v * 100)), '%')
  add('DNF rate', num(row, 'dnf_rate'), (v) => String(Math.round(v * 100)), '%')
  add('Pit lane loss', num(row, 'pit_lane_loss_time'), (v) => v.toFixed(1), 's')
  add('Braking zones', num(row, 'braking_zone_count'), (v) => String(Math.round(v)), '')
  return specs
}

export const isStreet = (row: Record<string, number | string | null> | null) => !!row && num(row, 'is_street_circuit') === 1

/** A driver's form inputs, as the row disclosure shows them. */
export function driverInputs(row: Record<string, number | string | null> | null, location: string) {
  if (!row) return []
  const p = (v: number | null) => orDash(v, (x) => `P${x.toFixed(1)}`)
  const pct = (v: number | null) => orDash(v, (x) => `${Math.round(x * 100)}%`)
  return [
    { k: 'Recent form', v: p(num(row, 'driver_recent_form')) },
    { k: `Form at ${location}`, v: p(num(row, 'driver_track_form')) },
    { k: 'Places gained', v: orDash(num(row, 'driver_positions_gained_form'), (x) => signed(x, 1)) },
    { k: 'DNF rate', v: pct(num(row, 'driver_dnf_rate')) },
    { k: 'Reliability', v: pct(num(row, 'team_reliability')) },
  ]
}

/** Which of the driver's own sessions the model has seen: grid, qualifying, practice. */
export function driverKnown(d: RaceDriver) {
  const row = d.featureRow
  return [
    { k: 'Grid', on: d.grid !== null },
    { k: 'Qualifying', on: !!row && num(row, 'quali_gap_pct') !== null },
    { k: 'Practice', on: !!row && num(row, 'practice_pace') !== null },
  ]
}

/** "13 Sep · 13:02 UTC" */
export function snapshotLabel(iso: string): string {
  const d = new Date(iso)
  const month = d.toLocaleString('en', { month: 'short', timeZone: 'UTC' })
  const time = d.toLocaleString('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'UTC' })
  return `${d.getUTCDate()} ${month} · ${time} UTC`
}

export const raceHref = (season: number, round: number, lens?: Target) =>
  `/race/${season}/${round}${lens && lens !== 'finish_position' ? `?lens=${lens}` : ''}`

export const driverHref = (season: number, round: number, code: string, lens?: Target) =>
  `/race/${season}/${round}/driver/${code}${lens && lens !== 'finish_position' ? `?lens=${lens}` : ''}`

/** Driver code -> surname, from the live standings. The forecast only carries codes;
 *  when the standings aren't reachable the code alone stands in. */
export function useDriverNames(): Map<string, string> {
  const champ = useChampionship()
  return useMemo(() => new Map((champ.data?.drivers ?? []).map((d) => [d.code, d.family_name])), [champ.data])
}

/** "the Bahrain Grand Prix" / "Montréal" / "this race", to drop into a question. */
export function placePhrase(name: string | null | undefined): string {
  if (!name) return 'this race'
  return /grand prix$/i.test(name) ? `the ${name}` : name
}
