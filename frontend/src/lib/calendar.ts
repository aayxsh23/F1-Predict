import { useMemo } from 'react'

import { useBacktestRaces, usePredictionsLatest, useRaces } from './queries'
import type { BacktestIndexEntry, RaceSummary } from './types'

export type RaceStatus = 'completed' | 'live' | 'upcoming'

export interface CalendarRace {
  season: number
  round: number
  name: string // "AZERBAIJAN", the Grand Prix without the suffix
  eventName: string // "Azerbaijan Grand Prix"
  location: string // "Baku"
  plate: string // "AZE", three letters like a timing screen; '' when unknown
  sprint: boolean
  dateLabel: string // "24-26 SEP"
  status: RaceStatus
  circuitId: string
}

const DAY = 86_400_000

// FIA/timing-screen style three-letter country codes. Flags are art the repo
// doesn't have; a plate reads like the timing screens this UI borrows from.
const PLATES: Record<string, string> = {
  Australia: 'AUS',
  Austria: 'AUT',
  Azerbaijan: 'AZE',
  Bahrain: 'BHR',
  Belgium: 'BEL',
  Brazil: 'BRA',
  Canada: 'CAN',
  China: 'CHN',
  France: 'FRA',
  Germany: 'GER',
  Hungary: 'HUN',
  Italy: 'ITA',
  Japan: 'JPN',
  Mexico: 'MEX',
  Monaco: 'MON',
  Netherlands: 'NED',
  Portugal: 'POR',
  Qatar: 'QAT',
  'Saudi Arabia': 'KSA',
  Singapore: 'SGP',
  Spain: 'ESP',
  'United Arab Emirates': 'UAE',
  'United Kingdom': 'GBR',
  'United States': 'USA',
}

export function slug(text: string): string {
  return text
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
}

// the API sends naive UTC timestamps ("2026-09-24T08:30:00"); parse them as UTC
function toMs(iso: string | null): number | null {
  if (!iso) return null
  const ms = Date.parse(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`)
  return Number.isNaN(ms) ? null : ms
}

function dateLabel(start: number | null, raceDay: number | null): string {
  if (raceDay === null) return ''
  const fmt = (ms: number, part: 'day' | 'month') =>
    part === 'day'
      ? String(new Date(ms).getUTCDate()).padStart(2, '0')
      : new Date(ms).toLocaleString('en', { month: 'short', timeZone: 'UTC' }).toUpperCase()
  const first = start ?? raceDay
  return fmt(first, 'month') === fmt(raceDay, 'month')
    ? `${fmt(first, 'day')}–${fmt(raceDay, 'day')} ${fmt(raceDay, 'month')}`
    : `${fmt(first, 'day')} ${fmt(first, 'month')}–${fmt(raceDay, 'day')} ${fmt(raceDay, 'month')}`
}

function statusOf(start: number | null, raceDay: number | null, now: number): RaceStatus {
  if (raceDay === null) return 'upcoming'
  if (now >= raceDay + DAY) return 'completed' // EventDate is 00:00 UTC of race day
  return start !== null && now >= start ? 'live' : 'upcoming'
}

function fromSchedule(season: number, r: RaceSummary, now: number): CalendarRace {
  const start = toMs(r.Session1DateUtc)
  const raceDay = toMs(r.EventDate)
  return {
    season,
    round: r.RoundNumber,
    name: r.EventName.replace(/\s+Grand Prix$/i, '').toUpperCase(),
    eventName: r.EventName,
    location: r.Location,
    plate: PLATES[r.Country] ?? r.Country.slice(0, 3).toUpperCase(),
    sprint: r.EventFormat.startsWith('sprint'),
    dateLabel: dateLabel(start, raceDay),
    status: statusOf(start, raceDay, now),
    circuitId: slug(r.Location),
  }
}

// /races needs a live FastF1 schedule call. When it fails, rebuild what the
// archive knows (decided races, plus the current forecast) rather than an empty ribbon.
function fromArchive(season: number, r: { round: number; location: string }, status: RaceStatus): CalendarRace {
  return {
    season,
    round: r.round,
    name: r.location.toUpperCase(),
    eventName: r.location,
    location: r.location,
    plate: '',
    sprint: false,
    dateLabel: '',
    status,
    circuitId: slug(r.location),
  }
}

export function useSeasonCalendar(season: number): { races: CalendarRace[]; isPending: boolean; degraded: boolean } {
  const schedule = useRaces(season)
  const archive = useBacktestRaces()
  const latest = usePredictionsLatest()

  return useMemo(() => {
    if (schedule.data) {
      const now = Date.now()
      return { races: schedule.data.map((r) => fromSchedule(season, r, now)), isPending: false, degraded: false }
    }
    if (schedule.isPending) return { races: [], isPending: true, degraded: false }

    const decided: BacktestIndexEntry[] = (archive.data ?? []).filter((r) => r.season === season)
    const races = decided.map((r) => fromArchive(season, r, 'completed'))
    const live = latest.data
    if (live?.season === season && !decided.some((r) => r.round === live.round)) {
      races.push(fromArchive(season, { round: live.round, location: live.location }, 'upcoming'))
    }
    return { races: races.sort((a, b) => a.round - b.round), isPending: false, degraded: true }
  }, [schedule.data, schedule.isPending, archive.data, latest.data, season])
}
