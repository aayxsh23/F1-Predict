import { createContext, useCallback, useContext, useMemo, type ReactNode } from 'react'
import { useSearchParams } from 'react-router-dom'

import { useSeasonCalendar, type CalendarRace } from './calendar'
import { useBacktestRaces, usePredictionsLatest } from './queries'
import { TARGETS, type Target } from './types'

export type Tab = 'predictor' | 'strategist' | 'shap' | 'rules'
export const TAB_ORDER: Tab[] = ['predictor', 'strategist', 'shap', 'rules']

export interface SelectedRace {
  season: number
  round: number
  circuitId: string
}

interface SelectionValue {
  selectedRace: SelectedRace | null
  season: number
  seasons: number[]
  calendar: CalendarRace[]
  calendarPending: boolean
  calendarDegraded: boolean
  tab: Tab
  driver: string | null // null = follow the predicted winner
  target: Target
  doc: string | null
  selectRace: (round: number, season?: number) => void
  selectSeason: (season: number) => void
  stepRace: (direction: -1 | 1) => void
  setTab: (tab: Tab) => void
  setTarget: (target: Target) => void
  setDoc: (filename: string | null) => void
  openDriver: (code: string, tab?: Tab) => void
  openDoc: (filename: string) => void
}

const SelectionContext = createContext<SelectionValue | null>(null)

const isTab = (v: string | null): v is Tab => TAB_ORDER.includes(v as Tab)
const isTarget = (v: string | null): v is Target => TARGETS.includes(v as Target)

/** The whole workbench's state lives in the URL (?season&round&tab&driver&target&doc),
 *  so every view is linkable and the old /race/... /regulations links can redirect
 *  into it. Defaults resolve from the latest forecast, then the calendar. */
export function SelectionProvider({ children }: { children: ReactNode }) {
  const [params, setParams] = useSearchParams()
  const latest = usePredictionsLatest()
  const archive = useBacktestRaces()

  const season = Number(params.get('season')) || latest.data?.season || new Date().getFullYear()
  const { races: calendar, isPending: calendarPending, degraded: calendarDegraded } = useSeasonCalendar(season)

  const seasons = useMemo(() => {
    const all = new Set((archive.data ?? []).map((r) => r.season))
    if (latest.data) all.add(latest.data.season)
    all.add(season)
    return [...all].sort((a, b) => a - b)
  }, [archive.data, latest.data, season])

  const defaultRound = useMemo(() => {
    if (latest.isPending) return null // don't open on the finale and then jump
    if (latest.data?.season === season) return latest.data.round
    // a decided season opens on its finale; a season with nothing decided on round 1
    return calendar.length ? calendar[calendar.length - 1].round : null
  }, [latest.isPending, latest.data, season, calendar])

  const round = Number(params.get('round')) || defaultRound
  const circuitId = calendar.find((r) => r.round === round)?.circuitId ?? ''

  const update = useCallback(
    (patch: Record<string, string | null>) =>
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev)
          for (const [key, value] of Object.entries(patch)) {
            if (value === null) next.delete(key)
            else next.set(key, value)
          }
          return next
        },
        { replace: true },
      ),
    [setParams],
  )

  const value = useMemo<SelectionValue>(() => {
    const tabParam = params.get('tab')
    const targetParam = params.get('target')
    return {
      selectedRace: round === null ? null : { season, round, circuitId },
      season,
      seasons,
      calendar,
      calendarPending,
      calendarDegraded,
      tab: isTab(tabParam) ? tabParam : 'predictor',
      driver: params.get('driver'),
      target: isTarget(targetParam) ? targetParam : 'finish_position',
      doc: params.get('doc'),
      selectRace: (next, nextSeason) => update({ round: String(next), season: String(nextSeason ?? season), driver: null }),
      selectSeason: (next) => update({ season: String(next), round: null, driver: null }),
      stepRace: (direction) => {
        const i = calendar.findIndex((r) => r.round === round)
        const target = calendar[i + direction]
        if (i >= 0 && target) update({ round: String(target.round), season: String(season), driver: null })
      },
      setTab: (tab) => update({ tab }),
      setTarget: (target) => update({ target }),
      setDoc: (filename) => update({ doc: filename }),
      openDriver: (code, tab) => update({ driver: code, ...(tab ? { tab } : {}) }),
      openDoc: (filename) => update({ doc: filename, tab: 'rules' }),
    }
  }, [params, season, seasons, calendar, calendarPending, calendarDegraded, round, circuitId, update])

  return <SelectionContext.Provider value={value}>{children}</SelectionContext.Provider>
}

export function useSelection(): SelectionValue {
  const ctx = useContext(SelectionContext)
  if (!ctx) throw new Error('useSelection must be used inside a SelectionProvider')
  return ctx
}
