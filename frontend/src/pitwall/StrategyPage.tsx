import { useEffect, useState } from 'react'
import { Navigate, useParams } from 'react-router-dom'

import { ApiError } from '@/lib/api'
import { cn } from '@/lib/cn'
import { formatPct } from '@/lib/format'
import { usePredictionsForRace, useStrategy } from '@/lib/queries'
import type { Compound, StrategyPlan, Stint } from '@/lib/types'

import { useChrome } from './chrome'
import { placePhrase, raceHref } from './data'
import { Mega, Notice, Offline, PAGE, Skeleton } from './parts'

// Pirelli's sidewall colours
const COMPOUND: Record<Compound, { fill: string; short: string; name: string }> = {
  SOFT: { fill: 'var(--color-tyre-soft)', short: 'S', name: 'Soft' },
  MEDIUM: { fill: 'var(--color-tyre-medium)', short: 'M', name: 'Medium' },
  HARD: { fill: 'var(--color-tyre-hard)', short: 'H', name: 'Hard' },
}

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms)
    return () => clearTimeout(t)
  }, [value, ms])
  return v
}

/** One plan as stints along the race distance, pit windows marked underneath. */
export function StintBar({ stints, laps, windows = [] }: { stints: Stint[]; laps: number; windows?: Array<[number, number]> }) {
  const x = (lap: number) => `${(lap / laps) * 100}%`
  return (
    <div
      className="relative h-6 w-full"
      role="img"
      aria-label={stints.map((s) => `${COMPOUND[s.compound].name} laps ${s.from_lap} to ${s.to_lap}`).join(', ')}
    >
      <div className="flex h-full w-full gap-[2px] overflow-hidden rounded-[4px]">
        {stints.map((s) => (
          <span
            key={s.from_lap}
            className="flex items-center justify-center text-[11px] font-bold leading-none text-asphalt"
            style={{ width: `${(s.laps / laps) * 100}%`, background: COMPOUND[s.compound].fill }}
          >
            {s.laps >= 6 ? `${COMPOUND[s.compound].short} ${s.laps}` : ''}
          </span>
        ))}
      </div>
      {windows.map(([lo, hi]) => (
        <span key={lo} className="absolute -bottom-2 h-[3px] rounded-full bg-live" style={{ left: x(lo - 1), width: x(hi - lo + 1) }} aria-hidden />
      ))}
    </div>
  )
}

function PlanCard({ plan, laps, best }: { plan: StrategyPlan; laps: number; best: boolean }) {
  return (
    <li className={cn('pw-glass flex flex-col gap-3.5 rounded-[10px] px-5 py-4', best && 'shadow-[0_0_0_1px_rgb(255_90_71/.35)]')}>
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <span className={cn('text-[15px] font-semibold', best ? 'text-ink' : 'text-ink-2')}>{plan.name}</span>
        <span className="pw-label">
          {plan.time_vs_best_s === 0 ? 'Fastest on paper' : `+${plan.time_vs_best_s.toFixed(1)} s`} · fastest in {formatPct(plan.chance_fastest)} of
          simulated races
        </span>
      </div>
      <StintBar stints={plan.stints} laps={laps} windows={plan.pit_windows} />
      <p className="pt-1 text-[13px] text-ink-2">
        Pit {plan.pit_windows.map(([lo, hi], i) => `on lap ${plan.pit_laps[i]} (window ${lo}–${hi})`).join(', then ')}
      </p>
    </li>
  )
}

/** The race's tyre strategy, with a safety car you can drop on any lap. */
export function StrategyPage() {
  const params = useParams()
  const season = Number(params.season)
  const round = Number(params.round)
  const valid = Number.isFinite(season) && Number.isFinite(round)
  const forecast = usePredictionsForRace(season, round, valid)
  const [scOn, setScOn] = useState(false)
  const [scDraft, setScDraft] = useState(15)
  const scLap = useDebounced(scOn ? scDraft : null, 250)
  const q = useStrategy(season, round, scLap, valid)
  const s = q.data
  const name = forecast.data?.event_name ?? forecast.data?.location

  useChrome({
    section: 'pred',
    crumbs: [{ label: 'Full grid', to: raceHref(season, round), back: true }, { label: String(season) }, { label: `Round ${round}` }, { label: 'Strategy' }],
    chat: { season, round, race_name: name ?? undefined },
    badge: `Strategy · ${forecast.data?.location ?? `R${round}`}`,
    quick: [`What's the best strategy for ${placePhrase(name)}?`, `What changes with a safety car on lap ${scDraft}?`],
  })

  if (!valid) return <Navigate to="/" replace />
  if (q.isPending) return <Skeleton label="Running the strategy model…" />
  if (q.isError) {
    return q.error instanceof ApiError && q.error.status === 404 ? (
      <Notice title="No strategy for this race">Strategy is modelled for races with a stored forecast.</Notice>
    ) : (
      <Offline />
    )
  }
  if (!s) return <Offline />

  return (
    <div className={cn(PAGE, 'gap-6 pt-[22px]')}>
      <div className="flex flex-wrap items-end justify-between gap-x-8 gap-y-3">
        <div className="flex min-w-0 flex-[1_1_320px] items-baseline gap-[18px]">
          <div className="min-w-0 max-w-[640px] flex-1">
            <Mega text={s.circuit} max={104} />
          </div>
          <span className="shrink-0 font-editorial text-[clamp(24px,3vw,40px)] italic leading-none text-ink-2">strategy</span>
        </div>
        <p className="pw-label leading-[1.7]">
          {s.laps} laps · a pit stop costs {s.pit_loss_s.toFixed(0)} s · safety car in {formatPct(s.safety_car_probability)} of races here
          <br />
          {s.races_of_data ? `fitted on ${s.races_of_data} past race${s.races_of_data === 1 ? '' : 's'} here` : 'a new venue: all-circuit tyre figures'}
        </p>
      </div>

      <ol className={cn('flex flex-col gap-3', q.isFetching && 'opacity-60')} aria-label="Fastest plans">
        {s.strategies.slice(0, 4).map((p, i) => (
          <PlanCard key={p.name} plan={p} laps={s.laps} best={i === 0} />
        ))}
      </ol>

      <section aria-label="Safety car scenario" className="pw-glass flex flex-col gap-3.5 rounded-[10px] px-5 py-4">
        <h2 className="pw-label">What a safety car changes</h2>
        <label className="flex flex-wrap items-center gap-3 text-[14px] text-ink">
          <input type="checkbox" checked={scOn} onChange={(e) => setScOn(e.target.checked)} className="size-4 accent-[var(--color-signal)]" />
          Safety car on lap <span className="font-mono tabular-nums">{scDraft}</span>
        </label>
        <input
          type="range"
          min={1}
          max={s.laps - 1}
          value={scDraft}
          aria-label="Safety car lap"
          onChange={(e) => {
            setScDraft(Number(e.target.value))
            setScOn(true)
          }}
          className="w-full accent-[var(--color-signal)]"
        />
        {s.scenario ? (
          <div className="flex flex-col gap-3">
            <p className="text-[14px] text-ink-2">
              The best plan becomes <span className="font-semibold text-ink">{s.scenario.best}</span>, pitting on lap {s.scenario.pit_laps.join(' and ')}
              {s.scenario.pits_under_safety_car ? ' (under the safety car)' : ''}.{' '}
              {s.scenario.gain_vs_sticking_to_plan_s > 0.05
                ? `About ${s.scenario.gain_vs_sticking_to_plan_s.toFixed(1)} s better than sticking to the original plan.`
                : 'No real gain over the original plan.'}
            </p>
            <StintBar stints={s.scenario.stints} laps={s.laps} />
          </div>
        ) : (
          <p className="text-[14px] text-ink-3">Tick the box or move the slider to put a safety car on a lap.</p>
        )}
      </section>

      <section aria-label="Tyres" className="pw-glass rounded-[10px] px-5 py-4">
        <h2 className="pw-label mb-3">The tyres here</h2>
        <table className="w-full text-[14px]">
          <thead>
            <tr className="text-left text-[10px] uppercase tracking-[.12em] text-ink-3">
              <th className="pb-2 font-medium">Compound</th>
              <th className="pb-2 text-right font-medium">Pace vs medium</th>
              <th className="pb-2 text-right font-medium">Wear per lap</th>
              <th className="pb-2 text-right font-medium">Longest usual stint</th>
            </tr>
          </thead>
          <tbody>
            {(Object.keys(COMPOUND) as Compound[]).map((c) => (
              <tr key={c} className="border-t border-white/[.07]">
                <td className="py-2">
                  <span className="mr-2 inline-block size-2.5 rounded-full align-middle" style={{ background: COMPOUND[c].fill }} aria-hidden />
                  {COMPOUND[c].name}
                </td>
                <td className="py-2 text-right font-mono tabular-nums">
                  {s.compounds[c].pace_vs_medium_s > 0 ? '+' : ''}
                  {s.compounds[c].pace_vs_medium_s.toFixed(2)} s
                </td>
                <td className="py-2 text-right font-mono tabular-nums">{s.compounds[c].deg_s_per_lap.toFixed(3)} s</td>
                <td className="py-2 text-right font-mono tabular-nums">{s.compounds[c].max_stint_laps} laps</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-3 text-[12px] text-ink-3">
          Fitted on real race laps since 2022. Known limits: tyre wear is a straight line (no cliff), tyres start new and traffic is ignored, so
          this compares plans rather than predicting gaps.
        </p>
      </section>
    </div>
  )
}
