import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { api } from '@/lib/api'
import { useModelCard } from '@/lib/queries'
import type { RaceDriver, RaceView } from '@/lib/raceView'

const pct = (p: number | null | undefined, digits = 0) => (p == null ? '—' : p > 0 && p < 0.005 ? '<1%' : `${(p * 100).toFixed(digits)}%`)

/** Each finishing place's chance, P1 first: the forecast as a distribution, not one decimal. */
export function FinishChances({ me }: { me: RaceDriver }) {
  const dist = me.finishChances
  if (!dist?.length) return null
  const top = Math.max(...dist)
  const likeliest = dist.indexOf(top) + 1
  return (
    <section aria-label={`${me.code}'s chances by finishing place`} className="pw-glass flex min-w-0 flex-[1_1_420px] flex-col gap-3 rounded-[10px] px-[22px] py-5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <h2 className="pw-label">Chances by finishing place</h2>
        <span className="text-[12px] text-ink-3">
          Likeliest: P{likeliest} ({pct(top)}).{me.retireRisk != null && ` A retirement counts as last (${pct(me.retireRisk)} chance).`}
        </span>
      </div>
      <div className="flex h-[92px] items-end gap-[3px]" aria-hidden>
        {dist.map((p, i) => (
          <div key={i} className="flex min-w-0 flex-1 flex-col items-center gap-1">
            <div className="w-full rounded-t-[2px] bg-signal/80" style={{ height: `${Math.max(2, (p / top) * 72)}px`, opacity: 0.35 + 0.65 * (p / top) }} />
            <span className="text-[9px] leading-none text-ink-3 tabular-nums">{i % 5 === 0 || i === dist.length - 1 ? i + 1 : ''}</span>
          </div>
        ))}
      </div>
      <p className="sr-only">{dist.map((p, i) => `P${i + 1} ${pct(p, 1)}`).join(', ')}</p>
    </section>
  )
}

/** Move the driver's grid slot and see the odds change. The models already ran for every slot; this only re-samples. */
export function WhatIf({ view, me }: { view: RaceView; me: RaceDriver }) {
  const field = view.drivers.length
  const [slot, setSlot] = useState<number>(me.grid ?? Math.min(field, Math.max(1, Math.round(me.qualiRank))))
  const q = useQuery({
    queryKey: ['whatif', view.season, view.round, me.code, slot],
    queryFn: () => api.whatIf(view.season, view.round, me.code, slot),
    staleTime: Infinity,
    retry: false,
  })
  if (q.isError) return null // forecasts published before the what-if tables have none
  const now = q.data?.current
  const then = q.data?.what_if
  const row = (label: string, a?: number, b?: number) => (
    <tr className="border-t border-white/[.07]">
      <th scope="row" className="py-1.5 pr-3 text-left font-medium text-ink-2">{label}</th>
      <td className="py-1.5 pr-3 tabular-nums text-ink-3">{pct(a)}</td>
      <td className="py-1.5 tabular-nums font-semibold">{pct(b)}</td>
    </tr>
  )
  const from = q.data?.current_grid
  const swing = now && then ? Math.round((then.win - now.win) * 100) : null
  return (
    <section aria-label="What if the grid were different" className="pw-glass flex min-w-0 flex-[1_1_320px] flex-col gap-3 rounded-[10px] px-[22px] py-5">
      <h2 className="pw-label">What if</h2>
      <label className="flex flex-col gap-2 text-[13px] text-ink-2">
        <span>
          {me.code} starts <span className="font-headline text-[26px] font-extrabold leading-none text-ink">P{slot}</span>
        </span>
        <input type="range" min={1} max={field} value={slot} onChange={(e) => setSlot(Number(e.target.value))} className="accent-[var(--color-signal)]" />
      </label>
      <table className="w-full text-[13px]" aria-live="polite">
        <thead className="text-[10px] uppercase tracking-[.12em] text-ink-3">
          <tr>
            <th className="py-1 text-left font-medium" />
            <th className="py-1 pr-3 text-left font-medium">Now</th>
            <th className="py-1 text-left font-medium">From P{slot}</th>
          </tr>
        </thead>
        <tbody>
          {row('Win', now?.win, then?.win)}
          {row('Podium', now?.podium, then?.podium)}
          {row('Points', now?.points, then?.points)}
        </tbody>
      </table>
      {swing !== null && (
        <p className="font-editorial text-[17px] italic leading-snug text-ink-2">
          {from
            ? `Starting P${slot} instead of P${from} ${swing === 0 ? 'barely changes' : swing > 0 ? `adds ${swing} points of` : `costs ${-swing} points of`} win chance.`
            : // before qualifying "now" still allows for qualifying going well or badly; a settled slot does not
              `A settled P${slot} gives ${pct(then?.win)} to win. The ${pct(now?.win)} now still allows for qualifying going either way.`}
        </p>
      )}
      {q.data?.note && <p className="text-[11px] text-ink-3">{q.data.note}</p>}
    </section>
  )
}

/** "When we said 30%, it happened 28% of the time": the win odds after qualifying on races the model never saw. */
export function OddsCalibration() {
  const card = useModelCard()
  const check = card.data?.odds?.post_quali
  if (!check?.win_reliability?.length) return null
  return (
    <section aria-label="How honest the odds are" className="pw-glass flex flex-col gap-3 rounded-[10px] px-[22px] py-5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <h2 className="pw-label">How honest the win odds are</h2>
        <span className="text-[12px] text-ink-3">After qualifying, {check.held_out_races} races the model never saw</span>
      </div>
      <table className="w-full text-left text-[13px] tabular-nums">
        <thead className="text-[10px] uppercase tracking-[.12em] text-ink-3">
          <tr>
            <th className="py-1.5 pr-3 font-medium">We said</th>
            <th className="py-1.5 pr-3 font-medium">It happened</th>
            <th className="py-1.5 font-medium">Drivers</th>
          </tr>
        </thead>
        <tbody>
          {check.win_reliability.map((r) => (
            <tr key={r.stated} className="border-t border-white/[.07]">
              <td className="py-1.5 pr-3">
                {pct(r.mean_stated)} <span className="text-[11px] text-ink-3">({r.stated})</span>
              </td>
              <td className="py-1.5 pr-3 font-semibold">{pct(r.happened)}</td>
              <td className="py-1.5 text-ink-3">{r.n}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}
