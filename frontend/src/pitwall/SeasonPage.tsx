import { useState } from 'react'

import { cn } from '@/lib/cn'
import { formatPct } from '@/lib/format'
import { useChampionship, useRatings } from '@/lib/queries'
import type { Ratings } from '@/lib/types'

import { useChrome } from './chrome'
import { Offline, PAGE, Skeleton } from './parts'

function Chance({ p, best }: { p: number; best: number }) {
  return (
    <td className="relative py-2 pl-3 text-right font-mono tabular-nums">
      {best > 0 && p > 0 && (
        <span className="absolute inset-y-1.5 right-0 rounded-[3px] bg-signal/20" style={{ width: `${(p / best) * 100}%` }} aria-hidden />
      )}
      <span className="relative">{p === 0 ? '<0.1%' : formatPct(p)}</span>
    </td>
  )
}

const th = 'pb-2 text-[10px] font-medium uppercase tracking-[.12em] text-ink-3'

type View = 'drivers' | 'constructors' | 'ratings'
const VIEWS: Array<[View, string]> = [
  ['drivers', 'Drivers'],
  ['constructors', 'Constructors'],
  ['ratings', 'Driver ratings'],
]

/** Who is fastest in the same car: every past teammate gap fitted at once. */
function RatingsTable({ ratings }: { ratings: Ratings }) {
  const fastest = Math.min(...ratings.drivers.map((d) => d.rating_pct))
  return (
    <>
      <table className="w-full min-w-[480px] text-[14px]">
        <thead>
          <tr className="text-left">
            <th className={th}>Rank</th>
            <th className={th}>Driver</th>
            <th className={th}>Team</th>
            <th className={cn(th, 'text-right')}>vs an average driver</th>
            <th className={cn(th, 'text-right')}>vs teammate</th>
          </tr>
        </thead>
        <tbody>
          {ratings.drivers.map((d) => (
            <tr key={d.driver} className="border-t border-white/[.07]">
              <td className="py-2 font-mono tabular-nums text-ink-3">{d.rank}</td>
              <td className="py-2 font-semibold">{d.driver}</td>
              <td className="py-2 text-ink-2">{d.team}</td>
              <td className="relative py-2 pl-3 text-right font-mono tabular-nums">
                {d.rating_pct < 0 && (
                  <span className="absolute inset-y-1.5 right-0 rounded-[3px] bg-gain/20" style={{ width: `${(d.rating_pct / fastest) * 100}%` }} aria-hidden />
                )}
                <span className="relative">
                  {d.rating_pct > 0 ? '+' : ''}
                  {d.rating_pct.toFixed(2)}%
                </span>
              </td>
              <td className="py-2 text-right font-mono tabular-nums text-ink-2">
                {d.vs_teammate_pct === null ? '—' : `${d.vs_teammate_pct > 0 ? '+' : ''}${d.vs_teammate_pct.toFixed(2)}%`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-3 text-[12px] text-ink-3">
        One-lap pace as a share of a lap, negative is faster, from every past qualifying gap between teammates (measured in the last session both ran),
        recent races counting more. Teammates share a car, so the gap between them is the cleanest measure of the drivers; a driver who changes team
        keeps his record. Rated before {ratings.event_name}.
      </p>
    </>
  )
}

/** The title race: live standings and each contender's chance, from simulating
 *  the rest of the season with every driver's form (not this weekend's grid). */
export function SeasonPage() {
  const q = useChampionship()
  const ratings = useRatings()
  const [kind, setKind] = useState<View>('drivers')

  useChrome({
    section: 'season',
    crumbs: [{ label: 'Title race' }],
    chat: {},
    badge: 'Title race',
    quick: ['What does the leader need to win the title?', 'Who is leading the constructors championship?'],
  })

  if (q.isPending) return <Skeleton label="Simulating the rest of the season…" />
  if (q.isError || !q.data) return <Offline />
  const { drivers, constructors, remaining } = q.data
  const bestDriver = Math.max(0, ...drivers.map((d) => d.title_chance))
  const bestTeam = Math.max(0, ...constructors.map((c) => c.title_chance))

  return (
    <div className={cn(PAGE, 'gap-6 pt-7')}>
      <div className="flex flex-wrap items-end justify-between gap-x-8 gap-y-4">
        <div className="flex max-w-[760px] flex-col gap-3.5">
          <h1 className="-ml-1 font-headline text-[clamp(72px,10vw,150px)] font-black uppercase leading-[.8]">Title race</h1>
          <p className="font-editorial text-[clamp(20px,2.2vw,26px)] italic leading-[1.3] text-ink-2 [text-wrap:pretty]">
            Who can still win it, and how likely, from simulating the rest of the season 5,000 times with every driver's current form.
          </p>
        </div>
        <span className="pw-label text-right leading-[1.6]">
          {remaining.races} race{remaining.races === 1 ? '' : 's'}
          {remaining.sprints ? ` and ${remaining.sprints} sprint${remaining.sprints === 1 ? '' : 's'}` : ''} left
        </span>
      </div>

      <div role="group" aria-label="Championship" className="flex w-max gap-1 rounded-[7px] border border-white/10 p-[3px]">
        {VIEWS.filter(([k]) => k !== 'ratings' || ratings.data).map(([k, label]) => (
          <button
            key={k}
            type="button"
            aria-pressed={kind === k}
            onClick={() => setKind(k)}
            className={cn(
              'h-8 rounded-[5px] px-[11px] text-[11px] font-semibold uppercase leading-none tracking-[.06em]',
              kind === k ? 'bg-white/[.09] text-ink' : 'text-ink-3 hover:text-ink',
            )}
          >
            {label}
          </button>
        ))}
      </div>

      {/* focusable so a keyboard user can scroll the table sideways on a phone */}
      <section
        tabIndex={0}
        className="pw-glass overflow-x-auto rounded-[10px] px-5 py-4"
        aria-label={kind === 'drivers' ? "Drivers' championship" : kind === 'constructors' ? "Constructors' championship" : 'Driver ratings'}
      >
        {kind === 'ratings' && ratings.data ? (
          <RatingsTable ratings={ratings.data} />
        ) : kind === 'drivers' ? (
          <table className="w-full min-w-[560px] text-[14px]">
            <thead>
              <tr className="text-left">
                <th className={th}>Pos</th>
                <th className={th}>Driver</th>
                <th className={th}>Team</th>
                <th className={cn(th, 'text-right')}>Points</th>
                <th className={cn(th, 'text-right')}>Wins</th>
                <th className={cn(th, 'text-right')}>Expected final</th>
                <th className={cn(th, 'text-right')}>Title chance</th>
              </tr>
            </thead>
            <tbody>
              {drivers.map((d) => (
                <tr key={d.code} className="border-t border-white/[.07]">
                  <td className="py-2 font-mono tabular-nums text-ink-3">{d.position}</td>
                  <td className="py-2 font-semibold">
                    {d.given_name} {d.family_name} <span className="text-ink-3">{d.code}</span>
                  </td>
                  <td className="py-2 text-ink-2">{d.team ?? '—'}</td>
                  <td className="py-2 text-right font-mono tabular-nums">{d.points.toFixed(0)}</td>
                  <td className="py-2 text-right font-mono tabular-nums">{d.wins}</td>
                  <td className="py-2 text-right font-mono tabular-nums text-ink-2">{d.expected_points.toFixed(0)}</td>
                  <Chance p={d.title_chance} best={bestDriver} />
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <table className="w-full min-w-[420px] text-[14px]">
            <thead>
              <tr className="text-left">
                <th className={th}>Pos</th>
                <th className={th}>Team</th>
                <th className={cn(th, 'text-right')}>Points</th>
                <th className={cn(th, 'text-right')}>Wins</th>
                <th className={cn(th, 'text-right')}>Title chance</th>
              </tr>
            </thead>
            <tbody>
              {constructors.map((c) => (
                <tr key={c.name} className="border-t border-white/[.07]">
                  <td className="py-2 font-mono tabular-nums text-ink-3">{c.position}</td>
                  <td className="py-2 font-semibold">{c.name}</td>
                  <td className="py-2 text-right font-mono tabular-nums">{c.points.toFixed(0)}</td>
                  <td className="py-2 text-right font-mono tabular-nums">{c.wins}</td>
                  <Chance p={c.title_chance} best={bestTeam} />
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {kind !== 'ratings' && <p className="mt-3 text-[12px] text-ink-3">
          Each driver's strength is their forecast before this weekend's practice (form, team, circuit), so one weekend's grid or penalty doesn't count in
          every remaining round. Ties on points split the chance evenly (the FIA's countback isn't modelled yet).
        </p>}
      </section>
    </div>
  )
}
