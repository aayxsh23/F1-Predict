import { motion } from 'framer-motion'
import { MessageCircleQuestion } from 'lucide-react'
import { Link, Navigate, useParams, useSearchParams } from 'react-router-dom'

import { ApiError } from '@/lib/api'
import { cn } from '@/lib/cn'
import { spring } from '@/lib/motion'
import { useBacktestForRace, useBacktestRaces, useLiveRecord } from '@/lib/queries'
import type { BacktestDriverEntry, LiveRecord, Target } from '@/lib/types'

import { useChrome, useChromeApi } from './chrome'
import { lensOf, signed, useLens } from './data'
import { LensTabs, Mega, Notice, Offline, PAGE, Skeleton } from './parts'

/** S4: every race the model has been scored on, by season. */
export function HistoryPage() {
  const archive = useBacktestRaces()
  const [params, setParams] = useSearchParams()
  const races = archive.data ?? []
  const seasons = [...new Set(races.map((r) => r.season))].sort((a, b) => b - a)
  const season = Number(params.get('season')) || seasons[0] || new Date().getFullYear()
  const inSeason = races.filter((r) => r.season === season).sort((a, b) => a.round - b.round)

  useChrome({
    section: 'hist',
    crumbs: [{ label: 'History' }, { label: String(season) }],
    chat: { season },
    badge: `History · ${season}`,
    quick: [`How accurate has the model been in ${season}?`, 'Who is leading the drivers championship?'],
  })

  if (archive.isPending) return <Skeleton label="Loading the archive…" />
  if (archive.isError) return <Offline />

  return (
    <div className={cn(PAGE, 'gap-6 pt-7')}>
      <div className="flex flex-wrap items-end justify-between gap-x-8 gap-y-4">
        <div className="flex max-w-[760px] flex-col gap-3.5">
          <h1 className="-ml-1 font-headline text-[clamp(72px,10vw,150px)] font-black uppercase leading-[.8]">Race history</h1>
          <p className="font-editorial text-[clamp(20px,2.2vw,26px)] italic leading-[1.3] text-ink-2 [text-wrap:pretty]">
            How the model's predictions compared to what actually happened, race by race. Each race is predicted by a model that had never seen it.
          </p>
        </div>
        <span className="pw-label text-right leading-[1.6]">
          {races.length} races backtested
          {seasons.length > 0 && (
            <>
              <br />
              {seasons[seasons.length - 1]} — {seasons[0]}
            </>
          )}
        </span>
      </div>

      <LiveTrackRecord />

      <div role="tablist" aria-label="Season" className="flex gap-[clamp(14px,3vw,36px)] overflow-x-auto border-b border-white/10 [scrollbar-width:none]">
        {seasons.map((y) => {
          const on = y === season
          return (
            <button
              key={y}
              type="button"
              role="tab"
              aria-selected={on}
              onClick={() => setParams({ season: String(y) }, { replace: true })}
              className={cn('relative pb-3.5 font-headline text-[clamp(34px,4vw,52px)] font-extrabold leading-none transition-colors duration-200 hover:text-ink', on ? 'text-ink' : 'text-ink-3')}
            >
              {y}
              {on && <motion.span layoutId="season-bar" transition={spring.snappy} className="absolute inset-x-0 -bottom-px h-0.5 bg-signal" aria-hidden />}
            </button>
          )
        })}
      </div>

      {inSeason.length === 0 ? (
        <p className="font-editorial text-[21px] italic text-ink-2">No races scored for {season} yet.</p>
      ) : (
        <ol className="grid grid-cols-[repeat(auto-fill,minmax(190px,1fr))] gap-px overflow-hidden rounded-[10px] border border-white/[.08] bg-white/[.08]">
          {inSeason.map((r) => (
            <li key={r.round}>
              <Link to={`/history/${r.season}/${r.round}`} className="relative flex h-[150px] flex-col overflow-hidden bg-[#0c0d10] px-[18px] py-4 text-ink transition-colors duration-200 hover:bg-[#15171b] hover:text-ink">
                <span className="pointer-events-none absolute -bottom-[34px] -right-1.5 select-none font-headline text-[150px] font-black leading-none text-ink/5" aria-hidden>
                  {r.round}
                </span>
                <span className="text-[11px] font-semibold leading-none tracking-[.14em] text-ink-3">R{r.round}</span>
                <span className="relative mt-2.5 font-headline text-[30px] font-extrabold uppercase leading-[.95]">{r.location}</span>
                <span className="relative mt-auto flex items-center gap-2 text-[10px] font-medium uppercase leading-none tracking-[.12em] text-gain">
                  <span className="size-1.5 rounded-full bg-gain" aria-hidden />
                  Predicted vs actual
                </span>
              </Link>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

/** The forecasts the app actually showed, scored after each race: the last one published before the start. */
function LiveTrackRecord() {
  const record = useLiveRecord()
  const races = record.data?.races ?? []
  if (!races.length) return null
  const last = (r: LiveRecord['races'][number]) => r.snapshots[r.snapshots.length - 1]
  return (
    <section aria-label="Live track record" className="pw-glass flex flex-col gap-3 rounded-[10px] px-[22px] py-5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
        <h2 className="pw-label">Live track record</h2>
        <span className="text-[12px] text-ink-3">What the app actually published, scored after the race. Not a simulation.</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[520px] text-left text-[13px] tabular-nums">
          <thead className="text-[10px] uppercase tracking-[.12em] text-ink-3">
            <tr>
              <th className="py-1.5 pr-3 font-medium">Race</th>
              <th className="py-1.5 pr-3 font-medium">Last forecast</th>
              <th className="py-1.5 pr-3 font-medium">Off by (places)</th>
              <th className="py-1.5 pr-3 font-medium">Grid guess</th>
              <th className="py-1.5 pr-3 font-medium">Winner</th>
              <th className="py-1.5 font-medium">Podium</th>
            </tr>
          </thead>
          <tbody>
            {races.map((r) => {
              const s = last(r)
              return (
                <tr key={`${r.season}-${r.round}`} className="border-t border-white/[.07]">
                  <td className="py-2 pr-3 font-semibold">
                    {r.season} {r.location}
                  </td>
                  <td className="py-2 pr-3 text-ink-2">{s.label}</td>
                  <td className="py-2 pr-3">{s.finish_mae?.toFixed(2) ?? '—'}</td>
                  <td className="py-2 pr-3 text-ink-2">{s.grid_baseline_mae?.toFixed(2) ?? '—'}</td>
                  <td className="py-2 pr-3">{s.winner_called === undefined ? '—' : s.winner_called ? 'Called' : 'Missed'}</td>
                  <td className="py-2">{s.podium_called === undefined ? '—' : `${s.podium_called} of 3`}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}

// actual values are whole places; predictions keep a decimal so "8.1" isn't passed off as a result
const fmtActual: Record<Target, (v: number) => string> = {
  qualifying: (v) => `${v.toFixed(2)}%`,
  finish_position: (v) => `P${Math.round(v)}`,
  quali_delta: (v) => signed(v, 0),
  race_time: (v) => `${v.toFixed(2)}%`,
}
const fmtPredicted: Record<Target, (v: number) => string> = {
  qualifying: (v) => `${v.toFixed(2)}%`,
  finish_position: (v) => v.toFixed(1),
  quali_delta: (v) => signed(v, 1),
  race_time: (v) => `${v.toFixed(2)}%`,
}
const MISS_UNIT: Record<Target, string> = { qualifying: '% lap', finish_position: 'places', quali_delta: 'places', race_time: '% race' }
const COLS = 'grid-cols-[minmax(70px,90px)_minmax(0,1fr)_46px_32px] wide:grid-cols-[minmax(150px,230px)_minmax(0,1fr)_64px_64px_32px]'

function Hindsight({ season, round, location, drivers }: { season: number; round: number; location: string; drivers: BacktestDriverEntry[] }) {
  const [lensId, setLens] = useLens()
  const { ask } = useChromeApi()
  const lens = lensOf(lensId)
  const pick = (d: BacktestDriverEntry) => d[lens.id]
  const rows = [...drivers].sort((a, b) => (a.finish_position.actual ?? 99) - (b.finish_position.actual ?? 99))
  const nums = rows.flatMap((d) => [pick(d).actual, pick(d).predicted]).filter((v): v is number => v !== null)
  const pad = (Math.max(...nums) - Math.min(...nums)) * 0.04 || 1
  const a0 = Math.min(...nums) - pad
  const a1 = Math.max(...nums) + pad
  const at = (v: number) => ((v - a0) / (a1 - a0)) * 100
  const pairs = rows.filter((d) => pick(d).actual !== null && pick(d).predicted !== null)
  const errs = pairs.map((d) => Math.abs(pick(d).actual! - pick(d).predicted!))
  const mae = errs.length ? errs.reduce((s, e) => s + e, 0) / errs.length : null
  const big = errs.length ? pairs[errs.indexOf(Math.max(...errs))].driver : null
  const bigMiss = (a1 - a0) * 0.25

  const question = (d: BacktestDriverEntry) => {
    const { actual, predicted } = pick(d)
    const a = actual === null ? 'no result' : fmtActual[lens.id](actual)
    const p = predicted === null ? '—' : fmtPredicted[lens.id](predicted)
    return lens.id === 'finish_position'
      ? `Why did ${d.driver} finish ${a} instead of the predicted ${p} at ${location} ${season}?`
      : `Why was ${d.driver}'s ${lens.title.toLowerCase()} ${a} instead of the predicted ${p} at ${location} ${season}?`
  }

  return (
    <div className={cn(PAGE, 'gap-[22px] pt-7')}>
      <div className="flex flex-col gap-3">
        <Mega text={location} max={150} className="-ml-1" />
        <div className="flex flex-wrap items-baseline gap-x-[18px] gap-y-1.5">
          <span className="font-editorial text-[clamp(28px,3vw,40px)] italic leading-none text-ink-2">Grand Prix, in hindsight</span>
          <span className="pw-label leading-[1.6]">
            Round {round} · {season}: actual result vs. what the model predicted
          </span>
        </div>
      </div>

      <div className="flex flex-wrap items-end justify-between gap-4">
        <dl className="grid grid-cols-3 gap-[clamp(18px,4vw,48px)]">
          {[
            { k: 'Mean miss', v: mae === null ? '—' : (lens.id === 'finish_position' || lens.id === 'quali_delta' ? mae.toFixed(1) : mae.toFixed(2)), u: mae === null ? '' : MISS_UNIT[lens.id], c: undefined },
            { k: 'Biggest miss', v: big ?? '—', u: '', c: 'var(--color-signal-ink)' },
            { k: 'Compared', v: `${pairs.length}/${rows.length}`, u: '', c: undefined },
          ].map((s) => (
            <div key={s.k} className="flex flex-col gap-2">
              <dt className="text-[10px] font-medium uppercase leading-none tracking-[.12em] text-ink-3">{s.k}</dt>
              <dd className="font-headline text-[48px] font-extrabold leading-[.9]" style={{ color: s.c }}>
                {s.v}
                {s.u && <span className="ml-1.5 font-mono text-[11px] font-medium text-ink-3">{s.u}</span>}
              </dd>
            </div>
          ))}
        </dl>
        <LensTabs id="hindsight" lens={lens.id} onChange={setLens} compact />
      </div>

      <section aria-label={`${lens.title}: actual against predicted`} className="pw-glass rounded-[10px] px-[clamp(12px,2vw,20px)] py-3.5">
        <div className={cn('grid items-end gap-3 border-b border-white/[.07] pb-2.5 text-[10px] font-medium uppercase leading-none tracking-[.12em] text-ink-3', COLS)}>
          <span>Driver</span>
          <span className="flex flex-wrap gap-x-4 gap-y-1.5">
            <span className="flex items-center gap-1.5">
              <span className="size-2 rounded-full bg-ink" aria-hidden />
              Actual
            </span>
            <span className="flex items-center gap-1.5">
              <span className="size-2 rounded-full border-[1.5px] border-ink-2" aria-hidden />
              Predicted
            </span>
            <span className="flex items-center gap-1.5">
              <span className="h-0.5 w-3.5 bg-signal" aria-hidden />
              Big miss
            </span>
          </span>
          <span className="text-right">Actual</span>
          <span className="hidden text-right wide:block">Pred.</span>
          <span />
        </div>
        {pairs.length === 0 && (
          <p className="py-7 font-editorial text-[21px] italic leading-[1.4] text-ink-2">
            No actual values were recorded for {lens.title.toLowerCase()} at this race; the model's predictions still appear below.
          </p>
        )}
        <ol>
          {rows.map((d) => {
            const { actual, predicted } = pick(d)
            const pL = predicted === null ? null : at(predicted)
            const aL = actual === null ? null : at(actual)
            const missed = actual !== null && predicted !== null && Math.abs(actual - predicted) > bigMiss
            return (
              <li key={d.driver} className={cn('grid h-[38px] items-center gap-3 border-b border-white/5 hover:bg-white/[.03]', COLS)}>
                <span className="flex min-w-0 items-baseline gap-2.5">
                  <span className="w-[22px] text-[11px] font-semibold leading-none text-ink-3">
                    {d.finish_position.actual === null ? '—' : String(d.finish_position.actual).padStart(2, '0')}
                  </span>
                  <span className="text-[14px] font-bold leading-none">{d.driver}</span>
                  <span className="hidden truncate text-[11px] leading-none text-ink-3 wide:inline">{d.team}</span>
                </span>
                <div className="relative h-full" aria-hidden>
                  {pL !== null && aL !== null && (
                    <span
                      className="absolute top-1/2 -mt-px h-0.5 transition-all duration-500 ease-[var(--ease-out-expo)]"
                      style={{ left: `${Math.min(aL, pL)}%`, width: `${Math.abs(aL - pL)}%`, background: missed ? 'var(--color-signal)' : 'rgb(242 241 237 / .3)' }}
                    />
                  )}
                  {pL !== null && (
                    <span className="absolute top-1/2 -ml-[5px] -mt-[5px] size-2.5 rounded-full border-[1.5px] border-ink-2 bg-[#0e1013] transition-[left] duration-500 ease-[var(--ease-out-expo)]" style={{ left: `${pL}%` }} />
                  )}
                  {aL !== null && <span className="absolute top-1/2 -ml-[5px] -mt-[5px] size-2.5 rounded-full bg-ink transition-[left] duration-500 ease-[var(--ease-out-expo)]" style={{ left: `${aL}%` }} />}
                </div>
                <span className="text-right text-[13px] font-semibold leading-none tabular-nums">{actual === null ? '—' : fmtActual[lens.id](actual)}</span>
                <span className="hidden text-right text-[13px] font-medium leading-none tabular-nums text-ink-2 wide:block">{predicted === null ? '—' : fmtPredicted[lens.id](predicted)}</span>
                <button
                  type="button"
                  onClick={() => ask(question(d))}
                  title="Ask Pit Radio why"
                  aria-label={`Ask Pit Radio about ${d.driver}`}
                  className="grid size-8 place-items-center justify-self-end rounded-[5px] text-ink-3 hover:bg-signal/10 hover:text-signal-ink"
                >
                  <MessageCircleQuestion className="size-[15px]" aria-hidden />
                </button>
              </li>
            )
          })}
        </ol>
        <div className={cn('grid gap-3 pt-2.5', COLS)} aria-hidden>
          <span />
          <div className="relative h-3">
            {[0, 0.25, 0.5, 0.75, 1].map((t, i) => (
              <span
                key={t}
                className={cn('absolute -translate-x-1/2 whitespace-nowrap text-[10px] font-medium leading-none text-ink-3', i % 2 && 'hidden wide:block')}
                style={{ left: `${t * 100}%` }}
              >
                {fmtPredicted[lens.id](a0 + t * (a1 - a0))}
              </span>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}

/** One past race: what the model said before it, against what happened. */
export function HistoryRacePage() {
  const params = useParams()
  const season = Number(params.season)
  const round = Number(params.round)
  const valid = Number.isInteger(season) && Number.isInteger(round) && round > 0
  const bt = useBacktestForRace(season, round, valid)
  const location = bt.data?.location ?? ''
  const notFound = bt.error instanceof ApiError && bt.error.status === 404

  useChrome({
    section: 'hist',
    crumbs: [{ label: 'History', to: `/history?season=${season}`, back: true }, { label: String(season) }, { label: `Round ${round}` }, { label: location || '…' }],
    chat: { season, round, race_name: location || undefined },
    badge: `${location || `Round ${round}`} · ${season}`,
    quick: [location ? `Why did the model miss the biggest calls at ${location} ${season}?` : `How accurate has the model been in ${season}?`],
  })

  if (!valid) return <Navigate to="/history" replace />
  if (bt.data) return <Hindsight season={season} round={round} location={bt.data.location} drivers={bt.data.drivers} />
  if (notFound)
    return (
      <Notice title="No backtest for this race" action={<Link to="/history" className="text-[13px] font-semibold text-signal-ink hover:text-ink">Back to race history →</Link>}>
        {season} round {round} hasn't been scored: it hasn't happened yet, or it isn't in the archive.
      </Notice>
    )
  if (bt.isError) return <Offline />
  return <Skeleton label="Loading the backtest…" />
}
