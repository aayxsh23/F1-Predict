import { motion } from 'framer-motion'
import { ArrowRight } from 'lucide-react'
import { Link } from 'react-router-dom'

import { cn } from '@/lib/cn'
import { usePredictionsLatest } from '@/lib/queries'
import { useRaceView, type RaceDriver, type RaceView } from '@/lib/raceView'
import type { KnownSessions } from '@/lib/types'

import { useChrome } from './chrome'
import { circuitSpecs, driverHref, isStreet, orDash, placePhrase, raceHref, signed, useDriverNames, weekendPhase } from './data'
import { MagneticLink, Mega, Notice, Offline, PAGE, Skeleton } from './parts'

const stats = (d: RaceDriver) => [
  { k: 'Finish', v: orDash(d.modelFinish, (v) => v.toFixed(2)) },
  { k: 'Quali gap', v: orDash(d.qualiGapPct, (v) => `${v.toFixed(2)}%`) },
  { k: 'Grid→fin', v: orDash(d.modelDelta, (v) => signed(v, 2)) },
  { k: 'Time gap', v: orDash(d.raceGapPct, (v) => `${v.toFixed(2)}%`) },
]

/** C1: how informed the forecast is, from known_sessions alone. Never a fake progress bar. */
function ConfidenceRail({ known }: { known: KnownSessions }) {
  const phase = weekendPhase(known)
  return (
    <section aria-label="Weekend confidence" className="pw-glass flex flex-col gap-4 rounded-[10px] px-[22px] py-5">
      <div className="flex items-baseline justify-between gap-4">
        <h2 className="pw-label">Weekend confidence</h2>
        <span className="pw-label">
          Information <span className="text-ink">{phase.count}/{phase.nodes.length}</span>
        </span>
      </div>
      <ol className={cn('grid grid-cols-2 gap-x-1.5 gap-y-4', phase.nodes.length > 3 ? 'sm:grid-cols-3 lg:grid-cols-5' : 'sm:grid-cols-3')}>
        {phase.nodes.map((n, i) => {
          const isKnown = known[n.key]
          const isNext = i === phase.next
          const [status, color] = isKnown ? ['Known', 'var(--color-gain)'] : isNext ? ['Next', 'var(--color-pending)'] : ['Pending', 'var(--color-ink-3)']
          return (
            <li key={n.key} className="flex min-w-0 flex-col gap-2.5">
              <div className="h-[3px] overflow-hidden rounded-sm bg-white/[.12]">
                {(isKnown || isNext) && (
                  <motion.div
                    className="h-full origin-left"
                    style={{ background: isKnown ? 'var(--color-gain)' : 'repeating-linear-gradient(90deg,var(--color-pending) 0 6px,transparent 6px 10px)' }}
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: 1 }}
                    transition={{ duration: 0.6, delay: i * 0.08, ease: [0.16, 1, 0.3, 1] }}
                  />
                )}
              </div>
              <div className="flex items-center gap-2">
                <span
                  className={cn('size-2.5 shrink-0 rounded-full border-[1.5px]', isNext && 'motion-safe:animate-[pw-pulse_1.6s_ease-out_infinite]')}
                  style={{ background: isKnown ? color : 'transparent', borderColor: isKnown || isNext ? color : 'rgb(255 255 255 / .28)' }}
                  aria-hidden
                />
                <span className="truncate text-[12px] font-semibold leading-none">{n.label}</span>
              </div>
              <span className="text-[10px] font-medium uppercase leading-none tracking-[.12em]" style={{ color }}>
                {status}
              </span>
            </li>
          )
        })}
      </ol>
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1.5 border-t border-white/[.07] pt-3.5">
        <span className="whitespace-nowrap font-headline text-[26px] font-extrabold uppercase leading-none">{phase.title}</span>
        <span className="font-editorial text-[19px] italic leading-[1.3] text-ink-2 [text-wrap:pretty]">{phase.copy}</span>
      </div>
    </section>
  )
}

const tile = 'relative overflow-hidden rounded-[10px] border transition-[transform,border-color] duration-[350ms] ease-[var(--ease-spring)] hover:-translate-y-[3px] hover:border-white/20'

function PodiumP1({ d, name, href }: { d: RaceDriver; name?: string; href: string }) {
  return (
    <article
      className={cn(
        tile,
        'flex min-h-[318px] flex-col border-white/10 bg-[linear-gradient(160deg,rgb(255_90_71/.10),rgb(255_255_255/.02)_55%)] px-6 py-[22px] shadow-[inset_0_1px_0_rgb(255_255_255/.07)] backdrop-blur-[18px]',
      )}
    >
      <span
        className="pointer-events-none absolute -right-[18px] -top-[74px] select-none bg-[linear-gradient(180deg,rgb(255_90_71/.6),rgb(255_90_71/0)_78%)] bg-clip-text font-headline text-[470px] font-black leading-none text-transparent"
        aria-hidden
      >
        1
      </span>
      <span className="relative text-[12px] font-semibold leading-none tracking-[.14em] text-signal-ink">P1</span>
      <span className="relative mt-3 font-headline text-[104px] font-extrabold leading-[.85]" style={{ viewTransitionName: `driver-${d.code}` }}>
        {d.code}
      </span>
      <span className="relative mt-2.5 text-[13px] leading-none text-ink-2">{name ? `${name} · ${d.team}` : d.team}</span>
      <div className="relative mt-auto flex flex-wrap items-end justify-between gap-5 pt-7">
        <dl className="grid grid-cols-2 gap-x-[22px] gap-y-3 sm:grid-cols-4">
          {stats(d).map((s) => (
            <div key={s.k} className="flex flex-col gap-1.5">
              <dt className="text-[10px] font-medium uppercase leading-none tracking-[.12em] text-ink-3">{s.k}</dt>
              <dd className="text-[18px] font-semibold leading-none tabular-nums">{s.v}</dd>
            </div>
          ))}
        </dl>
        <MagneticLink to={href}>Why {d.code}</MagneticLink>
      </div>
    </article>
  )
}

function PodiumTile({ d, n, href }: { d: RaceDriver; n: number; href: string }) {
  return (
    <Link
      to={href}
      viewTransition
      className={cn(tile, 'flex h-[142px] flex-col border-white/[.09] bg-[linear-gradient(180deg,rgb(255_255_255/.05),rgb(255_255_255/.015))] px-[18px] py-4')}
    >
      <span className="pointer-events-none absolute -top-[26px] right-1.5 select-none font-headline text-[190px] font-black leading-none text-ink/[.07]" aria-hidden>
        {n}
      </span>
      <span className="text-[11px] font-semibold leading-none tracking-[.14em] text-ink-3">P{n}</span>
      <span className="relative mt-2 font-headline text-[50px] font-extrabold leading-[.9]" style={{ viewTransitionName: `driver-${d.code}` }}>
        {d.code}
      </span>
      <span className="relative mt-auto flex items-baseline justify-between gap-2">
        <span className="truncate text-[12px] leading-none text-ink-2">{d.team}</span>
        <span className="text-[15px] font-semibold leading-none tabular-nums">{orDash(d.modelFinish, (v) => v.toFixed(2))}</span>
      </span>
    </Link>
  )
}

/** 22 dots on the average-finish axis: the honest pre-weekend story is "it's tight". */
function FieldSpread({ view, compressed }: { view: RaceView; compressed: boolean }) {
  const ds = view.drivers.filter((d) => d.modelFinish !== null)
  if (ds.length < 7) return null
  const vals = ds.map((d) => d.modelFinish!)
  const mn = Math.min(...vals)
  const mx = Math.max(...vals)
  const at = (v: number) => `${(((v - mn) / (mx - mn || 1)) * 96 + 2).toFixed(2)}%`
  return (
    <section aria-label="How spread out the field is" className="flex flex-col gap-3 rounded-[10px] border border-white/[.09] bg-white/[.02] px-[18px] py-4">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h2 className="font-headline text-[24px] font-bold uppercase leading-none">
          Top 7 within {(vals[6] - vals[0]).toFixed(2)} places
        </h2>
        <Link to={raceHref(view.season, view.round)} viewTransition className="flex items-center gap-1.5 text-[12px] font-semibold leading-none text-signal-ink hover:text-ink">
          Full grid · {view.drivers.length}
          <ArrowRight className="size-3.5" aria-hidden />
        </Link>
      </div>
      <div className="relative h-[22px] border-b border-white/[.14]" aria-hidden>
        {ds.map((d, i) => (
          <span
            key={d.code}
            title={`${d.code} ${d.modelFinish!.toFixed(2)}`}
            className="absolute -bottom-1 -ml-[3.5px] size-[7px] rounded-full"
            style={{ left: at(d.modelFinish!), background: i === 0 ? 'var(--color-signal)' : i < 7 ? 'var(--color-ink)' : 'rgb(242 241 237 / .35)' }}
          />
        ))}
      </div>
      <div className="flex items-baseline justify-between gap-4 text-[10px] font-medium leading-none tracking-[.1em] text-ink-3">
        <span>P{mn.toFixed(1)}</span>
        <span className="text-center font-editorial text-[15px] italic leading-snug tracking-normal text-ink-2">
          {compressed ? 'Pre-weekend the front pack is compressed. It spreads once practice lands.' : 'Each dot is a driver’s average predicted finish.'}
        </span>
        <span>P{mx.toFixed(1)}</span>
      </div>
    </section>
  )
}

function Weekend({ view }: { view: RaceView }) {
  const [p1, p2, p3] = view.drivers
  const names = useDriverNames()
  const row = p1?.featureRow ?? null
  const specs = circuitSpecs(row)
  const href = (d: RaceDriver) => driverHref(view.season, view.round, d.code)
  const preWeekend = !!view.knownSessions && weekendPhase(view.knownSessions).count === 0

  return (
    <div className={cn(PAGE, 'flex-row flex-wrap gap-7 pt-7')}>
      <div className="flex min-w-0 flex-[1.22_1_520px] flex-col gap-[22px]">
        <div>
          <Mega text={view.location} max={236} className="-ml-1.5" />
          <div className="mt-3.5 flex flex-wrap items-baseline gap-x-[18px] gap-y-2">
            <span className="font-editorial text-[clamp(36px,4vw,58px)] italic leading-none text-ink-2">{view.eventName ?? 'Grand Prix'}</span>
            <span className="pw-label leading-[1.6]">
              Round {view.round} of {view.season}
              <br />
              {row ? (isStreet(row) ? 'Street circuit' : 'Permanent circuit') : 'Archived race'} · {view.drivers.length} drivers
            </span>
          </div>
        </div>

        {specs.length > 0 && (
          <dl className="grid grid-cols-[repeat(auto-fit,minmax(118px,1fr))] gap-px border-y border-white/[.12] bg-white/[.09]">
            {specs.map((s) => (
              <div key={s.k} className="flex flex-col justify-between gap-1.5 bg-asphalt p-3.5">
                <dt className="text-[10px] font-medium uppercase leading-tight tracking-[.12em] text-ink-3">{s.k}</dt>
                <dd className="font-headline text-[34px] font-bold leading-none">
                  {s.v}
                  <span className="ml-1 font-mono text-[12px] font-medium text-ink-3">{s.u}</span>
                </dd>
              </div>
            ))}
          </dl>
        )}

        {view.knownSessions ? (
          <ConfidenceRail known={view.knownSessions} />
        ) : (
          <p className="pw-glass rounded-[10px] px-[22px] py-5 font-editorial text-[19px] italic leading-[1.35] text-ink-2">
            An archived race: this is the walk-forward prediction from a model that had never seen it.
          </p>
        )}
      </div>

      <div className="flex min-w-0 flex-[1_1_400px] flex-col gap-3.5">
        <div className="pw-label flex items-baseline justify-between gap-3">
          <h2>Predicted finish · top 3</h2>
          <span className="hidden sm:inline">avg finishing position</span>
        </div>
        {p1 && <PodiumP1 d={p1} name={names.get(p1.code)} href={href(p1)} />}
        <div className="grid grid-cols-2 gap-3.5">
          {p2 && <PodiumTile d={p2} n={2} href={href(p2)} />}
          {p3 && <PodiumTile d={p3} n={3} href={href(p3)} />}
        </div>
        <FieldSpread view={view} compressed={preWeekend} />
      </div>
    </div>
  )
}

/** S1: this weekend at a glance. Opens on the latest forecast the pipeline stored. */
export function RaceWeekendPage() {
  const latest = usePredictionsLatest()
  const sel = latest.data ? { season: latest.data.season, round: latest.data.round } : null
  const { status, view } = useRaceView(sel)
  const ready = status === 'ready' ? view : null
  const raceName = ready ? (ready.eventName ?? ready.location) : undefined

  useChrome({
    section: 'pred',
    crumbs: ready ? [{ label: String(ready.season) }, { label: `Round ${ready.round}` }, { label: ready.location }] : [{ label: 'This weekend' }],
    fresh: ready?.generatedAt,
    chat: { season: ready?.season, round: ready?.round, race_name: raceName },
    badge: ready ? `Round ${ready.round} · ${ready.season}` : 'This weekend',
    quick: [`What matters most for predictions at ${placePhrase(raceName)}${ready ? ` (round ${ready.round})` : ''}?`, 'Who is leading the drivers championship?'],
    codes: ready?.drivers.map((d) => d.code),
  })

  if (ready && ready.drivers.length > 0) return <Weekend view={ready} />
  if (latest.isError || status === 'error') return <Offline />
  if (ready) return <Notice title="No forecast yet">The pipeline hasn't stored a forecast for this weekend.</Notice>
  return <Skeleton />
}
