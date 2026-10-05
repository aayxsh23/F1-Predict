import { AnimatePresence, motion } from 'framer-motion'
import { ArrowRight, ChevronDown } from 'lucide-react'
import { useLayoutEffect, useRef, useState } from 'react'
import { Link, Navigate, useParams } from 'react-router-dom'

import { cn } from '@/lib/cn'
import { spring } from '@/lib/motion'
import { usePredictionsLatest } from '@/lib/queries'
import { useRaceView, type RaceDriver, type RaceView } from '@/lib/raceView'
import { TARGET_HIGHER_IS_BETTER, type Target } from '@/lib/types'

import { useChrome } from './chrome'
import { betterWord, driverHref, driverInputs, driverKnown, lensOf, lensRange, normaliser, orDash, placePhrase, rankBy, signed, useLens, type Lens } from './data'
import { LensTabs, Mega, Notice, Offline, PAGE, Skeleton } from './parts'

// Pos · Driver · lens · Quali · Finish · Grid→fin · Time gap · Inputs · chevron. Sized
// by the tower's own width (a container query), so an open Pit Radio column narrows it too.
const COLS = 'grid-cols-[40px_minmax(0,1fr)_minmax(0,1.3fr)_18px] @min-[900px]:grid-cols-[56px_170px_minmax(0,1fr)_80px_72px_88px_84px_70px_18px]'
const FULL = 'hidden @min-[900px]:block' // the four target columns and inputs, once the tower is wide enough

const COLUMNS: Array<{ id: Target; label: string; cell: (d: RaceDriver) => string }> = [
  { id: 'qualifying', label: 'Quali', cell: (d) => orDash(d.qualiGapPct, (v) => `${v.toFixed(2)}%`) },
  { id: 'finish_position', label: 'Finish', cell: (d) => orDash(d.modelFinish, (v) => v.toFixed(2)) },
  { id: 'quali_delta', label: 'Grid→fin', cell: (d) => orDash(d.modelDelta, (v) => signed(v, 2)) },
  { id: 'race_time', label: 'Time gap', cell: (d) => orDash(d.raceGapPct, (v) => `${v.toFixed(2)}%`) },
]

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [width, setWidth] = useState(1200)
  useLayoutEffect(() => {
    if (!ref.current) return
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])
  return [ref, width] as const
}

/** C2's field: every driver on the lens's axis, better to the left. Labels
 *  stack in lanes so a cluster reads as a cluster, not as tied numbers. */
function PredictionField({ drivers, lens, onPick }: { drivers: RaceDriver[]; lens: Lens; onPick: (code: string) => void }) {
  const [ref, width] = useWidth<HTMLDivElement>()
  const tn = normaliser(drivers, lens)
  const [mn, mx] = lensRange(drivers, lens)
  const hib = TARGET_HIGHER_IS_BETTER[lens.id]
  const leader = rankBy(drivers, lens)[0]?.code

  const pts = drivers
    .filter((d) => lens.value(d) !== null)
    .map((d) => ({ d, left: 1 + tn(lens.value(d)!) * 98 }))
    .sort((a, b) => a.left - b.left)
  const minGap = (32 / Math.max(width, 1)) * 100 // a label is ~28px wide
  const ends: number[] = []
  const lane = new Map<string, number>()
  for (const p of pts) {
    let l = ends.findIndex((e) => p.left - e > minGap)
    if (l < 0) l = ends.push(0) - 1
    ends[l] = p.left
    lane.set(p.d.code, l)
  }

  return (
    <section aria-label={`Prediction field, ${lens.title}`} className="pw-glass relative rounded-[10px] px-6 pt-4" style={{ height: Math.max(196, 110 + ends.length * 14) }}>
      <div className="flex flex-wrap justify-between gap-2 text-[10px] font-medium uppercase leading-none tracking-[.14em] text-ink-3">
        <h2>Prediction field · {lens.title}</h2>
        <span>
          <span className="text-gain">◂ Better</span> · each dot is a driver
        </span>
      </div>
      <div className="absolute inset-x-6 bottom-[42px] h-px bg-white/[.18]" aria-hidden />
      {[0, 0.25, 0.5, 0.75, 1].map((t, i) => (
        <span
          key={t}
          className={cn('absolute bottom-5 -translate-x-1/2 text-[10px] font-medium leading-none text-ink-3', i % 2 && 'hidden wide:block')}
          style={{ left: `calc(24px + (100% - 48px) * ${(1 + t * 98) / 100})` }}
        >
          {lens.fmt(hib ? mx - t * (mx - mn) : mn + t * (mx - mn))}
        </span>
      ))}
      <div ref={ref} className="absolute inset-x-6 bottom-[42px] top-10">
        {pts.map(({ d, left }) => {
          const l = lane.get(d.code)!
          const first = d.code === leader
          return (
            <div key={d.code} className="group absolute bottom-0 w-0 transition-[left] duration-700 ease-[cubic-bezier(.34,1.3,.64,1)]" style={{ left: `${left}%` }}>
              <button
                type="button"
                onClick={() => onPick(d.code)}
                aria-label={`${d.code}: ${orDash(lens.value(d), lens.fmt)}. Show in the table`}
                className="absolute -bottom-2.5 -left-2.5 grid size-5 place-items-center rounded-full"
              >
                <span className="size-[9px] rounded-full shadow-[0_0_0_3px_#0e1013] transition-transform group-hover:scale-150" style={{ background: first ? 'var(--color-signal)' : 'rgb(242 241 237 / .85)' }} />
              </button>
              <span className="absolute bottom-1.5 left-0 w-px bg-white/[.14] transition-[height] duration-400" style={{ height: 8 + l * 14 }} aria-hidden />
              <span
                onClick={() => onPick(d.code)}
                className="absolute -left-3.5 w-7 cursor-pointer text-center text-[10px] font-semibold leading-none transition-[bottom] duration-400"
                style={{ bottom: 16 + l * 14, color: first ? 'var(--color-signal-ink)' : 'var(--color-ink-2)' }}
                aria-hidden
              >
                {d.code}
              </span>
            </div>
          )
        })}
      </div>
    </section>
  )
}

function TowerRow({ d, i, lens, t, open, flash, onToggle, view }: { d: RaceDriver; i: number; lens: Lens; t: number | null; open: boolean; flash: boolean; onToggle: () => void; view: RaceView }) {
  const v = lens.value(d)
  const first = i === 0
  const inputs = driverInputs(d.featureRow, view.location)
  const known = driverKnown(d)
  return (
    <motion.li
      id={`tower-${d.code}`}
      layout="position"
      transition={{ ...spring.soft, delay: i * 0.012 }}
      className={cn('scroll-mt-24 border-t border-white/[.07] transition-shadow duration-300', open && 'bg-white/[.035]')}
      style={{ boxShadow: flash ? 'inset 0 0 0 1px rgb(255 90 71 / .7)' : 'none' }}
    >
      <button
        type="button"
        aria-expanded={open}
        aria-controls={`row-${d.code}`}
        aria-label={`${String(i + 1)}. ${d.code}, ${d.team}: ${lens.title} ${orDash(v, lens.fmt)}`}
        onClick={onToggle}
        className={cn('grid h-[46px] w-full items-center gap-x-3.5 px-4 text-left transition-colors duration-200 hover:bg-white/[.04]', COLS)}
      >
        <span className="font-headline text-[26px] font-extrabold leading-none" style={{ color: first ? 'var(--color-signal)' : undefined }}>
          {String(i + 1).padStart(2, '0')}
        </span>
        <span className="flex min-w-0 items-baseline gap-2.5">
          <span className="text-[15px] font-bold leading-none" style={{ viewTransitionName: `driver-${d.code}` }}>
            {d.code}
          </span>
          <span className="truncate text-[11px] leading-none text-ink-3">{d.team}</span>
        </span>
        <span className="flex min-w-0 items-center gap-3">
          <span className="w-14 shrink-0 text-[14px] font-semibold leading-none tabular-nums">{orDash(v, lens.fmt)}</span>
          <span className="h-1 flex-1 overflow-hidden rounded-sm bg-white/[.06]">
            <span
              className="block h-full transition-[width] duration-600 ease-[var(--ease-out-expo)]"
              style={{
                width: t === null ? 0 : `${(6 + (1 - t) * 94).toFixed(1)}%`,
                background: first ? 'linear-gradient(90deg,rgb(255 90 71/.4),var(--color-signal))' : 'linear-gradient(90deg,rgb(242 241 237/.12),rgb(242 241 237/.55))',
              }}
            />
          </span>
        </span>
        {COLUMNS.map((c) => (
          <span key={c.id} className={cn(FULL, 'text-right text-[13px] font-medium leading-none tabular-nums', c.id === lens.id ? 'text-ink' : 'text-ink-3')}>
            {c.cell(d)}
          </span>
        ))}
        <span className="hidden justify-center gap-[5px] @min-[900px]:flex" title={known.map((k) => `${k.k}: ${k.on ? 'known' : 'not yet'}`).join(' · ')}>
          {known.map((k) => (
            <span key={k.k} className="size-[7px] rounded-full border-[1.5px]" style={k.on ? { background: 'var(--color-gain)', borderColor: 'var(--color-gain)' } : { borderColor: 'rgb(255 255 255/.28)' }} />
          ))}
        </span>
        <ChevronDown className={cn('size-4 justify-self-end text-ink-3 transition-transform duration-300', open && 'rotate-180')} aria-hidden />
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            id={`row-${d.code}`}
            key="inputs"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden"
          >
            <div className="flex flex-wrap items-end gap-y-3.5 px-4 pb-4 pt-1">
              {inputs.length ? (
                inputs.map((ft, k) => (
                  <motion.div
                    key={ft.k}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.05 + k * 0.03 }}
                    className="flex flex-[1_1_110px] flex-col gap-2 border-l border-white/10 pl-3"
                  >
                    <span className="text-[10px] font-medium uppercase leading-none tracking-[.12em] text-ink-3">{ft.k}</span>
                    <span className="font-headline text-[30px] font-bold leading-none">{ft.v}</span>
                  </motion.div>
                ))
              ) : (
                <p className="flex-1 font-editorial text-[17px] italic leading-snug text-ink-2">
                  Form inputs aren't stored for archived races.{d.actual !== null && ` ${d.code} finished P${d.actual}.`}
                </p>
              )}
              {view.mode === 'forecast' && (
                <Link
                  to={driverHref(view.season, view.round, d.code, lens.id)}
                  viewTransition
                  className="ml-auto flex h-10 items-center gap-2 whitespace-nowrap rounded-md border border-signal/50 px-3.5 text-[12px] font-semibold leading-none text-signal-ink hover:bg-signal/[.12]"
                >
                  Why {d.code}
                  <ArrowRight className="size-3.5" aria-hidden />
                </Link>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.li>
  )
}

function Grid({ view }: { view: RaceView }) {
  const [lensId, setLens] = useLens()
  const [open, setOpen] = useState<string | null>(null)
  const [flash, setFlash] = useState<string | null>(null)
  const timer = useRef(0)
  const lens = lensOf(lensId)
  const ordered = rankBy(view.drivers, lens)
  const tn = normaliser(view.drivers, lens)

  return (
    <div className={cn(PAGE, 'gap-[18px] pt-[22px]')}>
      <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-[18px]">
        <div className="flex min-w-0 flex-[1_1_320px] items-baseline gap-[18px]">
          <div className="min-w-0 max-w-[640px] flex-1">
            <Mega text={view.location} max={104} />
          </div>
          <span className="shrink-0 font-editorial text-[clamp(24px,3vw,40px)] italic leading-none text-ink-2">full grid</span>
        </div>
        <div className="flex w-full min-w-0 flex-col gap-2 wide:w-auto">
          <span className="pw-label text-[10px]">Target lens · {betterWord(lens.id)}</span>
          <LensTabs id="grid" lens={lens.id} onChange={setLens} />
        </div>
      </div>

      <PredictionField
        drivers={view.drivers}
        lens={lens}
        onPick={(code) => {
          setOpen(code)
          setFlash(code)
          document.getElementById(`tower-${code}`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
          clearTimeout(timer.current)
          timer.current = window.setTimeout(() => setFlash(null), 1200)
        }}
      />

      <div className="@container">
        <div className={cn('grid gap-x-3.5 px-4 pb-2.5 text-[10px] font-medium uppercase leading-none tracking-[.12em] text-ink-3', COLS)} aria-hidden>
          <span>Pos</span>
          <span>Driver</span>
          <span>{lens.title}</span>
          {COLUMNS.map((c) => (
            <span key={c.id} className={cn(FULL, 'text-right', c.id === lens.id && 'text-ink')}>
              {c.label}
            </span>
          ))}
          <span className={cn(FULL, 'text-center')}>Inputs</span>
          <span />
        </div>
        <ol>
          {ordered.map((d, i) => {
            const v = lens.value(d)
            return (
              <TowerRow
                key={d.code}
                d={d}
                i={i}
                lens={lens}
                t={v === null ? null : tn(v)}
                open={open === d.code}
                flash={flash === d.code}
                onToggle={() => setOpen((o) => (o === d.code ? null : d.code))}
                view={view}
              />
            )
          })}
        </ol>
      </div>
    </div>
  )
}

/** S2: the full grid under one of the four targets, re-sorting as the lens changes. */
export function FullGridPage() {
  const params = useParams()
  const season = Number(params.season)
  const round = Number(params.round)
  const valid = Number.isInteger(season) && Number.isInteger(round) && round > 0
  const { status, view } = useRaceView(valid ? { season, round } : null)
  const latest = usePredictionsLatest()
  const ready = status === 'ready' ? view : null
  const raceName = ready ? (ready.eventName ?? ready.location) : undefined
  const isLatest = latest.data?.season === season && latest.data.round === round

  useChrome({
    section: 'pred',
    crumbs: [{ label: String(season) }, { label: `Round ${round}` }, { label: ready?.location || '…', to: isLatest ? '/' : undefined }, { label: 'Full grid' }],
    fresh: ready?.generatedAt,
    chat: { season, round, race_name: raceName },
    badge: `Round ${round} · ${season}`,
    quick: [`What matters most for predictions at ${placePhrase(raceName)} (round ${round})?`, "What's the gap between the top two constructors?"],
    codes: ready?.drivers.map((d) => d.code),
  })

  if (!valid) return <Navigate to="/" replace />
  if (ready && ready.drivers.length > 0) return <Grid view={ready} />
  if (status === 'error') return <Offline />
  if (ready)
    return (
      <Notice title="No forecast for this race" action={<Link to="/" className="text-[13px] font-semibold text-signal-ink hover:text-ink">Back to this weekend →</Link>}>
        Neither a stored forecast nor the archive covers {season} round {round}.
      </Notice>
    )
  return <Skeleton label="Loading the grid…" />
}
