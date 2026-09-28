import { Check, ChevronLeft, ChevronRight } from 'lucide-react'
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from 'react'

import type { CalendarRace, RaceStatus } from '@/lib/calendar'
import { cn } from '@/lib/cn'
import { useHealth } from '@/lib/queries'
import { useSelection } from '@/lib/selection'

const CARD_W = 176
const GAP = 8
const STRIDE = CARD_W + GAP
const PAD = 20
const OVERSCAN = 3 // cards rendered beyond each edge so a fast drag never shows a gap

const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches

function StatusMark({ status }: { status: RaceStatus }) {
  if (status === 'completed') {
    return (
      <span className="flex items-center gap-1 text-silver-400">
        <Check className="h-3 w-3" strokeWidth={2.5} aria-hidden /> COMPLETED
      </span>
    )
  }
  if (status === 'live') {
    return (
      <span className="flex items-center gap-1.5 font-semibold text-laser-300">
        <span className="h-1.5 w-1.5 rounded-full bg-laser-400 motion-safe:animate-[pulse-dot_1.6s_ease-in-out_infinite]" aria-hidden />
        LIVE
      </span>
    )
  }
  return (
    <span className="flex items-center gap-1.5 text-silver-300">
      <span className="h-1.5 w-1.5 rounded-full border border-silver-400" aria-hidden /> UPCOMING
    </span>
  )
}

function RaceCard({
  race,
  selected,
  style,
  onSelect,
}: {
  race: CalendarRace
  selected: boolean
  style: CSSProperties
  onSelect: () => void
}) {
  return (
    // the glow lives on this wrapper: filter runs before clip-path, so a
    // drop-shadow on the chamfered button itself would be cut off
    <div
      className={cn(
        'absolute top-3 transition-[transform,filter] duration-300 ease-out-expo',
        selected && '-translate-y-0.5 drop-shadow-[0_6px_14px_var(--color-laser-wash)]',
      )}
      style={style}
    >
      <button
        type="button"
        onClick={onSelect}
        aria-current={selected ? 'true' : undefined}
        aria-label={`Round ${race.round}, ${race.eventName}, ${race.dateLabel}, ${race.status}`}
        className={cn(
          'cut-frame block h-[72px] text-left',
          selected ? 'frame-laser' : 'hover:[--frame:var(--color-silver-400)]',
        )}
        style={{ width: CARD_W }}
      >
        <span className="flex h-full flex-col justify-between px-3 py-2">
          <span className="flex items-center gap-2">
            <span className="font-mono text-hud text-silver-400">R{String(race.round).padStart(2, '0')}</span>
            {race.plate && (
              <span className="border border-line-strong px-1 font-mono text-micro leading-4 text-silver-300">
                {race.plate}
              </span>
            )}
            {race.sprint && (
              <span className="hud-label ml-auto text-silver-300" title="Sprint weekend">
                SPRINT
              </span>
            )}
          </span>
          <span
            className={cn(
              'truncate font-display text-base font-semibold uppercase tracking-[0.06em]',
              selected ? 'text-silver-100' : 'text-silver-200',
            )}
          >
            {race.name}
          </span>
          <span className="flex items-center justify-between gap-2 text-hud text-silver-400">
            <span className="truncate">{race.dateLabel}</span>
            <StatusMark status={race.status} />
          </span>
        </span>
      </button>
    </div>
  )
}

function ApiLed() {
  const health = useHealth()
  const state = health.isSuccess ? 'ok' : health.isError ? 'down' : 'waking'
  const copy = { ok: 'API LIVE', down: 'API DOWN', waking: 'WAKING' }[state]
  return (
    <span className="hidden items-center gap-2 text-silver-400 xl:flex" title={`Backend: ${copy.toLowerCase()}`}>
      <span
        className={cn(
          'h-1.5 w-1.5 rounded-full',
          state === 'ok' && 'bg-laser-400',
          state === 'down' && 'bg-signal-coral',
          state === 'waking' && 'bg-signal-amber motion-safe:animate-[pulse-dot_1.6s_ease-in-out_infinite]',
        )}
      />
      <span className="hud-label">{copy}</span>
    </span>
  )
}

export function SeasonRaceBar() {
  const sel = useSelection()
  const scroller = useRef<HTMLDivElement>(null)
  const [win, setWin] = useState({ left: 0, width: 1200 })
  const drag = useRef({ moved: false, startX: 0, startLeft: 0, lastX: 0, lastT: 0, v: 0 })
  const glideFrame = useRef(0)

  const races = sel.calendar
  const selectedIdx = races.findIndex((r) => r.round === sel.selectedRace?.round)
  const seasonIdx = sel.seasons.indexOf(sel.season)

  const sync = useCallback(() => {
    const el = scroller.current
    if (el) setWin({ left: el.scrollLeft, width: el.clientWidth })
  }, [])

  useEffect(() => {
    const el = scroller.current
    if (!el) return
    sync()
    const observer = new ResizeObserver(sync)
    observer.observe(el)
    return () => observer.disconnect()
  }, [sync])

  // wheel scrolls the ribbon sideways; a non-passive listener is needed to claim it
  useEffect(() => {
    const el = scroller.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      userScrolled.current = true
      if (Math.abs(e.deltaY) <= Math.abs(e.deltaX)) return // trackpad sideways swipes already work natively
      e.preventDefault()
      el.scrollLeft += e.deltaY
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  // Snap the selected weekend to the centre. Driven by layout, not render order: the
  // track can still be laying out when the selection resolves, so a ResizeObserver
  // re-centres until the visitor scrolls the ribbon themselves.
  const track = useRef<HTMLDivElement>(null)
  const userScrolled = useRef(false)
  const selectedRef = useRef(selectedIdx)
  useLayoutEffect(() => {
    selectedRef.current = selectedIdx
  })

  const center = useCallback((smooth: boolean) => {
    const el = scroller.current
    const i = selectedRef.current
    if (!el || i < 0) return
    const left = Math.max(0, PAD + i * STRIDE - (el.clientWidth - CARD_W) / 2)
    el.scrollTo({ left, behavior: smooth && !reducedMotion() ? 'smooth' : 'auto' })
    sync()
  }, [sync])

  useEffect(() => {
    const t = track.current
    if (!t) return
    const observer = new ResizeObserver(() => !userScrolled.current && center(false))
    observer.observe(t)
    return () => observer.disconnect()
  }, [center])

  useEffect(() => {
    center(userScrolled.current) // first placement jumps; a later pick glides
  }, [selectedIdx, center])

  function onPointerDown(e: ReactPointerEvent<HTMLDivElement>) {
    if (e.pointerType !== 'mouse' || e.button !== 0) return // touch and pen scroll natively
    const el = scroller.current!
    cancelAnimationFrame(glideFrame.current)
    const d = drag.current
    userScrolled.current = true
    Object.assign(d, { moved: false, startX: e.clientX, startLeft: el.scrollLeft, lastX: e.clientX, lastT: performance.now(), v: 0 })

    const move = (ev: PointerEvent) => {
      const dx = ev.clientX - d.startX
      if (Math.abs(dx) > 3) d.moved = true
      if (!d.moved) return
      el.scrollLeft = d.startLeft - dx
      el.style.cursor = 'grabbing'
      const now = performance.now()
      d.v = (ev.clientX - d.lastX) / Math.max(1, now - d.lastT)
      d.lastX = ev.clientX
      d.lastT = now
    }
    const up = () => {
      window.removeEventListener('pointermove', move)
      el.style.cursor = ''
      if (d.moved && Math.abs(d.v) > 0.15 && !reducedMotion()) {
        let v = d.v * 16
        const step = () => {
          el.scrollLeft -= v
          v *= 0.94
          if (Math.abs(v) > 0.4) glideFrame.current = requestAnimationFrame(step)
        }
        glideFrame.current = requestAnimationFrame(step)
      }
      setTimeout(() => (d.moved = false), 0) // after the click that ends the drag has been ignored
    }
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up, { once: true })
  }

  const first = Math.max(0, Math.floor((win.left - PAD) / STRIDE) - OVERSCAN)
  const last = Math.min(races.length, Math.ceil((win.left + win.width) / STRIDE) + OVERSCAN)
  const trackWidth = PAD * 2 + Math.max(0, races.length) * STRIDE - GAP

  return (
    <header className="relative z-20 flex shrink-0 flex-wrap items-stretch border-b border-line bg-obsidian-950 md:h-[92px] md:flex-nowrap">
      <div className="flex h-12 w-full shrink-0 items-center justify-between gap-3 border-b border-line px-4 md:h-auto md:w-auto md:justify-start md:border-b-0 md:border-r lg:gap-5 lg:px-6">
        <p className="font-display text-base font-bold leading-none tracking-[0.2em] text-silver-100 lg:text-lg">
          F1<span className="text-laser-400">·</span>PREDICT
        </p>
        <div className="flex items-center gap-1" role="group" aria-label="Season">
          <button
            type="button"
            aria-label="Previous season"
            disabled={seasonIdx <= 0}
            onClick={() => sel.selectSeason(sel.seasons[seasonIdx - 1])}
            className="grid h-7 w-7 place-items-center text-silver-400 hover:text-silver-100 disabled:opacity-30"
          >
            <ChevronLeft className="h-4 w-4" aria-hidden />
          </button>
          <span className="w-11 text-center font-display text-base font-semibold tabular-nums text-silver-100">{sel.season}</span>
          <button
            type="button"
            aria-label="Next season"
            disabled={seasonIdx < 0 || seasonIdx >= sel.seasons.length - 1}
            onClick={() => sel.selectSeason(sel.seasons[seasonIdx + 1])}
            className="grid h-7 w-7 place-items-center text-silver-400 hover:text-silver-100 disabled:opacity-30"
          >
            <ChevronRight className="h-4 w-4" aria-hidden />
          </button>
        </div>
      </div>

      <div
        ref={scroller}
        role="group"
        aria-label={`${sel.season} calendar, ${races.length} races`}
        onScroll={sync}
        onPointerDown={onPointerDown}
        onTouchStart={() => (userScrolled.current = true)}
        className="scroll-none relative h-[96px] w-full min-w-0 touch-pan-x md:h-auto md:w-auto md:flex-1 select-none overflow-x-auto overflow-y-hidden [mask-image:linear-gradient(to_right,transparent,black_28px,black_calc(100%-28px),transparent)]"
      >
        <div ref={track} className="relative h-full" style={{ width: trackWidth }}>
          {sel.calendarPending &&
            Array.from({ length: 8 }, (_, i) => (
              <div
                key={i}
                className="absolute top-3 h-[72px] overflow-hidden border border-line bg-obsidian-800/60"
                style={{ left: PAD + i * STRIDE, width: CARD_W }}
                aria-hidden
              >
                <span className="absolute inset-y-0 w-1/2 bg-gradient-to-r from-transparent via-silver-400/10 to-transparent motion-safe:animate-[scan_1.6s_linear_infinite]" />
              </div>
            ))}
          {races.slice(first, last).map((race, k) => (
            <RaceCard
              key={race.round}
              race={race}
              selected={race.round === sel.selectedRace?.round}
              style={{ left: PAD + (first + k) * STRIDE }}
              onSelect={() => !drag.current.moved && sel.selectRace(race.round)}
            />
          ))}
        </div>
      </div>

      <div className="hidden shrink-0 items-center gap-1 border-l border-line px-2 md:flex lg:px-4">
        {sel.calendarDegraded && (
          <span className="hud-label mr-2 hidden text-signal-amber xl:block" title="The live schedule isn't reachable; showing races the archive knows">
            ARCHIVE VIEW
          </span>
        )}
        <ApiLed />
        <button
          type="button"
          aria-label="Previous race"
          title="Previous race  ["
          disabled={selectedIdx <= 0}
          onClick={() => sel.stepRace(-1)}
          className="grid h-9 w-9 place-items-center text-silver-300 hover:text-laser-400 disabled:opacity-30"
        >
          <ChevronLeft className="h-5 w-5" aria-hidden />
        </button>
        <button
          type="button"
          aria-label="Next race"
          title="Next race  ]"
          disabled={selectedIdx < 0 || selectedIdx >= races.length - 1}
          onClick={() => sel.stepRace(1)}
          className="grid h-9 w-9 place-items-center text-silver-300 hover:text-laser-400 disabled:opacity-30"
        >
          <ChevronRight className="h-5 w-5" aria-hidden />
        </button>
      </div>
    </header>
  )
}
