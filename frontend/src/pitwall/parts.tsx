import { useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { ArrowRight, RotateCw } from 'lucide-react'
import { useRef, type KeyboardEvent, type ReactNode } from 'react'
import { Link } from 'react-router-dom'

import { cn } from '@/lib/cn'
import { spring, useMagnet } from '@/lib/motion'
import type { Target } from '@/lib/types'

import { LENSES } from './data'

/** One control re-frames the page across the four targets. ←/→ move along it. */
export function LensTabs({ lens, onChange, compact = false, id }: { lens: Target; onChange: (t: Target) => void; compact?: boolean; id: string }) {
  const tabs = useRef<Array<HTMLButtonElement | null>>([])

  const onKeyDown = (e: KeyboardEvent, i: number) => {
    const step = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0
    if (!step) return
    e.preventDefault()
    const next = (i + step + LENSES.length) % LENSES.length
    onChange(LENSES[next].id)
    tabs.current[next]?.focus()
  }

  return (
    <div
      role="tablist"
      aria-label="Which prediction"
      className={cn(
        'relative border border-white/10',
        compact ? 'flex w-max max-w-full flex-wrap gap-1 rounded-[7px] p-[3px]' : 'grid w-full grid-cols-4 rounded-lg bg-white/[.03] p-1 wide:flex wide:w-max',
      )}
    >
      {LENSES.map((l, i) => {
        const active = l.id === lens
        return (
          <button
            key={l.id}
            ref={(el) => {
              tabs.current[i] = el
            }}
            type="button"
            role="tab"
            aria-selected={active}
            tabIndex={active ? 0 : -1}
            onClick={() => onChange(l.id)}
            onKeyDown={(e) => onKeyDown(e, i)}
            className={cn(
              'relative shrink-0 font-semibold uppercase transition-colors duration-200',
              compact ? 'h-8 px-[11px] text-[11px] leading-none tracking-[.06em]' : 'h-10 whitespace-nowrap text-[12px] leading-none tracking-[.06em] wide:w-[150px]',
              active ? 'text-ink' : 'text-ink-3 hover:text-ink',
            )}
          >
            {active && (
              <motion.span
                layoutId={`lens-pill-${id}`}
                transition={spring.snappy}
                className="absolute inset-0 rounded-[5px] bg-white/[.09] shadow-[inset_0_1px_0_rgb(255_255_255/.1),0_0_0_1px_rgb(255_255_255/.12)]"
                aria-hidden
              />
            )}
            {compact ? (
              <span className="relative">{l.label}</span>
            ) : (
              <>
                <span className="relative wide:hidden">{l.label.replace('Grid → finish', 'Grid→fin')}</span>
                <span className="relative hidden wide:inline">{l.label}</span>
              </>
            )}
          </button>
        )
      })}
    </div>
  )
}

const MotionLink = motion.create(Link)

/** The one primary action on a view: signal fill, leans toward the pointer. */
export function MagneticLink({ to, children, className }: { to: string; children: ReactNode; className?: string }) {
  const magnet = useMagnet(0.25)
  return (
    <MotionLink
      to={to}
      viewTransition
      {...magnet}
      className={cn(
        'flex h-11 shrink-0 items-center gap-2.5 rounded-md bg-signal px-[18px] text-[13px] font-bold leading-none tracking-[.04em] text-asphalt shadow-[0_0_0_1px_rgb(255_90_71/.4),0_10px_30px_rgb(255_90_71/.25)] transition-colors duration-200 hover:bg-signal-ink',
        className,
      )}
    >
      {children}
      <ArrowRight className="size-4" aria-hidden />
    </MotionLink>
  )
}

/** Race or driver identity at poster scale. Condensed caps run about half an em
 *  a glyph, so the word is sized to its container and capped at the design size. */
export function Mega({ text, max, className }: { text: string; max: number; className?: string }) {
  return (
    <div className="@container">
      <h1
        className={cn('font-headline font-black uppercase leading-[.78] tracking-[-.012em]', className)}
        style={{ fontSize: `min(${max}px, ${(100 / (Math.max(text.length, 3) * 0.5)).toFixed(1)}cqw)` }}
      >
        {text}
      </h1>
    </div>
  )
}

/** What a view says when it has nothing to show: what's missing, and where to go. */
export function Notice({ title, children, action, busy = false }: { title: string; children?: ReactNode; action?: ReactNode; busy?: boolean }) {
  return (
    <div role="status" aria-busy={busy} className="flex min-h-[60vh] flex-col items-start justify-center gap-4 px-4 md:px-8">
      <p className={cn('font-headline text-[44px] font-extrabold uppercase leading-none', busy && 'motion-safe:animate-[pw-live_1.6s_ease-in-out_infinite]')}>{title}</p>
      {children && <p className="max-w-[560px] font-editorial text-[22px] italic leading-[1.35] text-ink-2">{children}</p>}
      {action}
    </div>
  )
}

/** Page padding and the rise-in every pit-wall page enters with. */
export const PAGE = 'flex flex-col px-[clamp(16px,3vw,32px)] pb-36 motion-safe:animate-[pw-in_.4s_cubic-bezier(.16,1,.3,1)]'

const shimmer = 'bg-[linear-gradient(90deg,rgb(255_255_255/.04)_0,rgb(255_255_255/.09)_50%,rgb(255_255_255/.04)_100%)] bg-[size:800px_100%] motion-safe:animate-[pw-shim_1.4s_linear_infinite]'

/** Shimmering stand-ins shaped like the page that's coming, never a spinner. */
export function Skeleton({ label = 'Loading the latest snapshot…' }: { label?: string }) {
  return (
    <div role="status" className={cn(PAGE, 'gap-[22px] pt-7')}>
      <div className={cn('h-[clamp(90px,13vw,190px)] w-[min(520px,80%)] rounded-lg', shimmer)} />
      <div className={cn('h-[18px] w-[260px] rounded', shimmer)} />
      <div className="flex flex-wrap gap-[22px]">
        <div className={cn('h-[300px] flex-[1.2_1_420px] rounded-[10px] border border-white/[.06]', shimmer)} />
        <div className={cn('h-[300px] flex-[1_1_340px] rounded-[10px] border border-white/[.06]', shimmer)} />
      </div>
      <span className="pw-label">{label}</span>
    </div>
  )
}

/** A cold-starting free-tier backend: say so, and offer the retry. */
export function Offline() {
  const qc = useQueryClient()
  return (
    <div className="flex min-h-full items-center justify-center px-5 pb-36 pt-12">
      <div className="flex max-w-[560px] flex-col items-start gap-[18px]">
        <span className="flex items-center gap-2.5 text-[11px] font-semibold uppercase leading-none tracking-[.14em] text-pending">
          <span className="size-2.5 rounded-full border-[1.5px] border-pending motion-safe:animate-[pw-pulse_1.6s_ease-out_infinite]" aria-hidden />
          API waking up
        </span>
        <h1 className="font-headline text-[clamp(56px,8vw,96px)] font-black uppercase leading-[.85]">Pit Wall isn't reachable right now</h1>
        <p className="font-editorial text-[24px] italic leading-[1.35] text-ink-2">The backend may still be waking up. Try again shortly; the last snapshot stays on screen once it answers.</p>
        <button
          type="button"
          onClick={() => qc.refetchQueries({ type: 'active' })}
          className="flex h-11 items-center gap-2.5 rounded-md bg-signal px-[18px] text-[13px] font-bold leading-none text-asphalt hover:bg-signal-ink"
        >
          <RotateCw className="size-4" aria-hidden />
          Try again
        </button>
      </div>
    </div>
  )
}
