import { motion, useReducedMotion } from 'framer-motion'
import { ArrowDown, ArrowUp, ArrowUpRight, Minus } from 'lucide-react'
import type { ReactNode } from 'react'

import { cn } from '@/lib/cn'
import { formatPct } from '@/lib/format'
import type { RaceDriver } from '@/lib/raceView'
import { COMPOUNDS } from '@/lib/teams'

import { SLOTS, type Rank } from './slots'

function Pill({ tone = 'plain', children }: { tone?: 'plain' | 'laser'; children: ReactNode }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 border px-1.5 py-[3px] font-mono text-micro leading-none',
        tone === 'laser' ? 'border-laser-500 bg-laser-900 text-laser-300' : 'border-line-strong text-silver-300',
      )}
    >
      {children}
    </span>
  )
}

function GridDelta({ delta }: { delta: number }) {
  const Icon = delta > 0 ? ArrowUp : delta < 0 ? ArrowDown : Minus
  return (
    <Pill>
      <Icon className={cn('h-2.5 w-2.5', delta > 0 && 'text-laser-400', delta < 0 && 'text-signal-coral')} strokeWidth={3} aria-hidden />
      {delta === 0 ? 'HOLDS GRID' : `${Math.abs(delta)} VS GRID`}
    </Pill>
  )
}

/** One HUD callout pinned above a car. Only real model output is drawn: the odds
 *  pill appears once the API supplies probabilities, the tyre pill once a
 *  starting compound is known, the result pill once the race is decided. */
export function CarTelemetryBadge({
  driver,
  rank,
  hovered,
  onHover,
  onExplain,
}: {
  driver: RaceDriver
  rank: Rank
  hovered: boolean
  onHover: (rank: Rank | null) => void
  onExplain: () => void
}) {
  const reduced = useReducedMotion()
  const isWinner = rank === 1
  const odds = isWinner ? driver.win : driver.podium
  const compound = driver.compound ? COMPOUNDS[driver.compound] : undefined
  const hit = driver.actual !== null && driver.actual <= 3
  const full = isWinner // P1 carries the full callout; P2/P3 stay compact so the stage never crowds

  return (
    <motion.div
      initial={reduced ? false : { opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: reduced ? 0 : SLOTS[rank].delay + 0.9, ease: [0.16, 1, 0.3, 1] }}
      onPointerEnter={() => onHover(rank)}
      onPointerLeave={() => onHover(null)}
      className={cn('cut-frame cut-all w-[116px] text-left', full ? 'sm:w-[214px]' : 'sm:w-[172px]', hovered && 'frame-laser')}
      style={{ ['--fill' as string]: hovered ? undefined : 'color-mix(in srgb, var(--color-obsidian-900) 88%, transparent)' }}
    >
      <div className="px-2.5 pb-2 pt-1.5 sm:px-3 sm:pb-2.5 sm:pt-2">
        <div className="flex items-baseline gap-2">
          <span className="hud-label text-laser-300">P{rank}</span>
          <span className={cn('font-display font-bold leading-none tracking-wide text-silver-100', isWinner ? 'text-2xl sm:text-3xl' : 'text-xl sm:text-2xl')}>
            {driver.code}
          </span>
          {driver.number !== null && <span className="ml-auto hidden font-mono text-sm text-silver-300 sm:inline">#{driver.number}</span>}
        </div>

        <div className="mt-1.5 flex items-center gap-2 text-hud text-silver-400">
          <span className="h-[2px] w-6 shrink-0" style={{ background: driver.accent }} aria-hidden />
          <span className={cn('truncate', full ? 'hidden sm:inline' : 'sr-only')}>{driver.team}</span>
        </div>

        <div className={cn('mt-1.5 flex flex-wrap gap-1.5 [&>*:nth-child(n+2)]:hidden', full ? 'sm:[&>*:nth-child(n+2)]:inline-flex' : 'sm:[&>*:nth-child(2)]:inline-flex')}>
          {odds !== null && (
            <Pill tone="laser">
              {formatPct(odds)} {isWinner ? 'WIN' : 'PODIUM'}
            </Pill>
          )}
          {driver.actual !== null && (
            <Pill tone={hit ? 'laser' : 'plain'}>RESULT P{driver.actual}</Pill>
          )}
          {driver.delta !== null && <GridDelta delta={driver.delta} />}
          {compound && (
            <Pill>
              <span className="h-2 w-2 rounded-full" style={{ background: compound.color }} aria-hidden />
              {compound.label.toUpperCase()}
            </Pill>
          )}
        </div>

        <button
          type="button"
          onClick={onExplain}
          className="hud-label mt-1.5 hidden items-center gap-1 text-silver-300 hover:text-laser-300 sm:flex"
          aria-label={`Explain ${driver.code}'s prediction`}
        >
          Why this call <ArrowUpRight className="h-3 w-3" strokeWidth={2.5} aria-hidden />
        </button>
      </div>
    </motion.div>
  )
}
