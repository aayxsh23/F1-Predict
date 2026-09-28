import { motion, useReducedMotion } from 'framer-motion'
import type { CSSProperties, ReactNode } from 'react'

import { cn } from '@/lib/cn'
import { COMPOUNDS, UNKNOWN_COMPOUND_COLOR } from '@/lib/teams'

import { RANKS, SLOTS, type Rank, type StageProps } from './slots'

/* The low-power stage: the same podium as flat SVG cut-outs stacked at different
   depths and turned with a CSS perspective, so the 3/4 tilt still gives real
   parallax (front wing and wheels slide against the nose and rear wing). Also
   what shows while the 3D chunk downloads, so the hero is never empty. */

const CARBON = 'var(--color-obsidian-700)'
const EDGE = 'var(--color-obsidian-600)'
const SILVER = 'var(--color-silver-300)'
const LASER = 'var(--color-laser-400)'

// [layer, translateZ px]: rear wing furthest back, front wing nearest
const DEPTHS = [-70, -40, -10, 22, 56, 90] as const

function Layers({ accent, compound, ghost }: { accent: string; compound: string | null; ghost: boolean }) {
  const band = compound && COMPOUNDS[compound] ? COMPOUNDS[compound].color : UNKNOWN_COMPOUND_COLOR
  const layer = (i: number, children: ReactNode) => (
    <svg
      key={i}
      viewBox="0 0 400 230"
      className="absolute inset-0 h-full w-full overflow-visible"
      style={{ transform: `translateZ(${DEPTHS[i]}px)` }}
      aria-hidden
    >
      {children}
    </svg>
  )
  return (
    <div className={cn('relative h-full w-full [transform-style:preserve-3d]', ghost && 'opacity-30 saturate-0')}>
      {layer(0, <>
        <rect x="132" y="26" width="136" height="10" fill={CARBON} stroke={EDGE} />
        <rect x="140" y="14" width="120" height="7" fill={SILVER} opacity="0.85" />
        <rect x="128" y="8" width="6" height="36" fill={CARBON} stroke={EDGE} />
        <rect x="266" y="8" width="6" height="36" fill={CARBON} stroke={EDGE} />
        <rect x="128" y="8" width="6" height="4" fill={accent} />
        <rect x="266" y="8" width="6" height="4" fill={accent} />
        <path d="M195 44h10l2 54h-14z" fill={CARBON} stroke={EDGE} />
      </>)}
      {layer(1, <>
        <rect x="84" y="86" width="56" height="112" rx="9" fill="var(--color-obsidian-950)" stroke={EDGE} />
        <rect x="260" y="86" width="56" height="112" rx="9" fill="var(--color-obsidian-950)" stroke={EDGE} />
        <rect x="84" y="112" width="5" height="60" rx="2" fill={band} />
        <rect x="311" y="112" width="5" height="60" rx="2" fill={band} />
        <path d="M172 60h56l8 72h-72z" fill={CARBON} stroke={EDGE} />
        <rect x="184" y="50" width="32" height="26" rx="4" fill="var(--color-obsidian-950)" stroke={EDGE} />
      </>)}
      {layer(2, <>
        <path d="M120 114l46-10 4 72-52 4z" fill={CARBON} stroke={EDGE} />
        <path d="M280 114l-46-10-4 72 52 4z" fill={CARBON} stroke={EDGE} />
        <path d="M120 114l46-10" stroke={accent} strokeWidth="2" fill="none" />
        <path d="M280 114l-46-10" stroke={accent} strokeWidth="2" fill="none" />
        <path d="M108 200h184l10 10H98z" fill="var(--color-obsidian-950)" stroke={EDGE} />
      </>)}
      {layer(3, <>
        <path d="M172 82q28-24 56 0l-6 70q-22 44-44 0z" fill={CARBON} stroke={EDGE} />
        <path d="M186 148h28l-8 54h-12z" fill={CARBON} stroke={EDGE} />
        <path d="M170 102q30-46 60 0" stroke="var(--color-silver-400)" strokeWidth="5" fill="none" strokeLinecap="round" />
        <path d="M200 64v34" stroke="var(--color-silver-400)" strokeWidth="5" strokeLinecap="round" />
        <circle cx="200" cy="96" r="13" fill={accent} />
        <rect x="192" y="92" width="16" height="7" rx="3" fill="var(--color-obsidian-950)" />
      </>)}
      {layer(4, <>
        <path d="M172 152L110 160M172 170L110 178" stroke="var(--color-silver-400)" strokeWidth="2" fill="none" />
        <rect x="54" y="118" width="54" height="86" rx="9" fill="var(--color-obsidian-950)" stroke={EDGE} />
        <rect x="292" y="118" width="54" height="86" rx="9" fill="var(--color-obsidian-950)" stroke={EDGE} />
        <rect x="54" y="140" width="6" height="44" rx="2" fill={band} />
        <rect x="340" y="140" width="6" height="44" rx="2" fill={band} />
      </>)}
      {layer(5, <>
        <rect x="46" y="199" width="308" height="8" fill={SILVER} opacity="0.85" />
        <rect x="28" y="207" width="344" height="11" fill={CARBON} stroke={EDGE} />
        <rect x="26" y="180" width="8" height="42" fill={CARBON} stroke={EDGE} />
        <rect x="366" y="180" width="8" height="42" fill={CARBON} stroke={EDGE} />
        <rect x="26" y="180" width="8" height="5" fill={accent} />
        <rect x="366" y="180" width="8" height="5" fill={accent} />
        <rect x="28" y="217" width="344" height="2" fill={LASER} />
      </>)}
    </div>
  )
}

export default function PodiumStageLite({ cars, raceKey, hovered, onHover, onSelect, renderHud }: StageProps) {
  const reduced = useReducedMotion()
  return (
    <div className="absolute inset-0 [container-type:size]" key={raceKey}>
      {RANKS.map((rank: Rank) => {
        const slot = SLOTS[rank]
        const driver = cars.find((c) => c.rank === rank)?.driver ?? null
        const lit = hovered === rank
        const place: CSSProperties = {
          // spread toward the edges, but never so far that a 116px HUD leaves the stage
          left: `calc(50% + ${slot.ndc} * min(44%, ${(1 / Math.abs(slot.ndc || 1)).toFixed(3)} * (50% - 66px)))`,
          bottom: `${[6, 15, 22][rank - 1]}%`,
          zIndex: 40 - rank * 10,
          // limited by stage height as well as width, so a short stage never pushes the HUDs off the top
          width: `clamp(120px, min(${34 * slot.scale}%, ${56 * slot.scale}cqh), ${520 * slot.scale}px)`,
          // on the wrapper, not the 3D element: a filter would flatten the layers and kill the parallax
          filter: `brightness(${0.55 + slot.key * 0.45})`,
        }
        return (
          <motion.div
            key={rank}
            className="absolute flex flex-col items-center"
            style={{ ...place, x: '-50%' }}
            initial={reduced ? false : { opacity: 0, y: 46, scale: 0.92 }}
            animate={{ opacity: 1, y: lit ? -6 : 0, scale: 1 }}
            transition={{ duration: 0.9, delay: reduced ? 0 : slot.delay, ease: [0.16, 1, 0.3, 1] }}
          >
            {driver && <div className="mb-2 w-max">{renderHud(driver, rank)}</div>}
            <button
              type="button"
              tabIndex={-1}
              aria-hidden
              disabled={!driver}
              onPointerEnter={() => driver && onHover(rank)}
              onPointerLeave={() => onHover(null)}
              onClick={() => driver && onSelect(driver)}
              className="relative block aspect-[400/230] w-full disabled:cursor-default"
              style={{ transform: `perspective(1100px) rotateY(${slot.yaw * 52}deg) rotateX(5deg)`, transformStyle: 'preserve-3d' }}
            >
              <Layers accent={driver?.accent ?? UNKNOWN_COMPOUND_COLOR} compound={driver?.compound ?? null} ghost={!driver} />
            </button>
            {/* ground light */}
            <span
              aria-hidden
              className={cn('mt-[-4%] block h-[10%] w-[92%] rounded-[50%] border transition-[opacity,border-color] duration-300', lit ? 'border-laser-300' : 'border-laser-500')}
              style={{ opacity: (driver ? 0.85 : 0.25) * slot.key }}
            />
          </motion.div>
        )
      })}
    </div>
  )
}
