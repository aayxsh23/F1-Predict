import type { ReactNode } from 'react'

import type { RaceDriver } from '@/lib/raceView'

export type Rank = 1 | 2 | 3

/* The podium composition, shared by the 3D stage and the CSS fallback so both
   read as the same picture. `ndc` is the horizontal screen position (-1..1) the
   car should land on, whatever the stage aspect; `z` is depth behind P1; `scale`
   is the on-screen size relative to P1 (the brief: 1.0 / 0.84 / 0.72); `yaw` turns
   the nose in toward the winner; `delay` staggers the reveal, last place first. */
export interface Slot {
  rank: Rank
  ndc: number
  z: number
  scale: number
  yaw: number
  delay: number
  key: number // lighting key: P3 sits in a lower key than P1
}

export const SLOTS: Record<Rank, Slot> = {
  1: { rank: 1, ndc: 0, z: 0, scale: 1, yaw: -0.4, delay: 0.5, key: 1 },
  2: { rank: 2, ndc: 0.56, z: -2.5, scale: 0.84, yaw: -0.55, delay: 0.25, key: 0.78 },
  3: { rank: 3, ndc: -0.56, z: -5, scale: 0.72, yaw: 0.5, delay: 0, key: 0.58 },
}

export const RANKS: Rank[] = [3, 2, 1] // back to front

export interface StageCar {
  rank: Rank
  driver: RaceDriver | null // null draws an unlit ghost while there is nothing to show
}

export interface StageProps {
  cars: StageCar[]
  raceKey: string // a new race replays the reveal
  hovered: Rank | null
  onHover: (rank: Rank | null) => void
  onSelect: (driver: RaceDriver) => void
  renderHud: (driver: RaceDriver, rank: Rank) => ReactNode
}
