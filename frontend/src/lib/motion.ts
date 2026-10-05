import { useReducedMotion, useSpring, type Transition } from 'framer-motion'
import type { PointerEvent } from 'react'

export const spring = {
  snappy: { type: 'spring', stiffness: 520, damping: 34 },
  soft: { type: 'spring', stiffness: 260, damping: 26 },
} satisfies Record<string, Transition>

/** The primary CTA leans a little toward the pointer and springs home on leave. */
export function useMagnet(k = 0.25) {
  const reduce = useReducedMotion()
  const x = useSpring(0, spring.snappy)
  const y = useSpring(0, spring.snappy)
  return {
    style: { x, y },
    onPointerMove: (e: PointerEvent<HTMLElement>) => {
      if (reduce || e.pointerType !== 'mouse') return
      const r = e.currentTarget.getBoundingClientRect()
      x.set((e.clientX - r.left - r.width / 2) * k)
      y.set((e.clientY - r.top - r.height / 2) * k)
    },
    onPointerLeave: () => {
      x.set(0)
      y.set(0)
    },
  }
}
