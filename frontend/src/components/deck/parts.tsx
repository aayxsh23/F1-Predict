import type { Variants } from 'framer-motion'
import type { ReactNode } from 'react'

/* One entrance for every dossier sheet: the card's sections rise into place a
   beat apart. Sections are the unit, never individual rows. */
export const stagger: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.05, delayChildren: 0.08 } },
}

export const rise: Variants = {
  hidden: { opacity: 0, y: 14 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] } },
}

/** A sheet with nothing to show yet: says what is missing and what to do. */
export function SheetMessage({ title, children, action }: { title: string; children: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex h-full flex-col items-center justify-center px-6 text-center" role="status">
      <p className="font-display text-lg font-semibold uppercase tracking-[0.08em] text-silver-100">{title}</p>
      <p className="mt-1.5 max-w-md text-sm text-silver-400">{children}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

/** Scanning placeholder rows, so a cold-starting backend never leaves a blank sheet. */
export function SheetSkeleton({ rows = 8 }: { rows?: number }) {
  return (
    <div className="space-y-2 p-4 lg:p-6" aria-hidden>
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="relative h-7 overflow-hidden bg-obsidian-700/60">
          <span className="absolute inset-y-0 w-1/3 bg-gradient-to-r from-transparent via-silver-400/10 to-transparent motion-safe:animate-[scan_1.6s_linear_infinite]" />
        </div>
      ))}
    </div>
  )
}
