import type { ReactNode } from 'react'

import { cn } from '@/lib/cn'

type Tone = 'known' | 'pending' | 'neutral'

const toneClasses: Record<Tone, string> = {
  known: 'bg-badge-known-bg text-badge-known-text',
  pending: 'bg-badge-pending-bg text-badge-pending-text',
  neutral: 'bg-surface-sunken text-text-secondary',
}

export function Badge({ tone = 'neutral', children }: { tone?: Tone; children: ReactNode }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-sm px-2 py-0.5 text-xs font-medium leading-4',
        toneClasses[tone],
      )}
    >
      {children}
    </span>
  )
}
