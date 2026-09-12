import { Check, Circle } from 'lucide-react'

import { Badge } from '@/components/ui/Badge'
import type { KnownSessions } from '@/lib/types'

const LABELS: Record<keyof KnownSessions, string> = {
  practice: 'Practice',
  qualifying: 'Qualifying',
  grid: 'Grid',
  compound: 'Compound',
}

export function KnownSessionsBadges({ sessions }: { sessions: KnownSessions }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {(Object.keys(LABELS) as (keyof KnownSessions)[]).map((key) => {
        const known = sessions[key]
        return (
          <Badge key={key} tone={known ? 'known' : 'pending'}>
            {known ? <Check className="h-3 w-3" strokeWidth={2.5} /> : <Circle className="h-3 w-3" strokeWidth={2.5} />}
            {LABELS[key]}
          </Badge>
        )
      })}
    </div>
  )
}
