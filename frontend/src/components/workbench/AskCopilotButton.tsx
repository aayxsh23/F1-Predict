import { Sparkles } from 'lucide-react'

import { Tooltip } from '@/components/ui/Tooltip'
import { cn } from '@/lib/cn'

import { useCopilot } from './CopilotProvider'

export function AskCopilotButton({ prompt, className }: { prompt: string; className?: string }) {
  const { ask } = useCopilot()

  return (
    <Tooltip label="Ask Copilot">
      <button
        type="button"
        onClick={(e) => {
          e.preventDefault()
          e.stopPropagation()
          ask(prompt)
        }}
        aria-label="Ask Copilot"
        className={cn(
          'flex h-7 w-7 items-center justify-center rounded-sm text-text-secondary transition-colors duration-150 hover:bg-surface-sunken hover:text-accent-text',
          className,
        )}
      >
        <Sparkles className="h-4 w-4" strokeWidth={2} />
      </button>
    </Tooltip>
  )
}
