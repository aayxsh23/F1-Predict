import { PanelRightClose, Send, Trash2 } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { Tooltip } from '@/components/ui/Tooltip'
import { cn } from '@/lib/cn'

import { splitMentions } from './copilotMentions'
import { useCopilot, type CopilotViewContext } from './CopilotProvider'

const DEFAULT_EXAMPLES = [
  'Who is leading the drivers championship?',
  "What's the gap between the top two constructors?",
  'What does the championship leader need to do to win the title?',
]

function quickActions(viewContext: CopilotViewContext): string[] {
  if (viewContext.driver) {
    return [
      `Why is ${viewContext.driver} predicted to finish there?`,
      `How does ${viewContext.driver}'s grid position affect their outlook?`,
    ]
  }
  if (viewContext.round) {
    const race = viewContext.location ? `${viewContext.location} (round ${viewContext.round})` : `round ${viewContext.round}`
    return [`What matters most for predictions at ${race}?`, ...DEFAULT_EXAMPLES.slice(0, 1)]
  }
  return DEFAULT_EXAMPLES
}

function viewContextLabel(viewContext: CopilotViewContext): string | null {
  if (viewContext.driver) return `Driver: ${viewContext.driver}`
  if (viewContext.location) return `${viewContext.location}${viewContext.round ? ` · Round ${viewContext.round}` : ''}`
  if (viewContext.round) return `Round ${viewContext.round}${viewContext.season ? ` · ${viewContext.season}` : ''}`
  return null
}

export function CopilotDock({ className }: { className?: string }) {
  const { messages, isPending, ask, clearThread, dockCollapsed, setDockCollapsed, viewContext, triggerHighlight } =
    useCopilot()
  const [input, setInput] = useState('')
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, isPending])

  if (dockCollapsed) return null

  const badgeLabel = viewContextLabel(viewContext)

  return (
    <aside
      className={cn(
        'flex w-full flex-col border-border-default bg-surface',
        className,
      )}
    >
      <header className="flex shrink-0 items-center justify-between gap-2 border-b border-border-default px-4 py-3">
        <div className="flex min-w-0 items-center gap-2">
          <h2 className="text-sm font-semibold text-text-primary">Copilot</h2>
          {badgeLabel && <Badge tone="neutral">{badgeLabel}</Badge>}
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <Tooltip label="Clear conversation">
            <button
              onClick={clearThread}
              aria-label="Clear conversation"
              className="flex h-8 w-8 items-center justify-center rounded-md text-text-secondary transition-colors duration-150 hover:bg-surface-sunken hover:text-text-primary"
            >
              <Trash2 className="h-4 w-4" strokeWidth={2} />
            </button>
          </Tooltip>
          <Tooltip label="Collapse (Ctrl+\)">
            <button
              onClick={() => setDockCollapsed(true)}
              aria-label="Collapse copilot dock"
              className="flex h-8 w-8 items-center justify-center rounded-md text-text-secondary transition-colors duration-150 hover:bg-surface-sunken hover:text-text-primary"
            >
              <PanelRightClose className="h-4 w-4" strokeWidth={2} />
            </button>
          </Tooltip>
        </div>
      </header>

      <div ref={listRef} className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {messages.length === 0 && (
          <div className="flex flex-col gap-2">
            <p className="text-sm text-text-secondary">
              Live driver and constructor standings, and title-scenario questions — answered from real, current
              data.
            </p>
            {quickActions(viewContext).map((q) => (
              <button
                key={q}
                onClick={() => ask(q)}
                className="rounded-md border border-border-default bg-surface px-3 py-2.5 text-left text-sm text-text-secondary transition-colors duration-150 hover:bg-surface-sunken hover:text-text-primary"
              >
                {q}
              </button>
            ))}
          </div>
        )}

        {messages.map((m) => (
          <div key={m.id} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <p
              className={`max-w-[90%] whitespace-pre-line rounded-md px-4 py-2.5 text-sm leading-relaxed ${
                m.role === 'user'
                  ? 'bg-accent text-white'
                  : 'border border-border-default bg-surface text-text-primary'
              }`}
            >
              {m.role === 'assistant'
                ? splitMentions(m.text).map((seg, i) =>
                    seg.isMention ? (
                      <button
                        key={i}
                        onClick={() => triggerHighlight(seg.text)}
                        className="underline decoration-dotted text-accent-text hover:decoration-solid"
                      >
                        {seg.text}
                      </button>
                    ) : (
                      <span key={i}>{seg.text}</span>
                    ),
                  )
                : m.text}
            </p>
          </div>
        ))}

        {isPending && (
          <div className="flex justify-start">
            <div className="flex items-center gap-1 rounded-md border border-border-default bg-surface px-4 py-3">
              <span className="h-1.5 w-1.5 animate-[pulse_1.2s_ease-in-out_infinite] rounded-full bg-text-muted [animation-delay:-0.4s]" />
              <span className="h-1.5 w-1.5 animate-[pulse_1.2s_ease-in-out_infinite] rounded-full bg-text-muted [animation-delay:-0.2s]" />
              <span className="h-1.5 w-1.5 animate-[pulse_1.2s_ease-in-out_infinite] rounded-full bg-text-muted" />
            </div>
          </div>
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault()
          ask(input)
          setInput('')
        }}
        className="shrink-0 border-t border-border-default p-3"
      >
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                ask(input)
                setInput('')
              }
            }}
            rows={1}
            placeholder="Ask about this race, driver, or the standings…"
            className="max-h-32 min-h-10 flex-1 resize-none rounded-md border border-border-default bg-surface px-3 py-2 text-sm text-text-primary outline-none placeholder:text-text-muted focus-visible:ring-2 focus-visible:ring-accent"
          />
          <Button type="submit" variant="primary" size="md" disabled={!input.trim() || isPending}>
            <Send className="h-4 w-4" strokeWidth={2} />
            <span className="sr-only">Send</span>
          </Button>
        </div>
      </form>
    </aside>
  )
}
