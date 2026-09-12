import { Send } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/Button'
import { useAskAgent } from '@/lib/queries'
import type { ChatMessage } from '@/lib/types'

const EXAMPLES = [
  'Who is leading the drivers championship?',
  "What's the gap between the top two constructors?",
  'What does the championship leader need to do to win the title?',
]

export function Agent() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [conversationId, setConversationId] = useState<string | undefined>()
  const mutation = useAskAgent()
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, mutation.isPending])

  function send(text: string) {
    const trimmed = text.trim()
    if (!trimmed || mutation.isPending) return

    setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: 'user', text: trimmed }])
    setInput('')

    mutation.mutate(
      { message: trimmed, conversationId },
      {
        onSuccess: (res) => {
          setConversationId(res.conversation_id)
          setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: 'assistant', text: res.reply }])
        },
        onError: () => {
          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role: 'assistant',
              text: "Couldn't reach the live data right now — try again in a moment.",
            },
          ])
        },
      },
    )
  }

  return (
    <div className="mx-auto flex h-[calc(100dvh-3.5rem)] max-w-2xl flex-col px-4 md:h-[calc(100dvh-3.5rem)] md:px-6">
      <div className="shrink-0 py-6">
        <h1 className="text-2xl font-semibold tracking-tight text-text-primary md:text-3xl">Agent</h1>
        <p className="mt-1 text-sm text-text-secondary">
          Live driver and constructor standings, and title-scenario questions — answered from real, current data.
        </p>
      </div>

      <div ref={listRef} className="flex-1 space-y-3 overflow-y-auto pb-4">
        {messages.length === 0 && (
          <div className="flex flex-col gap-2">
            {EXAMPLES.map((q) => (
              <button
                key={q}
                onClick={() => send(q)}
                className="rounded-md border border-border-default bg-surface px-4 py-3 text-left text-sm text-text-secondary transition-colors duration-150 hover:bg-surface-sunken hover:text-text-primary"
              >
                {q}
              </button>
            ))}
          </div>
        )}

        {messages.map((m) => (
          <div key={m.id} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <p
              className={`max-w-[85%] whitespace-pre-line rounded-md px-4 py-2.5 text-sm leading-relaxed ${
                m.role === 'user'
                  ? 'bg-accent text-white'
                  : 'border border-border-default bg-surface text-text-primary'
              }`}
            >
              {m.text}
            </p>
          </div>
        ))}

        {mutation.isPending && (
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
          send(input)
        }}
        className="shrink-0 border-t border-border-default py-4"
      >
        <div className="flex items-center gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about standings or a title scenario…"
            className="h-10 flex-1 rounded-md border border-border-default bg-surface px-3 text-sm text-text-primary outline-none placeholder:text-text-muted focus-visible:ring-2 focus-visible:ring-accent"
          />
          <Button type="submit" variant="primary" size="md" disabled={!input.trim() || mutation.isPending}>
            <Send className="h-4 w-4" strokeWidth={2} />
            <span className="sr-only">Send</span>
          </Button>
        </div>
      </form>
    </div>
  )
}
