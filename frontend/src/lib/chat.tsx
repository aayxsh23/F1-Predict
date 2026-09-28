import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from 'react'

import { ApiError, streamChat } from './api'
import { useSelection } from './selection'
import type { ChatContext, ChatEvent, ChatMessage } from './types'

const HISTORY_SENT = 16 // recent messages the server gets; it keeps nothing between requests

interface ChatValue {
  messages: ChatMessage[]
  isPending: boolean
  ask: (prompt: string) => void
  retry: () => void
  stop: () => void
  clear: () => void
}

const ChatContextObj = createContext<ChatValue | null>(null)

function applyEvent(m: ChatMessage, e: ChatEvent): ChatMessage {
  switch (e.type) {
    case 'token':
      return { ...m, text: m.text + e.text }
    case 'tool_start':
      return { ...m, steps: [...(m.steps ?? []), { id: e.id, name: e.name, done: false, ok: true }] }
    case 'tool_end':
      return { ...m, steps: (m.steps ?? []).map((s) => (s.id === e.id ? { ...s, done: true, ok: e.ok } : s)) }
    case 'sources': {
      const seen = new Set((m.sources ?? []).map((s) => `${s.filename}|${s.article}`))
      return { ...m, sources: [...(m.sources ?? []), ...e.items.filter((s) => !seen.has(`${s.filename}|${s.article}`))] }
    }
    case 'error':
      return { ...m, text: m.text ? `${m.text}\n\n${e.message}` : e.message, error: true }
    default:
      return m
  }
}

/** The assistant's conversation. Lives above the tabs so it survives switching
 *  them, and sends the selected race and driver along so "why is he P3?" works. */
export function ChatProvider({ children }: { children: ReactNode }) {
  const sel = useSelection()
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isPending, setPending] = useState(false)
  const controller = useRef<AbortController | null>(null)

  const race = sel.calendar.find((r) => r.round === sel.selectedRace?.round)
  const context: ChatContext = {
    season: sel.selectedRace?.season,
    round: sel.selectedRace?.round,
    race_name: race?.eventName,
    driver: sel.driver ?? undefined,
  }
  const run = useCallback(async (history: ChatMessage[], ctx: ChatContext) => {
    const ctrl = new AbortController()
    controller.current = ctrl
    setPending(true)
    const reply: ChatMessage = { id: crypto.randomUUID(), role: 'assistant', text: '', steps: [], sources: [], streaming: true }
    setMessages([...history, reply])
    const patch = (fn: (m: ChatMessage) => ChatMessage) =>
      setMessages((prev) => prev.map((m) => (m.id === reply.id ? fn(m) : m)))
    const sent = history.filter((m) => !m.error && m.text).slice(-HISTORY_SENT).map((m) => ({ role: m.role, content: m.text }))
    try {
      await streamChat(sent, ctx, (e) => patch((m) => applyEvent(m, e)), ctrl.signal)
    } catch (err) {
      if (!ctrl.signal.aborted) {
        const message =
          err instanceof ApiError && err.status === 429
            ? err.message
            : "Couldn't reach the assistant. The backend may still be waking up."
        patch((m) => applyEvent(m, { type: 'error', message }))
      }
    } finally {
      patch((m) => ({ ...m, streaming: false }))
      setPending(false)
    }
  }, [])

  const value: ChatValue = {
    messages,
    isPending,
    ask: (prompt) => {
      const text = prompt.trim()
      if (!text || isPending) return
      void run([...messages, { id: crypto.randomUUID(), role: 'user', text }], context)
    },
    retry: () => {
      const lastUser = messages.map((m) => m.role).lastIndexOf('user')
      if (lastUser >= 0 && !isPending) void run(messages.slice(0, lastUser + 1), context)
    },
    stop: () => controller.current?.abort(),
    clear: () => {
      controller.current?.abort()
      setMessages([])
    },
  }
  return <ChatContextObj.Provider value={value}>{children}</ChatContextObj.Provider>
}

export function useChat(): ChatValue {
  const ctx = useContext(ChatContextObj)
  if (!ctx) throw new Error('useChat must be used inside a ChatProvider')
  return ctx
}
