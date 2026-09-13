import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'

import { useAskAgent } from '@/lib/queries'
import type { ChatMessage } from '@/lib/types'

export interface CopilotViewContext {
  season?: number
  round?: number
  driver?: string
  location?: string
}

interface HighlightToken {
  value: string
  nonce: number
}

interface CopilotContextValue {
  messages: ChatMessage[]
  isPending: boolean
  ask: (prompt: string) => void
  clearThread: () => void
  dockCollapsed: boolean
  setDockCollapsed: (collapsed: boolean) => void
  viewContext: CopilotViewContext
  highlightToken: HighlightToken | null
  triggerHighlight: (value: string) => void
}

const CopilotContext = createContext<CopilotContextValue | null>(null)

/* No react-router import here on purpose -- WorkbenchShell is the only
place router state (season/round/driver) feeds in, passed down as the
viewContext prop. Keeps this provider's own state machine easy to reason
about in isolation from routing. */
export function CopilotProvider({
  viewContext,
  children,
}: {
  viewContext: CopilotViewContext
  children: ReactNode
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [conversationId, setConversationId] = useState<string | undefined>()
  // Below lg: the dock renders as a full-screen (mobile) or slide-over
  // (tablet) overlay, not an inline pane -- defaulting it open there would
  // bury the entire page behind the chat on first load. Only the lg:+ 3-pane
  // layout is meant to start with the dock open.
  const [dockCollapsed, setDockCollapsed] = useState(
    () => typeof window !== 'undefined' && window.innerWidth < 1024,
  )
  const [highlightToken, setHighlightToken] = useState<HighlightToken | null>(null)
  const mutation = useAskAgent()

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if ((e.ctrlKey || e.metaKey) && e.key === '\\') {
        e.preventDefault()
        setDockCollapsed((prev) => !prev)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  function ask(prompt: string) {
    const trimmed = prompt.trim()
    if (!trimmed || mutation.isPending) return

    setDockCollapsed(false)
    setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: 'user', text: trimmed }])

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

  function clearThread() {
    setMessages([])
    setConversationId(undefined)
  }

  function triggerHighlight(value: string) {
    setHighlightToken({ value, nonce: Date.now() })
  }

  return (
    <CopilotContext.Provider
      value={{
        messages,
        isPending: mutation.isPending,
        ask,
        clearThread,
        dockCollapsed,
        setDockCollapsed,
        viewContext,
        highlightToken,
        triggerHighlight,
      }}
    >
      {children}
    </CopilotContext.Provider>
  )
}

export function useCopilot() {
  const ctx = useContext(CopilotContext)
  if (!ctx) throw new Error('useCopilot must be used inside a CopilotProvider')
  return ctx
}
