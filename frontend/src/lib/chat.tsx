import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from 'react'

import { useAskAgent } from './queries'
import type { ChatMessage } from './types'

interface ChatValue {
  messages: ChatMessage[]
  isPending: boolean
  ask: (prompt: string) => void
  retry: () => void
  clear: () => void
}

const ChatContext = createContext<ChatValue | null>(null)

/** The live-standings agent's thread. Lives above the folder deck so a
 *  conversation survives switching tabs (the inactive card unmounts). */
export function ChatProvider({ children }: { children: ReactNode }) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [conversationId, setConversationId] = useState<string | undefined>()
  const lastPrompt = useRef<string | null>(null)
  const mutation = useAskAgent()

  const send = useCallback(
    (prompt: string, echo: boolean) => {
      const text = prompt.trim()
      if (!text || mutation.isPending) return
      lastPrompt.current = text
      if (echo) setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: 'user', text }])
      mutation.mutate(
        { message: text, conversationId },
        {
          onSuccess: (res) => {
            setConversationId(res.conversation_id)
            setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: 'assistant', text: res.reply }])
          },
          onError: () =>
            setMessages((prev) => [
              ...prev,
              {
                id: crypto.randomUUID(),
                role: 'assistant',
                text: "Couldn't reach the live standings right now. The backend may still be waking up.",
                error: true,
              },
            ]),
        },
      )
    },
    [mutation, conversationId],
  )

  const value: ChatValue = {
    messages,
    isPending: mutation.isPending,
    ask: (prompt) => send(prompt, true),
    retry: () => lastPrompt.current && send(lastPrompt.current, false),
    clear: () => {
      setMessages([])
      setConversationId(undefined)
      lastPrompt.current = null
    },
  }
  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>
}

export function useChat(): ChatValue {
  const ctx = useContext(ChatContext)
  if (!ctx) throw new Error('useChat must be used inside a ChatProvider')
  return ctx
}
