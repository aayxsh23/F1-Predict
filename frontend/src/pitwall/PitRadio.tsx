import { motion } from 'framer-motion'
import { Check, LoaderCircle, RotateCcw, Send, Square, Trash2, X } from 'lucide-react'
import { lazy, Suspense, useEffect, useRef, useState, type FormEvent, type KeyboardEvent, type RefObject } from 'react'
import { Link } from 'react-router-dom'

import { TOOL_LABELS } from './toolLabels'
import { useChat } from '@/lib/chat'
import { useHealth } from '@/lib/queries'
import { cn } from '@/lib/cn'
import { useMagnet } from '@/lib/motion'
import type { ChatMessage } from '@/lib/types'

/** Driver codes in a reply become links to that driver's "why" page. */
function linkCodes(text: string, codes: string[], href: (code: string) => string): string {
  if (!codes.length) return text
  return text.replace(new RegExp(`\\b(${codes.join('|')})\\b(?![\\]/])`, 'g'), (c) => `[${c}](${href(c)})`)
}

// The markdown renderer (react-markdown + GFM tables) is about a quarter of
// the app's JavaScript and only answers need it, so it loads with the first answer.
const AnswerMarkdown = lazy(() => import('./AnswerMarkdown'))

function Answer({ text }: { text: string }) {
  return (
    <Suspense fallback={<p className="whitespace-pre-wrap font-editorial text-[19px] leading-[1.45] text-ink">{text}</p>}>
      <AnswerMarkdown text={text} />
    </Suspense>
  )
}

function Steps({ m }: { m: ChatMessage }) {
  if (!m.steps?.length) return null
  return (
    <ul className="space-y-1" aria-label="Lookups">
      {m.steps.map((s) => (
        <li key={s.id} className="flex items-center gap-1.5 text-[11px] text-ink-3">
          {!s.done ? (
            <LoaderCircle className="size-3 motion-safe:animate-spin" aria-hidden />
          ) : s.ok ? (
            <Check className="size-3 text-gain" aria-hidden />
          ) : (
            <X className="size-3 text-signal-ink" aria-hidden />
          )}
          {TOOL_LABELS[s.name] ?? s.name}
          {!s.done && '…'}
        </li>
      ))}
    </ul>
  )
}

function Sources({ m }: { m: ChatMessage }) {
  if (!m.sources?.length || m.streaming) return null
  return (
    <div className="flex flex-col gap-1">
      <span className="pw-label text-[10px]">Grounded in</span>
      {m.sources.slice(0, 6).map((s, i) => (
        <Link
          key={`${s.filename}|${s.article}`}
          to={`/regulations?doc=${encodeURIComponent(s.filename)}${s.article ? `&article=${encodeURIComponent(s.article)}` : ''}`}
          className="text-[12px] leading-snug text-ink-2 hover:text-ink"
        >
          <span className="text-signal-ink">{i + 1}</span> {s.title}
          {s.article ? ` · Art. ${s.article}` : ''}
        </Link>
      ))}
    </div>
  )
}

interface PitRadioProps {
  badge: string
  suggestions: string[]
  codes: string[]
  driverHref: (code: string) => string
  inputRef: RefObject<HTMLTextAreaElement | null>
  onClose: () => void
}

/** The analyst, as the pit wall's radio channel: same conversation everywhere, answers set as editorial copy. */
export function PitRadio({ badge, suggestions, codes, driverHref, inputRef, onClose }: PitRadioProps) {
  const chat = useChat()
  const openQuestions = useHealth().data?.chat_model === true
  const [draft, setDraft] = useState('')
  const log = useRef<HTMLDivElement>(null)
  const magnet = useMagnet(0.25)

  useEffect(() => {
    log.current?.scrollTo({ top: log.current.scrollHeight, behavior: 'smooth' })
  }, [chat.messages])

  function submit(e?: FormEvent) {
    e?.preventDefault()
    if (!draft.trim() || chat.isPending) return
    chat.ask(draft)
    setDraft('')
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <motion.aside
      aria-label="Pit Radio: ask the race analyst"
      initial={{ opacity: 0, x: 24 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.32, ease: [0.16, 1, 0.3, 1] }}
      className="absolute inset-0 z-30 flex flex-col bg-[#0e1013]/[.97] backdrop-blur-[22px] wide:relative wide:inset-auto wide:w-[392px] wide:shrink-0 wide:border-l wide:border-white/[.08] wide:bg-carbon/60"
    >
      <div className="flex h-14 shrink-0 items-center justify-between gap-2 border-b border-white/[.07] pl-[18px] pr-2.5">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="font-headline text-[22px] font-extrabold uppercase leading-none tracking-[.02em]">Pit Radio</span>
          <span className="truncate rounded-[4px] border border-white/[.12] px-[7px] py-1 text-[11px] leading-none text-ink-2">{badge}</span>
        </div>
        <div className="flex gap-0.5">
          <button type="button" onClick={chat.clear} title="Clear conversation" aria-label="Clear conversation" className="grid size-9 place-items-center rounded-md text-ink-3 hover:bg-white/[.06] hover:text-ink">
            <Trash2 className="size-4" aria-hidden />
          </button>
          <button type="button" onClick={onClose} title="Close (Ctrl+\)" aria-label="Close Pit Radio" className="grid size-9 place-items-center rounded-md text-ink-3 hover:bg-white/[.06] hover:text-ink">
            <X className="size-4" aria-hidden />
          </button>
        </div>
      </div>

      <div ref={log} role="log" aria-live="polite" className="min-h-0 flex-1 space-y-4 overflow-y-auto px-[18px] py-5 [scrollbar-width:thin]">
        {chat.messages.length === 0 && (
          <p className="font-editorial text-[20px] italic leading-[1.4] text-ink-2">
            {openQuestions
              ? 'Ask anything about Formula 1: this weekend, the title fight, 75 years of results and records, the rules. Every number is looked up, never remembered; open questions take a few seconds.'
              : "Live driver and constructor standings, title scenarios, and the reasoning behind every prediction. Ask in plain words; every number comes from this app's own models, and rules are cited by article."}
          </p>
        )}
        {chat.messages.map((m) =>
          m.role === 'user' ? (
            <p key={m.id} className="ml-auto max-w-[88%] whitespace-pre-line rounded-lg bg-signal px-3.5 py-2.5 text-[13px] font-semibold leading-normal text-asphalt">
              {m.text}
            </p>
          ) : (
            <div key={m.id} className={cn('flex flex-col gap-2.5 border-l pl-3.5', m.error ? 'border-pending' : 'border-signal/50')}>
              <span className="pw-label text-[10px]">Pit wall</span>
              <Steps m={m} />
              {m.text ? (
                <Answer text={linkCodes(m.text, codes, driverHref)} />
              ) : (
                m.streaming && (
                  <span className="flex gap-1.5 py-1.5" role="status" aria-label="The pit wall is answering">
                    {[0, 1, 2].map((i) => (
                      <span key={i} className="size-1.5 rounded-full bg-signal motion-safe:animate-[pw-live_1.2s_ease-in-out_infinite]" style={{ animationDelay: `${i * 0.18}s` }} />
                    ))}
                  </span>
                )
              )}
              <Sources m={m} />
              {m.error && !chat.isPending && (
                <button type="button" onClick={chat.retry} className="pw-label flex items-center gap-1.5 text-signal-ink hover:text-ink">
                  <RotateCcw className="size-3" aria-hidden /> Try again
                </button>
              )}
            </div>
          ),
        )}
      </div>

      <form onSubmit={submit} className="flex flex-col gap-2 border-t border-white/[.07] px-[18px] py-3.5">
        {suggestions.map((s) => (
          <button
            key={s}
            type="button"
            disabled={chat.isPending}
            onClick={() => chat.ask(s)}
            className="rounded-md border border-white/10 px-3 py-[9px] text-left text-[12px] font-medium leading-snug text-ink-2 transition-[border-color,color,transform] duration-300 ease-[var(--ease-spring)] hover:translate-x-[3px] hover:border-white/30 hover:text-ink disabled:opacity-40"
          >
            {s}
          </button>
        ))}
        <div className="mt-1.5 flex items-end gap-2">
          <label className="min-w-0 flex-1">
            <span className="sr-only">Ask the pit wall</span>
            <textarea
              ref={inputRef}
              rows={1}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKeyDown}
              maxLength={4000}
              placeholder={openQuestions ? 'Ask anything about F1…' : 'Ask about this race, driver, or standings…'}
              className="block max-h-28 min-h-11 w-full resize-none rounded-md border border-white/[.14] bg-black/25 px-3.5 py-2.5 font-editorial text-[17px] leading-snug text-ink placeholder:italic placeholder:text-ink-3 focus:border-white/30 focus:outline-none"
            />
          </label>
          {chat.isPending ? (
            <button type="button" onClick={chat.stop} aria-label="Stop answering" className="grid size-11 shrink-0 place-items-center rounded-md border border-white/[.18] text-ink hover:border-white/30">
              <Square className="size-3.5" aria-hidden />
            </button>
          ) : (
            <motion.button
              type="submit"
              disabled={!draft.trim()}
              aria-label="Send"
              {...magnet}
              className="grid size-11 shrink-0 place-items-center rounded-md bg-signal text-asphalt disabled:opacity-40"
            >
              <Send className="size-4" aria-hidden />
            </motion.button>
          )}
        </div>
      </form>
    </motion.aside>
  )
}
