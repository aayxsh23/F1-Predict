import { motion } from 'framer-motion'
import { Lock, RotateCcw, Send } from 'lucide-react'
import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'

import { useChat } from '@/lib/chat'
import { cn } from '@/lib/cn'
import { useRaceView } from '@/lib/raceView'
import { useSelection } from '@/lib/selection'

import { rise, stagger } from './parts'

const SCENARIOS = ['Safety car', 'Virtual safety car', 'Red flag', 'Wet track']

/** Race-strategy scenario injection (safety car on a lap, pit window, stint timeline)
 *  needs a lap-by-lap simulator this project doesn't have. The controls are drawn so
 *  the surface is ready for it, but they are disabled and say why: no toggle here
 *  pretends to change a prediction. */
function LockedSimulator() {
  return (
    <aside aria-label="Scenario injection, unavailable" className="lg:border-l lg:border-line lg:pl-6">
      <div className="flex items-center justify-between gap-3">
        <p className="hud-label text-silver-200">Scenario injection</p>
        <span className="hud-label inline-flex items-center gap-1.5 text-signal-amber">
          <Lock className="h-3 w-3" strokeWidth={2.5} aria-hidden /> Simulator offline
        </span>
      </div>
      <div className="mt-3 hidden flex-wrap gap-2 lg:flex">
        {SCENARIOS.map((s) => (
          <button key={s} type="button" disabled className="hud-label cursor-not-allowed border border-line px-2.5 py-1.5 text-silver-500">
            {s}
          </button>
        ))}
        <button type="button" disabled className="hud-label cursor-not-allowed border border-dashed border-line px-2.5 py-1.5 font-mono text-silver-500">
          Lap 24
        </button>
        <button type="button" disabled className="hud-label cursor-not-allowed border border-dashed border-line px-2.5 py-1.5 text-silver-500">
          Pit window
        </button>
      </div>
      <svg viewBox="0 0 300 30" className="mt-4 hidden h-7 w-full lg:block" aria-hidden>
        <line x1="0" y1="15" x2="300" y2="15" className="stroke-line-strong" strokeDasharray="4 4" />
        {Array.from({ length: 13 }, (_, i) => (
          <line key={i} x1={i * 25} x2={i * 25} y1="9" y2="21" className="stroke-line" />
        ))}
      </svg>
      <p className="mt-2 hidden text-hud text-silver-400 lg:block">
        Race-strategy scenarios aren't available yet. Championship scenarios are live in the chat.
      </p>
    </aside>
  )
}

export function StrategistCard() {
  const sel = useSelection()
  const { view } = useRaceView(sel.selectedRace)
  const chat = useChat()
  const [draft, setDraft] = useState('')
  const log = useRef<HTMLDivElement>(null)

  const code = sel.driver ?? view?.drivers[0]?.code ?? null
  const chips = [
    { label: 'Driver standings', prompt: 'Who leads the driver standings?' },
    { label: 'Constructor standings', prompt: 'Show the constructor standings' },
    ...(code
      ? [
          { label: `${code} title fight`, prompt: `What does ${code} need to do to win the title?` },
          { label: `${code} standing`, prompt: `Where is ${code} in the driver standings?` },
        ]
      : []),
  ]

  useEffect(() => {
    log.current?.scrollTo({ top: log.current.scrollHeight, behavior: 'smooth' })
  }, [chat.messages.length, chat.isPending])

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
    <motion.div variants={stagger} initial="hidden" animate="show" className="grid h-full grid-rows-[auto_minmax(0,1fr)] lg:grid-cols-[minmax(0,1fr)_340px] lg:grid-rows-1 lg:gap-x-6 lg:px-6">
      <motion.div variants={rise} className="order-first border-b border-line px-4 py-3 lg:order-last lg:border-0 lg:px-0 lg:py-4">
        <LockedSimulator />
      </motion.div>

      <motion.div variants={rise} className="flex min-h-0 flex-col">
        <div ref={log} role="log" aria-live="polite" aria-label="Copilot conversation" className="scroll-thin min-h-0 flex-1 space-y-3 overflow-y-auto px-4 py-4 lg:px-0">
          {chat.messages.length === 0 && (
            <div className="max-w-md">
              <p className="font-display text-lg font-semibold uppercase tracking-[0.06em] text-silver-100">Ask about the championship</p>
              <p className="mt-1.5 text-sm text-silver-400">
                Live standings and title scenarios. Try what a driver needs to win the title, or how the constructors stand.
              </p>
            </div>
          )}
          {chat.messages.map((m) => (
            <div
              key={m.id}
              className={cn(
                'max-w-[88%] whitespace-pre-line border px-3 py-2 text-sm leading-relaxed',
                m.role === 'user' ? 'ml-auto border-line-strong bg-obsidian-700 text-silver-100' : m.error ? 'border-signal-coral text-silver-200' : 'border-laser-700 bg-laser-900/50 text-silver-200',
              )}
            >
              {m.text}
              {m.error && (
                <button type="button" onClick={chat.retry} className="hud-label mt-2 flex items-center gap-1.5 text-laser-300 hover:text-laser-400">
                  <RotateCcw className="h-3 w-3" aria-hidden /> Try again
                </button>
              )}
            </div>
          ))}
          {chat.isPending && (
            <div className="flex w-16 items-center gap-1.5 border border-laser-700 bg-laser-900/50 px-3 py-3" role="status" aria-label="Copilot is answering">
              {[0, 1, 2].map((i) => (
                <span key={i} className="h-1.5 w-1.5 rounded-full bg-laser-400 motion-safe:animate-[pulse-dot_1.2s_ease-in-out_infinite]" style={{ animationDelay: `${i * 0.18}s` }} />
              ))}
            </div>
          )}
        </div>

        <form onSubmit={submit} className="border-t border-line px-4 py-3 lg:px-0">
          <div className="scroll-none mb-2 flex gap-2 overflow-x-auto">
            {chips.map((c) => (
              <button
                key={c.label}
                type="button"
                disabled={chat.isPending}
                onClick={() => chat.ask(c.prompt)}
                className="hud-label shrink-0 border border-line-strong px-2.5 py-1.5 text-silver-300 hover:border-laser-500 hover:text-laser-300 disabled:opacity-40"
              >
                {c.label}
              </button>
            ))}
            {chat.messages.length > 0 && (
              <button type="button" onClick={chat.clear} className="hud-label ml-auto shrink-0 px-2.5 py-1.5 text-silver-400 hover:text-silver-200">
                Clear
              </button>
            )}
          </div>
          <div className="flex items-end gap-2">
            <label className="min-w-0 flex-1">
              <span className="sr-only">Ask the strategist</span>
              <textarea
                rows={1}
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder="What does NOR need to win the title?"
                className="block max-h-28 min-h-10 w-full resize-none border border-line-strong bg-obsidian-900 px-3 py-2 text-sm text-silver-100 placeholder:text-silver-400 focus:border-laser-500"
              />
            </label>
            <button
              type="submit"
              disabled={!draft.trim() || chat.isPending}
              aria-label="Send"
              className="grid h-10 w-10 shrink-0 place-items-center border border-laser-500 text-laser-300 hover:bg-laser-900 disabled:opacity-40"
            >
              <Send className="h-4 w-4" aria-hidden />
            </button>
          </div>
        </form>
      </motion.div>
    </motion.div>
  )
}
