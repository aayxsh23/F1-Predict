import { motion } from 'framer-motion'
import { Check, ChevronDown, LoaderCircle, RotateCcw, Send, Square, X } from 'lucide-react'
import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import { useChat } from '@/lib/chat'
import { cn } from '@/lib/cn'
import { formatPct } from '@/lib/format'
import { useStrategy } from '@/lib/queries'
import { useRaceView } from '@/lib/raceView'
import { useSelection } from '@/lib/selection'
import type { ChatMessage, Compound, StrategyPlan, Stint } from '@/lib/types'

import { rise, stagger } from './parts'

// what each lookup is doing, in words a fan reads while waiting
export const TOOL_LABELS: Record<string, string> = {
  race_forecast: 'Reading the forecast',
  explain_prediction: 'Breaking down the prediction',
  head_to_head: 'Simulating the head-to-head',
  championship: 'Simulating the title race',
  title_scenario_for: 'Doing the title maths',
  race_strategy: 'Running the strategy model',
  forecast_timeline: 'Checking how the forecast moved',
  search_rules: 'Searching the rulebook',
  circuit_guide: 'Reading the circuit guide',
  driver_history: 'Looking up past results',
  past_race_prediction: 'Checking a past prediction',
  model_track_record: "Checking the model's track record",
  season_schedule: 'Checking the calendar',
}

// Pirelli's sidewall colours ("The Tyre Code Rule" in DESIGN.md)
const COMPOUND: Record<Compound, { fill: string; ink: string; short: string }> = {
  SOFT: { fill: 'var(--color-tyre-soft)', ink: 'var(--color-obsidian-950)', short: 'S' },
  MEDIUM: { fill: 'var(--color-tyre-medium)', ink: 'var(--color-obsidian-950)', short: 'M' },
  HARD: { fill: 'var(--color-tyre-hard)', ink: 'var(--color-obsidian-950)', short: 'H' },
}

function StintBar({ stints, laps, windows = [] }: { stints: Stint[]; laps: number; windows?: Array<[number, number]> }) {
  const x = (lap: number) => `${(lap / laps) * 100}%`
  return (
    <div className="relative h-5 w-full" role="img" aria-label={stints.map((s) => `${s.compound.toLowerCase()} laps ${s.from_lap}-${s.to_lap}`).join(', ')}>
      <div className="flex h-full w-full gap-px">
        {stints.map((s) => (
          <span
            key={s.from_lap}
            className="flex items-center justify-center font-mono text-micro font-semibold"
            style={{ width: `${(s.laps / laps) * 100}%`, background: COMPOUND[s.compound].fill, color: COMPOUND[s.compound].ink }}
          >
            {s.laps >= 6 ? `${COMPOUND[s.compound].short} ${s.laps}` : ''}
          </span>
        ))}
      </div>
      {windows.map(([lo, hi]) => (
        <span key={lo} className="absolute -bottom-1.5 h-1 bg-laser-400/80" style={{ left: x(lo - 1), width: x(hi - lo + 1) }} aria-hidden />
      ))}
    </div>
  )
}

function PlanRow({ plan, laps, best }: { plan: StrategyPlan; laps: number; best: boolean }) {
  return (
    <li className="border-t border-line py-2.5">
      <div className="mb-1.5 flex items-baseline justify-between gap-3">
        <span className={cn('text-sm', best ? 'text-silver-100' : 'text-silver-300')}>{plan.name}</span>
        <span className="shrink-0 font-mono text-hud tabular-nums text-silver-400">
          {plan.time_vs_best_s === 0 ? 'fastest' : `+${plan.time_vs_best_s.toFixed(1)}s`} · {formatPct(plan.chance_fastest)} best
        </span>
      </div>
      <StintBar stints={plan.stints} laps={laps} windows={plan.pit_windows} />
      <p className="mt-2.5 text-hud text-silver-400">
        Pit {plan.pit_windows.map(([lo, hi], i) => `lap ${plan.pit_laps[i]} (window ${lo}–${hi})`).join(', then ')}
      </p>
    </li>
  )
}

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms)
    return () => clearTimeout(t)
  }, [value, ms])
  return v
}

function StrategyPanel() {
  const sel = useSelection()
  const { view } = useRaceView(sel.selectedRace)
  const [open, setOpen] = useState(false)
  const [scOn, setScOn] = useState(false)
  const [scDraft, setScDraft] = useState(15)
  const scLap = useDebounced(scOn ? scDraft : null, 250)
  const enabled = view?.mode === 'forecast'
  const q = useStrategy(view?.season ?? 0, view?.round ?? 0, scLap, enabled)
  const s = q.data

  return (
    <aside aria-label="Tyre strategy" className="lg:border-l lg:border-line lg:pl-6">
      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open} className="flex w-full items-center justify-between gap-3 lg:pointer-events-none">
        <span className="hud-label text-silver-200">Tyre strategy</span>
        {s && <span className="hud-label hidden text-silver-400 sm:inline">{s.laps} laps · pit loss {s.pit_loss_s.toFixed(0)}s · SC {formatPct(s.safety_car_probability)}</span>}
        <ChevronDown className={cn('h-4 w-4 text-silver-400 transition-transform lg:hidden', open && 'rotate-180')} aria-hidden />
      </button>
      <div className={cn('mt-2', open ? 'block' : 'hidden lg:block')}>
        {!enabled ? (
          <p className="text-sm text-silver-400">Strategy is modelled for races with a stored forecast. Pick the current round on the ribbon.</p>
        ) : q.isPending ? (
          <p className="text-sm text-silver-400">Running the strategy model…</p>
        ) : q.isError || !s ? (
          <p className="text-sm text-silver-400">The strategy model isn't reachable right now.</p>
        ) : (
          <>
            <ol className={cn(q.isFetching && 'opacity-60')}>
              {s.strategies.slice(0, 3).map((p, i) => <PlanRow key={p.name} plan={p} laps={s.laps} best={i === 0} />)}
            </ol>
            <div className="mt-3 border-t border-line pt-3">
              <label className="flex items-center gap-2 text-sm text-silver-200">
                <input type="checkbox" checked={scOn} onChange={(e) => setScOn(e.target.checked)} className="accent-[var(--color-laser-400)]" />
                Safety car on lap <span className="font-mono tabular-nums text-silver-100">{scDraft}</span>
              </label>
              <input
                type="range"
                min={1}
                max={s.laps - 1}
                value={scDraft}
                onChange={(e) => {
                  setScDraft(Number(e.target.value))
                  setScOn(true)
                }}
                aria-label="Safety car lap"
                className="mt-2 w-full accent-[var(--color-laser-400)]"
              />
              {s.scenario && (
                <div className="mt-2">
                  <StintBar stints={s.scenario.stints} laps={s.laps} />
                  <p className="mt-2 text-sm leading-relaxed text-silver-300">
                    Best now: {s.scenario.best}, pitting on lap {s.scenario.pit_laps.join(' and ')}
                    {s.scenario.pits_under_safety_car ? ', under the safety car' : ''}.{' '}
                    {s.scenario.gain_vs_sticking_to_plan_s > 0.05
                      ? `About ${s.scenario.gain_vs_sticking_to_plan_s.toFixed(1)}s better than sticking to the plan.`
                      : 'No gain over sticking to the plan.'}
                  </p>
                </div>
              )}
            </div>
            <p className="mt-3 text-hud text-silver-400">
              Compares plans for a typical car, from {s.races_of_data ? `${s.races_of_data} past races here` : 'all circuits (no history here yet)'}: tyre wear, pit-lane time and safety cars. Traffic and track position aren't modelled.
            </p>
          </>
        )}
      </div>
    </aside>
  )
}

function Steps({ m }: { m: ChatMessage }) {
  if (!m.steps?.length) return null
  return (
    <ul className="mb-2 space-y-1" aria-label="Lookups">
      {m.steps.map((s) => (
        <li key={s.id} className="flex items-center gap-1.5 text-hud text-silver-400">
          {!s.done ? (
            <LoaderCircle className="h-3 w-3 motion-safe:animate-spin" aria-hidden />
          ) : s.ok ? (
            <Check className="h-3 w-3 text-laser-400" aria-hidden />
          ) : (
            <X className="h-3 w-3 text-signal-coral" aria-hidden />
          )}
          {TOOL_LABELS[s.name] ?? s.name}
          {!s.done && '…'}
        </li>
      ))}
    </ul>
  )
}

function Sources({ m }: { m: ChatMessage }) {
  const sel = useSelection()
  if (!m.sources?.length || m.streaming) return null
  return (
    <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
      <span className="text-hud text-silver-400">Sources</span>
      {m.sources.slice(0, 6).map((s) => (
        <button
          key={`${s.filename}|${s.article}`}
          type="button"
          onClick={() => sel.openDoc(s.filename, s.article)}
          className="border border-line-strong px-2 py-0.5 font-mono text-micro text-laser-300 hover:border-laser-500"
          title={s.title}
        >
          {s.article ? `Art. ${s.article}` : s.title}
        </button>
      ))}
    </div>
  )
}

function Answer({ text }: { text: string }) {
  return (
    <div className="space-y-2 text-sm leading-relaxed text-silver-200 [&_a]:text-laser-300 [&_a]:underline [&_code]:font-mono [&_li]:ml-4 [&_ol]:list-decimal [&_strong]:font-semibold [&_strong]:text-silver-100 [&_ul]:list-disc">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: (props) => (
            <div className="scroll-thin my-1 overflow-x-auto">
              <table className="w-full border-collapse text-sm" {...props} />
            </div>
          ),
          th: (props) => <th className="border-b border-line-strong px-2 py-1 text-left font-semibold text-silver-300" {...props} />,
          td: (props) => <td className="border-b border-line px-2 py-1 font-mono tabular-nums" {...props} />,
          a: (props) => <a target="_blank" rel="noreferrer" {...props} />,
        }}
      >
        {text}
      </Markdown>
    </div>
  )
}

export function StrategistCard() {
  const sel = useSelection()
  const { view } = useRaceView(sel.selectedRace)
  const chat = useChat()
  const [draft, setDraft] = useState('')
  const log = useRef<HTMLDivElement>(null)

  const me = view?.drivers.find((d) => d.code === sel.driver) ?? view?.drivers[0]
  const race = view?.eventName ?? (view?.location ? `race at ${view.location}` : 'this race')
  const chips = [
    { label: 'Who wins?', prompt: `Who is most likely to win the ${race}, and how sure is the model?` },
    ...(me ? [{ label: `Why ${me.code} P${me.rank}?`, prompt: `Why is ${me.code} predicted to finish P${me.rank}?` }] : []),
    { label: 'Best strategy', prompt: 'What is the best tyre strategy for this race, and what changes if there is an early safety car?' },
    ...(me ? [{ label: `${me.code} title chances`, prompt: `Can ${me.code} still win the championship? What would it take?` }] : []),
    { label: 'Pole favourite', prompt: 'Who is favourite for pole position, and what lap time is expected?' },
    { label: 'How accurate?', prompt: 'How accurate has this model been on races it had not seen?' },
  ]

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
    <motion.div variants={stagger} initial="hidden" animate="show" className="grid h-full grid-cols-[minmax(0,1fr)] grid-rows-[auto_minmax(0,1fr)] lg:grid-cols-[minmax(0,1fr)_380px] lg:grid-rows-1 lg:gap-x-6 lg:px-6">
      <motion.div variants={rise} className="scroll-thin order-first max-h-[45dvh] overflow-y-auto border-b border-line px-4 py-3 lg:order-last lg:max-h-none lg:border-0 lg:px-0 lg:py-4">
        <StrategyPanel />
      </motion.div>

      <motion.div variants={rise} className="flex min-h-0 flex-col">
        <div ref={log} role="log" aria-live="polite" aria-label="Conversation with the race analyst" className="scroll-thin min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-4 lg:px-0">
          {chat.messages.length === 0 && (
            <div className="max-w-lg">
              <p className="font-display text-lg font-semibold uppercase tracking-[0.06em] text-silver-100">Ask the race analyst</p>
              <p className="mt-1.5 text-sm text-silver-400">
                Forecasts, qualifying, tyre strategy, the title fight and the rulebook. Every number in an answer comes from this app's own models and data, and rules are cited by article.
              </p>
            </div>
          )}
          {chat.messages.map((m) =>
            m.role === 'user' ? (
              <p key={m.id} className="ml-auto max-w-[85%] whitespace-pre-line border border-line-strong bg-obsidian-700 px-3 py-2 text-sm text-silver-100">
                {m.text}
              </p>
            ) : (
              <div key={m.id} className={cn('max-w-[95%] border-l-2 pl-3', m.error ? 'border-signal-coral' : 'border-laser-500')}>
                <Steps m={m} />
                {m.text ? (
                  <Answer text={m.text} />
                ) : (
                  m.streaming && (
                    <span className="flex gap-1.5 py-1.5" role="status" aria-label="The analyst is answering">
                      {[0, 1, 2].map((i) => (
                        <span key={i} className="h-1.5 w-1.5 rounded-full bg-laser-400 motion-safe:animate-[pulse-dot_1.2s_ease-in-out_infinite]" style={{ animationDelay: `${i * 0.18}s` }} />
                      ))}
                    </span>
                  )
                )}
                <Sources m={m} />
                {m.error && !chat.isPending && (
                  <button type="button" onClick={chat.retry} className="hud-label mt-2 flex items-center gap-1.5 text-laser-300 hover:text-laser-400">
                    <RotateCcw className="h-3 w-3" aria-hidden /> Try again
                  </button>
                )}
              </div>
            ),
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
                New chat
              </button>
            )}
          </div>
          <div className="flex items-end gap-2">
            <label className="min-w-0 flex-1">
              <span className="sr-only">Ask the race analyst</span>
              <textarea
                rows={1}
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={onKeyDown}
                maxLength={4000}
                placeholder={me ? `Ask anything, e.g. "Can ${me.code} beat their teammate?"` : 'Ask anything about this race'}
                className="block max-h-28 min-h-10 w-full resize-none border border-line-strong bg-obsidian-900 px-3 py-2 text-sm text-silver-100 placeholder:text-silver-400 focus:border-laser-500"
              />
            </label>
            {chat.isPending ? (
              <button type="button" onClick={chat.stop} aria-label="Stop answering" className="grid h-10 w-10 shrink-0 place-items-center border border-line-strong text-silver-200 hover:border-laser-500">
                <Square className="h-3.5 w-3.5" aria-hidden />
              </button>
            ) : (
              <button type="submit" disabled={!draft.trim()} aria-label="Send" className="grid h-10 w-10 shrink-0 place-items-center border border-laser-500 text-laser-300 hover:bg-laser-900 disabled:opacity-40">
                <Send className="h-4 w-4" aria-hidden />
              </button>
            )}
          </div>
        </form>
      </motion.div>
    </motion.div>
  )
}
