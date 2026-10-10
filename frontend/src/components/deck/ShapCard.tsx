import { motion } from 'framer-motion'
import { useState } from 'react'

import { cn } from '@/lib/cn'
import { formatFeatureName, formatNumber, formatPct, formatSigned } from '@/lib/format'
import { useDriverExplain, useModelCard, useNaturalExplanation, usePredictionsLatest, useRegulationsList, useTimeline } from '@/lib/queries'
import { useRaceView, type RaceDriver, type RaceView } from '@/lib/raceView'
import { useSelection } from '@/lib/selection'
import {
  TARGET_HIGHER_IS_BETTER,
  TARGET_LABELS,
  TARGET_MEANING,
  TARGET_UNITS,
  TARGETS,
  type ShapContribution,
  type ShapExplanation,
  type Stage,
  type Target,
} from '@/lib/types'

import { rise, SheetMessage, SheetSkeleton, stagger } from './parts'

const label = (c: ShapContribution) => c.label ?? formatFeatureName(c.feature)

function formatValue(v: number | string | null): string {
  if (v === null) return '—'
  if (typeof v === 'string') return v.toUpperCase()
  return Number.isInteger(v) ? String(v) : v.toFixed(2)
}

interface Step {
  key: string
  label: string
  feature: string | null
  value: number | string | null
  from: number
  to: number
  shap: number
}

function buildSteps(exp: ShapExplanation): Step[] {
  let running = exp.base_value
  const steps: Step[] = exp.top_contributions.map((c) => {
    const step = { key: c.feature, label: label(c), feature: c.feature, value: c.value, from: running, to: running + c.shap, shap: c.shap }
    running += c.shap
    return step
  })
  const rest = exp.predicted_value - running // SHAP is additive: whatever the top few don't explain is everything else
  if (Math.abs(rest) > 0.005) steps.push({ key: 'rest', label: 'Everything else', feature: null, value: null, from: running, to: running + rest, shap: rest })
  return steps
}

/** One sentence a fan can repeat: the prediction and its two biggest reasons. */
function PlainSummary({ exp, target, me }: { exp: ShapExplanation; target: Target; me: RaceDriver | undefined }) {
  const good = (s: number) => (TARGET_HIGHER_IS_BETTER[target] ? s > 0 : s < 0)
  const helps = exp.top_contributions.find((c) => good(c.shap))
  const hurts = exp.top_contributions.find((c) => !good(c.shap))
  const headline =
    target === 'finish_position' && me
      ? `${me.code} is predicted to finish P${me.rank}${me.band ? `, most likely between P${me.band[0]} and P${me.band[1]}` : ''}.`
      : target === 'qualifying' && me
        ? `${me.code} is predicted to qualify P${me.qualiRank}${me.qualiGapPredicted ? `, ${formatNumber(me.qualiGapPredicted, 3, 's')} off pole` : ''}.`
        : `${TARGET_LABELS[target]}: ${formatNumber(exp.predicted_value, 2, TARGET_UNITS[target])} (an average driver here: ${formatNumber(exp.base_value, 2, TARGET_UNITS[target])}).`
  return (
    <p className="max-w-[72ch] text-base leading-relaxed text-silver-200">
      {headline}
      {helps && <> Helping most: <span className="text-laser-300">{label(helps).toLowerCase()}</span>.</>}
      {hurts && <> Holding back most: <span className="text-signal-coral">{label(hurts).toLowerCase()}</span>.</>}
    </p>
  )
}

function Waterfall({ exp, target }: { exp: ShapExplanation; target: Target }) {
  const unit = TARGET_UNITS[target]
  const steps = buildSteps(exp)
  const marks = [exp.base_value, exp.predicted_value, ...steps.flatMap((s) => [s.from, s.to])]
  const lo = Math.min(...marks)
  const hi = Math.max(...marks)
  const pad = (hi - lo) * 0.06 || 1
  const min = lo - pad
  const span = hi + pad - min
  const at = (v: number) => ((v - min) / span) * 100
  const higherIsBetter = TARGET_HIGHER_IS_BETTER[target]

  const guides = (
    <>
      <span className="absolute inset-y-0 w-px bg-line-strong" style={{ left: `${at(exp.base_value)}%` }} aria-hidden />
      <span className="absolute inset-y-0 w-px bg-laser-500/70" style={{ left: `${at(exp.predicted_value)}%` }} aria-hidden />
    </>
  )
  const row = 'grid min-h-[30px] py-0.5 grid-cols-[minmax(0,44%)_minmax(0,1fr)_52px] sm:grid-cols-[minmax(140px,250px)_minmax(0,1fr)_64px] items-center gap-x-3'

  return (
    <div>
      <div className="mb-2 flex flex-wrap gap-x-5 gap-y-1 text-hud text-silver-400">
        <span className="flex items-center gap-1.5"><span className="h-2 w-3 bg-laser-400" aria-hidden />helps</span>
        <span className="flex items-center gap-1.5"><span className="h-2 w-3 bg-signal-coral" aria-hidden />hurts</span>
        <span>{TARGET_MEANING[target]}; {higherIsBetter ? 'higher is better' : 'lower is better'}</span>
      </div>
      <ol>
        <li className={row}>
          <span className="hud-label text-silver-400" title="What the model predicts for a typical driver before looking at this one">Typical driver</span>
          <div className="relative h-full">{guides}</div>
          <span className="text-right font-mono text-sm tabular-nums text-silver-300">{formatNumber(exp.base_value, 2, unit)}</span>
        </li>
        {steps.map((s, i) => {
          const good = higherIsBetter ? s.shap > 0 : s.shap < 0
          return (
            <li key={s.key} className={row}>
              <span className="flex min-w-0 items-center gap-2">
                <span className="text-sm leading-tight text-silver-200 sm:truncate" title={s.label}>{s.label}</span>
                <span className="ml-auto hidden shrink-0 font-mono text-hud text-silver-400 sm:inline">{s.feature ? formatValue(s.value) : ''}</span>
              </span>
              <div className="relative h-full">
                {guides}
                <motion.span
                  className={cn('absolute top-1/2 h-3 -translate-y-1/2', good ? 'bg-laser-400' : 'bg-signal-coral')}
                  style={{ left: `${at(Math.min(s.from, s.to))}%`, width: `${Math.max(0.6, Math.abs(at(s.to) - at(s.from)))}%`, transformOrigin: s.to >= s.from ? 'left' : 'right' }}
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: 1 }}
                  transition={{ duration: 0.45, delay: 0.12 + i * 0.045, ease: [0.16, 1, 0.3, 1] }}
                />
              </div>
              <span className={cn('text-right font-mono text-sm tabular-nums', good ? 'text-laser-300' : 'text-signal-coral')}>{formatSigned(s.shap, 2)}</span>
            </li>
          )
        })}
        <li className={cn(row, 'mt-1 border-t border-line-strong')}>
          <span className="hud-label text-silver-100">Prediction</span>
          <div className="relative h-full">{guides}</div>
          <span className="text-right font-mono text-sm font-medium tabular-nums text-silver-100">{formatNumber(exp.predicted_value, 2, unit)}</span>
        </li>
      </ol>
    </div>
  )
}

function TeammatePanel({ me, mate }: { me: RaceDriver; mate: RaceDriver | undefined }) {
  if (!mate) return <p className="text-sm text-silver-400">No teammate in this race's data.</p>
  const rows: Array<{ label: string; mine: string; theirs: string; leads: 'me' | 'mate' | null }> = [
    { label: 'Predicted finish', mine: `P${me.rank}`, theirs: `P${mate.rank}`, leads: me.rank < mate.rank ? 'me' : 'mate' },
    { label: 'Predicted quali', mine: `P${me.qualiRank}`, theirs: `P${mate.qualiRank}`, leads: me.qualiRank < mate.qualiRank ? 'me' : 'mate' },
    ...(me.grid !== null || mate.grid !== null
      ? [{ label: 'Grid', mine: me.grid === null ? '—' : `P${me.grid}`, theirs: mate.grid === null ? '—' : `P${mate.grid}`, leads: (me.grid === null || mate.grid === null || me.grid === mate.grid ? null : me.grid < mate.grid ? 'me' : 'mate') as 'me' | 'mate' | null }]
      : []),
    ...(me.win !== null && mate.win !== null ? [{ label: 'Win chance', mine: formatPct(me.win), theirs: formatPct(mate.win), leads: (me.win > mate.win ? 'me' : 'mate') as 'me' | 'mate' }] : []),
  ]
  return (
    <div>
      <table className="w-full text-sm">
        <thead>
          <tr className="hud-label text-silver-400">
            <th scope="col" className="pb-1.5 text-left font-semibold">vs teammate</th>
            <th scope="col" className="pb-1.5 text-right font-semibold">{me.code}</th>
            <th scope="col" className="pb-1.5 text-right font-semibold">{mate.code}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} className="border-t border-line">
              <th scope="row" className="py-1.5 text-left font-normal text-silver-300">{r.label}</th>
              <td className={cn('py-1.5 text-right font-mono tabular-nums', r.leads === 'me' ? 'text-laser-300' : 'text-silver-300')}>{r.mine}</td>
              <td className={cn('py-1.5 text-right font-mono tabular-nums', r.leads === 'mate' ? 'text-laser-300' : 'text-silver-300')}>{r.theirs}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {me.beatsTeammate !== null && (
        <p className="mt-2 text-hud text-silver-400">
          {me.code} finishes ahead of {mate.code} in {formatPct(me.beatsTeammate)} of simulated races.
        </p>
      )}
    </div>
  )
}

/** How this driver's predicted finish moved through the weekend's sessions. */
function WeekendTimeline({ view, code }: { view: RaceView; code: string }) {
  const q = useTimeline(view.season, view.round, view.mode === 'forecast')
  // rank within each snapshot, not the raw model number
  const points = (q.data ?? [])
    .map((s) => {
      const order = s.drivers.filter((d) => d.predicted_finish_position !== null).sort((a, b) => a.predicted_finish_position! - b.predicted_finish_position!)
      return { label: s.label, rank: order.findIndex((d) => d.driver === code) + 1 }
    })
    .filter((pt) => pt.rank > 0)
  if (points.length < 2) return null
  return (
    <div>
      <p className="hud-label mb-1.5 text-silver-400">Through the weekend</p>
      <ol className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
        {points.map((pt, i) => (
          <li key={pt.label} className="flex items-center gap-2">
            {i > 0 && <span className="text-silver-500" aria-hidden>→</span>}
            <span className="text-silver-400">{pt.label}</span>
            <span className="font-mono tabular-nums text-silver-100">P{pt.rank}</span>
          </li>
        ))}
      </ol>
    </div>
  )
}

const STAGE_WORDS: Record<Stage, string> = {
  pre_weekend: 'before practice',
  post_practice: 'after practice',
  post_sprint_quali: 'after Sprint Qualifying',
  post_sprint: 'after the Sprint',
  post_quali: 'after qualifying',
  race_day: 'on race day',
}

/** The honest numbers: error on races the model had never seen, next to a naive guess. */
function Accuracy({ target, stage }: { target: Target; stage: Stage | null }) {
  const card = useModelCard()
  const m = card.data?.[target]
  if (!m) return null
  const s = stage ?? 'post_quali'
  const acc = m.stages[s] ?? m.stages.post_practice ?? Object.values(m.stages)[0]
  if (!acc) return null
  const unit = target === 'qualifying' || target === 'race_time' ? '%' : ' places'
  return (
    <div className="border-t border-line pt-3">
      <p className="hud-label mb-1 text-silver-400">How accurate is this?</p>
      <p className="text-sm leading-relaxed text-silver-300">
        On races it had never seen, this prediction made {STAGE_WORDS[s]} was off by{' '}
        <span className="font-mono text-silver-100">{acc.mae.toFixed(2)}{unit}</span> on average. The simple guess ({m.baseline}) was off by{' '}
        <span className="font-mono text-silver-100">{acc.baseline_mae.toFixed(2)}{unit}</span>.
      </p>
    </div>
  )
}

function ExplainInWords({ season, round, driver, target }: { season: number; round: number; driver: string; target: Target }) {
  const sel = useSelection()
  const [asked, setAsked] = useState<string | null>(null)
  const key = `${season}-${round}-${driver}-${target}`
  const q = useNaturalExplanation(season, round, driver, target, asked === key)
  const docs = useRegulationsList()

  return (
    <div>
      {q.data ? (
        <>
          <p className="max-w-[68ch] whitespace-pre-line text-base leading-relaxed text-silver-200">{q.data.explanation}</p>
          {q.data.sources.length > 0 && (
            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              <span className="text-hud text-silver-400">Grounded in</span>
              {[...new Set(q.data.sources)].map((file) =>
                docs.data?.some((d) => d.filename === file) ? (
                  <button key={file} type="button" onClick={() => sel.openDoc(file)} className="border border-line-strong px-2 py-0.5 font-mono text-micro text-laser-300 hover:border-laser-500">
                    {file.replace(/\.(pdf|txt)$/i, '')}
                  </button>
                ) : (
                  <span key={file} className="border border-line px-2 py-0.5 font-mono text-micro text-silver-400">{file.replace(/\.(pdf|txt)$/i, '').replace(/_/g, ' ')}</span>
                ),
              )}
            </div>
          )}
        </>
      ) : q.isFetching ? (
        <div role="status" aria-live="polite">
          <div className="space-y-2" aria-hidden>
            {[100, 96, 62].map((w) => (
              <div key={w} className="relative h-3 overflow-hidden bg-obsidian-700" style={{ width: `${w}%` }}>
                <span className="absolute inset-y-0 w-1/3 bg-gradient-to-r from-transparent via-silver-400/15 to-transparent motion-safe:animate-[scan_1.4s_linear_infinite]" />
              </div>
            ))}
          </div>
          <p className="mt-2 text-hud text-silver-400">Writing an explanation from these numbers and the circuit notes…</p>
        </div>
      ) : q.isError ? (
        <div>
          <p className="text-sm text-silver-300">{q.error.message.includes('local model') ? "Written explanations aren't set up on this server yet." : "The explanation service isn't reachable right now."} The breakdown on the left still holds.</p>
          <button type="button" onClick={() => q.refetch()} className="hud-label mt-2 border border-laser-500 px-3 py-1.5 text-laser-300 hover:bg-laser-900">Try again</button>
        </div>
      ) : (
        <button type="button" onClick={() => setAsked(key)} className="hud-label border border-laser-500 px-3 py-2 text-laser-300 hover:bg-laser-900">
          Explain in words
        </button>
      )}
    </div>
  )
}

export function ShapCard() {
  const sel = useSelection()
  const { status, view } = useRaceView(sel.selectedRace)
  const latest = usePredictionsLatest()

  const race = sel.selectedRace
  const drivers = view?.drivers ?? []
  const code = sel.driver ?? drivers[0]?.code ?? null
  const me = drivers.find((d) => d.code === code)
  const mate = me ? drivers.find((d) => d.team === me.team && d.code !== me.code) : undefined
  const enabled = status === 'ready' && view.mode === 'forecast' && !!code && !!race
  const shap = useDriverExplain(race?.season ?? 0, race?.round ?? 0, code ?? '', sel.target, enabled)

  if (status === 'pending') return <SheetSkeleton rows={6} />
  if (status === 'error') return <SheetMessage title="The breakdown isn't reachable right now">The backend may still be waking up. Try again in a moment.</SheetMessage>
  if (view.mode !== 'forecast') {
    return (
      <SheetMessage
        title="The breakdown needs a stored forecast"
        action={
          latest.data && (
            <button type="button" onClick={() => sel.selectRace(latest.data.round, latest.data.season)} className="hud-label border border-laser-500 px-3 py-2 text-laser-300 hover:bg-laser-900">
              Go to the latest forecast
            </button>
          )
        }
      >
        The reasons behind a prediction come from the forecast stored before the race. Recent rounds have one; older races only have the prediction-vs-result archive.
      </SheetMessage>
    )
  }

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="flex h-full flex-col">
      <motion.div variants={rise} className="flex flex-wrap items-center gap-x-5 gap-y-2 border-b border-line px-4 py-3 lg:px-6">
        <label className="flex items-center gap-2">
          <span className="hud-label text-silver-400">Driver</span>
          <select
            value={code ?? ''}
            onChange={(e) => sel.openDriver(e.target.value)}
            className="h-9 border border-line-strong bg-obsidian-900 px-2.5 font-display text-base font-semibold tracking-wide text-silver-100 hover:border-laser-500"
          >
            {drivers.map((d) => (
              <option key={d.code} value={d.code}>P{d.rank} · {d.code}</option>
            ))}
          </select>
        </label>
        <div role="group" aria-label="Which prediction" className="scroll-none -mb-px flex min-w-0 gap-1 overflow-x-auto">
          {TARGETS.map((t) => (
            <button
              key={t}
              type="button"
              aria-pressed={sel.target === t}
              onClick={() => sel.setTarget(t)}
              className={cn('hud-label h-9 shrink-0 border px-3', sel.target === t ? 'border-laser-400 bg-laser-900 text-laser-300' : 'border-line text-silver-400 hover:border-line-strong hover:text-silver-200')}
            >
              {TARGET_LABELS[t]}
            </button>
          ))}
        </div>
      </motion.div>

      <div className="scroll-thin min-h-0 flex-1 overflow-auto">
        <div className="grid gap-x-10 gap-y-6 px-4 py-4 lg:grid-cols-[minmax(0,1fr)_340px] lg:px-6">
          <motion.div variants={rise} className="min-w-0 space-y-4">
            {shap.isPending ? (
              <SheetSkeleton rows={6} />
            ) : shap.isError || !shap.data ? (
              <p className="text-sm text-silver-300">No breakdown for {code} on this prediction right now.</p>
            ) : (
              <>
                <PlainSummary exp={shap.data} target={sel.target} me={me} />
                <Waterfall key={`${code}-${sel.target}`} exp={shap.data} target={sel.target} />
              </>
            )}
            {code && <WeekendTimeline view={view} code={code} />}
          </motion.div>
          <motion.div variants={rise} className="min-w-0 space-y-6">
            {me && <TeammatePanel me={me} mate={mate} />}
            {code && race && <ExplainInWords season={race.season} round={race.round} driver={code} target={sel.target} />}
            <Accuracy target={sel.target} stage={view.stage} />
          </motion.div>
        </div>
      </div>
    </motion.div>
  )
}
