import type { UseQueryResult } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { Radio } from 'lucide-react'
import { useState } from 'react'
import { Link, Navigate, useParams } from 'react-router-dom'

import { cn } from '@/lib/cn'
import { formatFeatureName } from '@/lib/format'
import { useDriverExplain, useNaturalExplanation, usePredictionsLatest, useRegulationsList } from '@/lib/queries'
import { useRaceView, type RaceDriver, type RaceView } from '@/lib/raceView'
import { TARGET_HIGHER_IS_BETTER, type ExplainResult, type ShapContribution, type ShapExplanation, type Target } from '@/lib/types'

import { betterWord, driverHref, lensOf, orDash, raceHref, signed, useDriverNames, useLens, type Lens } from './data'
import { useChrome, useChromeApi } from './chrome'
import { LensTabs, Notice, Offline, PAGE, Skeleton } from './parts'

const COLS = 'grid-cols-[minmax(120px,250px)_minmax(0,1fr)_64px]'
const UNIT: Record<Target, string> = { qualifying: '% lap', finish_position: 'places', quali_delta: 'places', race_time: '% race' }

const labelOf = (c: ShapContribution) => c.label ?? formatFeatureName(c.feature)

function valueOf(v: number | string | null): string {
  if (v === null) return 'not known yet'
  if (typeof v === 'string') return v.toUpperCase()
  return Number.isInteger(v) ? String(v) : v.toFixed(2)
}

interface Step {
  key: string
  label: string
  value: string
  from: number
  to: number
  shap: number
}

/** SHAP is additive: model average + every contribution = the prediction. What
 *  the top few don't explain becomes one "everything else" step so the sum closes. */
function buildSteps(exp: ShapExplanation): Step[] {
  let running = exp.base_value
  const steps: Step[] = exp.top_contributions.map((c) => {
    const s = { key: c.feature, label: labelOf(c), value: valueOf(c.value), from: running, to: running + c.shap, shap: c.shap }
    running += c.shap
    return s
  })
  const rest = exp.predicted_value - running
  if (Math.abs(rest) > 0.005) steps.push({ key: 'rest', label: 'Everything else', value: '', from: running, to: exp.predicted_value, shap: rest })
  return steps
}

/** C3: a waterfall from the model average to the prediction, on the target's own
 *  axis. Colour is good/bad for the driver, never the sign. */
function ForceLedger({ exp, lens, focus, setFocus }: { exp: ShapExplanation; lens: Lens; focus: string | null; setFocus: (f: string | null) => void }) {
  const steps = buildSteps(exp)
  const marks = [exp.base_value, exp.predicted_value, ...steps.flatMap((s) => [s.from, s.to])]
  const lo = Math.min(...marks)
  const hi = Math.max(...marks)
  const pad = (hi - lo) * 0.08 || 1
  const min = lo - pad
  const span = hi + pad - min
  const at = (v: number) => ((v - min) / span) * 100
  const good = (s: number) => (TARGET_HIGHER_IS_BETTER[lens.id] ? s > 0 : s < 0)
  const diamond = (v: number, final = false) => (
    <span
      className={cn('absolute top-1/2 rotate-45', final ? '-ml-1.5 -mt-1.5 size-3 bg-signal' : '-ml-[5px] -mt-[5px] size-2.5 border-[1.5px] border-ink-2')}
      style={{ left: `${at(v)}%` }}
    />
  )

  return (
    <section aria-label="What moved the number" className="pw-glass rounded-[10px] px-4 pb-3 pt-4 sm:px-[22px]">
      <div aria-hidden>
        <div className={cn('grid items-end gap-4 border-b border-white/[.07] pb-2', COLS)}>
          <span className="pw-label text-[10px] leading-[1.3]">Force ledger · what moved the number</span>
          <div className="relative h-3">
            {[0.1, 0.3, 0.5, 0.7, 0.9].map((t, i) => (
              <span key={t} className={cn('absolute -translate-x-1/2 whitespace-nowrap text-[10px] font-medium leading-none text-ink-3', i % 2 && 'hidden wide:block')} style={{ left: `${t * 100}%` }}>
                {lens.id === 'finish_position' ? `P${(min + t * span).toFixed(1)}` : lens.fmt(min + t * span)}
              </span>
            ))}
          </div>
          <span className="pw-label text-right text-[10px]">{UNIT[lens.id]}</span>
        </div>
        <div className={cn('grid h-[30px] items-center gap-4', COLS)}>
          <span className="text-[12px] font-semibold leading-none text-ink-2">Model average</span>
          <div className="relative h-full">{diamond(exp.base_value)}</div>
          <span className="text-right text-[13px] font-semibold leading-none tabular-nums">{lens.fmt(exp.base_value)}</span>
        </div>
        <ol>
          {steps.map((s, i) => {
            const g = good(s.shap)
            const on = focus === s.key
            return (
              <li
                key={s.key}
                onPointerEnter={() => setFocus(s.key)}
                onPointerLeave={() => setFocus(null)}
                className={cn('grid h-8 items-center gap-4 rounded-[4px] transition-opacity duration-150 hover:bg-white/[.03]', COLS)}
                style={{ opacity: focus && !on ? 0.32 : 1 }}
              >
                <span className="flex min-w-0 items-baseline gap-2.5">
                  <span className="truncate text-[12px] font-semibold leading-none" title={s.label}>
                    {s.label}
                  </span>
                  <span className="hidden truncate text-[11px] leading-none text-ink-3 sm:inline">{s.value}</span>
                </span>
                <div className="relative h-full">
                  <span className="absolute inset-y-0 w-px bg-white/[.08]" style={{ left: `${at(exp.base_value)}%` }} />
                  <motion.span
                    className="absolute top-2 h-4 rounded-sm transition-shadow duration-200"
                    style={{
                      left: `${at(Math.min(s.from, s.to))}%`,
                      width: `${Math.max(0.6, Math.abs(at(s.to) - at(s.from)))}%`,
                      transformOrigin: s.to >= s.from ? 'left' : 'right',
                      background: g ? 'var(--color-gain)' : 'var(--color-signal)',
                      boxShadow: on ? `0 0 18px ${g ? 'rgb(61 220 151/.55)' : 'rgb(255 90 71/.55)'}` : 'none',
                    }}
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: 1 }}
                    transition={{ duration: 0.45, delay: 0.1 + i * 0.06, ease: [0.16, 1, 0.3, 1] }}
                  />
                </div>
                <span className="text-right text-[13px] font-semibold leading-none tabular-nums" style={{ color: g ? 'var(--color-gain)' : 'var(--color-signal-ink)' }}>
                  {signed(s.shap, 2)}
                </span>
              </li>
            )
          })}
        </ol>
        <div className={cn('mt-1 grid h-[34px] items-center gap-4 border-t border-white/[.07]', COLS)}>
          <span className="text-[12px] font-bold leading-none">Prediction</span>
          <div className="relative h-full">{diamond(exp.predicted_value, true)}</div>
          <span className="text-right text-[13px] font-bold leading-none tabular-nums text-signal-ink">{lens.fmt(exp.predicted_value)}</span>
        </div>
      </div>

      <table className="sr-only">
        <caption>
          {lens.title}: model average {lens.fmt(exp.base_value)}, prediction {lens.fmt(exp.predicted_value)}, {betterWord(lens.id)}
        </caption>
        <thead>
          <tr>
            <th scope="col">Factor</th>
            <th scope="col">Value</th>
            <th scope="col">Change</th>
            <th scope="col">Effect</th>
          </tr>
        </thead>
        <tbody>
          {steps.map((s) => (
            <tr key={s.key}>
              <th scope="row">{s.label}</th>
              <td>{s.value}</td>
              <td>{signed(s.shap, 2)}</td>
              <td>{good(s.shap) ? 'helps' : 'hurts'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}

/** The two biggest reasons each way, as a sentence; each phrase lights its ledger row. */
function LinkedSummary({ exp, lens, code, focus, setFocus }: { exp: ShapExplanation; lens: Lens; code: string; focus: string | null; setFocus: (f: string | null) => void }) {
  const good = (c: ShapContribution) => (TARGET_HIGHER_IS_BETTER[lens.id] ? c.shap > 0 : c.shap < 0)
  const helps = exp.top_contributions.filter(good).slice(0, 2)
  const hurts = exp.top_contributions.filter((c) => !good(c)).slice(0, 2)
  const phrase = (c: ShapContribution) => (
    <span
      key={c.feature}
      onPointerEnter={() => setFocus(c.feature)}
      onPointerLeave={() => setFocus(null)}
      className="rounded-[3px] transition-colors duration-200"
      style={{ background: focus === c.feature ? 'rgb(255 90 71/.24)' : 'transparent' }}
    >
      {labelOf(c).toLowerCase()}
      {c.value === null && ' (not known yet)'}
    </span>
  )
  const list = (cs: ShapContribution[]) => cs.flatMap((c, i) => (i ? [' and ', phrase(c)] : [phrase(c)]))
  return (
    <p className="font-editorial text-[clamp(19px,2vw,23px)] leading-[1.45] [text-wrap:pretty]">
      {helps.length > 0 && <>Helping {code} most: {list(helps)}. </>}
      {hurts.length > 0 && (
        <>
          <em>What holds it back:</em> {list(hurts)}.
        </>
      )}
    </p>
  )
}

/** The written explanation costs a model call, so it's asked for, not automatic. */
function InWords({ q, onAsk }: { q: UseQueryResult<ExplainResult>; onAsk: () => void }) {
  return (
    <>
      {q.data ? (
        <p className="whitespace-pre-line font-editorial text-[19px] leading-[1.45] text-ink-2">{q.data.explanation}</p>
      ) : q.isFetching ? (
        <p role="status" className="font-editorial text-[19px] italic text-ink-3 motion-safe:animate-[pw-live_1.6s_ease-in-out_infinite]">
          Writing it up from these numbers and the circuit notes…
        </p>
      ) : q.isError ? (
        <div className="flex flex-wrap items-baseline gap-3">
          <p className="font-editorial text-[19px] italic text-ink-2">
            {q.error.message.includes('local model') ? "Written explanations aren't set up on this server yet." : "The explanation service isn't reachable right now."}
          </p>
          <button type="button" onClick={() => q.refetch()} className="text-[12px] font-semibold text-signal-ink hover:text-ink">
            Try again
          </button>
        </div>
      ) : (
        <button
          type="button"
          onClick={onAsk}
          className="flex h-[38px] w-max items-center rounded-md border border-signal/50 px-3.5 text-[12px] font-semibold leading-none text-signal-ink hover:bg-signal/[.12]"
        >
          Explain in words
        </button>
      )}
    </>
  )
}

function GroundedIn({ code, view, sources, onAsk }: { code: string; view: RaceView; sources: string[]; onAsk: () => void }) {
  const docs = useRegulationsList()
  const unique = [...new Set(sources)]
  return (
    <aside className="flex flex-[0_1_250px] flex-col gap-3 border-l border-white/10 pl-[18px]">
      <h2 className="pw-label text-[10px]">Grounded in</h2>
      <span className="text-[12px] leading-normal text-ink-2">
        Model inputs for {code} · {view.location} R{view.round}
      </span>
      {unique.map((file) => {
        const name = file.replace(/\.(pdf|txt|md)$/i, '').replace(/_/g, ' ')
        return docs.data?.some((d) => d.filename === file) ? (
          <Link key={file} to={`/regulations?doc=${encodeURIComponent(file)}`} className="text-[12px] leading-normal text-signal-ink hover:text-ink">
            {name} →
          </Link>
        ) : (
          <span key={file} className="text-[12px] leading-normal text-ink-2">
            {name}
          </span>
        )
      })}
      <Link to="/regulations" className="text-[12px] leading-normal text-signal-ink hover:text-ink">
        Regulations →
      </Link>
      <button
        type="button"
        onClick={onAsk}
        className="mt-1 flex h-10 w-max items-center gap-2 rounded-md border border-white/[.14] px-3.5 text-[12px] font-semibold leading-none text-ink hover:border-white/35"
      >
        <Radio className="size-3.5 text-signal-ink" aria-hidden />
        Ask Pit Radio
      </button>
    </aside>
  )
}

function Why({ view, me }: { view: RaceView; me: RaceDriver }) {
  const [lensId, setLens] = useLens()
  const [focus, setFocus] = useState<string | null>(null)
  const { ask } = useChromeApi()
  const names = useDriverNames()
  const lens = lensOf(lensId)
  const shap = useDriverExplain(view.season, view.round, me.code, lens.id)
  const key = `${view.season}-${view.round}-${me.code}-${lens.id}`
  const [asked, setAsked] = useState<string | null>(null)
  const words = useNaturalExplanation(view.season, view.round, me.code, lens.id, asked === key)
  const v = lens.value(me)
  // step through the field in predicted order, wrapping at either end
  const at = view.drivers.findIndex((d) => d.code === me.code)
  const prev = view.drivers[(at - 1 + view.drivers.length) % view.drivers.length].code
  const next = view.drivers[(at + 1) % view.drivers.length].code
  const name = names.get(me.code)
  const step = 'flex h-7 items-center rounded-[5px] border border-white/[.12] px-[9px] text-[11px] font-semibold leading-none text-ink-2 hover:border-white/30 hover:text-ink'

  return (
    <div className={cn(PAGE, 'gap-5 pt-6')}>
      <div className="flex flex-wrap items-end gap-x-7 gap-y-[18px]">
        <h1 className="-ml-1.5 font-headline text-[clamp(88px,10vw,168px)] font-black leading-[.78]" style={{ viewTransitionName: `driver-${me.code}` }}>
          {me.code}
        </h1>
        <div className="flex min-w-0 flex-[1_1_320px] flex-col gap-3.5 pb-1">
          <div className="flex flex-wrap items-center gap-x-3.5 gap-y-2">
            <span className="text-[13px] leading-none text-ink-2">
              {name ? `${name} · ` : ''}
              {me.team} · {view.location} R{view.round}
            </span>
            <span className="flex gap-1">
              <Link to={driverHref(view.season, view.round, prev, lens.id)} viewTransition title="Previous driver" className={step}>
                ← {prev}
              </Link>
              <Link to={driverHref(view.season, view.round, next, lens.id)} viewTransition title="Next driver" className={step}>
                {next} →
              </Link>
            </span>
          </div>
          <LensTabs id="why" lens={lens.id} onChange={setLens} compact />
        </div>
        <div className="ml-auto flex flex-col items-end gap-1.5">
          <span className="pw-label text-[10px]">Predicted · {lens.label}</span>
          <span className="font-headline text-[clamp(64px,7vw,96px)] font-extrabold leading-[.8]">{orDash(v, lens.fmt)}</span>
          <span className="text-[11px] leading-none text-ink-3">
            {shap.data ? `model avg ${lens.fmt(shap.data.base_value)} · ` : ''}
            {betterWord(lens.id)}
          </span>
        </div>
      </div>

      {shap.isPending ? (
        <div className="pw-glass h-[320px] rounded-[10px] motion-safe:animate-[pw-live_1.6s_ease-in-out_infinite]" role="status" aria-label="Loading the breakdown" />
      ) : shap.isError || !shap.data ? (
        <p className="font-editorial text-[19px] italic text-ink-2">No breakdown for {me.code} on this prediction right now.</p>
      ) : (
        <ForceLedger key={`${me.code}-${lens.id}`} exp={shap.data} lens={lens} focus={focus} setFocus={setFocus} />
      )}

      <div className="flex flex-wrap items-start gap-x-8 gap-y-6">
        <div className="flex min-w-0 flex-[1_1_480px] flex-col items-start gap-5">
          {shap.data && <LinkedSummary exp={shap.data} lens={lens} code={me.code} focus={focus} setFocus={setFocus} />}
          <InWords q={words} onAsk={() => setAsked(key)} />
        </div>
        <GroundedIn
          code={me.code}
          view={view}
          sources={words.data?.sources ?? []}
          onAsk={() => ask(`Why is ${me.code} predicted ${orDash(v, lens.fmt)} for ${lens.label.toLowerCase()}?`)}
        />
      </div>
    </div>
  )
}

/** S3: why the model sees this driver where it does, on any of the four targets. */
export function DriverWhyPage() {
  const params = useParams()
  const season = Number(params.season)
  const round = Number(params.round)
  const code = (params.driver ?? '').toUpperCase()
  const valid = Number.isInteger(season) && Number.isInteger(round) && round > 0 && /^[A-Z]{3}$/.test(code)
  const { status, view } = useRaceView(valid ? { season, round } : null)
  const latest = usePredictionsLatest()
  const [lensId] = useLens()
  const ready = status === 'ready' ? view : null
  const me = ready?.drivers.find((d) => d.code === code)
  const back = raceHref(season, round)
  const lens = lensOf(lensId)
  const v = me ? orDash(lens.value(me), lens.fmt) : '…'

  useChrome({
    section: 'pred',
    crumbs: [{ label: 'Full grid', to: back, back: true }, { label: code }],
    fresh: ready?.generatedAt,
    chat: { season, round, race_name: ready ? (ready.eventName ?? ready.location) : undefined, driver: code },
    badge: `Driver: ${code}`,
    quick: [`How does ${code}'s grid position affect their outlook?`, `Why is ${code} predicted ${v} for ${lens.label.toLowerCase()}?`],
    codes: ready?.drivers.map((d) => d.code),
  })

  if (!valid) return <Navigate to="/" replace />
  if (ready && me && ready.mode === 'forecast') return <Why view={ready} me={me} />
  if (status === 'error') return <Offline />
  if (ready && !me)
    return <Notice title={`No ${code} in this race`} action={<Link to={back} className="text-[13px] font-semibold text-signal-ink hover:text-ink">Back to the full grid →</Link>} />
  if (ready)
    return (
      <Notice
        title="The breakdown needs a stored forecast"
        action={
          latest.data && (
            <Link to={raceHref(latest.data.season, latest.data.round)} className="text-[13px] font-semibold text-signal-ink hover:text-ink">
              Go to the latest forecast →
            </Link>
          )
        }
      >
        The reasons behind a prediction come from the forecast stored before the race. For older races, Race history has predicted against actual.
      </Notice>
    )
  return <Skeleton label="Loading the breakdown…" />
}
