import { motion } from 'framer-motion'
import { useState } from 'react'

import { cn } from '@/lib/cn'
import { formatFeatureName, formatNumber, formatPct, formatSigned } from '@/lib/format'
import { useDriverExplain, useNaturalExplanation, usePredictionsLatest, useRegulationsList } from '@/lib/queries'
import { useRaceView, type RaceDriver } from '@/lib/raceView'
import { useSelection } from '@/lib/selection'
import { TARGET_HIGHER_IS_BETTER, TARGET_LABELS, TARGET_UNITS, TARGETS, type ShapExplanation, type Target } from '@/lib/types'

import { rise, SheetMessage, SheetSkeleton, stagger } from './parts'

// The model's inputs in a fan's words (PRODUCT.md: "grid position", not "feature")
const FEATURE_LABELS: Record<string, string> = {
  overtaking_difficulty: 'How hard it is to pass here',
  is_street_circuit: 'Street circuit',
  pit_lane_loss_time: 'Time lost in the pit lane',
  safety_car_frequency: 'Safety cars at this track',
  dnf_rate: 'Retirements at this track',
  longest_straight_m: 'Longest straight',
  braking_zone_count: 'Heavy braking zones',
  tyre_degradation_level: 'Tyre wear at this track',
  rain_race_frequency: 'How often it rains here',
  track_length_km: 'Lap length',
  air_temp: 'Air temperature',
  track_temp: 'Track temperature',
  rain_probability: 'Rain',
  wind_speed: 'Wind',
  wet_track_probability: 'Wet track',
  grid_position: 'Starting grid slot',
  quali_gap_to_pole: 'Qualifying gap to pole',
  practice_pace: 'Practice pace',
  driver_recent_form: 'Recent results',
  driver_track_form: 'Past results here',
  driver_positions_gained_form: 'Usual places gained',
  driver_dnf_rate: 'Recent retirements',
  team_recent_form: "Team's recent results",
  team_quali_pace: "Team's qualifying pace",
  team_race_pace: "Team's race pace",
  team_reliability: "Team's reliability",
  team_track_type_form: "Team's form on tracks like this",
  teammate_quali_gap: 'Qualifying vs teammate',
  teammate_race_pace_gap: 'Race pace vs teammate',
  grid_vs_expected_position: 'Grid slot vs usual form',
  starting_tire_compound: 'Starting tyre',
  expected_stops: 'Usual pit stops here',
  historical_compound_performance: 'Past results on this tyre',
  grid_x_overtaking_difficulty: 'Grid slot on a hard-to-pass track',
}

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
    const step = { key: c.feature, label: FEATURE_LABELS[c.feature] ?? formatFeatureName(c.feature), feature: c.feature, value: c.value, from: running, to: running + c.shap, shap: c.shap }
    running += c.shap
    return step
  })
  const rest = exp.predicted_value - running // SHAP is additive: whatever the top few don't explain is everything else
  if (Math.abs(rest) > 0.005) steps.push({ key: 'rest', label: 'Everything else', feature: null, value: null, from: running, to: running + rest, shap: rest })
  return steps
}

function Waterfall({ exp, target }: { exp: ShapExplanation; target: Target }) {
  const unit = TARGET_UNITS[target]
  const decimals = target === 'finish_position' || target === 'quali_delta' ? 2 : 2
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
        <span className="flex items-center gap-1.5"><span className="h-2 w-3 bg-laser-400" aria-hidden />pushes {TARGET_LABELS[target].toLowerCase()} the right way</span>
        <span className="flex items-center gap-1.5"><span className="h-2 w-3 bg-signal-coral" aria-hidden />pushes it the wrong way</span>
        <span>{higherIsBetter ? 'higher is better' : 'lower is better'}</span>
      </div>
      <ol>
        <li className={row}>
          <span className="hud-label text-silver-400">Model average</span>
          <div className="relative h-full">{guides}</div>
          <span className="text-right font-mono text-sm tabular-nums text-silver-300">{formatNumber(exp.base_value, decimals, unit)}</span>
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
          <span className="text-right font-mono text-sm font-medium tabular-nums text-silver-100">{formatNumber(exp.predicted_value, decimals, unit)}</span>
        </li>
      </ol>
    </div>
  )
}

function TeammatePanel({ me, mate }: { me: RaceDriver; mate: RaceDriver | undefined }) {
  if (!mate) return <p className="text-sm text-silver-400">No teammate in this race's data.</p>
  const pace = me.featureRow?.teammate_race_pace_gap
  const rows: Array<{ label: string; mine: string; theirs: string; leads: 'me' | 'mate' | null }> = [
    { label: 'Predicted finish', mine: `P${me.rank}`, theirs: `P${mate.rank}`, leads: me.rank < mate.rank ? 'me' : 'mate' },
    { label: 'Grid', mine: me.grid === null ? '—' : `P${me.grid}`, theirs: mate.grid === null ? '—' : `P${mate.grid}`, leads: me.grid === null || mate.grid === null || me.grid === mate.grid ? null : me.grid < mate.grid ? 'me' : 'mate' },
    { label: 'Quali gap', mine: formatNumber(me.qualiGap, 3, 's'), theirs: formatNumber(mate.qualiGap, 3, 's'), leads: me.qualiGap === null || mate.qualiGap === null ? null : me.qualiGap < mate.qualiGap ? 'me' : 'mate' },
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
      {typeof pace === 'number' && (
        <p className="mt-2 text-hud text-silver-400">
          Recent race pace {formatSigned(pace, 2, '%')} {pace > 0 ? 'slower' : 'faster'} than {mate.code}
        </p>
      )}
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
              {[...new Set(q.data.sources.map((s) => s.split(/[/\\]/).pop() ?? s))].map((file) =>
                docs.data?.some((d) => d.filename === file) ? (
                  <button key={file} type="button" onClick={() => sel.openDoc(file)} className="border border-line-strong px-2 py-0.5 font-mono text-micro text-laser-300 hover:border-laser-500">
                    {file.replace(/\.(pdf|txt)$/i, '')}
                  </button>
                ) : (
                  <span key={file} className="border border-line px-2 py-0.5 font-mono text-micro text-silver-400">{file.replace(/\.(pdf|txt)$/i, '')}</span>
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
          <p className="mt-2 text-hud text-silver-400">Writing a grounded explanation. This runs on the local model and can take 10–20 seconds.</p>
        </div>
      ) : q.isError ? (
        <div>
          <p className="text-sm text-silver-300">The explanation service isn't reachable right now. The breakdown on the left still holds.</p>
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
  if (status === 'error') return <SheetMessage title="Attribution isn't reachable right now">The backend may still be waking up. Try again in a moment.</SheetMessage>
  if (view.mode !== 'forecast') {
    return (
      <SheetMessage
        title="Attribution needs the pre-race forecast"
        action={
          latest.data && (
            <button type="button" onClick={() => sel.selectRace(latest.data.round, latest.data.season)} className="hud-label border border-laser-500 px-3 py-2 text-laser-300 hover:bg-laser-900">
              Go to the latest forecast
            </button>
          )
        }
      >
        Feature attribution is computed from the forecast cached before a race. It exists for the most recent rounds; older races only have the replay.
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
        <div role="group" aria-label="Prediction target" className="scroll-none -mb-px flex min-w-0 gap-1 overflow-x-auto">
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
          <motion.div variants={rise} className="min-w-0">
            {shap.isPending ? (
              <SheetSkeleton rows={6} />
            ) : shap.isError || !shap.data ? (
              <p className="text-sm text-silver-300">No feature breakdown for {code} on this target right now.</p>
            ) : (
              <Waterfall key={`${code}-${sel.target}`} exp={shap.data} target={sel.target} />
            )}
          </motion.div>
          <motion.div variants={rise} className="min-w-0 space-y-6">
            {me && <TeammatePanel me={me} mate={mate} />}
            {code && race && (
              <ExplainInWords season={race.season} round={race.round} driver={code} target={sel.target} />
            )}
          </motion.div>
        </div>
      </div>
    </motion.div>
  )
}
