import { motion } from 'framer-motion'
import { ArrowDown, ArrowUp, ArrowUpRight, Minus } from 'lucide-react'
import { useState } from 'react'

import { cn } from '@/lib/cn'
import { describeChance, formatDuration, formatLapTime, formatNumber, formatPct, formatSigned } from '@/lib/format'
import { useRaceView, type RaceDriver, type RaceView } from '@/lib/raceView'
import { useSelection } from '@/lib/selection'

import { rise, SheetMessage, SheetSkeleton, stagger } from './parts'

type Mode = 'race' | 'quali'

function Delta({ delta }: { delta: number | null }) {
  if (delta === null) return <span className="text-silver-400">—</span>
  const Icon = delta > 0 ? ArrowUp : delta < 0 ? ArrowDown : Minus
  return (
    <span className={cn('inline-flex items-center gap-1', delta > 0 && 'text-laser-300', delta < 0 && 'text-signal-coral', delta === 0 && 'text-silver-400')}>
      <Icon className="h-3 w-3" strokeWidth={3} aria-hidden />
      {Math.abs(delta)}
      <span className="sr-only">{delta > 0 ? ' places gained' : delta < 0 ? ' places lost' : ' places, holds grid'}</span>
    </span>
  )
}

/** The cell carries its own bar, scaled to the column's best value, so a scan down
 *  the column reads as a chart without a second visual. */
function OddsCell({ p, best, className }: { p: number | null; best: number; className?: string }) {
  return (
    <td className={cn('relative border-b border-line px-3 py-1.5 text-right font-mono tabular-nums', className)}>
      {p !== null && best > 0 && (
        <span className="absolute inset-y-1 left-0 bg-laser-500/25" style={{ width: `${Math.min(100, (p / best) * 100)}%` }} aria-hidden />
      )}
      <span className="relative text-silver-200">{formatPct(p)}</span>
    </td>
  )
}

function RangeBar({ band, expected, field }: { band: [number, number]; expected: number | null; field: number }) {
  const x = (p: number) => ((p - 1) / Math.max(1, field - 1)) * 100
  return (
    <svg viewBox="0 0 100 10" className="h-2.5 w-24" role="img" aria-label={`Finishes between P${band[0]} and P${band[1]} in 8 of 10 simulated races`}>
      <line x1="0" y1="5" x2="100" y2="5" className="stroke-line-strong" />
      <rect x={x(band[0])} y="2" width={Math.max(2, x(band[1]) - x(band[0]))} height="6" className="fill-laser-500/60" />
      {expected !== null && <line x1={x(expected)} x2={x(expected)} y1="0" y2="10" className="stroke-silver-100" strokeWidth="1.5" />}
    </svg>
  )
}

const th = 'hud-label sticky top-0 z-10 bg-obsidian-800 px-3 py-2 text-left font-semibold text-silver-400 border-b border-line-strong'
const td = 'border-b border-line px-3 py-1.5'

function DriverCell({ d }: { d: RaceDriver }) {
  const sel = useSelection()
  return (
    <td className={td}>
      <button type="button" onClick={() => sel.openDriver(d.code)} className="flex items-center gap-2 text-left" aria-label={`Select ${d.code}`}>
        <span className="h-2 w-2 shrink-0" style={{ background: d.accent }} aria-hidden />
        <span className="font-display text-base font-semibold tracking-wide text-silver-100">{d.code}</span>
        {d.number !== null && <span className="hidden font-mono text-hud text-silver-400 sm:inline">#{d.number}</span>}
        <span className="ml-1 hidden truncate text-hud text-silver-400 lg:inline">{d.team}</span>
      </button>
    </td>
  )
}

function ExplainCell({ code }: { code: string }) {
  const sel = useSelection()
  return (
    <td className="border-b border-line px-1 py-1.5 text-center">
      <button type="button" onClick={() => sel.openDriver(code, 'shap')} aria-label={`Why is ${code} predicted here?`} className="grid h-7 w-7 place-items-center text-silver-400 hover:text-laser-300">
        <ArrowUpRight className="h-4 w-4" aria-hidden />
      </button>
    </td>
  )
}

function RaceTable({ view, selected }: { view: RaceView; selected: string }) {
  const { drivers } = view
  const hasOdds = drivers.some((d) => d.win !== null)
  const hasBand = drivers.some((d) => d.band !== null)
  const hasGrid = drivers.some((d) => d.grid !== null) // columns with nothing in them yet stay hidden
  const hasGap = drivers.some((d) => d.gapToLeader !== null)
  const bestWin = Math.max(0, ...drivers.map((d) => d.win ?? 0))
  const bestPodium = Math.max(0, ...drivers.map((d) => d.podium ?? 0))
  return (
    <table className="w-full border-separate border-spacing-0 text-sm sm:min-w-[620px]">
      <caption className="sr-only">Predicted finishing order for {view.location}</caption>
      <thead>
        <tr>
          <th scope="col" className={cn(th, 'w-12')}>Pos</th>
          <th scope="col" className={th}>Driver</th>
          {hasOdds && <th scope="col" className={cn(th, 'text-right')} title="Chance of winning, from 10,000 simulated races">Win</th>}
          {hasOdds && <th scope="col" className={cn(th, 'text-right')} title="Chance of a top-three finish">Podium</th>}
          {hasOdds && <th scope="col" className={cn(th, 'hidden text-right 2xl:table-cell')} title="Chance of finishing in the points (top 10)">Points</th>}
          {hasGrid && <th scope="col" className={cn(th, 'text-right', hasOdds && 'hidden sm:table-cell')}>Grid</th>}
          {hasGrid && <th scope="col" className={cn(th, 'text-right', hasOdds && 'hidden sm:table-cell')} title="Places gained (up) or lost (down) against the starting grid">± Grid</th>}
          {hasBand && <th scope="col" className={cn(th, 'hidden xl:table-cell')} title="Where the car finishes in 8 of 10 simulated races; the tick is the average">Likely range</th>}
          {hasGap && <th scope="col" className={cn(th, 'hidden text-right lg:table-cell')} title="Predicted gap to the winner at the flag">Gap</th>}
          {view.hasActuals && <th scope="col" className={cn(th, 'text-right')}>Result</th>}
          <th scope="col" className={cn(th, 'w-10')}><span className="sr-only">Why</span></th>
        </tr>
      </thead>
      <tbody>
        {drivers.map((d) => (
          <tr key={d.code} aria-current={d.code === selected ? 'true' : undefined} className={cn('group', d.code === selected ? 'bg-laser-900/70' : 'hover:bg-obsidian-700/60')}>
            <td className={cn(td, 'font-display text-lg font-semibold tabular-nums text-silver-100')}>{d.rank}</td>
            <DriverCell d={d} />
            {hasOdds && <OddsCell p={d.win} best={bestWin} />}
            {hasOdds && <OddsCell p={d.podium} best={bestPodium} />}
            {hasOdds && <td className={cn(td, 'hidden text-right font-mono tabular-nums text-silver-300 2xl:table-cell')}>{formatPct(d.top10)}</td>}
            {hasGrid && <td className={cn(td, 'text-right font-mono tabular-nums text-silver-300', hasOdds && 'hidden sm:table-cell')}>{d.grid ?? '—'}</td>}
            {hasGrid && <td className={cn(td, 'text-right font-mono tabular-nums', hasOdds && 'hidden sm:table-cell')}><Delta delta={d.delta} /></td>}
            {hasBand && <td className={cn(td, 'hidden xl:table-cell')}>{d.band && <RangeBar band={d.band} expected={d.expected} field={drivers.length} />}</td>}
            {hasGap && (
              <td className={cn(td, 'hidden text-right font-mono tabular-nums text-silver-300 lg:table-cell')}>
                {d.gapToLeader === null ? '—' : d.rank === 1 ? 'WINNER' : formatSigned(d.gapToLeader, 1, 's')}
              </td>
            )}
            {view.hasActuals && (
              <td className={cn(td, 'text-right font-mono tabular-nums', d.actual === d.rank ? 'text-laser-300' : 'text-silver-300')}>{d.actual === null ? '—' : `P${d.actual}`}</td>
            )}
            <ExplainCell code={d.code} />
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function QualiTable({ view, selected }: { view: RaceView; selected: string }) {
  const drivers = [...view.drivers].sort((a, b) => a.qualiRank - b.qualiRank)
  const hasOdds = drivers.some((d) => d.pole !== null)
  const hasLap = drivers.some((d) => d.qualiLap !== null)
  const hasGap = drivers.some((d) => d.qualiGapPredicted !== null)
  const happened = drivers.some((d) => !d.qualiGapIsEstimate)
  const bestPole = Math.max(0, ...drivers.map((d) => d.pole ?? 0))
  return (
    <table className="w-full border-separate border-spacing-0 text-sm sm:min-w-[560px]">
      <caption className="sr-only">Predicted qualifying order for {view.location}</caption>
      <thead>
        <tr>
          <th scope="col" className={cn(th, 'w-12')}>Pos</th>
          <th scope="col" className={th}>Driver</th>
          {hasLap && <th scope="col" className={cn(th, 'text-right')} title="Predicted best qualifying lap">Lap</th>}
          {hasGap && <th scope="col" className={cn(th, 'text-right')} title="Predicted gap to the fastest car">Gap</th>}
          {hasOdds && <th scope="col" className={cn(th, 'text-right')} title="Chance of pole position">Pole</th>}
          {hasOdds && <th scope="col" className={cn(th, 'hidden text-right sm:table-cell')} title="Chance of reaching the top-10 shootout">Q3</th>}
          {hasOdds && <th scope="col" className={cn(th, 'hidden text-right lg:table-cell')} title="Chance of being knocked out in the first session">Out in Q1</th>}
          {happened && <th scope="col" className={cn(th, 'text-right')} title="The real gap to pole">Real gap</th>}
          <th scope="col" className={cn(th, 'w-10')}><span className="sr-only">Why</span></th>
        </tr>
      </thead>
      <tbody>
        {drivers.map((d) => (
          <tr key={d.code} aria-current={d.code === selected ? 'true' : undefined} className={cn(d.code === selected ? 'bg-laser-900/70' : 'hover:bg-obsidian-700/60')}>
            <td className={cn(td, 'font-display text-lg font-semibold tabular-nums text-silver-100')}>{d.qualiRank}</td>
            <DriverCell d={d} />
            {hasLap && <td className={cn(td, 'text-right font-mono tabular-nums text-silver-200')}>{formatLapTime(d.qualiLap)}</td>}
            {hasGap && (
              <td className={cn(td, 'text-right font-mono tabular-nums text-silver-300')}>
                {d.qualiGapPredicted === null ? '—' : d.qualiRank === 1 ? 'POLE' : formatSigned(d.qualiGapPredicted, 3, 's')}
              </td>
            )}
            {hasOdds && <OddsCell p={d.pole} best={bestPole} />}
            {hasOdds && <td className={cn(td, 'hidden text-right font-mono tabular-nums text-silver-300 sm:table-cell')}>{formatPct(d.q3)}</td>}
            {hasOdds && <td className={cn(td, 'hidden text-right font-mono tabular-nums text-silver-300 lg:table-cell')}>{formatPct(d.q1Out)}</td>}
            {happened && (
              <td className={cn(td, 'text-right font-mono tabular-nums text-silver-200')}>{d.qualiGapIsEstimate ? '—' : formatNumber(d.qualiGap, 3, 's')}</td>
            )}
            <ExplainCell code={d.code} />
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Summary({ view, mode }: { view: RaceView; mode: Mode }) {
  const r = view.race
  if (view.mode === 'replay') return <>Predicted before the race by a model that hadn't seen it, next to what happened.</>
  if (mode === 'quali') {
    return (
      <>
        {r?.pole_time_estimate_s ? <>Pole lap around {formatLapTime(r.pole_time_estimate_s)}. </> : <>Lap times appear once practice has run. </>}
        {view.drivers.some((d) => d.pole !== null) && 'Odds from 10,000 simulated qualifying sessions.'}
      </>
    )
  }
  return (
    <>
      {r?.laps ? `${r.laps} laps` : ''}
      {r?.expected_duration_s ? ` · about ${formatDuration(r.expected_duration_s)}` : ''}
      {r?.safety_car_probability !== null && r?.safety_car_probability !== undefined ? ` · safety car ${describeChance(r.safety_car_probability)}` : ''}
      {view.drivers.some((d) => d.win !== null) ? ' · odds from 10,000 simulated races' : ''}
    </>
  )
}

export function PredictorCard() {
  const sel = useSelection()
  const { status, view } = useRaceView(sel.selectedRace)
  const [mode, setMode] = useState<Mode>('race')

  if (status === 'pending') return <SheetSkeleton />
  if (status === 'error') return <SheetMessage title="Predictions aren't reachable right now">The backend may still be waking up. Try again in a moment.</SheetMessage>
  if (view.mode === 'none' || view.drivers.length === 0) {
    return <SheetMessage title="No forecast for this race yet">Forecasts appear once the race is next on the calendar. Pick a past round on the ribbon to see how the model called it.</SheetMessage>
  }

  const selected = sel.driver ?? view.drivers[0].code
  const hasBand = view.drivers.some((d) => d.band !== null)

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="flex h-full flex-col">
      <motion.div variants={rise} className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2 border-b border-line px-4 py-2.5 lg:px-6">
        <div role="group" aria-label="Show" className="flex gap-1">
          {(['race', 'quali'] as const).map((m) => (
            <button
              key={m}
              type="button"
              aria-pressed={mode === m}
              onClick={() => setMode(m)}
              className={cn('hud-label h-8 border px-3', mode === m ? 'border-laser-400 bg-laser-900 text-laser-300' : 'border-line text-silver-400 hover:border-line-strong hover:text-silver-200')}
            >
              {m === 'race' ? 'Race' : 'Qualifying'}
            </button>
          ))}
        </div>
        <p className="min-w-0 flex-1 text-right text-hud text-silver-400">
          <Summary view={view} mode={mode} />
        </p>
        {mode === 'race' && hasBand && (
          <p className="hidden w-full text-hud text-silver-400 xl:block">Likely range: where the car finishes in 8 of 10 simulated races; the tick is the average.</p>
        )}
      </motion.div>

      <motion.div variants={rise} className="scroll-thin min-h-0 flex-1 overflow-auto">
        {mode === 'race' ? <RaceTable view={view} selected={selected} /> : <QualiTable view={view} selected={selected} />}
      </motion.div>
    </motion.div>
  )
}
