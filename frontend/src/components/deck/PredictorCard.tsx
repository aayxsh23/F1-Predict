import { motion } from 'framer-motion'
import { ArrowDown, ArrowUp, ArrowUpRight, Minus } from 'lucide-react'

import { cn } from '@/lib/cn'
import { formatNumber, formatPct, formatSigned } from '@/lib/format'
import { useRaceView, type RaceDriver } from '@/lib/raceView'
import { useSelection } from '@/lib/selection'

import { rise, SheetMessage, SheetSkeleton, stagger } from './parts'

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
function OddsCell({ p, best }: { p: number | null; best: number }) {
  return (
    <td className="relative px-3 py-1.5 text-right font-mono tabular-nums">
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

export function PredictorCard() {
  const sel = useSelection()
  const { status, view } = useRaceView(sel.selectedRace)

  if (status === 'pending') return <SheetSkeleton />
  if (status === 'error') return <SheetMessage title="Predictions aren't reachable right now">The backend may still be waking up. Try again in a moment.</SheetMessage>
  if (view.mode === 'none' || view.drivers.length === 0) {
    return <SheetMessage title="No forecast for this race yet">Forecasts appear as the weekend's sessions arrive. Pick a decided round on the ribbon to see how the model called it.</SheetMessage>
  }

  const { drivers } = view
  const selected = sel.driver ?? drivers[0].code
  const hasOdds = drivers.some((d) => d.win !== null)
  const hasBand = drivers.some((d) => d.band !== null)
  const bestWin = Math.max(0, ...drivers.map((d) => d.win ?? 0))
  const bestPodium = Math.max(0, ...drivers.map((d) => d.podium ?? 0))

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="flex h-full flex-col">
      <motion.div variants={rise} className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 border-b border-line px-4 py-3 lg:px-6">
        <h2 className="font-display text-lg font-semibold uppercase tracking-[0.06em] text-silver-100">Predicted finishing order</h2>
        <p className="text-hud text-silver-400">
          {drivers.length} cars ·{' '}
          {view.mode === 'replay' ? 'the model was fitted on this race, so a replay flatters it' : hasOdds ? 'odds from 10,000 simulated races' : 'model order'}
        </p>
        <p className="w-full text-hud text-silver-400">
          {hasBand && <>Range: where the car finishes in 8 of 10 simulated races, tick = average. </>}
          {drivers.some((d) => d.qualiGapIsEstimate && d.qualiGap !== null) && <>~ marks the model's estimate before qualifying.</>}
        </p>
      </motion.div>

      <motion.div variants={rise} className="scroll-thin min-h-0 flex-1 overflow-auto">
        <table className="w-full border-separate sm:min-w-[620px] border-spacing-0 text-sm">
          <caption className="sr-only">Predicted finishing order for {view.location}</caption>
          <thead>
            <tr>
              <th scope="col" className={cn(th, 'w-12')}>Pos</th>
              <th scope="col" className={th}>Driver</th>
              {hasOdds && <th scope="col" className={cn(th, 'text-right')} title="Chance of winning, from 10,000 simulated races">Win</th>}
              {hasOdds && <th scope="col" className={cn(th, 'text-right')} title="Chance of a top-three finish">Podium</th>}
              <th scope="col" className={cn(th, 'text-right', hasOdds && 'hidden sm:table-cell')}>Grid</th>
              <th scope="col" className={cn(th, 'text-right', hasOdds && 'hidden sm:table-cell')} title="Places gained (up) or lost (down) against the starting grid">Δ Grid</th>
              {hasBand && <th scope="col" className={cn(th, 'hidden xl:table-cell')} title="Where the car finishes in 8 of 10 simulated races; the tick is the average">Range</th>}
              <th scope="col" className={cn(th, 'hidden text-right lg:table-cell')} title="Model's estimated gap to the predicted winner">Gap to P1</th>
              <th scope="col" className={cn(th, 'hidden text-right xl:table-cell')} title="Qualifying gap to pole; a tilde marks the model's estimate">Quali gap</th>
              {view.hasActuals && <th scope="col" className={cn(th, 'text-right')}>Result</th>}
              <th scope="col" className={cn(th, 'w-10')}><span className="sr-only">Explain</span></th>
            </tr>
          </thead>
          <tbody>
            {drivers.map((d: RaceDriver) => {
              const isSelected = d.code === selected
              return (
                <tr key={d.code} aria-current={isSelected ? 'true' : undefined} className={cn('group border-b border-line', isSelected ? 'bg-laser-900/70' : 'hover:bg-obsidian-700/60')}>
                  <td className="border-b border-line px-3 py-1.5 font-display text-lg font-semibold tabular-nums text-silver-100">{d.rank}</td>
                  <td className="border-b border-line px-3 py-1.5">
                    <button type="button" onClick={() => sel.openDriver(d.code)} className="flex items-center gap-2 text-left" aria-label={`Select ${d.code}`}>
                      <span className="h-2 w-2 shrink-0" style={{ background: d.accent }} aria-hidden />
                      <span className="font-display text-base font-semibold tracking-wide text-silver-100">{d.code}</span>
                      {d.number !== null && <span className="hidden font-mono text-hud text-silver-400 sm:inline">#{d.number}</span>}
                      <span className="ml-1 hidden truncate text-hud text-silver-400 lg:inline">{d.team}</span>
                    </button>
                  </td>
                  {hasOdds && <OddsCell p={d.win} best={bestWin} />}
                  {hasOdds && <OddsCell p={d.podium} best={bestPodium} />}
                  <td className={cn('border-b border-line px-3 py-1.5 text-right font-mono tabular-nums text-silver-300', hasOdds && 'hidden sm:table-cell')}>{d.grid ?? '—'}</td>
                  <td className={cn('border-b border-line px-3 py-1.5 text-right font-mono tabular-nums', hasOdds && 'hidden sm:table-cell')}><Delta delta={d.delta} /></td>
                  {hasBand && (
                    <td className="hidden border-b border-line px-3 py-1.5 xl:table-cell">
                      {d.band && <RangeBar band={d.band} expected={d.expected} field={drivers.length} />}
                    </td>
                  )}
                  <td className="hidden border-b border-line px-3 py-1.5 text-right font-mono tabular-nums text-silver-300 lg:table-cell">
                    {d.gapToLeader === null ? '—' : d.rank === 1 ? 'LEADER' : formatSigned(d.gapToLeader, 1, 's')}
                  </td>
                  <td className={cn('hidden border-b border-line px-3 py-1.5 text-right font-mono tabular-nums xl:table-cell', d.qualiGapIsEstimate ? 'text-silver-400' : 'text-silver-300')}>
                    {d.qualiGap === null ? '—' : `${d.qualiGapIsEstimate ? '~' : ''}${formatNumber(d.qualiGap, 3, 's')}`}
                  </td>
                  {view.hasActuals && (
                    <td className={cn('border-b border-line px-3 py-1.5 text-right font-mono tabular-nums', d.actual === d.rank ? 'text-laser-300' : 'text-silver-300')}>
                      {d.actual === null ? '—' : `P${d.actual}`}
                    </td>
                  )}
                  <td className="border-b border-line px-1 py-1.5 text-center">
                    <button
                      type="button"
                      onClick={() => sel.openDriver(d.code, 'shap')}
                      aria-label={`Explain ${d.code}'s prediction`}
                      className="grid h-7 w-7 place-items-center text-silver-400 hover:text-laser-300"
                    >
                      <ArrowUpRight className="h-4 w-4" aria-hidden />
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </motion.div>
    </motion.div>
  )
}
