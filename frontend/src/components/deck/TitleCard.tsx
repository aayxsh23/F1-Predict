import { motion } from 'framer-motion'
import { useState } from 'react'

import { cn } from '@/lib/cn'
import { formatPct } from '@/lib/format'
import { useChampionship } from '@/lib/queries'
import { useSelection } from '@/lib/selection'
import { teamAccent } from '@/lib/teams'

import { rise, SheetMessage, SheetSkeleton, stagger } from './parts'

const th = 'hud-label sticky top-0 z-10 bg-obsidian-800 px-3 py-2 text-left font-semibold text-silver-400 border-b border-line-strong'
const td = 'border-b border-line px-3 py-1.5'

function ChanceCell({ p, best }: { p: number; best: number }) {
  return (
    <td className={cn(td, 'relative text-right font-mono tabular-nums')}>
      {best > 0 && p > 0 && <span className="absolute inset-y-1 left-0 bg-laser-500/25" style={{ width: `${(p / best) * 100}%` }} aria-hidden />}
      <span className="relative text-silver-200">{p === 0 ? '<0.1%' : formatPct(p)}</span>
    </td>
  )
}

/** Live standings plus each contender's chance of the title, from simulating the
 *  rest of the season thousands of times with every driver's current form. */
export function TitleCard() {
  const sel = useSelection()
  const q = useChampionship()
  const [kind, setKind] = useState<'drivers' | 'constructors'>('drivers')

  if (q.isPending) return <SheetSkeleton />
  if (q.isError || !q.data) {
    return <SheetMessage title="Live standings aren't reachable right now">The standings service may be rate-limiting or waking up. Try again in a minute.</SheetMessage>
  }
  const { drivers, constructors, remaining } = q.data
  const bestDriver = Math.max(0, ...drivers.map((d) => d.title_chance))
  const bestTeam = Math.max(0, ...constructors.map((c) => c.title_chance))

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="flex h-full flex-col">
      <motion.div variants={rise} className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2 border-b border-line px-4 py-2.5 lg:px-6">
        <div role="group" aria-label="Championship" className="flex gap-1">
          {(['drivers', 'constructors'] as const).map((k) => (
            <button
              key={k}
              type="button"
              aria-pressed={kind === k}
              onClick={() => setKind(k)}
              className={cn('hud-label h-8 border px-3', kind === k ? 'border-laser-400 bg-laser-900 text-laser-300' : 'border-line text-silver-400 hover:border-line-strong hover:text-silver-200')}
            >
              {k === 'drivers' ? 'Drivers' : 'Constructors'}
            </button>
          ))}
        </div>
        <p className="min-w-0 flex-1 text-right text-hud text-silver-400">
          {remaining.races} race{remaining.races === 1 ? '' : 's'}{remaining.sprints ? ` and ${remaining.sprints} sprint${remaining.sprints === 1 ? '' : 's'}` : ''} left · title chances from 5,000 simulated seasons at current form
        </p>
      </motion.div>

      <motion.div variants={rise} className="scroll-thin min-h-0 flex-1 overflow-auto">
        {kind === 'drivers' ? (
          <table className="w-full border-separate border-spacing-0 text-sm">
            <caption className="sr-only">Drivers' championship with title chances</caption>
            <thead>
              <tr>
                <th scope="col" className={cn(th, 'w-12')}>Pos</th>
                <th scope="col" className={th}>Driver</th>
                <th scope="col" className={cn(th, 'text-right')}>Points</th>
                <th scope="col" className={cn(th, 'hidden text-right sm:table-cell')}>Wins</th>
                <th scope="col" className={cn(th, 'text-right')} title="Chance of winning the championship">Title</th>
                <th scope="col" className={cn(th, 'hidden text-right md:table-cell')} title="Average final points across the simulated seasons">Expected final</th>
              </tr>
            </thead>
            <tbody>
              {drivers.map((d) => (
                <tr key={d.code} className="hover:bg-obsidian-700/60">
                  <td className={cn(td, 'font-display text-lg font-semibold tabular-nums text-silver-100')}>{d.position}</td>
                  <td className={td}>
                    <button type="button" onClick={() => sel.openDriver(d.code)} className="flex items-center gap-2 text-left">
                      <span className="h-2 w-2 shrink-0" style={{ background: teamAccent(d.team ?? '') }} aria-hidden />
                      <span className="font-display text-base font-semibold tracking-wide text-silver-100">{d.code}</span>
                      <span className="hidden text-hud text-silver-400 sm:inline">{d.given_name} {d.family_name}</span>
                    </button>
                  </td>
                  <td className={cn(td, 'text-right font-mono tabular-nums text-silver-200')}>{d.points}</td>
                  <td className={cn(td, 'hidden text-right font-mono tabular-nums text-silver-300 sm:table-cell')}>{d.wins}</td>
                  <ChanceCell p={d.title_chance} best={bestDriver} />
                  <td className={cn(td, 'hidden text-right font-mono tabular-nums text-silver-300 md:table-cell')}>{Math.round(d.expected_points)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <table className="w-full border-separate border-spacing-0 text-sm">
            <caption className="sr-only">Constructors' championship with title chances</caption>
            <thead>
              <tr>
                <th scope="col" className={cn(th, 'w-12')}>Pos</th>
                <th scope="col" className={th}>Team</th>
                <th scope="col" className={cn(th, 'text-right')}>Points</th>
                <th scope="col" className={cn(th, 'hidden text-right sm:table-cell')}>Wins</th>
                <th scope="col" className={cn(th, 'text-right')}>Title</th>
              </tr>
            </thead>
            <tbody>
              {constructors.map((c) => (
                <tr key={c.name} className="hover:bg-obsidian-700/60">
                  <td className={cn(td, 'font-display text-lg font-semibold tabular-nums text-silver-100')}>{c.position}</td>
                  <td className={td}>
                    <span className="flex items-center gap-2">
                      <span className="h-2 w-2 shrink-0" style={{ background: teamAccent(c.name) }} aria-hidden />
                      <span className="text-silver-100">{c.name}</span>
                    </span>
                  </td>
                  <td className={cn(td, 'text-right font-mono tabular-nums text-silver-200')}>{c.points}</td>
                  <td className={cn(td, 'hidden text-right font-mono tabular-nums text-silver-300 sm:table-cell')}>{c.wins}</td>
                  <ChanceCell p={c.title_chance} best={bestTeam} />
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="px-4 py-3 text-hud text-silver-400 lg:px-6">
          Chances assume today's form holds for the rest of the season. Ties on points are split rather than decided on countback. Ask the analyst for the exact points a driver needs.
        </p>
      </motion.div>
    </motion.div>
  )
}
