import { motion } from 'framer-motion'
import { CornerDownLeft, FileText, Gauge, History, ListOrdered, Radio, Scale, Search, User, Wrench, type LucideIcon } from 'lucide-react'
import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { useNavigate } from 'react-router-dom'

import { useBacktestRaces, usePredictionsLatest, useRegulationsList } from '@/lib/queries'
import { useRaceView } from '@/lib/raceView'

import { driverHref, raceHref, useDriverNames } from './data'

interface Item {
  group: string
  label: string
  meta: string
  icon: LucideIcon
  run: () => void
}

/** ⌘K: jump to a page, a driver, a past race or a document. */
export function Palette({ onClose, openRadio }: { onClose: () => void; openRadio: () => void }) {
  const navigate = useNavigate()
  const latest = usePredictionsLatest()
  const { view } = useRaceView(latest.data ? { season: latest.data.season, round: latest.data.round } : null)
  const archive = useBacktestRaces()
  const docs = useRegulationsList()
  const names = useDriverNames()
  const [q, setQ] = useState('')
  const [active, setActive] = useState(0)
  const input = useRef<HTMLInputElement>(null)
  const list = useRef<HTMLDivElement>(null)

  useEffect(() => input.current?.focus(), [])

  const go = (to: string) => () => navigate(to, { viewTransition: true })
  const pq = q.trim().toLowerCase()
  const pastRaces = [...(archive.data ?? [])].sort((a, b) => b.season - a.season || b.round - a.round)
  const all: Item[] = [
    { group: 'Pages', label: 'Race weekend', meta: view ? `${view.location} · R${view.round}` : '', icon: Gauge, run: go('/') },
    ...(view ? [{ group: 'Pages', label: 'Full grid', meta: `${view.drivers.length} drivers`, icon: ListOrdered, run: go(raceHref(view.season, view.round)) }] : []),
    { group: 'Pages', label: 'Race history', meta: archive.data ? `${archive.data.length} races` : '', icon: History, run: go('/history') },
    { group: 'Pages', label: 'Regulations', meta: docs.data ? `${docs.data.length} docs` : '', icon: Scale, run: go('/regulations') },
    { group: 'Pages', label: 'Workbench', meta: 'strategy · title odds', icon: Wrench, run: go('/workbench') },
    ...(view?.drivers ?? []).map((d) => ({
      group: 'Drivers',
      label: names.get(d.code) ? `${d.code} · ${names.get(d.code)}` : d.code,
      meta: `${d.team} · P${d.rank}`,
      icon: User,
      run: go(driverHref(view!.season, view!.round, d.code)),
    })),
    ...pastRaces.map((r) => ({ group: 'Past races', label: `${r.location} · ${r.season} R${r.round}`, meta: 'predicted vs actual', icon: History, run: go(`/history/${r.season}/${r.round}`) })),
    ...(docs.data ?? []).map((d) => ({
      group: 'Documents',
      label: d.title,
      meta: d.doc_type === 'regulation' ? 'Regulation' : 'Decision',
      icon: FileText,
      run: go(`/regulations?doc=${encodeURIComponent(d.filename)}`),
    })),
    { group: 'Actions', label: 'Open Pit Radio', meta: 'Ctrl+\\', icon: Radio, run: openRadio },
  ]
  // with no query: every page, the top five drivers, the six latest races, every document
  const cap: Record<string, number> = { Drivers: 5, 'Past races': 6 }
  const seen: Record<string, number> = {}
  // a query matches names, plus a driver's team ("ferrari" finds both)
  const matches = (it: Item) => it.label.toLowerCase().includes(pq) || (it.group === 'Drivers' && it.meta.toLowerCase().includes(pq))
  const items = (pq ? all.filter(matches) : all.filter((it) => (seen[it.group] = (seen[it.group] ?? 0) + 1) <= (cap[it.group] ?? Infinity))).slice(0, 40)
  const at = Math.min(active, Math.max(0, items.length - 1))

  const run = (it: Item) => {
    onClose()
    it.run()
  }

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault()
      const next = e.key === 'ArrowDown' ? Math.min(at + 1, items.length - 1) : Math.max(at - 1, 0)
      setActive(next)
      list.current?.querySelector(`[data-i="${next}"]`)?.scrollIntoView({ block: 'nearest' })
    } else if (e.key === 'Enter' && items[at]) {
      e.preventDefault()
      run(items[at])
    }
  }

  return (
    <>
      <div onClick={onClose} className="absolute inset-0 z-50 bg-[#050607]/55 backdrop-blur-[6px]" aria-hidden />
      <motion.div
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
        className="absolute left-1/2 top-[12%] z-[51] w-[min(640px,calc(100%-24px))] -translate-x-1/2 overflow-hidden rounded-xl border border-white/[.14] bg-carbon/95 shadow-[0_30px_80px_rgb(0_0_0/.6)] backdrop-blur-[24px]"
      >
        <div className="flex items-center gap-3 border-b border-white/[.08] px-4">
          <Search className="size-[18px] shrink-0 text-ink-3" aria-hidden />
          <input
            ref={input}
            value={q}
            onChange={(e) => {
              setQ(e.target.value)
              setActive(0)
            }}
            onKeyDown={onKeyDown}
            role="combobox"
            aria-expanded="true"
            aria-controls="pw-palette-list"
            aria-activedescendant={items[at] ? `pw-pal-${at}` : undefined}
            aria-label="Jump to a driver, race, document, or page"
            placeholder="Jump to a driver, race, document, or page…"
            className="h-[58px] min-w-0 flex-1 bg-transparent font-editorial text-[21px] italic leading-none text-ink outline-none placeholder:text-ink-3 focus-visible:outline-none"
          />
          <kbd className="rounded-[4px] border border-white/[.12] px-1.5 py-[5px] font-mono text-[10px] font-semibold leading-none text-ink-3">Esc</kbd>
        </div>
        <div ref={list} id="pw-palette-list" role="listbox" className="max-h-[min(420px,60vh)] overflow-y-auto p-1.5">
          {items.length === 0 && <p className="px-3 py-[18px] font-editorial text-[19px] italic leading-none text-ink-3">Nothing matches.</p>}
          {items.map((it, i) => {
            const on = i === at
            const Icon = it.icon
            return (
              <div key={`${it.group}-${it.label}-${i}`}>
                {(i === 0 || items[i - 1].group !== it.group) && <p className="pw-label px-2.5 pb-1.5 pt-2.5 text-[10px]">{it.group}</p>}
                <div
                  id={`pw-pal-${i}`}
                  data-i={i}
                  role="option"
                  aria-selected={on}
                  onClick={() => run(it)}
                  onPointerMove={() => !on && setActive(i)}
                  className="flex w-full cursor-pointer items-center gap-3 rounded-md p-2.5"
                  style={{ background: on ? 'rgb(255 255 255 / .07)' : 'transparent' }}
                >
                  <Icon className="size-4 shrink-0" style={{ color: on ? 'var(--color-signal-ink)' : 'var(--color-ink-3)' }} aria-hidden />
                  <span className="min-w-0 flex-1 truncate text-[13px] font-semibold leading-tight">{it.label}</span>
                  <span className="hidden whitespace-nowrap text-[11px] leading-none text-ink-3 sm:inline">{it.meta}</span>
                  <CornerDownLeft className="size-3.5 text-ink-3" style={{ opacity: on ? 1 : 0 }} aria-hidden />
                </div>
              </div>
            )
          })}
        </div>
        <div className="flex gap-4 border-t border-white/[.07] px-4 py-2.5 text-[10px] font-medium uppercase leading-none tracking-[.1em] text-ink-3">
          <span>↑↓ move</span>
          <span>↵ open</span>
          <span>esc close</span>
        </div>
      </motion.div>
    </>
  )
}
