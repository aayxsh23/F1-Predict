import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { Activity, ListOrdered, MessageSquare, Scale, Trophy, type LucideIcon } from 'lucide-react'
import type { KeyboardEvent } from 'react'

import { cn } from '@/lib/cn'
import { TAB_ORDER, useSelection, type Tab } from '@/lib/selection'

import { PredictorCard } from './PredictorCard'
import { RulesCard } from './RulesCard'
import { ShapCard } from './ShapCard'
import { StrategistCard } from './StrategistCard'
import { TitleCard } from './TitleCard'

const TABS: Record<Tab, { label: string; short: string; Icon: LucideIcon }> = {
  predictor: { label: 'FORECAST', short: 'FORECAST', Icon: ListOrdered },
  shap: { label: 'WHY THIS PREDICTION', short: 'WHY', Icon: Activity },
  strategist: { label: 'STRATEGY & ANALYST', short: 'ANALYST', Icon: MessageSquare },
  title: { label: 'TITLE RACE', short: 'TITLE', Icon: Trophy },
  rules: { label: 'RULES & RULINGS', short: 'RULES', Icon: Scale },
}

function ActiveCard({ tab }: { tab: Tab }) {
  switch (tab) {
    case 'predictor':
      return <PredictorCard />
    case 'strategist':
      return <StrategistCard />
    case 'shap':
      return <ShapCard />
    case 'title':
      return <TitleCard />
    case 'rules':
      return <RulesCard />
  }
}

/** A telemetry dossier: four folder tabs along the top, one sheet open at a time.
 *  The active tab is a shared-layout element (layoutId="activeFolder") that
 *  slides between tabs and is filled like the sheet, so it reads as part of it. */
export function TelemetryFolderDeck() {
  const sel = useSelection()
  const reduced = useReducedMotion()

  function onKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    const i = TAB_ORDER.indexOf(sel.tab)
    const n = TAB_ORDER.length
    const next = { ArrowRight: (i + 1) % n, ArrowLeft: (i + n - 1) % n, Home: 0, End: n - 1 }[e.key as string]
    if (next === undefined) return
    e.preventDefault()
    sel.setTab(TAB_ORDER[next])
    document.getElementById(`tab-${TAB_ORDER[next]}`)?.focus()
  }

  return (
    <section aria-label="Telemetry dossier" className="relative z-10 shrink-0 pb-3 lg:pb-5">
      <div role="tablist" aria-label="Dossier" onKeyDown={onKeyDown} className="scroll-none relative z-10 flex gap-1 overflow-x-auto px-3 pt-1 lg:px-6">
        {TAB_ORDER.map((tab) => {
          const { label, short, Icon } = TABS[tab]
          const n = String(TAB_ORDER.indexOf(tab) + 1).padStart(2, '0')
          const active = sel.tab === tab
          return (
            <button
              key={tab}
              id={`tab-${tab}`}
              type="button"
              role="tab"
              aria-selected={active}
              aria-controls="dossier-sheet"
              tabIndex={active ? 0 : -1}
              onClick={() => sel.setTab(tab)}
              className={cn('relative -mb-px h-11 shrink-0 text-left sm:min-w-[150px] sm:flex-1 lg:max-w-[268px]', active ? 'z-10' : 'z-0')}
            >
              {active ? (
                <motion.span
                  layoutId="activeFolder"
                  transition={reduced ? { duration: 0 } : { type: 'spring', stiffness: 520, damping: 42 }}
                  className="cut-frame cut-top frame-carbon absolute inset-0"
                  style={{ ['--frame' as string]: 'var(--color-laser-400)', ['--inset-b' as string]: '0px' }}
                />
              ) : (
                <span
                  className="cut-frame cut-top absolute inset-x-0 bottom-0 top-1.5"
                  style={{ ['--fill' as string]: 'var(--color-obsidian-950)' }}
                />
              )}
              <span className={cn('relative flex h-full items-center gap-2.5 px-4', active ? 'text-silver-100' : 'text-silver-400 hover:text-silver-200')}>
                <Icon className={cn('h-4 w-4 shrink-0', active && 'text-laser-400')} strokeWidth={2} aria-hidden />
                <span className="hud-label whitespace-nowrap">
                  <span className={active ? 'text-laser-400' : ''}>{n}</span> // <span className="sm:hidden">{short}</span>
                  <span className="hidden sm:inline">{label}</span>
                </span>
              </span>
            </button>
          )
        })}
      </div>

      <div
        id="dossier-sheet"
        role="tabpanel"
        aria-labelledby={`tab-${sel.tab}`}
        className={cn(
          'cut-frame cut-bottom frame-carbon relative mx-3 h-[min(74dvh,640px)] lg:mx-6',
          // a conversation needs room to read; the stage gives way while it's open
          sel.tab === 'strategist' ? 'lg:h-[clamp(360px,50dvh,560px)]' : 'lg:h-[clamp(240px,34dvh,380px)]',
        )}
        style={{ ['--c-tr' as string]: '0px' }}
      >
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={sel.tab}
            className="absolute inset-0 overflow-hidden"
            initial={reduced ? false : { opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={reduced ? undefined : { opacity: 0, y: 24, transition: { duration: 0.16, ease: 'easeIn' } }}
            transition={{ duration: 0.36, ease: [0.16, 1, 0.3, 1] }}
          >
            <ActiveCard tab={sel.tab} />
          </motion.div>
        </AnimatePresence>
      </div>
    </section>
  )
}
