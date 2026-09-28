import { useEffect } from 'react'

import { TelemetryFolderDeck } from '@/components/deck/TelemetryFolderDeck'
import { PodiumHero } from '@/components/hero/PodiumHero'
import { SeasonRaceBar } from '@/components/season/SeasonRaceBar'
import { ChatProvider } from '@/lib/chat'
import { SelectionProvider, TAB_ORDER, useSelection } from '@/lib/selection'

/** 1-5 open a dossier tab, [ and ] step through the calendar. Never while typing. */
function Shortcuts() {
  const sel = useSelection()
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      if ((e.target as HTMLElement).closest('input, textarea, select, [contenteditable]')) return
      const tab = TAB_ORDER[Number(e.key) - 1]
      if (tab) sel.setTab(tab)
      else if (e.key === '[') sel.stepRace(-1)
      else if (e.key === ']') sel.stepRace(1)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [sel])
  return null
}

/** The whole product on one surface: season ribbon on top, the predicted podium
 *  on stage, and the dossier of tools below. Everything reads one selected race. */
export function WorkbenchPage() {
  return (
    <SelectionProvider>
      <ChatProvider>
        <Shortcuts />
        <a
          href="#dossier-sheet"
          className="hud-label fixed left-3 top-3 z-50 -translate-y-16 bg-obsidian-950 px-3 py-2 text-laser-300 focus:translate-y-0"
        >
          Skip to the dossier
        </a>
        <div className="flex min-h-dvh flex-col bg-obsidian-900 lg:h-dvh lg:overflow-hidden">
          <SeasonRaceBar />
          <main className="flex min-h-0 flex-1 flex-col">
            <PodiumHero />
            <TelemetryFolderDeck />
          </main>
        </div>
      </ChatProvider>
    </SelectionProvider>
  )
}
