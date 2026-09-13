import { Sparkles } from 'lucide-react'
import { Outlet, useParams } from 'react-router-dom'

import { BottomNav } from '@/components/layout/BottomNav'
import { MobileTopBar } from '@/components/layout/MobileTopBar'

import { ActivityBar } from './ActivityBar'
import { CopilotDock } from './CopilotDock'
import { CopilotProvider, useCopilot, type CopilotViewContext } from './CopilotProvider'
import { StatusStrip } from './StatusStrip'

/*
THESIS: reasoning (the copilot) and the data it explains are never in
different places you have to navigate between -- refuses the category
default of a chatbot bolted onto a separate page that forgets you
mid-conversation.
OWN-WORLD: the existing restrained-neutral/one-accent/dense-timing-tower
system, read as a cockpit instrument panel -- an icon-only activity rail
instead of a labeled sidebar, a permanently-docked copilot instead of a
routed chat page, a telemetry micro-palette reserved strictly for live
status signals, layered on the one-accent rule, never replacing it.
STORY: a fan hovers a predicted-finish row, asks the copilot "why" without
losing their place, clicks the driver code in the reply, and watches that
exact row flash back into view.
FIRST VIEWPORT (lg:+): activity rail + predictions center pane + open
copilot dock with view-aware badges, all visible at once.
FORM: brief-pinned like the original Shell.tsx thesis -- a scoped extension
of an already-approved token system and nav model, not a new visual-world
exploration.
FINISH: unreviewed and undocumented is unfinished; this build ends with the
finish review, the verdict, and DESIGN.md's Layout + Telemetry Accent
sections updated to match what shipped.
*/
export function WorkbenchShell() {
  const params = useParams<{ season?: string; round?: string; driver?: string }>()

  const viewContext: CopilotViewContext = {
    season: params.season ? Number(params.season) : undefined,
    round: params.round ? Number(params.round) : undefined,
    driver: params.driver,
  }

  return (
    <CopilotProvider viewContext={viewContext}>
      <WorkbenchLayout />
    </CopilotProvider>
  )
}

function WorkbenchLayout() {
  const { dockCollapsed, setDockCollapsed } = useCopilot()

  return (
    <div className="flex h-dvh flex-col bg-canvas">
      <div className="flex min-h-0 flex-1">
        <ActivityBar />

        <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
          <MobileTopBar />
          <main className="flex-1 overflow-y-auto pb-20 md:pb-0">
            <Outlet />
          </main>
        </div>

        {!dockCollapsed && (
          <div
            onClick={() => setDockCollapsed(true)}
            aria-hidden
            className="fixed inset-0 z-30 hidden bg-black/30 md:block lg:hidden"
          />
        )}

        <CopilotDock
          className={
            !dockCollapsed
              ? 'fixed inset-0 z-50 md:inset-y-0 md:left-auto md:right-0 md:w-full md:max-w-[420px] md:border-l lg:static lg:inset-auto lg:w-[400px] lg:max-w-none lg:shrink-0'
              : ''
          }
        />

        {dockCollapsed && (
          <button
            onClick={() => setDockCollapsed(false)}
            aria-label="Open Copilot"
            className="fixed bottom-20 right-4 z-40 flex h-12 w-12 items-center justify-center rounded-full bg-accent text-white shadow-md transition-colors duration-150 hover:bg-accent-hover md:hidden"
          >
            <Sparkles className="h-5 w-5" strokeWidth={2} />
          </button>
        )}
      </div>

      <StatusStrip />
      <BottomNav />
    </div>
  )
}
