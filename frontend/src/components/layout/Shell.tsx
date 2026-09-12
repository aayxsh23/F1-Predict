import { Outlet } from 'react-router-dom'

import { BottomNav } from './BottomNav'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'

/*
THESIS: A prediction's number and its reasoning are never more than one click
apart — refuses the category default of a bare stats dashboard bolted to a
separate, disconnected "insights" page.
OWN-WORLD: Restrained neutrals (gray-50/900 canvas) + one F1-red accent used
only for the current selection, primary actions, and links — never chrome.
Inter for UI text, JetBrains Mono for every numeral (driver codes, positions,
gaps, timing data) so columns never jitter. Dense, timing-tower-style rows
over icon-card grids.
STORY: A fan opens the app, immediately sees the next race's predicted top 3
and how much real session data backs that prediction (known-sessions strip),
and is one tap from the SHAP chart + grounded natural-language explanation
for any driver.
FIRST VIEWPORT: Round/season label, race name, freshness line, known-sessions
badges, then ONE card holding the predicted top 3 as timing-sheet rows (big
mono rank digit, driver code, team, "Why" affordance) — not three duplicate
stat cards.
FORM: Brief-pinned, not concept-seed-rolled — phase7-ui-backend-plan.md
already specified this exact token system (restrained neutrals + accent,
Inter/JetBrains Mono, dense component tokens) as a deliberate, approved
Operate-mode direction; user confirmed building it as-is over running the
direction-exploration ritual.
FINISH: unreviewed and undocumented is unfinished; this build ends with the
finish review, the verdict, and DESIGN.md.
*/
export function Shell() {
  return (
    <div className="flex min-h-dvh bg-canvas">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar />
        <main className="flex-1 pb-20 md:pb-0">
          <Outlet />
        </main>
      </div>
      <BottomNav />
    </div>
  )
}
