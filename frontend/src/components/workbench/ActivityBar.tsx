import { PanelRightClose, PanelRightOpen } from 'lucide-react'
import { NavLink, useLocation } from 'react-router-dom'

import { Tooltip } from '@/components/ui/Tooltip'
import { NAV_ITEMS } from '@/components/layout/navItems'
import { cn } from '@/lib/cn'

import { useCopilot } from './CopilotProvider'
import { SettingsPopover } from './SettingsPopover'

const ITEM_HEIGHT = 48

function isActivePath(pathname: string, to: string, end: boolean) {
  return end ? pathname === to : pathname === to || pathname.startsWith(`${to}/`)
}

export function ActivityBar() {
  const { pathname } = useLocation()
  const { dockCollapsed, setDockCollapsed } = useCopilot()
  const activeIndex = NAV_ITEMS.findIndex((item) => isActivePath(pathname, item.to, item.end))

  return (
    <aside className="hidden w-14 shrink-0 flex-col items-center border-r border-border-default bg-surface-sunken py-3 md:flex">
      <nav className="relative flex flex-col gap-1">
        {activeIndex >= 0 && (
          <span
            aria-hidden
            className="absolute left-0 h-8 w-1 rounded-r-full bg-accent transition-transform duration-200 ease-out"
            style={{ transform: `translateY(${activeIndex * ITEM_HEIGHT + (ITEM_HEIGHT - 32) / 2}px)` }}
          />
        )}
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <Tooltip key={to} label={label}>
            <NavLink
              to={to}
              end={end}
              style={{ height: ITEM_HEIGHT }}
              className={({ isActive }) =>
                cn(
                  'flex w-12 items-center justify-center rounded-md text-text-secondary transition-colors duration-150 hover:bg-surface hover:text-text-primary',
                  isActive && 'text-accent-text',
                )
              }
            >
              <Icon className="h-5 w-5" strokeWidth={2} />
              <span className="sr-only">{label}</span>
            </NavLink>
          </Tooltip>
        ))}
      </nav>

      <div className="mt-auto flex flex-col items-center gap-1">
        <Tooltip label={dockCollapsed ? 'Open Copilot (Ctrl+\\)' : 'Collapse Copilot (Ctrl+\\)'}>
          <button
            onClick={() => setDockCollapsed(!dockCollapsed)}
            aria-label="Toggle Copilot dock"
            className="flex h-10 w-10 items-center justify-center rounded-md text-text-secondary transition-colors duration-150 hover:bg-surface hover:text-text-primary"
          >
            {dockCollapsed ? (
              <PanelRightOpen className="h-5 w-5" strokeWidth={2} />
            ) : (
              <PanelRightClose className="h-5 w-5" strokeWidth={2} />
            )}
          </button>
        </Tooltip>
        <SettingsPopover />
      </div>
    </aside>
  )
}
