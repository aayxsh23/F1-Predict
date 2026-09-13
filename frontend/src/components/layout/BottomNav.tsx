import { Settings } from 'lucide-react'
import { NavLink } from 'react-router-dom'

import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import { cn } from '@/lib/cn'

import { NAV_ITEMS } from './navItems'
import { SettingsBody } from '@/components/workbench/SettingsPopover'

export function BottomNav() {
  return (
    <nav className="fixed inset-x-0 bottom-0 z-30 flex border-t border-border-default bg-surface pb-[env(safe-area-inset-bottom)] md:hidden">
      {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          className={({ isActive }) =>
            cn(
              'flex flex-1 flex-col items-center gap-1 py-2.5 text-xs font-medium text-text-muted transition-colors duration-150',
              isActive && 'text-accent-text',
            )
          }
        >
          <Icon className="h-5 w-5" strokeWidth={2} />
          {label}
        </NavLink>
      ))}
      <Popover>
        <PopoverTrigger asChild>
          <button className="flex flex-1 flex-col items-center gap-1 py-2.5 text-xs font-medium text-text-muted transition-colors duration-150">
            <Settings className="h-5 w-5" strokeWidth={2} />
            Settings
          </button>
        </PopoverTrigger>
        <PopoverContent align="end">
          <SettingsBody />
        </PopoverContent>
      </Popover>
    </nav>
  )
}
