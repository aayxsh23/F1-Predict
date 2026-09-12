import { NavLink } from 'react-router-dom'

import { cn } from '@/lib/cn'

import { NAV_ITEMS } from './navItems'

export function Sidebar() {
  return (
    <aside className="hidden w-56 shrink-0 flex-col border-r border-border-default bg-surface-sunken px-4 py-6 md:flex">
      <div className="mb-8 px-2">
        <p className="text-lg font-semibold tracking-tight text-text-primary">Pit Wall</p>
        <p className="mt-0.5 text-xs text-text-muted">F1 race predictions &amp; why</p>
      </div>
      <nav className="flex flex-col gap-1">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium text-text-secondary transition-colors duration-150 hover:bg-surface hover:text-text-primary',
                isActive && 'bg-surface text-text-primary shadow-sm',
              )
            }
          >
            <Icon className="h-4 w-4" strokeWidth={2} />
            {label}
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}
