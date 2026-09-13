import { Check, Monitor, Moon, Settings, Sun } from 'lucide-react'

import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import { Tooltip } from '@/components/ui/Tooltip'
import { cn } from '@/lib/cn'
import { useTheme } from '@/lib/theme'

const OPTIONS = [
  { value: 'system', label: 'Matching system', icon: Monitor },
  { value: 'light', label: 'Light', icon: Sun },
  { value: 'dark', label: 'Dark', icon: Moon },
] as const

/* The theme-picker list, split out so BottomNav can host it inside its own
labeled trigger instead of the bare icon-button trigger below. */
export function SettingsBody() {
  const { theme, setTheme } = useTheme()

  return (
    <>
      <p className="px-2 py-1.5 text-xs font-medium uppercase tracking-wide text-text-muted">Theme</p>
      {OPTIONS.map(({ value, label, icon: Icon }) => (
        <button
          key={value}
          onClick={() => setTheme(value)}
          className={cn(
            'flex w-full items-center gap-2.5 rounded-sm px-2 py-1.5 text-left text-sm text-text-secondary transition-colors duration-150 hover:bg-surface-sunken hover:text-text-primary',
            theme === value && 'text-text-primary',
          )}
        >
          <Icon className="h-4 w-4 shrink-0" strokeWidth={2} />
          <span className="flex-1">{label}</span>
          {theme === value && <Check className="h-4 w-4 shrink-0 text-accent-text" strokeWidth={2} />}
        </button>
      ))}
    </>
  )
}

export function SettingsPopover({ hoverClassName = 'hover:bg-surface' }: { hoverClassName?: string }) {
  return (
    <Popover>
      <Tooltip label="Settings">
        <PopoverTrigger asChild>
          <button
            aria-label="Settings"
            className={cn(
              'flex h-10 w-10 items-center justify-center rounded-md text-text-secondary transition-colors duration-150 hover:text-text-primary',
              hoverClassName,
            )}
          >
            <Settings className="h-5 w-5" strokeWidth={2} />
          </button>
        </PopoverTrigger>
      </Tooltip>
      <PopoverContent align="start">
        <SettingsBody />
      </PopoverContent>
    </Popover>
  )
}
