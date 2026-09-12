import { Monitor, Moon, Sun } from 'lucide-react'

import { Tooltip } from '@/components/ui/Tooltip'
import { useTheme } from '@/lib/theme'

const ORDER = ['system', 'light', 'dark'] as const
const ICON = { system: Monitor, light: Sun, dark: Moon } as const
const LABEL = { system: 'Matching system', light: 'Light', dark: 'Dark' } as const

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  const Icon = ICON[theme]

  function cycle() {
    const next = ORDER[(ORDER.indexOf(theme) + 1) % ORDER.length]
    setTheme(next)
  }

  return (
    <Tooltip label={`Theme: ${LABEL[theme]}. Click to change.`}>
      <button
        onClick={cycle}
        aria-label={`Theme: ${LABEL[theme]}. Click to change.`}
        className="flex h-9 w-9 items-center justify-center rounded-md text-text-secondary transition-colors duration-150 hover:bg-surface-sunken hover:text-text-primary"
      >
        <Icon className="h-[18px] w-[18px]" strokeWidth={2} />
      </button>
    </Tooltip>
  )
}
