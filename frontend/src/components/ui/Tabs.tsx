import * as RadixTabs from '@radix-ui/react-tabs'
import type { ReactNode } from 'react'

import { cn } from '@/lib/cn'

export const Tabs = RadixTabs.Root

export function TabsList({ children }: { children: ReactNode }) {
  return (
    <RadixTabs.List className="flex w-full gap-1 overflow-x-auto rounded-md bg-surface-sunken p-1 md:inline-flex md:w-auto">
      {children}
    </RadixTabs.List>
  )
}

export function TabsTrigger({ value, children }: { value: string; children: ReactNode }) {
  return (
    <RadixTabs.Trigger
      value={value}
      className={cn(
        'shrink-0 whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium text-text-secondary transition-colors duration-150',
        'hover:text-text-primary',
        'data-[state=active]:bg-surface data-[state=active]:text-text-primary data-[state=active]:shadow-sm',
      )}
    >
      {children}
    </RadixTabs.Trigger>
  )
}

export const TabsContent = RadixTabs.Content
