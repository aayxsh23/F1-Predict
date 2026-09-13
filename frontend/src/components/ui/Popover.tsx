import * as RadixPopover from '@radix-ui/react-popover'
import type { ReactNode } from 'react'

export const Popover = RadixPopover.Root
export const PopoverTrigger = RadixPopover.Trigger

export function PopoverContent({
  children,
  align = 'center',
}: {
  children: ReactNode
  align?: 'start' | 'center' | 'end'
}) {
  return (
    <RadixPopover.Portal>
      <RadixPopover.Content
        align={align}
        sideOffset={8}
        className="z-50 min-w-40 rounded-md border border-border-default bg-surface p-1 shadow-md"
      >
        {children}
      </RadixPopover.Content>
    </RadixPopover.Portal>
  )
}
