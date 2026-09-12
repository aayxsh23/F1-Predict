import * as RadixSelect from '@radix-ui/react-select'
import { Check, ChevronDown } from 'lucide-react'

export interface SelectOption {
  value: string
  label: string
}

export function Select({
  value,
  onValueChange,
  options,
  ariaLabel,
}: {
  value: string
  onValueChange: (value: string) => void
  options: SelectOption[]
  ariaLabel: string
}) {
  return (
    <RadixSelect.Root value={value} onValueChange={onValueChange}>
      <RadixSelect.Trigger
        aria-label={ariaLabel}
        className="inline-flex h-9 items-center gap-2 rounded-md border border-border-default bg-surface px-3 text-sm text-text-primary outline-none hover:bg-surface-sunken focus-visible:ring-2 focus-visible:ring-accent"
      >
        <RadixSelect.Value />
        <RadixSelect.Icon>
          <ChevronDown className="h-4 w-4 text-text-muted" />
        </RadixSelect.Icon>
      </RadixSelect.Trigger>
      <RadixSelect.Portal>
        <RadixSelect.Content
          position="popper"
          sideOffset={4}
          className="z-50 overflow-hidden rounded-md border border-border-default bg-surface shadow-md"
        >
          <RadixSelect.Viewport className="p-1">
            {options.map((opt) => (
              <RadixSelect.Item
                key={opt.value}
                value={opt.value}
                className="relative flex h-9 cursor-pointer select-none items-center rounded-sm pl-8 pr-3 text-sm text-text-primary outline-none data-[highlighted]:bg-surface-sunken"
              >
                <RadixSelect.ItemIndicator className="absolute left-2 inline-flex items-center">
                  <Check className="h-4 w-4 text-accent" />
                </RadixSelect.ItemIndicator>
                <RadixSelect.ItemText>{opt.label}</RadixSelect.ItemText>
              </RadixSelect.Item>
            ))}
          </RadixSelect.Viewport>
        </RadixSelect.Content>
      </RadixSelect.Portal>
    </RadixSelect.Root>
  )
}
