import { SettingsPopover } from '@/components/workbench/SettingsPopover'

export function MobileTopBar() {
  return (
    <header className="sticky top-0 z-20 flex h-14 items-center justify-between border-b border-border-default bg-surface/90 px-4 backdrop-blur-sm md:hidden">
      <p className="text-base font-semibold tracking-tight text-text-primary">Pit Wall</p>
      <SettingsPopover hoverClassName="hover:bg-surface-sunken" />
    </header>
  )
}
