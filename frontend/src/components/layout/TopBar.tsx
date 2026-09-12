import { ThemeToggle } from './ThemeToggle'

export function TopBar() {
  return (
    <header className="sticky top-0 z-20 flex h-14 items-center justify-between border-b border-border-default bg-surface/90 px-4 backdrop-blur-sm md:px-6">
      <p className="text-base font-semibold tracking-tight text-text-primary md:hidden">Pit Wall</p>
      <div className="hidden md:block" />
      <ThemeToggle />
    </header>
  )
}
