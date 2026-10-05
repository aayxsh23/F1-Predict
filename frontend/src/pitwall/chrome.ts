import { createContext, useContext, useLayoutEffect } from 'react'

import type { ChatContext } from '@/lib/types'

export interface Crumb {
  label: string
  to?: string
  back?: boolean // a "← Full grid" style crumb
}

export type Section = 'pred' | 'hist' | 'regs' | null

/** What a page tells the layout around it: where it sits, how fresh it is, and
 *  what Pit Radio should know and suggest while it's on screen. */
export interface Chrome {
  section: Section
  crumbs: Crumb[]
  fresh?: string | null // the forecast snapshot time, prediction pages only
  chat: ChatContext
  badge: string
  quick: string[]
  codes?: string[] // driver codes in replies link to their "why" page
}

export interface ChromeApi {
  setChrome: (c: Chrome) => void
  /** ask Pit Radio something and open it */
  ask: (prompt: string) => void
  openRadio: () => void
}

export const ChromeContext = createContext<ChromeApi | null>(null)

export function useChromeApi(): ChromeApi {
  const api = useContext(ChromeContext)
  if (!api) throw new Error('pit wall pages render inside PitWallLayout')
  return api
}

/** Declare this page's chrome. Keyed on the serialised value, so a page can pass a fresh object every render. */
export function useChrome(chrome: Chrome) {
  const { setChrome } = useChromeApi()
  const key = JSON.stringify(chrome)
  useLayoutEffect(() => setChrome(JSON.parse(key) as Chrome), [key, setChrome])
}
