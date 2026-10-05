import { MotionConfig, motion } from 'framer-motion'
import { ArrowLeft, Check, Gauge, History, Radio, Scale, Search, Settings, type LucideIcon } from 'lucide-react'
import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type PointerEvent, type ReactNode } from 'react'
import { Link, Outlet, useLocation } from 'react-router-dom'

import { ChatProvider, useChat } from '@/lib/chat'
import { cn } from '@/lib/cn'
import { formatRelativeTime } from '@/lib/format'
import { useHealth } from '@/lib/queries'

import { ChromeContext, type Chrome, type ChromeApi, type Section } from './chrome'
import { driverHref, snapshotLabel } from './data'
import { Palette } from './Palette'
import { PitRadio } from './PitRadio'

const MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform)
const MOD = MAC ? '⌘' : 'Ctrl '
const KEY_K = MAC ? '⌘K' : 'Ctrl K'

const NAV: Array<{ section: Section; to: string; icon: LucideIcon; label: string; short: string }> = [
  { section: 'pred', to: '/', icon: Gauge, label: 'Predictions', short: 'Predictions' },
  { section: 'hist', to: '/history', icon: History, label: 'Race history', short: 'History' },
  { section: 'regs', to: '/regulations', icon: Scale, label: 'Regulations', short: 'Rules' },
]

type Ambient = 'Full' | 'Subtle' | 'Off'
const AMBIENT: Record<Ambient, number> = { Full: 1, Subtle: 0.45, Off: 0 }
const AMBIENT_KEY = 'pw-ambient'

function readAmbient(): Ambient {
  try {
    const v = localStorage.getItem(AMBIENT_KEY)
    return v === 'Subtle' || v === 'Off' ? v : 'Full'
  } catch {
    return 'Full' // storage blocked: the default still renders
  }
}

function useApiStatus() {
  const health = useHealth()
  if (health.data) return { dot: 'var(--color-gain)', color: 'var(--color-ink-3)', label: 'API connected', short: 'Live' }
  if (health.isError) return { dot: 'var(--color-pending)', color: 'var(--color-pending)', label: 'API waking up', short: 'Waking' }
  return { dot: 'var(--color-live)', color: 'var(--color-ink-2)', label: 'Connecting…', short: 'Sync' }
}

const EMPTY: Chrome = { section: null, crumbs: [], chat: {}, badge: 'Pit Wall', quick: [] }

function SettingsMenu({ ambient, setAmbient, onClose }: { ambient: Ambient; setAmbient: (a: Ambient) => void; onClose: () => void }) {
  return (
    <>
      <div onClick={onClose} className="absolute inset-0 z-40" aria-hidden />
      <motion.div
        role="dialog"
        aria-label="Settings"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
        className="absolute bottom-[74px] right-3 z-[41] w-[260px] rounded-[10px] border border-white/[.12] bg-carbon/[.92] p-2 shadow-[0_20px_50px_rgb(0_0_0/.55)] backdrop-blur-[22px] wide:bottom-4 wide:left-20 wide:right-auto"
      >
        <p className="pw-label px-2 pb-1.5 pt-2 text-[10px]">Atmosphere</p>
        {(Object.keys(AMBIENT) as Ambient[]).map((a) => (
          <button
            key={a}
            type="button"
            onClick={() => setAmbient(a)}
            aria-pressed={ambient === a}
            className={cn('flex w-full items-center gap-2.5 rounded-[5px] px-2 py-[9px] text-left text-[13px] font-medium leading-none hover:bg-white/5 hover:text-ink', ambient === a ? 'text-ink' : 'text-ink-2')}
          >
            <span className="flex-1">{a === 'Full' ? 'Full glow' : a === 'Subtle' ? 'Subtle' : 'Off (plain asphalt)'}</span>
            <Check className="size-4 text-signal-ink" style={{ opacity: ambient === a ? 1 : 0 }} aria-hidden />
          </button>
        ))}
        <p className="pw-label mt-1.5 border-t border-white/[.07] px-2 pb-1.5 pt-2 text-[10px]">Shortcuts</p>
        <div className="flex flex-col gap-2 px-2 pb-2 pt-1 text-[12px] font-medium leading-none text-ink-2">
          {[['Command palette', `${MOD}K`], ['Pit Radio', 'Ctrl+\\'], ['Close', 'Esc']].map(([k, v]) => (
            <span key={k} className="flex justify-between">
              {k}
              <span className="text-ink-3">{v}</span>
            </span>
          ))}
        </div>
        <Link to="/workbench" onClick={onClose} className="mt-1 block border-t border-white/[.07] px-2 pb-1.5 pt-2.5 text-[12px] font-semibold text-signal-ink hover:text-ink">
          Workbench: strategy & title odds →
        </Link>
      </motion.div>
    </>
  )
}

/** Night Session chrome around every pit-wall page: the rail (or a bottom tab bar
 *  below 860px), a top bar that states freshness first, the ambient asphalt, Pit
 *  Radio, the ⌘K palette and settings. The chat lives here so it survives navigation. */
export function PitWallLayout() {
  const [chrome, setChrome] = useState<Chrome>(EMPTY)
  const [radio, setRadio] = useState(false)
  const [palette, setPalette] = useState(false)
  const [settings, setSettings] = useState(false)
  const [ambient, setAmbientState] = useState<Ambient>(readAmbient)
  const scroller = useRef<HTMLElement>(null)
  const input = useRef<HTMLTextAreaElement>(null)
  const raf = useRef(0)
  const { pathname } = useLocation()
  const api = useApiStatus()

  const setAmbient = (a: Ambient) => {
    setAmbientState(a)
    try {
      localStorage.setItem(AMBIENT_KEY, a)
    } catch {
      /* the choice just won't persist */
    }
  }

  useEffect(() => {
    scroller.current?.scrollTo({ top: 0 })
  }, [pathname])

  useEffect(() => {
    const narrow = () => window.matchMedia('(max-width: 859px)').matches
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setSettings(false)
        setPalette((p) => !p)
      } else if ((e.metaKey || e.ctrlKey) && e.key === '\\') {
        e.preventDefault()
        setRadio((r) => !r)
      } else if (e.key === 'Escape') {
        setPalette(false)
        setSettings(false)
        if (narrow()) setRadio(false)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => {
    if (radio && !palette) input.current?.focus({ preventScroll: true })
  }, [radio, palette])

  // cursor glow: CSS variables written once a frame, no React re-render
  const glow = (e: PointerEvent<HTMLDivElement>) => {
    const el = e.currentTarget
    const { clientX, clientY } = e
    cancelAnimationFrame(raf.current)
    raf.current = requestAnimationFrame(() => {
      const r = el.getBoundingClientRect()
      el.style.setProperty('--mx', `${(((clientX - r.left) / r.width) * 100).toFixed(1)}%`)
      el.style.setProperty('--my', `${(((clientY - r.top) / r.height) * 100).toFixed(1)}%`)
    })
  }

  const openRadio = useCallback(() => {
    setPalette(false)
    setRadio(true)
  }, [])
  const op = AMBIENT[ambient]
  const { season, round } = chrome.chat

  return (
    <MotionConfig reducedMotion="user">
      <ChatProvider context={chrome.chat}>
        <Api setChrome={setChrome} openRadio={openRadio}>
          <div onPointerMove={glow} className="pw-shell relative flex h-dvh overflow-hidden bg-asphalt font-mono text-ink antialiased">
            <div
              className="pointer-events-none absolute inset-0 bg-[radial-gradient(900px_620px_at_84%_4%,rgb(255_90_71/.16),transparent_60%),radial-gradient(760px_520px_at_0%_104%,rgb(76_201_240/.08),transparent_60%)]"
              style={{ opacity: op }}
              aria-hidden
            />
            <div
              className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgb(255_255_255/.026)_1px,transparent_1px),linear-gradient(90deg,rgb(255_255_255/.026)_1px,transparent_1px)] bg-[size:96px_96px] [mask-image:radial-gradient(1100px_700px_at_60%_25%,#000,transparent_80%)]"
              style={{ opacity: op }}
              aria-hidden
            />
            <div
              className="pointer-events-none absolute inset-0 bg-[radial-gradient(520px_circle_at_var(--mx,70%)_var(--my,30%),rgb(255_255_255/.05),transparent_45%)]"
              style={{ opacity: op }}
              aria-hidden
            />

            <nav aria-label="Pit wall" className="relative z-[2] hidden w-[72px] shrink-0 flex-col items-center gap-1.5 border-r border-white/[.07] bg-white/[.015] py-4 wide:flex">
              <Link to="/" title="Pit Wall" className="mb-[18px] grid size-[38px] place-items-center rounded-md border border-hairline-strong font-headline text-[17px] font-black leading-none text-ink hover:text-ink">
                PW
              </Link>
              {NAV.map((n) => {
                const on = chrome.section === n.section
                return (
                  <Link
                    key={n.to}
                    to={n.to}
                    title={n.label}
                    aria-label={n.label}
                    aria-current={on ? 'page' : undefined}
                    className={cn('relative grid size-11 place-items-center rounded-md transition-colors duration-200 hover:text-ink', on ? 'bg-signal/[.12] text-signal-ink' : 'text-ink-3')}
                  >
                    <span className="absolute -left-3.5 top-2.5 h-6 w-0.5 bg-signal transition-opacity duration-200" style={{ opacity: on ? 1 : 0 }} aria-hidden />
                    <n.icon className="size-5" aria-hidden />
                  </Link>
                )
              })}
              <div className="mt-auto flex flex-col items-center gap-2">
                <button
                  type="button"
                  onClick={() => setPalette((p) => !p)}
                  title={`Command palette (${MOD}K)`}
                  className="rounded-[4px] border border-white/[.12] px-[7px] py-1.5 font-mono text-[10px] font-semibold leading-none text-ink-3 hover:border-white/30 hover:text-ink"
                >
                  {KEY_K}
                </button>
                <button
                  type="button"
                  onClick={() => setRadio((r) => !r)}
                  title="Pit Radio (Ctrl+\)"
                  aria-label="Toggle Pit Radio"
                  aria-pressed={radio}
                  className={cn('grid size-11 place-items-center rounded-md hover:text-ink', radio ? 'text-signal-ink' : 'text-ink-3')}
                >
                  <Radio className="size-5" aria-hidden />
                </button>
                <button type="button" onClick={() => setSettings((s) => !s)} title="Settings" aria-label="Settings" aria-expanded={settings} className="grid size-11 place-items-center rounded-md text-ink-3 hover:text-ink">
                  <Settings className="size-5" aria-hidden />
                </button>
              </div>
            </nav>

            <div className="relative z-[1] flex min-w-0 flex-1 flex-col">
              <header className="pw-label hidden h-14 shrink-0 items-center justify-between gap-5 whitespace-nowrap border-b border-white/[.07] px-8 tracking-[.12em] wide:flex">
                <ol className="flex min-w-0 flex-auto items-center gap-2.5 overflow-hidden">
                  {chrome.crumbs.map((c, i) => {
                    const last = i === chrome.crumbs.length - 1
                    return (
                      <Fragment key={`${c.label}-${i}`}>
                        {i > 0 && (
                          <li className="shrink-0 text-[#3a3d42]" aria-hidden>
                            /
                          </li>
                        )}
                        <li className={cn('min-w-0 truncate', last ? 'shrink text-ink' : 'shrink-0')} aria-current={last ? 'page' : undefined}>
                          {c.to ? (
                            <Link to={c.to} viewTransition className="flex items-center gap-1.5 text-ink-2 hover:text-ink">
                              {c.back && <ArrowLeft className="size-3.5" aria-hidden />}
                              {c.label}
                            </Link>
                          ) : (
                            c.label
                          )}
                        </li>
                      </Fragment>
                    )
                  })}
                </ol>
                <div className="flex shrink-0 items-center gap-[22px]">
                  {chrome.fresh && (
                    <span className="flex items-center gap-2">
                      <span className="size-1.5 rounded-full bg-live motion-safe:animate-[pw-live_2s_ease-in-out_infinite]" aria-hidden />
                      <span className="text-ink">
                        <span className="hidden min-[1040px]:inline">Snapshot </span>
                        {snapshotLabel(chrome.fresh)}
                      </span>
                      <span className="hidden min-[1240px]:inline">· {formatRelativeTime(chrome.fresh)}</span>
                    </span>
                  )}
                  <span className="flex items-center gap-2">
                    <span className="size-1.5 rounded-full" style={{ background: api.dot }} aria-hidden />
                    <span style={{ color: api.color }}>{api.label}</span>
                  </span>
                  <span className="hidden min-[1240px]:inline">XGBoost · 4 targets</span>
                </div>
              </header>

              <header className="flex h-14 shrink-0 items-center justify-between border-b border-white/[.07] bg-asphalt/80 pl-4 pr-3 backdrop-blur-[14px] wide:hidden">
                <Link to="/" className="flex items-center gap-2.5 text-ink hover:text-ink">
                  <span className="grid size-[30px] place-items-center rounded-[5px] border border-hairline-strong font-headline text-[14px] font-black leading-none">PW</span>
                  <span className="font-headline text-[20px] font-extrabold uppercase leading-none tracking-[.02em]">Pit Wall</span>
                </Link>
                <div className="flex items-center gap-1">
                  <span className="mr-1.5 flex items-center gap-1.5 text-[10px] font-medium uppercase leading-none tracking-[.1em]" style={{ color: api.color }}>
                    <span className="size-1.5 rounded-full" style={{ background: api.dot }} aria-hidden />
                    {api.short}
                  </span>
                  <button type="button" onClick={() => setPalette(true)} aria-label="Search" className="grid size-11 place-items-center text-ink-2">
                    <Search className="size-5" aria-hidden />
                  </button>
                </div>
              </header>

              <main ref={scroller} className="relative min-h-0 flex-1 overflow-y-auto overflow-x-hidden [scrollbar-width:thin]">
                <Outlet />
              </main>

              {!radio && <CommandBar quick={chrome.quick[0]} onOpen={openRadio} onPalette={() => setPalette(true)} />}
              {!radio && (
                <button
                  type="button"
                  onClick={openRadio}
                  aria-label="Open Pit Radio"
                  className="absolute bottom-[84px] right-4 z-[6] grid size-14 place-items-center rounded-[14px] bg-signal text-asphalt shadow-[0_12px_30px_rgb(255_90_71/.35)] wide:hidden"
                >
                  <Radio className="size-[22px]" aria-hidden />
                </button>
              )}

              <nav aria-label="Pit wall" className="flex h-16 shrink-0 border-t border-white/[.08] bg-asphalt/[.92] pb-[env(safe-area-inset-bottom)] backdrop-blur-[14px] wide:hidden">
                {NAV.map((n) => {
                  const on = chrome.section === n.section
                  return (
                    <Link
                      key={n.to}
                      to={n.to}
                      aria-current={on ? 'page' : undefined}
                      className={cn('relative flex flex-1 flex-col items-center justify-center gap-[5px] text-[10px] font-semibold uppercase leading-none tracking-[.08em] hover:text-ink', on ? 'text-signal-ink' : 'text-ink-3')}
                    >
                      <span className="absolute inset-x-[30%] -top-px h-0.5 bg-signal" style={{ opacity: on ? 1 : 0 }} aria-hidden />
                      <n.icon className="size-5" aria-hidden />
                      {n.short}
                    </Link>
                  )
                })}
                <button
                  type="button"
                  onClick={() => setSettings((s) => !s)}
                  aria-expanded={settings}
                  className="flex flex-1 flex-col items-center justify-center gap-[5px] text-[10px] font-semibold uppercase leading-none tracking-[.08em] text-ink-3"
                >
                  <Settings className="size-5" aria-hidden />
                  Settings
                </button>
              </nav>
            </div>

            {radio && (
              <PitRadio
                badge={chrome.badge}
                suggestions={chrome.quick}
                codes={chrome.codes ?? []}
                driverHref={(code) => (season && round ? driverHref(season, round, code) : '/')}
                inputRef={input}
                onClose={() => setRadio(false)}
              />
            )}
            {settings && <SettingsMenu ambient={ambient} setAmbient={setAmbient} onClose={() => setSettings(false)} />}
            {palette && <Palette onClose={() => setPalette(false)} openRadio={openRadio} />}
          </div>
        </Api>
      </ChatProvider>
    </MotionConfig>
  )
}

/** Hands pages the chrome setter plus "ask Pit Radio", which needs the chat above it. */
function Api({ setChrome, openRadio, children }: { setChrome: (c: Chrome) => void; openRadio: () => void; children: ReactNode }) {
  const chat = useChat()
  const askRef = useRef(chat.ask)
  useEffect(() => {
    askRef.current = chat.ask
  })
  const api = useMemo<ChromeApi>(
    () => ({
      setChrome,
      openRadio,
      ask: (prompt) => {
        askRef.current(prompt)
        openRadio()
      },
    }),
    [setChrome, openRadio],
  )
  return <ChromeContext.Provider value={api}>{children}</ChromeContext.Provider>
}

/** The closed state of Pit Radio: a floating bar. A chip asks straight away; ⌘K opens the palette. */
function CommandBar({ quick, onOpen, onPalette }: { quick?: string; onOpen: () => void; onPalette: () => void }) {
  const chat = useChat()
  return (
    <div className="absolute inset-x-[clamp(16px,3vw,32px)] bottom-6 z-[5] hidden h-[62px] items-center gap-3 rounded-[10px] border border-white/[.12] bg-carbon/[.74] pl-[18px] pr-2.5 shadow-[0_20px_50px_rgb(0_0_0/.5),inset_0_1px_0_rgb(255_255_255/.06)] backdrop-blur-[22px] wide:flex">
      <Radio className="size-[18px] shrink-0 text-signal-ink" aria-hidden />
      <button type="button" onClick={onOpen} className="min-w-0 flex-1 cursor-text truncate text-left font-editorial text-[21px] italic leading-none text-ink-3">
        Ask the pit wall about this race, a driver, or the standings…
      </button>
      {quick && (
        <button
          type="button"
          onClick={() => {
            chat.ask(quick)
            onOpen()
          }}
          className="hidden max-w-[340px] truncate rounded-md border border-white/10 px-3 py-[9px] text-[12px] font-medium leading-none text-ink-2 transition-colors duration-200 hover:border-white/30 hover:text-ink min-[1100px]:block"
        >
          {quick}
        </button>
      )}
      <button type="button" onClick={onPalette} className="rounded-[5px] border border-white/[.14] px-[9px] py-2 font-mono text-[11px] font-semibold leading-none text-ink-3 hover:text-ink">
        {KEY_K}
      </button>
    </div>
  )
}
