import { useCallback, useState } from 'react'

export type FxMode = '3d' | 'lite'

const STORAGE_KEY = 'f1-predict-fx'

function webglAvailable(): boolean {
  try {
    const canvas = document.createElement('canvas')
    return !!(canvas.getContext('webgl2') ?? canvas.getContext('webgl'))
  } catch {
    return false
  }
}

function saved(): FxMode | null {
  try {
    const value = localStorage.getItem(STORAGE_KEY)
    return value === '3d' || value === 'lite' ? value : null
  } catch {
    return null // storage blocked (private window, embedded preview)
  }
}

/** First-visit choice: an explicit ?fx= or a saved preference wins; otherwise
 *  the 3D stage only where it will run well -- WebGL present, no reduced-motion
 *  or data-saver request, not a device reporting <=2 GB RAM or <=2 cores, and not a
 *  phone-width screen (three HUDs need the room there, and so does the battery). */
function detect(): FxMode {
  const fromUrl = new URLSearchParams(window.location.search).get('fx')
  if (fromUrl === '3d' || fromUrl === 'lite') return fromUrl
  const stored = saved()
  if (stored) return stored

  const nav = navigator as Navigator & { deviceMemory?: number; connection?: { saveData?: boolean } }
  const constrained =
    window.matchMedia('(prefers-reduced-motion: reduce)').matches ||
    nav.connection?.saveData === true ||
    (nav.deviceMemory ?? 8) <= 2 ||
    (navigator.hardwareConcurrency ?? 8) <= 2 ||
    window.matchMedia('(max-width: 639px)').matches
  return !constrained && webglAvailable() ? '3d' : 'lite'
}

export function useFxMode(): [FxMode, (mode: FxMode) => void] {
  const [mode, setMode] = useState<FxMode>(detect)
  const update = useCallback((next: FxMode) => {
    setMode(next)
    try {
      localStorage.setItem(STORAGE_KEY, next)
    } catch {
      /* preference just won't persist */
    }
  }, [])
  return [mode, update]
}
