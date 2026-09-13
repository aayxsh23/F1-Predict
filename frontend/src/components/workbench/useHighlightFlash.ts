import { useEffect, useRef, useState } from 'react'

import { useCopilot } from './CopilotProvider'

/* Rows call this with their own identifying value (a driver code, a corpus
filename) to know when to scroll into view and flash -- triggered by
clicking the matching mention inside a copilot reply. */
export function useHighlightFlash(value: string) {
  const { highlightToken } = useCopilot()
  const ref = useRef<HTMLElement>(null)
  const [flashing, setFlashing] = useState(false)

  useEffect(() => {
    if (!highlightToken || highlightToken.value !== value) return
    ref.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    setFlashing(true)
    const timer = setTimeout(() => setFlashing(false), 1200)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- re-fire on a
    // new nonce even if the same value is clicked twice in a row
  }, [highlightToken?.nonce])

  return { ref, flashing }
}
