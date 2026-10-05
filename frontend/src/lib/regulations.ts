import { useEffect, useState } from 'react'

// Shared by the workbench's Rules card and the pit wall's Regulations page.

export function useDebounced<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), ms)
    return () => clearTimeout(id)
  }, [value, ms])
  return debounced
}

export const SPORTING_REGS = 'fia_2026_sporting_regulations.pdf'
export const ARTICLE = /\b[A-C]\d+(?:\.\d+)+\b/g
export const FIELDS = ['Session', 'Fact', 'Infringement', 'Decision', 'Reason'] as const
export type Field = (typeof FIELDS)[number]

/** A steward decision's PDF text is a letterhead, then labelled fields. Pull out the
 *  fields so the reader opens on the ruling, not on "From The Stewards / To The Team Manager". */
export function parseDecision(text: string): { driver: string | null; fields: Partial<Record<Field, string>> } | null {
  // FIA PDFs join words with non-breaking spaces; left in, a whole ruling is one unbreakable line
  const lines = text.replace(/ /g, ' ').split(/\r?\n/)
  const fields: Partial<Record<Field, string>> = {}
  let driver: string | null = null
  let current: Field | null = null
  for (const line of lines) {
    if (/^Competitors are reminded/.test(line)) break
    const d = line.match(/^No \/ Driver\s+(.*)$/)
    if (d) driver = d[1].trim()
    const f = FIELDS.find((name) => line.startsWith(`${name} `))
    if (f) {
      current = f
      fields[f] = line.slice(f.length + 1).trim()
    } else if (current === 'Reason') {
      fields.Reason += `\n${line}`
    } else if (current && current !== 'Session' && line.trim() && !/^(Time|Competitor|No \/ Driver)\b/.test(line)) {
      fields[current] += ` ${line.trim()}`
    }
  }
  return fields.Decision ? { driver, fields } : null
}

/** PDF text breaks lines mid-sentence; a short line ending a sentence ends a paragraph. */
export function paragraphs(text: string): string[] {
  const lines = text.split('\n').map((l) => l.trim()).filter(Boolean)
  const width = Math.max(...lines.map((l) => l.length))
  const out: string[] = []
  let buf = ''
  for (const line of lines) {
    buf = buf ? `${buf} ${line}` : line
    if (/[.:]$/.test(line) && line.length < width * 0.8) {
      out.push(buf)
      buf = ''
    }
  }
  if (buf) out.push(buf)
  return out
}

// "B6.2.1 ..." or "2.1 ..." -- a bare "5" is a table cell or page number, not an article
const CLAUSE_START = /^((?:[A-C]\d+(?:\.\d+)*|\d+(?:\.\d+)+)[a-z]?)\s+(.*)$/

/** A regulation PDF's text as numbered clauses: a line opening with an article
 *  number ("B6.2.1 Other than...") starts one. Page furniture (headers and footers
 *  repeated on every page, stray page tokens) is dropped first; long unnumbered
 *  stretches still break at a sentence end so they stay readable. */
export function clauses(text: string): Array<{ a: string; t: string }> {
  const lines = text.replace(/ /g, ' ').split(/\r?\n/).map((l) => l.trim())
  const seen = new Map<string, number>()
  for (const l of lines) seen.set(l, (seen.get(l) ?? 0) + 1)
  const out: Array<{ a: string; t: string }> = []
  for (const line of lines) {
    if (line.length <= 3 || line.startsWith('©') || (seen.get(line) ?? 0) >= 8) continue
    const m = line.match(CLAUSE_START)
    const last = out[out.length - 1]
    if (m) out.push({ a: m[1], t: m[2] })
    else if (!last || (last.t.length > 400 && /[.:]$/.test(last.t))) out.push({ a: '', t: line })
    else last.t += ` ${line}`
  }
  return out
}
