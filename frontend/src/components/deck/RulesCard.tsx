import { motion } from 'framer-motion'
import { ArrowLeft, FileText, Search } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { cn } from '@/lib/cn'
import { useRegulationDetail, useRegulationsList, useRegulationsSearch } from '@/lib/queries'
import { useSelection } from '@/lib/selection'
import type { RegulationDocument } from '@/lib/types'

import { rise, SheetMessage, SheetSkeleton, stagger } from './parts'

function useDebounced<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), ms)
    return () => clearTimeout(id)
  }, [value, ms])
  return debounced
}

function Highlighted({ text, query }: { text: string; query: string }) {
  const q = query.trim()
  const parts = useMemo(() => {
    if (q.length < 3) return null
    const split = text.split(new RegExp(`(${q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'))
    return split.length > 800 ? null : split // a term this common in a 270k-character document isn't worth 800 nodes
  }, [text, q])
  if (!parts) return <>{text}</>
  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <mark key={i} className="bg-laser-500/30 text-silver-100">
            {part}
          </mark>
        ) : (
          part
        ),
      )}
    </>
  )
}

function DocRow({ doc, selected, onOpen }: { doc: RegulationDocument; selected: boolean; onOpen: () => void }) {
  const isDecision = doc.doc_type === 'steward_decision'
  return (
    <button
      type="button"
      onClick={onOpen}
      aria-current={selected ? 'true' : undefined}
      className={cn('flex w-full items-start gap-3 border-b border-line px-3 py-2.5 text-left', selected ? 'bg-laser-900/70' : 'hover:bg-obsidian-700/60')}
    >
      <FileText className={cn('mt-0.5 h-4 w-4 shrink-0', selected ? 'text-laser-400' : 'text-silver-400')} aria-hidden />
      <span className="min-w-0">
        <span className="block truncate text-sm text-silver-100">{doc.title}</span>
        <span className="mt-0.5 block text-hud text-silver-400">
          {isDecision && doc.car_number !== undefined ? `Steward decision · car ${doc.car_number}` : isDecision ? 'Steward decision' : 'Regulation'}
        </span>
      </span>
    </button>
  )
}

const SPORTING_REGS = 'fia_2026_sporting_regulations.pdf'
const ARTICLE = /\b[A-C]\d+(?:\.\d+)+\b/g
const FIELDS = ['Session', 'Fact', 'Infringement', 'Decision', 'Reason'] as const
type Field = (typeof FIELDS)[number]

/** A steward decision's PDF text is a letterhead, then labelled fields. Pull out the
 *  fields so the reader opens on the ruling, not on "From The Stewards / To The Team Manager". */
function parseDecision(text: string): { driver: string | null; fields: Partial<Record<Field, string>> } | null {
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
function paragraphs(text: string): string[] {
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

function Ruling({ text, onCite }: { text: string; onCite: (article: string) => void }) {
  const parsed = parseDecision(text)
  if (!parsed) return null
  const { driver, fields } = parsed
  const cited = [...new Set(text.match(ARTICLE) ?? [])]
  return (
    <div className="max-w-[72ch] space-y-4">
      <div className="cut-frame cut-all frame-laser">
        <div className="px-4 py-3">
          <p className="hud-label text-laser-300">Decision</p>
          <p className="mt-1 font-display text-lg font-semibold leading-snug text-silver-100">{fields.Decision}</p>
        </div>
      </div>
      <dl className="grid grid-cols-[112px_minmax(0,1fr)] gap-x-4 gap-y-2 text-sm">
        {driver && (
          <>
            <dt className="hud-label pt-0.5 text-silver-400">Driver</dt>
            <dd className="text-silver-200">{driver}</dd>
          </>
        )}
        {(['Session', 'Fact', 'Infringement'] as const).map((f) =>
          fields[f] ? (
            <div key={f} className="contents">
              <dt className="hud-label pt-0.5 text-silver-400">{f}</dt>
              <dd className="text-silver-200">{fields[f]}</dd>
            </div>
          ) : null,
        )}
      </dl>
      {cited.length > 0 && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="hud-label text-silver-400">Cites</span>
          {cited.map((a) => (
            <button
              key={a}
              type="button"
              onClick={() => onCite(a)}
              title={`Open the Sporting Regulations at ${a}`}
              className="border border-laser-500 px-2 py-1 font-mono text-hud text-laser-300 hover:bg-laser-900"
            >
              Art. {a}
            </button>
          ))}
        </div>
      )}
      {fields.Reason && (
        <div className="space-y-3 border-t border-line pt-4">
          <p className="hud-label text-silver-400">Reason</p>
          {paragraphs(fields.Reason).map((para, i) => (
            <p key={i} className="text-sm leading-relaxed text-silver-300">
              {para}
            </p>
          ))}
        </div>
      )}
    </div>
  )
}

function Reader({ doc, query, highlight, onCite, onBack }: { doc: RegulationDocument | undefined; query: string; highlight: string; onCite: (article: string) => void; onBack: () => void }) {
  const sel = useSelection()
  const filename = sel.doc!
  const detail = useRegulationDetail(filename)
  const [showFull, setShowFull] = useState(false)
  const body = useRef<HTMLDivElement>(null)
  const isDecision = doc?.doc_type === 'steward_decision'
  const term = highlight || query

  // a citation jump lands on its article, not at the top of a 90-page document
  useEffect(() => {
    if (highlight && detail.data) body.current?.querySelector('mark')?.scrollIntoView({ block: 'center' })
  }, [highlight, detail.data])

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center gap-3 border-b border-line px-4 py-2.5 lg:px-6">
        <button type="button" onClick={onBack} className="hud-label flex items-center gap-1.5 text-silver-300 hover:text-silver-100 lg:hidden">
          <ArrowLeft className="h-3.5 w-3.5" aria-hidden /> Back
        </button>
        <h3 className="min-w-0 flex-1 truncate font-display text-base font-semibold tracking-wide text-silver-100">{doc?.title ?? filename}</h3>
        {doc?.grand_prix && <span className="hud-label hidden text-silver-400 sm:block">{doc.grand_prix} GP</span>}
      </div>
      <div ref={body} className="scroll-thin min-h-0 flex-1 overflow-y-auto px-4 py-4 lg:px-6">
        {detail.isPending && <SheetSkeleton rows={6} />}
        {detail.isError && <p className="text-sm text-silver-300">Couldn't load this document right now.</p>}
        {detail.data && isDecision && !showFull && parseDecision(detail.data.text) ? (
          <>
            <Ruling text={detail.data.text} onCite={onCite} />
            <button type="button" onClick={() => setShowFull(true)} className="hud-label mt-5 text-silver-400 hover:text-silver-200">
              Show the full document
            </button>
          </>
        ) : (
          detail.data && (
            <p className="max-w-[72ch] whitespace-pre-wrap text-sm leading-relaxed text-silver-300">
              <Highlighted text={detail.data.text} query={term} />
            </p>
          )
        )}
      </div>
    </div>
  )
}

export function RulesCard() {
  const sel = useSelection()
  const docs = useRegulationsList()
  const [query, setQuery] = useState('')
  const [localHighlight, setHighlight] = useState('')
  const highlight = localHighlight || sel.article || ''
  const debounced = useDebounced(query, 250)
  const hits = useRegulationsSearch(debounced)

  if (docs.isPending) return <SheetSkeleton />
  if (docs.isError || !docs.data) return <SheetMessage title="Regulations aren't reachable right now">The backend may still be waking up. Try again in a moment.</SheetMessage>

  const regulations = docs.data.filter((d) => d.doc_type === 'regulation')
  const decisions = docs.data.filter((d) => d.doc_type === 'steward_decision')
  const open = docs.data.find((d) => d.filename === sel.doc)
  const searching = debounced.trim().length > 0

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="grid h-full lg:grid-cols-[340px_minmax(0,1fr)]">
      <motion.div variants={rise} className={cn('min-h-0 flex-col border-line lg:flex lg:border-r', sel.doc ? 'hidden' : 'flex')}>
        <div className="relative border-b border-line p-3">
          <Search className="pointer-events-none absolute left-6 top-1/2 h-4 w-4 -translate-y-1/2 text-silver-400" aria-hidden />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search regulations and decisions"
            aria-label="Search regulations and steward decisions"
            className="h-10 w-full border border-line-strong bg-obsidian-900 pl-9 pr-3 text-sm text-silver-100 placeholder:text-silver-400 focus:border-laser-500"
          />
        </div>
        <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
          {searching ? (
            hits.isFetching && !hits.data ? (
              <SheetSkeleton rows={4} />
            ) : hits.data && hits.data.length > 0 ? (
              hits.data.map((h, i) => (
                <button key={`${h.filename}-${i}`} type="button" onClick={() => { setHighlight(h.article ?? ''); sel.setDoc(h.filename) }} className="block w-full border-b border-line px-3 py-2.5 text-left hover:bg-obsidian-700/60">
                  <span className="block truncate text-sm text-silver-100">
                    {h.article && <span className="mr-1.5 font-mono text-hud text-laser-300">Art. {h.article}</span>}
                    {docs.data.find((d) => d.filename === h.filename)?.title ?? h.filename}
                  </span>
                  <span className="mt-0.5 line-clamp-2 block text-hud text-silver-400">{h.snippet}</span>
                </button>
              ))
            ) : (
              <p className="px-4 py-4 text-sm text-silver-400">No matches for “{debounced}”.</p>
            )
          ) : (
            <>
              <p className="hud-label px-3 pb-1 pt-3 text-silver-400">Regulations</p>
              {regulations.map((d) => <DocRow key={d.filename} doc={d} selected={d.filename === sel.doc} onOpen={() => { setHighlight(''); sel.setDoc(d.filename) }} />)}
              <p className="hud-label px-3 pb-1 pt-4 text-silver-400">Steward decisions</p>
              {decisions.map((d) => <DocRow key={d.filename} doc={d} selected={d.filename === sel.doc} onOpen={() => { setHighlight(''); sel.setDoc(d.filename) }} />)}
            </>
          )}
        </div>
      </motion.div>

      <motion.div variants={rise} className={cn('min-h-0', sel.doc ? 'block' : 'hidden lg:block')}>
        {sel.doc ? (
          <Reader
            key={sel.doc}
            doc={open}
            query={debounced}
            highlight={highlight}
            onCite={(article) => {
              setHighlight(article)
              sel.setDoc(SPORTING_REGS)
            }}
            onBack={() => sel.setDoc(null)}
          />
        ) : (
          <SheetMessage title="Select a document">The FIA sporting regulations, penalty guidelines and real 2026 steward decisions the explainer draws on.</SheetMessage>
        )}
      </motion.div>
    </motion.div>
  )
}
