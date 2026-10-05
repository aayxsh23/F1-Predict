import { ArrowLeft, FileText, Radio, Search } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { useSearchParams } from 'react-router-dom'

import { cn } from '@/lib/cn'
import { formatRelativeTime } from '@/lib/format'
import { useRegulationDetail, useRegulationsList, useRegulationsSearch } from '@/lib/queries'
import { ARTICLE, clauses as toClauses, paragraphs, parseDecision, SPORTING_REGS, useDebounced } from '@/lib/regulations'
import type { RegulationDocument } from '@/lib/types'

import { useChrome, useChromeApi } from './chrome'
import { Offline, PAGE, Skeleton } from './parts'

function Mark({ text, term }: { text: string; term: string }) {
  const i = term ? text.toLowerCase().indexOf(term.toLowerCase()) : -1
  if (i < 0) return <>{text}</>
  return (
    <>
      {text.slice(0, i)}
      <mark className="rounded-sm bg-signal/25 text-ink">{text.slice(i, i + term.length)}</mark>
      {text.slice(i + term.length)}
    </>
  )
}

function docMeta(d: RegulationDocument): string {
  if (d.doc_type === 'regulation') return `updated ${formatRelativeTime(d.modified_at)}`
  return [d.grand_prix && `${d.grand_prix} GP`, d.car_number !== undefined && `car ${d.car_number}`].filter(Boolean).join(' · ') || 'Steward decision'
}

/** A margin column for the article, the text set as editorial copy. */
function Clause({ a, children, hit = false, id, wide = false }: { a: string; children: ReactNode; hit?: boolean; id?: string; wide?: boolean }) {
  return (
    <div id={id} className={cn('grid scroll-mt-24 items-baseline gap-4 rounded', wide ? 'grid-cols-[minmax(0,108px)_minmax(0,1fr)]' : 'grid-cols-[72px_minmax(0,1fr)]', hit && 'bg-signal/10 outline outline-8 outline-signal/10')}>
      <span className="text-[12px] font-semibold leading-none text-signal-ink">{a}</span>
      <div className="font-editorial text-[20px] leading-normal text-ink [text-wrap:pretty]">{children}</div>
    </div>
  )
}

function Reader({ filename, doc, article, onBack, onCite, onSummarize }: { filename: string; doc?: RegulationDocument; article: string | null; onBack: () => void; onCite: (a: string) => void; onSummarize: () => void }) {
  const detail = useRegulationDetail(filename)
  const body = useRef<HTMLDivElement>(null)
  const isDecision = doc?.doc_type === 'steward_decision'
  const ruling = detail.data && isDecision ? parseDecision(detail.data.text) : null
  const clauses = detail.data && !ruling ? toClauses(detail.data.text) : []
  // the last match: a table of contents lists the article before the body does
  const target = article ? clauses.findLastIndex((c) => c.a === article) : -1

  // a citation lands on its article, not at the top of a 90-page document
  useEffect(() => {
    if (target >= 0) body.current?.querySelector('#pw-article')?.scrollIntoView({ block: 'center' })
  }, [target, detail.data])

  return (
    <article ref={body} className="pw-glass flex w-full flex-col gap-5 rounded-[10px] p-[clamp(18px,3vw,32px)]">
      <div className="flex flex-wrap items-center justify-between gap-2.5 border-b border-white/[.08] pb-4">
        <button type="button" onClick={onBack} className="flex items-center gap-1.5 text-[12px] font-semibold leading-none text-ink-2 hover:text-ink wide:hidden">
          <ArrowLeft className="size-3.5" aria-hidden /> Back
        </button>
        <span className="min-w-0 flex-1 truncate text-[11px] leading-none text-ink-3">{filename}</span>
        <button type="button" onClick={onSummarize} className="flex h-9 items-center gap-2 rounded-md border border-white/[.14] px-3 text-[11px] font-semibold leading-none text-ink hover:border-white/35">
          <Radio className="size-3.5 text-signal-ink" aria-hidden />
          Summarize
        </button>
      </div>
      <div className="flex flex-col gap-2.5">
        <span className="text-[10px] font-medium uppercase leading-none tracking-[.14em]" style={{ color: isDecision ? 'var(--color-pending)' : 'var(--color-live)' }}>
          {isDecision ? 'Steward decision' : 'Regulation'}
          {doc && ` · updated ${formatRelativeTime(doc.modified_at)}`}
        </span>
        <h2 className="font-headline text-[clamp(36px,4vw,56px)] font-extrabold uppercase leading-[.92] [text-wrap:balance]">{doc?.title ?? filename}</h2>
      </div>

      {detail.isPending && <p className="font-editorial text-[19px] italic text-ink-3 motion-safe:animate-[pw-live_1.6s_ease-in-out_infinite]">Opening the document…</p>}
      {detail.isError && <p className="font-editorial text-[19px] italic text-ink-2">Couldn't load this document right now.</p>}
      {ruling && (
        <>
          {ruling.driver && (
            <Clause a="Driver" wide>
              {ruling.driver}
            </Clause>
          )}
          {(['Session', 'Fact', 'Infringement', 'Decision'] as const).map(
            (f) =>
              ruling.fields[f] && (
                <Clause key={f} a={f} wide>
                  {ruling.fields[f]}
                </Clause>
              ),
          )}
          {ruling.fields.Reason && (
            <Clause a="Reason" wide>
              {paragraphs(ruling.fields.Reason).map((p, i) => (
                <p key={i} className={i ? 'mt-3' : undefined}>
                  {p}
                </p>
              ))}
            </Clause>
          )}
          {(() => {
            const cited = [...new Set(detail.data!.text.match(ARTICLE) ?? [])]
            return cited.length > 0 && (
              <Clause a="Cites" wide>
                <span className="flex flex-wrap gap-2">
                  {cited.map((a) => (
                    <button key={a} type="button" onClick={() => onCite(a)} className="rounded-md border border-signal/50 px-2.5 py-1 font-mono text-[12px] text-signal-ink hover:bg-signal/[.12]">
                      Art. {a}
                    </button>
                  ))}
                </span>
              </Clause>
            )
          })()}
        </>
      )}
      {clauses.map((c, i) => (
        <Clause key={i} a={c.a} hit={i === target} id={i === target ? 'pw-article' : undefined}>
          {c.t}
        </Clause>
      ))}
    </article>
  )
}

/** S5: the FIA texts the explainer draws on, searchable and readable. */
export function RegulationsPage() {
  const docs = useRegulationsList()
  const { ask } = useChromeApi()
  const [params, setParams] = useSearchParams()
  const [q, setQ] = useState('')
  const term = useDebounced(q.trim(), 250)
  const hits = useRegulationsSearch(term)
  const filename = params.get('doc')
  const article = params.get('article')
  const open = docs.data?.find((d) => d.filename === filename)
  const shortTitle = open?.title.split(' — ')[0]

  useChrome({
    section: 'regs',
    crumbs: [{ label: 'Regulations' }, ...(shortTitle ? [{ label: shortTitle }] : [])],
    chat: {},
    badge: shortTitle ? `Doc: ${shortTitle}` : 'Regulations',
    quick: open ? [`Summarize the key points of ${open.title}`] : ['What is the penalty for an unsafe release?'],
  })

  const show = (doc: string | null, art?: string | null) => {
    const next = new URLSearchParams()
    if (doc) next.set('doc', doc)
    if (art) next.set('article', art)
    setParams(next)
  }

  if (docs.isPending) return <Skeleton label="Loading the regulations…" />
  if (docs.isError || !docs.data) return <Offline />

  const groups = [
    { label: 'Regulations', docs: docs.data.filter((d) => d.doc_type === 'regulation') },
    { label: 'Steward decisions', docs: docs.data.filter((d) => d.doc_type !== 'regulation') },
  ]
  const row = (selected: boolean) => cn('relative flex w-full gap-3 border-b border-white/[.06] px-4 py-3 text-left transition-colors duration-200 hover:bg-white/[.04]', selected && 'bg-white/5')

  return (
    <div className={cn(PAGE, 'gap-[22px] pt-7')}>
      <div className="flex max-w-[820px] flex-col gap-3.5">
        <h1 className="-ml-1 font-headline text-[clamp(72px,10vw,150px)] font-black uppercase leading-[.8]">Regulations</h1>
        <p className="font-editorial text-[clamp(20px,2.2vw,26px)] italic leading-[1.3] text-ink-2 [text-wrap:pretty]">
          The FIA sporting regulations, penalty guidelines, and this season's steward decisions the explainer draws on.
        </p>
      </div>

      <label className="relative block">
        <span className="sr-only">Search regulation and decision text</span>
        <Search className="pointer-events-none absolute left-4 top-1/2 size-[18px] -translate-y-1/2 text-ink-3" aria-hidden />
        <input
          type="search"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search regulation and decision text…"
          className="h-[52px] w-full rounded-lg border border-white/[.12] bg-black/25 pl-[46px] pr-4 font-editorial text-[20px] italic leading-none text-ink outline-none placeholder:text-ink-3 focus:border-signal/60"
        />
      </label>

      <div className="flex flex-wrap items-start gap-[22px]">
        <div className={cn('min-w-0 flex-[1_1_300px] flex-col gap-5 wide:flex wide:max-w-[380px]', filename ? 'hidden' : 'flex')}>
          {term ? (
            <div className="flex flex-col overflow-hidden rounded-[10px] border border-white/[.09]">
              <p className="pw-label border-b border-white/[.07] px-4 py-3 text-[10px]">
                {hits.isFetching && !hits.data ? 'Searching…' : hits.data?.length ? `${hits.data.length} matches` : 'No matches'}
              </p>
              {(hits.data ?? []).map((h, i) => {
                const title = docs.data.find((d) => d.filename === h.filename)?.title ?? h.filename
                return (
                  <button key={`${h.filename}-${i}`} type="button" onClick={() => show(h.filename, h.article)} className={cn(row(h.filename === filename), 'flex-col gap-1.5')}>
                    <span className="flex flex-wrap items-center gap-2 text-[12px] font-semibold leading-snug">
                      {h.article && <span className="text-signal-ink">Art. {h.article}</span>}
                      {title}
                      <span className="rounded-[3px] border border-white/[.12] px-[5px] py-[3px] text-[9px] font-medium uppercase leading-none tracking-[.12em] text-ink-3">
                        {h.doc_type === 'regulation' ? 'Regulation' : 'Decision'}
                      </span>
                    </span>
                    <span className="line-clamp-3 text-[12px] leading-normal text-ink-3">
                      <Mark text={h.snippet} term={term} />
                    </span>
                  </button>
                )
              })}
            </div>
          ) : (
            groups.map((g) => (
              <div key={g.label} className="flex flex-col gap-2.5">
                <h2 className="pw-label">{g.label}</h2>
                <ul className="flex flex-col overflow-hidden rounded-[10px] border border-white/[.09] bg-white/[.015]">
                  {g.docs.map((d) => {
                    const selected = d.filename === filename
                    return (
                      <li key={d.filename}>
                        <button type="button" onClick={() => show(d.filename)} aria-current={selected ? 'true' : undefined} className={cn(row(selected), 'items-start')}>
                          <span className="absolute inset-y-2.5 left-0 w-0.5 bg-signal" style={{ opacity: selected ? 1 : 0 }} aria-hidden />
                          <FileText className="mt-0.5 size-[15px] shrink-0 text-ink-3" aria-hidden />
                          <span className="flex min-w-0 flex-col gap-[5px]">
                            <span className="text-[12px] font-semibold leading-[1.35]">{d.title}</span>
                            <span className="text-[11px] leading-none text-ink-3">{docMeta(d)}</span>
                          </span>
                        </button>
                      </li>
                    )
                  })}
                </ul>
              </div>
            ))
          )}
        </div>

        <div className={cn('min-w-0 flex-[2_1_420px] wide:flex', filename ? 'flex' : 'hidden')}>
          {filename ? (
            <Reader
              key={`${filename}-${article}`}
              filename={filename}
              doc={open}
              article={article}
              onBack={() => show(null)}
              onCite={(a) => show(SPORTING_REGS, a)}
              onSummarize={() => ask(`Summarize the key points of ${open?.title ?? filename}`)}
            />
          ) : (
            <div className="flex min-h-[340px] w-full flex-col items-center justify-center gap-2.5 rounded-[10px] border border-dashed border-white/[.14] p-6 text-center">
              <span className="font-headline text-[40px] font-extrabold uppercase leading-none text-ink-3">Pick a document</span>
              <span className="font-editorial text-[19px] italic leading-[1.3] text-ink-3">Select one on the left to read it here.</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
