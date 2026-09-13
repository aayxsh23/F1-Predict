import { ArrowLeft, FileText, Search } from 'lucide-react'
import { useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { AskCopilotButton } from '@/components/workbench/AskCopilotButton'
import { useHighlightFlash } from '@/components/workbench/useHighlightFlash'
import { Badge } from '@/components/ui/Badge'
import { Card } from '@/components/ui/Card'
import { Skeleton } from '@/components/ui/Skeleton'
import { formatRelativeTime } from '@/lib/format'
import { useRegulationDetail, useRegulationsList, useRegulationsSearch } from '@/lib/queries'
import type { RegulationDocument } from '@/lib/types'

function DocRow({ doc }: { doc: RegulationDocument }) {
  const [params, setParams] = useSearchParams()
  const { ref, flashing } = useHighlightFlash(doc.filename)
  const isSelected = params.get('doc') === doc.filename

  return (
    <button
      ref={ref as React.RefObject<HTMLButtonElement>}
      onClick={() => setParams({ doc: doc.filename })}
      className={`flex w-full items-start gap-3 border-b border-border-default px-4 py-3 text-left text-sm transition-colors duration-150 last:border-0 hover:bg-surface-sunken ${
        isSelected ? 'bg-surface-sunken' : ''
      } ${flashing ? 'animate-[flash-highlight_1.2s_ease-out]' : ''}`}
    >
      <FileText className="mt-0.5 h-4 w-4 shrink-0 text-text-muted" strokeWidth={2} />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-text-primary">{doc.title}</span>
        <span className="mt-0.5 block text-xs text-text-muted">{formatRelativeTime(doc.modified_at)}</span>
      </span>
    </button>
  )
}

function DocList({ docs }: { docs: RegulationDocument[] }) {
  const regulations = docs.filter((d) => d.doc_type === 'regulation')
  const decisions = docs.filter((d) => d.doc_type === 'steward_decision')

  return (
    <div className="space-y-6">
      <section>
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-text-muted">Regulations</h2>
        <Card className="p-0">
          {regulations.map((doc) => (
            <DocRow key={doc.filename} doc={doc} />
          ))}
        </Card>
      </section>
      <section>
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-text-muted">Steward decisions</h2>
        <Card className="p-0">
          {decisions.map((doc) => (
            <DocRow key={doc.filename} doc={doc} />
          ))}
        </Card>
      </section>
    </div>
  )
}

function DocReader({ filename, onBack }: { filename: string; onBack: () => void }) {
  const { data, isPending, isError } = useRegulationDetail(filename)

  return (
    <Card>
      <div className="flex items-center justify-between gap-2 border-b border-border-default pb-3">
        <button
          onClick={onBack}
          className="flex items-center gap-1.5 text-sm text-text-secondary hover:text-text-primary md:hidden"
        >
          <ArrowLeft className="h-4 w-4" strokeWidth={2} /> Back
        </button>
        <p className="min-w-0 flex-1 truncate text-sm font-semibold text-text-primary">{filename}</p>
        <AskCopilotButton prompt={`Summarize the key points of ${filename}`} />
      </div>
      {isPending && <Skeleton className="mt-4 h-80 w-full" />}
      {isError && <p className="mt-4 text-sm text-text-secondary">Couldn't load this document right now.</p>}
      {data && (
        <pre className="mt-4 max-h-[70vh] overflow-y-auto whitespace-pre-wrap font-sans text-sm leading-relaxed text-text-secondary">
          {data.text}
        </pre>
      )}
    </Card>
  )
}

export function Regulations() {
  const { data: docs, isPending, isError } = useRegulationsList()
  const [params, setParams] = useSearchParams()
  const [query, setQuery] = useState('')
  const { data: searchHits } = useRegulationsSearch(query)
  const selected = params.get('doc')

  if (isPending) {
    return (
      <div className="mx-auto max-w-5xl px-4 py-8 md:px-6 md:py-12">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="mt-6 h-96 w-full" />
      </div>
    )
  }

  if (isError || !docs) {
    return (
      <div className="mx-auto max-w-5xl px-4 py-16 text-center md:px-6">
        <p className="text-lg font-semibold text-text-primary">Regulations aren't reachable right now</p>
        <p className="mt-2 text-sm text-text-secondary">The backend may still be waking up. Try again shortly.</p>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8 md:px-6 md:py-12">
      <h1 className="text-2xl font-semibold tracking-tight text-text-primary md:text-3xl">Regulations</h1>
      <p className="mt-2 text-sm text-text-secondary">
        The FIA sporting regulations, penalty guidelines, and real 2026 steward decisions this model's explainer
        draws on.
      </p>

      <div className="relative mt-6">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-muted" strokeWidth={2} />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search regulation and decision text…"
          className="h-10 w-full rounded-md border border-border-default bg-surface pl-9 pr-3 text-sm text-text-primary outline-none placeholder:text-text-muted focus-visible:ring-2 focus-visible:ring-accent"
        />
      </div>

      <div className="mt-6 grid gap-6 md:grid-cols-[320px_1fr]">
        <div className={selected ? 'hidden md:block' : ''}>
          {query.trim() && searchHits ? (
            <Card className="p-0">
              {searchHits.length === 0 && <p className="px-4 py-3 text-sm text-text-secondary">No matches.</p>}
              {searchHits.map((hit, i) => (
                <button
                  key={`${hit.filename}-${i}`}
                  onClick={() => setParams({ doc: hit.filename })}
                  className="block w-full border-b border-border-default px-4 py-3 text-left text-sm last:border-0 hover:bg-surface-sunken"
                >
                  <span className="flex items-center gap-2 text-text-primary">
                    {hit.filename}
                    <Badge tone="neutral">{hit.doc_type === 'regulation' ? 'Regulation' : 'Decision'}</Badge>
                  </span>
                  <span className="mt-1 line-clamp-2 block text-xs text-text-muted">{hit.snippet}</span>
                </button>
              ))}
            </Card>
          ) : (
            <DocList docs={docs} />
          )}
        </div>

        <div className={selected ? '' : 'hidden md:block'}>
          {selected ? (
            <DocReader filename={selected} onBack={() => setParams({})} />
          ) : (
            <Card className="flex h-full min-h-64 items-center justify-center text-sm text-text-muted">
              Select a document to read it.
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
