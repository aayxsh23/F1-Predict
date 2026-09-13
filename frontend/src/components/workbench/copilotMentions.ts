/* Splits a copilot reply into plain-text and clickable-mention segments --
driver codes (bare 3-letter caps, e.g. "VER") and corpus citations
(bracketed filenames, matching format_retrieved_context()'s "[{source}]"
convention in src/rag/ingest_corpus.py). Driver-code matches are NOT
validated against a known-codes list -- a false positive ("FIA", "DRS") is
a harmless no-op click since nothing reads an unmatched value; upgrade path
if this ever gets noisy is filtering against the current prediction
payload's driver list. */

export interface MentionSegment {
  text: string
  isMention: boolean
}

const MENTION_RE = /\[([^\]]+\.(?:pdf|txt))\]|\b([A-Z]{3})\b/g

export function splitMentions(text: string): MentionSegment[] {
  const segments: MentionSegment[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null

  MENTION_RE.lastIndex = 0
  while ((match = MENTION_RE.exec(text)) !== null) {
    if (match.index > lastIndex) {
      segments.push({ text: text.slice(lastIndex, match.index), isMention: false })
    }
    const value = match[1] ?? match[2]
    segments.push({ text: value, isMention: true })
    lastIndex = match.index + match[0].length
  }
  if (lastIndex < text.length) {
    segments.push({ text: text.slice(lastIndex), isMention: false })
  }
  return segments
}
