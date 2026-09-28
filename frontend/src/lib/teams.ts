// Constructor accent colours: a thin identity line on a HUD and one detail on a
// car, never a livery (PRODUCT.md: no team livery reproduction). Matched loosely
// on the team name FastF1 reports, so a rename or a new entrant falls back to
// neutral silver instead of a wrong colour.
const ACCENTS: Array<[RegExp, string]> = [
  [/mercedes/i, '#27F4D2'],
  [/red bull/i, '#3671C6'],
  [/ferrari/i, '#E8002D'],
  [/mclaren/i, '#FF8000'],
  [/aston/i, '#229971'],
  [/alpine/i, '#FF87BC'],
  [/williams/i, '#64C4FF'],
  [/racing bulls|alphatauri|toro rosso|^rb\b/i, '#6692FF'],
  [/sauber|audi|kick/i, '#52E252'],
  [/haas/i, '#B6BABD'],
  [/cadillac/i, '#D5D9DE'],
]

export const FALLBACK_ACCENT = '#94A3B8'

export function teamAccent(team: string): string {
  return ACCENTS.find(([pattern]) => pattern.test(team))?.[1] ?? FALLBACK_ACCENT
}

// The tyre compound sidewall colours as broadcast graphics use them.
export const COMPOUNDS: Record<string, { label: string; color: string }> = {
  SOFT: { label: 'Soft', color: '#E5384A' },
  MEDIUM: { label: 'Medium', color: '#F5C518' },
  HARD: { label: 'Hard', color: '#F4F7FB' },
  INTERMEDIATE: { label: 'Inter', color: '#22A559' },
  WET: { label: 'Wet', color: '#3B82F6' },
}

export const UNKNOWN_COMPOUND_COLOR = '#4B5563'
