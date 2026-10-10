import type {
  BacktestIndexEntry,
  BacktestRacePayload,
  Championship,
  ChatContext,
  ChatEvent,
  ExplainResult,
  LiveRecord,
  ModelCard,
  PredictionPayload,
  RaceSummary,
  RegulationDetail,
  RegulationDocument,
  RegulationSearchHit,
  ShapExplanation,
  StrategyResult,
  Target,
  TimelineSnapshot,
} from './types'

// deployed: the API is served from the same origin under /api (vercel.json)
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? (import.meta.env.DEV ? 'http://localhost:8000' : '/api')

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // Content-Type only with a body: on a GET it isn't CORS-safelisted and would
  // force a preflight round trip on every cross-origin read
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: init?.body ? { 'Content-Type': 'application/json', ...init.headers } : init?.headers,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new ApiError(res.status, body?.detail ?? `${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
}

const q = encodeURIComponent

export const api = {
  health: () => request<{ status: string }>('/health'),
  races: (season: number) => request<RaceSummary[]>(`/races?season=${season}`),
  predictionsLatest: () => request<PredictionPayload>('/predictions/latest'),
  predictionsForRace: (season: number, round: number) => request<PredictionPayload>(`/predictions/${season}/${round}`),
  timeline: (season: number, round: number) => request<TimelineSnapshot[]>(`/predictions/${season}/${round}/timeline`),
  driverExplain: (season: number, round: number, driver: string, target: Target) =>
    request<ShapExplanation>(`/predictions/${season}/${round}/explain?driver=${q(driver)}&target=${target}`),
  backtestRaces: () => request<BacktestIndexEntry[]>('/backtest/races'),
  backtestForRace: (season: number, round: number) => request<BacktestRacePayload>(`/backtest/${season}/${round}`),
  modelCard: () => request<ModelCard>('/model'),
  liveRecord: () => request<LiveRecord>('/live-record'),
  strategy: (season: number, round: number, scLap: number | null) =>
    request<StrategyResult>(`/strategy/${season}/${round}${scLap === null ? '' : `?sc_lap=${scLap}`}`),
  championship: () => request<Championship>('/championship'),
  explain: (season: number, round: number, driver: string, target: Target) =>
    request<ExplainResult>('/explain', { method: 'POST', body: JSON.stringify({ season, round, driver, target }) }),
  regulationsList: () => request<RegulationDocument[]>('/regulations'),
  regulationsGet: (filename: string) => request<RegulationDetail>(`/regulations/${q(filename)}`),
  regulationsSearch: (query: string, k = 6) => request<RegulationSearchHit[]>(`/regulations/search?query=${q(query)}&k=${k}`),
}

/** POST /chat and hand each server-sent event to `onEvent` as it arrives. */
export async function streamChat(
  messages: Array<{ role: 'user' | 'assistant'; content: string }>,
  context: ChatContext,
  onEvent: (e: ChatEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, context }),
    signal,
  })
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => null)
    throw new ApiError(res.status, body?.detail ?? `${res.status} ${res.statusText}`)
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += value
    const events = buffer.split('\n\n')
    buffer = events.pop() ?? ''
    for (const e of events) {
      const data = e.replace(/^data: /, '').trim()
      if (data) onEvent(JSON.parse(data) as ChatEvent)
    }
  }
}
