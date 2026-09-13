import type {
  AgentResponse,
  BacktestIndexEntry,
  BacktestRacePayload,
  PredictionPayload,
  RegulationDetail,
  RegulationDocument,
  RegulationSearchHit,
  ShapExplanation,
  Target,
} from './types'

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    throw new ApiError(res.status, body?.detail ?? `${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string }>('/health'),
  predictionsLatest: () => request<PredictionPayload>('/predictions/latest'),
  predictionsForRace: (season: number, round: number) =>
    request<PredictionPayload>(`/predictions/${season}/${round}`),
  driverExplain: (season: number, round: number, driver: string, target: Target) =>
    request<ShapExplanation>(
      `/predictions/${season}/${round}/explain?driver=${encodeURIComponent(driver)}&target=${target}`,
    ),
  backtestRaces: () => request<BacktestIndexEntry[]>('/backtest/races'),
  backtestForRace: (season: number, round: number) =>
    request<BacktestRacePayload>(`/backtest/${season}/${round}`),
  explain: (season: number, round: number, driver: string, target: Target) =>
    request<import('./types').ExplainResult>('/explain', {
      method: 'POST',
      body: JSON.stringify({ season, round, driver, target }),
    }),
  askAgent: (message: string, conversationId?: string) =>
    request<AgentResponse>('/ask-agent', {
      method: 'POST',
      body: JSON.stringify({ message, conversation_id: conversationId }),
    }),
  regulationsList: () => request<RegulationDocument[]>('/regulations'),
  regulationsGet: (filename: string) => request<RegulationDetail>(`/regulations/${encodeURIComponent(filename)}`),
  regulationsSearch: (query: string, k = 5) =>
    request<RegulationSearchHit[]>(`/regulations/search?query=${encodeURIComponent(query)}&k=${k}`),
}
