import { keepPreviousData, useMutation, useQuery } from '@tanstack/react-query'

import { api } from './api'
import type { Target } from './types'

export function usePredictionsLatest() {
  return useQuery({
    queryKey: ['predictions', 'latest'],
    queryFn: api.predictionsLatest,
    placeholderData: keepPreviousData,
  })
}

export function usePredictionsForRace(season: number, round: number) {
  return useQuery({
    queryKey: ['predictions', season, round],
    queryFn: () => api.predictionsForRace(season, round),
    placeholderData: keepPreviousData,
  })
}

export function useDriverExplain(season: number, round: number, driver: string, target: Target) {
  return useQuery({
    queryKey: ['explain-shap', season, round, driver, target],
    queryFn: () => api.driverExplain(season, round, driver, target),
    placeholderData: keepPreviousData,
  })
}

export function useNaturalExplanation(season: number, round: number, driver: string, target: Target) {
  return useQuery({
    queryKey: ['explain-nl', season, round, driver, target],
    queryFn: () => api.explain(season, round, driver, target),
    staleTime: Infinity, // an LLM generation, not a live value -- one fetch per (race, driver, target) is enough
    retry: 1,
  })
}

export function useBacktestRaces() {
  return useQuery({ queryKey: ['backtest', 'races'], queryFn: api.backtestRaces })
}

export function useBacktestForRace(season: number, round: number) {
  return useQuery({
    queryKey: ['backtest', season, round],
    queryFn: () => api.backtestForRace(season, round),
  })
}

export function useAskAgent() {
  return useMutation({
    mutationFn: ({ message, conversationId }: { message: string; conversationId?: string }) =>
      api.askAgent(message, conversationId),
  })
}
