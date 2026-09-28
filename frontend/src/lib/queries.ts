import { keepPreviousData, useQuery } from '@tanstack/react-query'

import { api } from './api'
import type { Target } from './types'

export function usePredictionsLatest() {
  return useQuery({ queryKey: ['predictions', 'latest'], queryFn: api.predictionsLatest, placeholderData: keepPreviousData })
}

export function usePredictionsForRace(season: number, round: number, enabled = true) {
  return useQuery({ queryKey: ['predictions', season, round], queryFn: () => api.predictionsForRace(season, round), enabled })
}

export function useTimeline(season: number, round: number, enabled = true) {
  return useQuery({ queryKey: ['timeline', season, round], queryFn: () => api.timeline(season, round), enabled })
}

export function useRaces(season: number) {
  return useQuery({ queryKey: ['races', season], queryFn: () => api.races(season), staleTime: 5 * 60_000 })
}

export function useDriverExplain(season: number, round: number, driver: string, target: Target, enabled = true) {
  return useQuery({
    queryKey: ['explain-shap', season, round, driver, target],
    queryFn: () => api.driverExplain(season, round, driver, target),
    enabled,
  })
}

export function useNaturalExplanation(season: number, round: number, driver: string, target: Target, enabled = true) {
  return useQuery({
    queryKey: ['explain-nl', season, round, driver, target],
    queryFn: () => api.explain(season, round, driver, target),
    staleTime: Infinity, // generated text, not a live value: once per (race, driver, target)
    retry: 1,
    enabled,
  })
}

export function useBacktestRaces() {
  return useQuery({ queryKey: ['backtest', 'races'], queryFn: api.backtestRaces })
}

export function useBacktestForRace(season: number, round: number, enabled = true) {
  return useQuery({ queryKey: ['backtest', season, round], queryFn: () => api.backtestForRace(season, round), enabled })
}

export function useModelCard() {
  return useQuery({ queryKey: ['model'], queryFn: api.modelCard, staleTime: Infinity })
}

export function useStrategy(season: number, round: number, scLap: number | null, enabled = true) {
  return useQuery({
    queryKey: ['strategy', season, round, scLap],
    queryFn: () => api.strategy(season, round, scLap),
    placeholderData: keepPreviousData, // moving the safety-car lap shouldn't blank the panel
    enabled,
  })
}

export function useChampionship() {
  return useQuery({ queryKey: ['championship'], queryFn: api.championship, staleTime: 5 * 60_000, retry: 1 })
}

export function useHealth() {
  return useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 30_000, retry: false })
}

export function useRegulationsList() {
  return useQuery({ queryKey: ['regulations', 'list'], queryFn: api.regulationsList })
}

export function useRegulationDetail(filename: string | null) {
  return useQuery({
    queryKey: ['regulations', 'detail', filename],
    queryFn: () => api.regulationsGet(filename!),
    enabled: filename !== null,
  })
}

export function useRegulationsSearch(query: string) {
  return useQuery({
    queryKey: ['regulations', 'search', query],
    queryFn: () => api.regulationsSearch(query),
    enabled: query.trim().length > 0,
  })
}
