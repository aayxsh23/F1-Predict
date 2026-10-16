import { QueryClientProvider } from '@tanstack/react-query'
import { lazy, Suspense } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'

import { queryClient } from '@/lib/queryClient'
import { PitWallLayout } from '@/pitwall/Layout'
import { NotFoundPage } from '@/pitwall/NotFoundPage'
import { RaceWeekendPage } from '@/pitwall/RaceWeekendPage'
import { Skeleton } from '@/pitwall/parts'

// The front page loads with the app; every other page is fetched when first opened.
const named = <T extends string>(load: () => Promise<Record<T, React.ComponentType>>, name: T) =>
  lazy(() => load().then((m) => ({ default: m[name] })))
const FullGridPage = named(() => import('@/pitwall/FullGridPage'), 'FullGridPage')
const DriverWhyPage = named(() => import('@/pitwall/DriverWhyPage'), 'DriverWhyPage')
const StrategyPage = named(() => import('@/pitwall/StrategyPage'), 'StrategyPage')
const SeasonPage = named(() => import('@/pitwall/SeasonPage'), 'SeasonPage')
const HistoryPage = named(() => import('@/pitwall/HistoryPage'), 'HistoryPage')
const HistoryRacePage = named(() => import('@/pitwall/HistoryPage'), 'HistoryRacePage')
const RegulationsPage = named(() => import('@/pitwall/RegulationsPage'), 'RegulationsPage')

const page = (el: React.ReactNode) => <Suspense fallback={<Skeleton />}>{el}</Suspense>

// One app: predictions, strategy, the title race, race history and regulations
// under one layout that keeps Pit Radio and the palette alive across pages.
// The older workbench (strategy and title odds) moved in here on 2026-10-11.
export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<PitWallLayout />}>
            <Route index element={<RaceWeekendPage />} />
            <Route path="race/:season/:round" element={page(<FullGridPage />)} />
            <Route path="race/:season/:round/driver/:driver" element={page(<DriverWhyPage />)} />
            <Route path="race/:season/:round/strategy" element={page(<StrategyPage />)} />
            <Route path="season" element={page(<SeasonPage />)} />
            <Route path="history" element={page(<HistoryPage />)} />
            <Route path="history/:season/:round" element={page(<HistoryRacePage />)} />
            <Route path="regulations" element={page(<RegulationsPage />)} />
            <Route path="*" element={<NotFoundPage />} />
          </Route>
          <Route path="workbench" element={<Navigate to="/" replace />} />
          <Route path="agent" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
