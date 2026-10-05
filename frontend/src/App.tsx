import { QueryClientProvider } from '@tanstack/react-query'
import { lazy, Suspense } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'

import { queryClient } from '@/lib/queryClient'
import { DriverWhyPage } from '@/pitwall/DriverWhyPage'
import { FullGridPage } from '@/pitwall/FullGridPage'
import { HistoryPage, HistoryRacePage } from '@/pitwall/HistoryPage'
import { PitWallLayout } from '@/pitwall/Layout'
import { NotFoundPage } from '@/pitwall/NotFoundPage'
import { RaceWeekendPage } from '@/pitwall/RaceWeekendPage'
import { RegulationsPage } from '@/pitwall/RegulationsPage'

const WorkbenchPage = lazy(() => import('./routes/WorkbenchPage').then((m) => ({ default: m.WorkbenchPage })))

// The pit wall is the app: predictions, race history and regulations under one
// layout that keeps Pit Radio and the palette alive across pages. The older
// Night Garage workbench stays at /workbench for tyre strategy and title odds.
export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<PitWallLayout />}>
            <Route index element={<RaceWeekendPage />} />
            <Route path="race/:season/:round" element={<FullGridPage />} />
            <Route path="race/:season/:round/driver/:driver" element={<DriverWhyPage />} />
            <Route path="history" element={<HistoryPage />} />
            <Route path="history/:season/:round" element={<HistoryRacePage />} />
            <Route path="regulations" element={<RegulationsPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Route>
          <Route
            path="workbench"
            element={
              <Suspense fallback={null}>
                <WorkbenchPage />
              </Suspense>
            }
          />
          <Route path="agent" element={<Navigate to="/workbench?tab=strategist" replace />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
