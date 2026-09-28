import { QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Navigate, Route, Routes, useLocation, useParams } from 'react-router-dom'

import { queryClient } from '@/lib/queryClient'
import type { Tab } from '@/lib/selection'

import { WorkbenchPage } from './routes/WorkbenchPage'

// The per-page routes (race, driver, history, regulations, agent) are now the
// workbench's four dossier tabs. Old links keep working: they redirect into it,
// carrying the season, round, driver and ?doc= they named.
function Legacy({ tab }: { tab: Tab }) {
  const { season, round, driver } = useParams()
  const { search } = useLocation()
  const to = new URLSearchParams(search)
  to.set('tab', tab)
  if (season) to.set('season', season)
  if (round) to.set('round', round)
  if (driver) to.set('driver', driver)
  return <Navigate to={`/?${to}`} replace />
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route index element={<WorkbenchPage />} />
          <Route path="race/:season/:round" element={<Legacy tab="predictor" />} />
          <Route path="race/:season/:round/driver/:driver" element={<Legacy tab="shap" />} />
          <Route path="history" element={<Legacy tab="predictor" />} />
          <Route path="history/:season/:round" element={<Legacy tab="predictor" />} />
          <Route path="regulations" element={<Legacy tab="rules" />} />
          <Route path="agent" element={<Legacy tab="strategist" />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
