import { QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'

import { TooltipProvider } from '@/components/ui/Tooltip'
import { queryClient } from '@/lib/queryClient'
import { ThemeProvider } from '@/lib/theme'

import { Shell } from './components/layout/Shell'
import { Agent } from './routes/Agent'
import { Dashboard } from './routes/Dashboard'
import { DriverExplain } from './routes/DriverExplain'
import { History } from './routes/History'
import { HistoryDetail } from './routes/HistoryDetail'
import { NotFound } from './routes/NotFound'
import { RaceDetail } from './routes/RaceDetail'

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <TooltipProvider>
          <BrowserRouter>
            <Routes>
              <Route element={<Shell />}>
                <Route index element={<Dashboard />} />
                <Route path="race/:season/:round" element={<RaceDetail />} />
                <Route path="race/:season/:round/driver/:driver" element={<DriverExplain />} />
                <Route path="history" element={<History />} />
                <Route path="history/:season/:round" element={<HistoryDetail />} />
                <Route path="agent" element={<Agent />} />
                <Route path="404" element={<NotFound />} />
                <Route path="*" element={<Navigate to="/404" replace />} />
              </Route>
            </Routes>
          </BrowserRouter>
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  )
}
