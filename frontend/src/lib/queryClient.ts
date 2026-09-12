import { QueryClient } from '@tanstack/react-query'

// A free-tier backend cold-starts after ~15 min idle; a first request can take
// several seconds to wake it. Retry with backoff instead of failing fast, and
// keep the last-good data on screen (keepPreviousData at the call site) so a
// route change never blanks the screen while a woken backend catches up.
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,
      retry: 3,
      retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 8000),
    },
  },
})
