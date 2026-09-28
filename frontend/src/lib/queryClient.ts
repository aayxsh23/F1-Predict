import { QueryClient } from '@tanstack/react-query'

import { ApiError } from './api'

// A free-tier backend cold-starts after ~15 min idle; a first request can take
// several seconds to wake it. Retry with backoff instead of failing fast, and
// keep the last-good data on screen (keepPreviousData at the call site) so a
// selection change never blanks the screen while a woken backend catches up.
// A 404 is an answer, not a wake-up: "no forecast for this race" must not sit
// behind seven seconds of retries.
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,
      retry: (failureCount, error) => !(error instanceof ApiError && error.status === 404) && failureCount < 3,
      retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 8000),
    },
  },
})
