import { defineConfig, devices } from '@playwright/test'

// Browser smoke tests: the built app (vite preview) with every /api call
// answered from e2e/fixtures (recorded by scripts/e2e_fixtures.py), so they
// need no backend. Run after `npm run build`: `npm run e2e`.
export default defineConfig({
  testDir: 'e2e',
  timeout: 30_000,
  retries: process.env.CI ? 1 : 0,
  // reduced motion: checks see the settled page, not text mid-fade (which fails contrast)
  use: { baseURL: 'http://localhost:4173', trace: 'retain-on-failure', reducedMotion: 'reduce' },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'] } },
    { name: 'phone', use: { ...devices['Pixel 7'] } },
  ],
  webServer: { command: 'npm run preview -- --port 4173 --strictPort', url: 'http://localhost:4173', reuseExistingServer: !process.env.CI },
})
