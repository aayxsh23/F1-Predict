import AxeBuilder from '@axe-core/playwright'
import { expect, test, type Page } from '@playwright/test'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), 'fixtures')
const meta = JSON.parse(readFileSync(join(FIXTURES, '_meta.json'), 'utf8')) as {
  season: number
  round: number
  driver: string
  past: { season: number; round: number }
}

/** '/predictions/2026/17?x=1' -> 'predictions_2026_17_x_1.json', as scripts/e2e_fixtures.py names them. */
const fixtureName = (path: string) => `${path.replace(/^\/+/, '').replace(/[^A-Za-z0-9]+/g, '_').replace(/^_+|_+$/g, '')}.json`

async function serveFixtures(page: Page) {
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url())
    const file = join(FIXTURES, fixtureName(url.pathname.replace(/^\/api/, '') + url.search))
    if (existsSync(file)) await route.fulfill({ status: 200, contentType: 'application/json', body: readFileSync(file, 'utf8') })
    else await route.fulfill({ status: 404, contentType: 'application/json', body: JSON.stringify({ detail: 'no fixture' }) })
  })
}

/** Serious or critical accessibility problems on the page (axe-core, WCAG 2 A/AA rules). */
async function axeProblems(page: Page) {
  const r = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa']).analyze()
  return r.violations.filter((v) => v.impact === 'serious' || v.impact === 'critical').map((v) => `${v.id}: ${v.help}`)
}

test.beforeEach(async ({ page }) => {
  await serveFixtures(page)
})

test('the front page shows this weekend\'s forecast', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByText(meta.driver).first()).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Weekend confidence' })).toBeVisible()
  expect(await axeProblems(page)).toEqual([])
})

test('the full grid links to the race strategy, which takes a safety car', async ({ page }) => {
  await page.goto(`/race/${meta.season}/${meta.round}`)
  await page.getByRole('link', { name: /tyre strategy/i }).click()
  await expect(page).toHaveURL(new RegExp(`/race/${meta.season}/${meta.round}/strategy`))
  await expect(page.getByRole('list', { name: 'Fastest plans' }).getByRole('listitem').first()).toBeVisible()
  await page.getByRole('checkbox', { name: /safety car on lap/i }).check() // lap 15 is the default
  await expect(page.getByText(/best plan becomes/i)).toBeVisible()
  expect(await axeProblems(page)).toEqual([])
})

test('the title race lists drivers and constructors', async ({ page }) => {
  await page.goto('/season')
  await expect(page.getByRole('heading', { name: 'Title race' })).toBeVisible()
  await expect(page.getByRole('table')).toContainText('Title chance')
  await page.getByRole('button', { name: 'Constructors' }).click()
  await expect(page.getByRole('table')).toContainText('Team')
  await page.getByRole('button', { name: 'Driver ratings' }).click()
  await expect(page.getByRole('table')).toContainText('vs teammate')
  expect(await axeProblems(page)).toEqual([])
})

test('a driver page explains the prediction', async ({ page }) => {
  await page.goto(`/race/${meta.season}/${meta.round}/driver/${meta.driver}`)
  await expect(page.getByText(meta.driver).filter({ visible: true }).first()).toBeVisible()
  expect(await axeProblems(page)).toEqual([])
})

test('race history opens a past race', async ({ page }) => {
  await page.goto('/history')
  await expect(page.getByRole('heading', { name: 'Race history' })).toBeVisible()
  await page.goto(`/history/${meta.past.season}/${meta.past.round}`)
  await expect(page.locator('main, body').first()).not.toContainText('isn\'t reachable')
})

test('the old workbench address lands on the app', async ({ page }) => {
  await page.goto('/workbench')
  await expect(page).toHaveURL(/\/$/)
})
