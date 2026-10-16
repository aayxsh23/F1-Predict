// Fails the build when the JavaScript and CSS a first visit downloads (the
// files dist/index.html references) grow past the budget, gzipped. Routes and
// the chat's markdown renderer load later and aren't counted.
// Run after `npm run build`: `npm run size`.
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { gzipSync } from 'node:zlib'

const BUDGET_KB = { js: 150, css: 20 }
const dist = fileURLToPath(new URL('../dist/', import.meta.url))
const html = readFileSync(join(dist, 'index.html'), 'utf8')
const files = [...html.matchAll(/(?:src|href)="\/(assets\/[^"]+\.(js|css))"/g)].map((m) => ({ path: m[1], kind: m[2] }))

let failed = false
for (const kind of ['js', 'css']) {
  const kb = files
    .filter((f) => f.kind === kind)
    .reduce((sum, f) => sum + gzipSync(readFileSync(join(dist, f.path))).length / 1024, 0)
  const ok = kb <= BUDGET_KB[kind]
  failed ||= !ok
  console.log(`${ok ? 'ok  ' : 'OVER'} first-load ${kind}: ${kb.toFixed(1)} KB gzipped (budget ${BUDGET_KB[kind]} KB)`)
}
if (failed) process.exit(1)
