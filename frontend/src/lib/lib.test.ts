import { afterEach, describe, expect, it, vi } from 'vitest'

import { betterWord, lensOf, normaliser, rankBy, signed, weekendPhase } from '../pitwall/data'
import { streamChat } from './api'
import { describeChance, formatLapTime, formatPct } from './format'
import type { RaceDriver } from './raceView'
import { parseDecision } from './regulations'
import type { ChatEvent } from './types'

describe('format', () => {
  it('keeps a real but tiny chance visibly non-zero', () => {
    expect(formatPct(0.142)).toBe('14.2%')
    expect(formatPct(0.0004)).toBe('<0.1%')
    expect(formatPct(null)).toBe('—')
  })

  it('formats lap times as m:ss.sss', () => {
    expect(formatLapTime(92.274)).toBe('1:32.274')
    expect(formatLapTime(null)).toBe('—')
  })

  it('describes chances in words for every range', () => {
    for (const p of [0, 0.03, 0.2, 0.5, 0.9]) expect(describeChance(p).length).toBeGreaterThan(0)
  })

  it('signs numbers with a real minus', () => {
    expect(signed(1.234, 1)).toBe('+1.2')
    expect(signed(-0.5, 2)).toBe('−0.50')
    expect(signed(0, 1)).toBe('0.0')
  })
})

describe('lenses', () => {
  const drivers = [
    { code: 'AAA', modelFinish: 3.2, qualiGapPct: 0.4, modelDelta: 1, raceGapPct: 0.3 },
    { code: 'BBB', modelFinish: 1.4, qualiGapPct: 0.1, modelDelta: -2, raceGapPct: 0.0 },
    { code: 'CCC', modelFinish: null, qualiGapPct: null, modelDelta: null, raceGapPct: null },
  ] as unknown as RaceDriver[]

  it('ranks better-first and leaves missing predictions at the back', () => {
    expect(rankBy(drivers, lensOf('finish_position')).map((d) => d.code)).toEqual(['BBB', 'AAA', 'CCC'])
    expect(rankBy(drivers, lensOf('quali_delta')).map((d) => d.code)[0]).toBe('AAA') // places gained: higher is better
    expect(betterWord('quali_delta')).toBe('higher is better')
  })

  it('normalises the field onto 0..1 with the better end at 0', () => {
    const n = normaliser(drivers, lensOf('finish_position'))
    expect(n(1.4)).toBeCloseTo(0)
    expect(n(3.2)).toBeCloseTo(1)
  })
})

describe('weekend phase', () => {
  it('counts what is known and points at the next session', () => {
    const phase = weekendPhase({ practice: true, qualifying: false, grid: false })
    expect(phase.count).toBe(1)
    expect(phase.nodes[phase.next].label).toMatch(/quali/i)
  })
})

describe('steward decisions', () => {
  it('reads the fields of a decision document', () => {
    const d = parseDecision('No / Driver 63 - George Russell\nSession Race\nFact Unsafe release\nInfringement Breach of Article B1.8.5\nDecision 5 second time penalty\nReason The car was released unsafely.')
    expect(d?.fields.Session).toBe('Race')
    expect(d?.fields.Decision).toMatch(/5 second/)
  })
})

describe('chat stream', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('hands over every server-sent event, even one split across network chunks', async () => {
    const chunks = ['data: {"type":"tool_start","id":"r1","name":"race_forecast","args":{}}\n\n', 'data: {"type":"token","te', 'xt":"VER is favourite"}\n\ndata: {"type":"done"}\n\n']
    const body = new ReadableStream({
      start(c) {
        for (const ch of chunks) c.enqueue(new TextEncoder().encode(ch))
        c.close()
      },
    })
    vi.stubGlobal('fetch', vi.fn(async () => new Response(body, { status: 200 })))
    const events: ChatEvent[] = []
    await streamChat([{ role: 'user', content: 'who wins?' }], {}, (e) => events.push(e), new AbortController().signal)
    expect(events.map((e) => e.type)).toEqual(['tool_start', 'token', 'done'])
    expect(events[1]).toMatchObject({ text: 'VER is favourite' })
  })

  it('turns an API error into a readable message', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ detail: 'Too many questions' }), { status: 429 })))
    await expect(streamChat([{ role: 'user', content: 'x' }], {}, () => {}, new AbortController().signal)).rejects.toThrow('Too many questions')
  })
})
