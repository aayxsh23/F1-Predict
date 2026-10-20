---
name: F1 Predict
description: The pit wall at night. A timing tower set like a race-day magazine read under floodlights.
colors:
  asphalt: "#0a0b0d"
  carbon: "#121418"
  ink: "#f2f1ed"
  ink-2: "#b3b5b9"
  ink-3: "#82858b"
  signal: "#ff5a47"
  signal-ink: "#ff7a6a"
  gain: "#3ddc97"
  pending: "#f2b544"
  live: "#4cc9f0"
  hairline: "rgb(255 255 255 / 0.09)"
  hairline-strong: "rgb(255 255 255 / 0.18)"
  tyre-soft: "#e5384a"
  tyre-medium: "#f5c518"
  tyre-hard: "#f4f7fb"
typography:
  headline: "Big Shoulders Display 700-900"
  editorial: "Instrument Serif 400, italic"
  data: "JetBrains Mono 500-600, tabular-nums"
  body: "Barlow 400-600"
---

# Design System: F1 Predict, "Night Session"

The whole app is the pit wall (`frontend/src/pitwall/`). The earlier "Night Garage" workbench (3D podium, folder deck, laser-teal palette) was retired on 2026-10-11 when strategy and the title race moved into the pit wall; its tokens were removed from `styles/tokens.css` on 2026-10-19.

**Pages.** This weekend (`RaceWeekendPage`), the full grid (`FullGridPage`), a driver's "why" (`DriverWhyPage`), strategy (`StrategyPage`), the title race and driver ratings (`SeasonPage`), history (`HistoryPage`), regulations (`RegulationsPage`), the command palette (`Palette`) and Pit Radio (`PitRadio`).

**Creative North Star: "the timing tower, set like a race-day magazine read under floodlights."** Source: the Claude Design project *Pit Wall - Night Session* (concept, three screen blueprints S1-S3, components C1-C3, implementation specs) and its follow-up *Pit Wall - Night Session App* (the whole app: history, regulations, palette, settings, phone layout). Code: `frontend/src/pitwall/`.

**App shell.** One layout route (`Layout.tsx`) around every page: a 72px rail (Predictions, History, Regulations; ⌘K, Pit Radio, Settings at the foot) and a 56px top bar with crumbs on the left and freshness + API state on the right. Below 860px (the `wide` breakpoint) the rail becomes a 64px bottom tab bar, the top bar shows "PW Pit Wall", the API state and a search button, and Pit Radio opens full screen from a signal-filled FAB. Pages enter with an 8px rise; loading is a shimmering skeleton shaped like the page, never a spinner; an unreachable API gets its own state with a retry. Settings offer the ambient glow (Full / Subtle / Off).

**Habits worth keeping:** mono numerals that never jitter (tabular-nums on every column), one accent spent only on P1, action and links, fan vocabulary (never "SHAP"), freshness stated on every view, no hover shadow-lift.

**Tokens** (`styles/tokens.css`): asphalt #0A0B0D (canvas), carbon #121418 (raised), ink #F2F1ED (17.4:1), ink-2 #B3B5B9 (9.5:1), ink-3 #82858B (5.2:1, the floor for any text), signal #FF5A47 (fills; text uses signal-ink #FF7A6A), gain #3DDC97 (better-for-driver, known), pending #F2B544 (next session, waking API), live #4CC9F0 (freshness), hairline white 9% / 18%. Ambient mesh only: ember 16%, coolant 7%. Glass panels (`.pw-glass`) are white 5.5%→1.8% with an 18px blur and a 1px inset highlight.

**Type ramp** (deliberately literal pixel sizes, from the design spec):
- Mega: Big Shoulders Display 900, up to 236px (race) / 168px (driver code), leading .78, sized to its container (`Mega` in `pitwall/parts.tsx`). One identity per screen.
- Display: Big Shoulders 800, 50-104px, driver codes on the podium tiles.
- Readout: Big Shoulders 700, 26-44px, single values (circuit facts, input tiles, phase title). Never columns.
- Editorial: Instrument Serif 400 and italic, 15-23px, the "why", phase copy and Pit Radio replies.
- Data: JetBrains Mono 500-600, 12-18px, tabular-nums.
- Label: JetBrains Mono 500, 10-11px, caps, +0.12-0.14em (`.pw-label`).

**Components.** C1 Weekend Confidence Rail: four nodes from `known_sessions` only (known = gain fill, next = pulsing amber ring and dashed segment, pending = hollow). C2 Target Lens + Prediction Field: a `role="tablist"` with a spring pill (`layoutId`), URL state `?lens=`, driving the beeswarm field (labels stacked in lanes) and the timing tower (rows re-sort with Framer `layout`). C3 Force Ledger: a waterfall from the model average to the prediction on the target's own axis; colour means good or bad for the driver, never the sign; a visually hidden table carries the same rows. Pit Radio: the chat as a 392px column (open by default on the driver page) that collapses to a floating ⌘K bar; Ctrl/⌘+K or Ctrl/⌘+\ toggles it, and driver codes in replies link to that driver's page.

**Motion.** Hover: translateY(-3px) on a spring curve, border 9%→18%. Magnetic primary CTA (`useMagnet`, `lib/motion.ts`). Cursor glow from CSS variables written once a frame. Driver codes share a `view-transition-name` between the grid, the podium and the driver page (320ms). `MotionConfig reducedMotion="user"` plus the global reduced-motion rule.

**Pit Radio copy follows the tier.** `/health` says whether the local chat model runs (`chat_model`). With it, the intro and placeholder invite any F1 question and warn that open answers take a few seconds; without it they describe what the fixed lookups answer. Lookup steps show while they run (`toolLabels.ts` names every tool in fan words), so a 20-40 s answer never looks frozen.

**Checks.** `npm run size` holds first-load JavaScript under 150 KB gzipped; the Playwright smoke tests run axe on desktop and phone (`e2e/`).
