---
name: Pit Wall
description: F1 race predictions and the reasoning behind them, in a fan's vocabulary.
colors:
  gray-0: "#ffffff"
  gray-50: "#f8f9fb"
  gray-100: "#eef0f3"
  gray-200: "#dde1e6"
  gray-300: "#c3c9d1"
  gray-400: "#9aa2ad"
  gray-550: "#67717d"
  gray-600: "#4f5761"
  gray-700: "#363c44"
  gray-800: "#22262b"
  gray-900: "#131519"
  gray-950: "#0a0b0d"
  red-400: "#ff6b5e"
  red-500: "#e8483a"
  red-600: "#c22e22"
  red-700: "#961f17"
  green-400: "#34b46a"
  green-500: "#1f9d55"
  green-700: "#147842"
typography:
  display:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "36px"
    fontWeight: 600
    lineHeight: "42px"
    letterSpacing: "-0.01em"
  title:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "28px"
    fontWeight: 600
    lineHeight: "34px"
  body:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: "20px"
  label:
    fontFamily: "Inter, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 600
    lineHeight: "16px"
    letterSpacing: "0.05em"
  numeral:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: "14px"
    fontWeight: 500
    lineHeight: "20px"
rounded:
  sm: "4px"
  md: "8px"
  lg: "12px"
spacing:
  1: "4px"
  2: "8px"
  3: "12px"
  4: "16px"
  5: "24px"
  6: "32px"
  7: "48px"
  8: "64px"
components:
  button-primary:
    backgroundColor: "{colors.red-500}"
    textColor: "#ffffff"
    rounded: "{rounded.md}"
    padding: "0 16px"
    height: "40px"
  button-primary-hover:
    backgroundColor: "{colors.red-600}"
  button-secondary:
    backgroundColor: "{colors.gray-0}"
    textColor: "{colors.gray-900}"
    rounded: "{rounded.md}"
    padding: "0 16px"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.gray-600}"
    rounded: "{rounded.md}"
  badge-known:
    backgroundColor: "{colors.green-500}"
    textColor: "{colors.green-700}"
    rounded: "{rounded.sm}"
    padding: "2px 8px"
  badge-pending:
    backgroundColor: "{colors.gray-100}"
    textColor: "{colors.gray-550}"
    rounded: "{rounded.sm}"
    padding: "2px 8px"
  card:
    backgroundColor: "{colors.gray-0}"
    rounded: "{rounded.md}"
    padding: "20px"
---

# Design System: Pit Wall

## Overview

**Creative North Star: "The Timing Tower"**

Pit Wall is built like the data screens a race engineer actually reads during a session: a restrained neutral canvas, one controlled accent, and every number set in a monospace face so columns of positions, gaps, and times never jitter against each other. It is not a marketing dashboard of icon cards; it is dense, tabular, and built to be scanned in seconds — the FIRST VIEWPORT direction pins the podium as one card of timing-sheet rows, not three duplicate stat tiles, and the shipped Dashboard, RaceDetail, and History routes hold that line.

The palette is almost entirely gray. F1 red exists only where it means something — the current selection, a primary action, a link — never as chrome, never as decoration on a heading. This is a "brief-pinned" system: the token set was specified in phase7-ui-backend-plan.md and approved as-is rather than explored through a visual-world workshop, so its authority comes from deliberate specification and from what actually shipped, not from a comp file.

**Key Characteristics:**
- Restrained neutral canvas (gray-50 light / gray-950 dark) with a single red accent reserved for selection, primary actions, and links.
- Inter for all UI text; JetBrains Mono for every numeral so tabular data never jitters column-to-column.
- Dense, timing-tower-style rows and tables over icon-card grids.
- Flat surfaces (hairline borders + a near-invisible ambient shadow), not lifted cards.
- Full light/dark parity via the same semantic token names, driven by `prefers-color-scheme` with an explicit `[data-theme]` override.

## Colors

The palette is a single neutral gray ramp plus one accent hue plus two data-signal hues; there is no secondary or tertiary brand color.

### Primary
- **F1 Red** (`#e8483a` light / `#ff6b5e` dark, tokens `--red-500`/`--red-400` via `--accent-default`): primary buttons, active nav/tab state, links, the current-selection rank digit on the dashboard podium. Never used on chrome (headers, sidebars, borders).

### Neutral
- **Canvas** (`#f8f9fb` light / `#0a0b0d` dark, `--bg-canvas`): page background.
- **Surface** (`#ffffff` light / `#131519` dark, `--bg-surface`): cards, table backgrounds, the top bar.
- **Surface Sunken** (`#eef0f3` light / `#22262b` dark, `--bg-surface-sunken`): the sidebar, row hover, skeleton fill, tab-list track.
- **Border Default** (`#dde1e6` light / `#363c44` dark): the only border color in the system — hairline dividers between rows, card outlines, the top-bar rule.
- **Text Primary** (`#131519` light / `#f8f9fb` dark): headings, driver codes, primary data values.
- **Text Secondary** (`#4f5761` light / `#c3c9d1` dark): team names, supporting sentences.
- **Text Muted** (`--gray-550` `#67717d` light / `--gray-400` `#9aa2ad` dark): freshness lines, table headers, section labels, placeholders — deliberately re-picked off the raw 100-step gray scale on both ends to clear WCAG AA 4.5:1 for small text (gray-500 cleared only ~4.35:1 light; gray-500 cleared only ~4.2:1 dark). This is the system's contrast floor for informational text.

### Data-Signal Colors (chart & badge use only — not general UI colors)
- **Chart Positive / Known** (`--green-500` `#1f9d55`): SHAP bars pushing toward the target's "better" direction (see the Target-Relative SHAP Color rule below); the known-session badge fill.
- **Chart Negative** (`--red-500`): SHAP bars pushing toward the "worse" direction.
- **Chart Pending** (`--gray-400`): not-yet-known chart/badge state.
- **Badge Known Text** (`--green-700` `#147842` light / `--green-400` dark): re-picked off the raw green scale for the same reason as `--text-muted` — green-500 as small badge text cleared only ~3.5:1.

### Named Rules
**The One Accent Rule.** Red is spent on exactly three things: the primary action, the active/selected state, and a link. It never appears on a container, a border, or a heading. A screen where red decorates chrome instead of signaling selection or action is off-system.

**The Re-Picked Contrast Step Rule.** Where a token is used as small informational text (muted labels, badge text) rather than as a large/decorative element, its value is deliberately stepped off the raw numeric scale (`gray-550` instead of `gray-500`; `green-700` instead of `green-500`) to clear AA 4.5:1 on its actual background. Don't substitute the "nearby" raw scale step back in — it was rejected for a measured reason.

## Typography

**UI Font:** Inter (with `system-ui, sans-serif` fallback)
**Numeral Font:** JetBrains Mono (with `ui-monospace, monospace` fallback)

**Character:** A dense data-tool pairing, not an editorial one. Inter carries every word; JetBrains Mono is reserved entirely for numerals and driver/session codes so tabular values align vertically and never reflow mid-digit.

### Hierarchy
- **Display** (600, 36px/42px, tracking tight): the top-level page identity — the Grand Prix name on Dashboard/RaceDetail. Scales down to Title (28px) on the same element at the mobile breakpoint via responsive text-size classes, not a separate typographic role.
- **Title** (600, 28px/34px): secondary page headings ("Race history").
- **Body** (400, 14px/20px, base document size): descriptive sentences, supporting copy, table cell text.
- **Label** (600, 12px/16px, `letter-spacing: 0.05em`, uppercase): section labels within a page ("Predicted top 3", a season-group header, table column headers) — always paired with `text-muted`. This is a compact section divider inside body content, not a kicker sitting above a page's `<h1>`.
- **Numeral** (JetBrains Mono, 400–600, `tabular-nums`): every driver code, position digit, gap, and timing value across the app, regardless of surrounding font size.

### Named Rules
**The No-Jitter Rule.** Any value that is a number, a driver code, or session/timing data renders in JetBrains Mono with `tabular-nums`. Prose never does. This is what keeps the timing-sheet rows and result tables legible at a glance.

## Layout

Content is centered in a max-width column per route (`max-w-3xl` for single-column reading views like Dashboard/History, `max-w-5xl` for the RaceDetail results table), with `px-4 md:px-6` horizontal padding and `py-8 md:py-12` vertical rhythm — generous on desktop, tighter on mobile. Vertical spacing between page sections steps through the 4px-based scale in a consistent block sequence (`mt-2` → `mt-6` → `mt-8`/`mt-10`), never arbitrary values.

The app shell is a persistent sidebar on desktop (`md:` and up, 224px/`w-56`, fixed, `surface-sunken` background) and a fixed bottom tab bar on mobile (`md:hidden`, `surface` background, safe-area-aware), sharing one `NAV_ITEMS` source so the two chrome components never drift out of sync. A sticky, blurred top bar (`bg-surface/90 backdrop-blur-sm`) holds only the mobile wordmark and the theme toggle — it carries no page title, since the page title lives in the content column itself.

Tables collapse to stacked cards below `md:` (RaceDetail's driver grid becomes `MobileDriverRow` cards) rather than horizontally scrolling or truncating columns — a squeezed table is treated as a non-solution for dense tabular data on narrow screens.

## Elevation & Depth

The system is flat by default and uses borders, not shadows, as the primary separator between surfaces. Both shadow tokens that exist are near-imperceptible ambient shims (`--shadow-sm: 0 1px 2px rgba(0,0,0,0.06)`, `--shadow-md: 0 4px 12px rgba(0,0,0,0.12)`), applied at rest to cards and the active nav/tab item — never as a hover-triggered "lift." Depth between canvas, surface, and sunken-surface is conveyed by the three-step neutral tone ramp (`--bg-canvas` / `--bg-surface` / `--bg-surface-sunken`), not by shadow escalation.

### Shadow Vocabulary
- **`--shadow-sm`** (`0 1px 2px rgba(0,0,0,0.06)`): default resting shadow on Card and on the active sidebar/tab item — a hairline separation, not a lift.
- **`--shadow-md`** (`0 4px 12px rgba(0,0,0,0.12)`): declared as a heavier step in the token file; not observed in use by any shipped component. Available for a future overlay/popover surface, not yet spent.

### Named Rules
**The Border-Over-Shadow Rule.** The default way to separate one surface from another is a 1px `--border-default` line, not a shadow. Shadows are reserved for the resting Card token and active-state highlighting, never for implying hover elevation.

## Shapes

Corners are consistently soft and small: `--radius-sm` (4px) on badges and table-adjacent small controls, `--radius-md` (8px) on buttons, cards, nav items, and tab triggers, `--radius-lg` (12px) declared in tokens but not yet spent by a shipped component. There is no sharp-cornered or fully-rounded (pill) shape anywhere in the system — every interactive surface uses the same small-radius vocabulary, which reinforces the dense, tool-like character over a softer consumer-app look. Borders are always the single `--border-default` hairline; there is no double-border or outlined-badge pattern.

## Components

### Buttons
- **Shape:** `rounded-md` (8px), heights `h-8` (sm, 32px) / `h-10` (md, 40px).
- **Primary:** `bg-accent` (red-500/red-400) with white text; `padding` follows the size scale (`px-3` sm / `px-4` md).
- **Hover / Focus:** primary darkens to `--accent-hover`; all variants transition color/background over 150ms; focus-visible gets a 2px solid accent-colored outline with 2px offset, applied globally, not per-component.
- **Secondary:** white/`gray-900` surface with a `border-default` outline, hovers to `surface-sunken`.
- **Ghost:** transparent, `text-secondary` at rest, hovers to `text-primary` + `surface-sunken` fill. No dedicated tertiary variant exists.

### Chips / Badges
- **Style:** two semantic tones only — `known` (green tint background + AA-safe green text) and `pending` (sunken-gray background + muted text) — plus a generic `neutral` tone for non-session use. `rounded-sm` (4px), `text-xs`, always paired with a small check/circle glyph from lucide-react.
- **State:** Badges are display-only in this build (no dismiss/select interaction); tone is the only variant axis, driven by boolean session-known data, never invented status language.

### Cards / Containers
- **Corner Style:** `rounded-md` (8px).
- **Background:** `--bg-surface` (white/gray-900).
- **Shadow Strategy:** `--shadow-sm` at rest (see Elevation & Depth); no hover elevation change.
- **Border:** 1px `--border-default` on every card.
- **Internal Padding:** `p-5` (20px) by default; list-style cards (podium rows, history rows) override to `p-0` and let internal rows carry their own `px-5 py-4`/`py-3` padding with hairline dividers between them instead of gaps between separate cards.

### Navigation
- Sidebar (desktop, `md:` and up): fixed 224px column, `surface-sunken` background, vertical list of icon + label items; active item gets `surface` background, `text-primary`, and the resting shadow token; inactive items are `text-secondary` and hover to the same active treatment.
- Bottom nav (mobile, below `md:`): fixed full-width bar, `surface` background, icon-over-label items in a flex row; active item is text-only accent-colored (`text-accent-text`), no background change, distinguishing the mobile active state from the desktop one.
- Both share one `NAV_ITEMS` list (icons from lucide-react) so item order and labels can't drift between the two breakpoint-specific components.

### Signature Component: Timing-Sheet Row
The recurring pattern across Dashboard's podium, RaceDetail's mobile driver cards, and History's race list: a large mono rank/position digit on the left, a two-line identity block (bold primary line, muted secondary line) in the middle, and a trailing affordance (a "Why" label + chevron, or just a chevron) on the right, the whole row as one tappable `Link`. This is the app's core visual signature — direct grounding of the THESIS/FIRST VIEWPORT commitment that a prediction and its reasoning are never more than a tap apart.

## Do's and Don'ts

### Do:
- **Do** put every number, driver code, or timing value in JetBrains Mono with `tabular-nums` — never in Inter.
- **Do** reserve red for the primary action, the active/selected state, and links only.
- **Do** use the `--border-default` hairline as the default surface separator; reach for `--shadow-sm` only on Card and active nav/tab, never as a hover-triggered lift.
- **Do** re-pick a color's small-text step off its neutral or signal scale (`gray-550`, `green-700`) rather than reusing the visually-nearest raw step, whenever that token will render as small informational text — check the actual contrast ratio, don't assume adjacency is safe.
- **Do** collapse dense tables to stacked cards below `md:`, not a horizontally-scrolling or truncated table.

### Don't:
- **Don't** put a kicker/eyebrow label above a page's `<h1>`. Three route headers shipped with an invented eyebrow pattern during this build and it was removed in finish review; it is not part of this system going forward. (The `text-muted` uppercase label style that remains — "Predicted top 3", a season-group header, table column headers — is a section label living *inside* body content, not a device sitting above a headline; don't conflate the two.)
- **Don't** add a second accent hue. The system is one neutral ramp plus one accent plus two data-signal colors (green/red for SHAP polarity) — a second brand color has no precedent in the build.
- **Don't** use a pill (fully-rounded) shape or a sharp (0px) corner anywhere. The radius vocabulary is `sm`/`md`/`lg` (4/8/12px) only.
- **Don't** introduce a drop shadow heavier than `--shadow-md`, or one that appears only on hover to simulate lift — depth in this system comes from the border + tone-ramp model, not shadow escalation.
