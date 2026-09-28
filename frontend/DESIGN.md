---
name: F1 Predict
description: A pit-garage telemetry wall after dark; three predicted cars staged over the season, a dossier of tools below.
colors:
  obsidian-950: "#05070a"
  obsidian-900: "#080a0e"
  obsidian-800: "#0f141c"
  obsidian-700: "#151c27"
  obsidian-600: "#1c2532"
  line: "rgb(148 163 184 / 0.16)"
  line-strong: "rgb(148 163 184 / 0.34)"
  silver-100: "#f4f7fb"
  silver-200: "#e2e8f0"
  silver-300: "#c3ccd8"
  silver-400: "#94a3b8"
  silver-500: "#64748b"
  laser-300: "#5ff0df"
  laser-400: "#00d2be"
  laser-500: "#00a19b"
  laser-700: "#00615e"
  laser-900: "#072a2b"
  signal-amber: "#f5b84b"
  signal-coral: "#ff6b5e"
  weave-light: "rgb(255 255 255 / 0.03)"
  weave-dark: "rgb(0 0 0 / 0.3)"
  grid-line: "rgb(148 163 184 / 0.07)"
  laser-wash: "rgb(0 210 190 / 0.32)"
typography:
  display:
    fontFamily: "Chakra Petch, Barlow, ui-sans-serif, system-ui, sans-serif"
    fontSize: "1.875rem"
    fontWeight: 600
    lineHeight: 1.25
    letterSpacing: "0.04em"
  headline:
    fontFamily: "Chakra Petch, Barlow, ui-sans-serif, system-ui, sans-serif"
    fontSize: "1.125rem"
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: "0.06em"
  title:
    fontFamily: "Chakra Petch, Barlow, ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.9375rem"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "0.06em"
  body:
    fontFamily: "Barlow, ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.9375rem"
    fontWeight: 400
    lineHeight: 1.5
  body-small:
    fontFamily: "Barlow, ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Chakra Petch, Barlow, ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.6875rem"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "0.14em"
  data:
    fontFamily: "JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace"
    fontSize: "0.6875rem"
    fontWeight: 400
    lineHeight: 1.4
  micro:
    fontFamily: "JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace"
    fontSize: "0.625rem"
    fontWeight: 400
    lineHeight: 1
rounded:
  none: "0px"
  cut: "12px"
  dot: "9999px"
spacing:
  hairline: "1px"
  ribbon-gap: "8px"
  gutter: "16px"
  gutter-lg: "24px"
  ribbon-height: "92px"
components:
  button-laser:
    backgroundColor: "transparent"
    textColor: "{colors.laser-300}"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "6px 12px"
  button-laser-hover:
    backgroundColor: "{colors.laser-900}"
    textColor: "{colors.laser-300}"
  chip-prompt:
    backgroundColor: "transparent"
    textColor: "{colors.silver-300}"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "6px 10px"
  chip-prompt-hover:
    textColor: "{colors.laser-300}"
  race-card:
    backgroundColor: "{colors.obsidian-800}"
    textColor: "{colors.silver-200}"
    rounded: "{rounded.cut}"
    width: "176px"
    height: "72px"
    padding: "8px 12px"
  race-card-selected:
    backgroundColor: "{colors.laser-900}"
    textColor: "{colors.silver-100}"
  telemetry-pill:
    backgroundColor: "transparent"
    textColor: "{colors.silver-300}"
    typography: "{typography.micro}"
    rounded: "{rounded.none}"
    padding: "3px 6px"
  telemetry-pill-laser:
    backgroundColor: "{colors.laser-900}"
    textColor: "{colors.laser-300}"
  dossier-sheet:
    backgroundColor: "{colors.obsidian-800}"
    textColor: "{colors.silver-200}"
    rounded: "{rounded.cut}"
  input-field:
    backgroundColor: "{colors.obsidian-900}"
    textColor: "{colors.silver-100}"
    typography: "{typography.body-small}"
    rounded: "{rounded.none}"
    height: "40px"
    padding: "8px 12px"
---

# Design System: F1 Predict

## Overview

**Creative North Star: "The Night Garage"**

The product is one surface: a pit-garage telemetry wall after dark. A 92px season ribbon runs across the top, three procedurally built cars stand on a lit stage, and a four-tab dossier of tools sits beneath them. The ground is obsidian carbon, the only light is one laser teal, and every piece of type is etched silver. Depth comes from light and material (weave, micro-grid, radial glow), not from shadowed cards.

Density is HUD density: small caps labels, mono numbers, hairline frames cut at the corners. Every number is one click from its explanation, so the chrome stays quiet and the teal is spent on what is live, selected, or actionable. The cars carry no liveries; team identity is a thin line of colour, and teal reaches the cars only as light.

The world refuses the tab-and-card dashboard: no rounded card grid, no soft drop-shadow tiles, no light theme.

**Key Characteristics:**
- Obsidian carbon ground (#080A0E / #0F141C) lit by one laser teal (#00D2BE / #00A19B).
- Chakra Petch uppercase for display and HUD labels, Barlow for prose, JetBrains Mono for data.
- Laser-cut chamfered hairline frames (12px cut, 1px line) instead of radius and shadow.
- Carbon twill weave and a faded 32px HUD micro-grid as material, kept under 4% contrast.
- Procedural cars under studio light; team colour as 2px trim only.
- Dark only (`color-scheme: dark`).

## Colors

A single-accent dark palette: five obsidian steps, a silver type ramp, one laser-teal ramp, and two signal colours with fixed meanings.

### Primary
- **Laser Teal** (laser-400): the one light in the world. Focus rings, the selected race card's frame, the active dossier tab frame, live-status dots, positive SHAP bars, and the rim light and stage rings in the 3D scene.
- **Laser Glint** (laser-300): teal as text. Rank tags (P1), live source chip, laser-button labels, citation links. 9.6:1-class legibility on obsidian.
- **Deep Laser** (laser-500): outline strokes for laser buttons and inputs on focus, probability bars at 25-60% opacity, text highlight marks.
- **Laser Ember** (laser-700) and **Laser Well** (laser-900): the assistant message border and the fill behind laser-framed surfaces and laser pills.
- **Laser Wash** (laser-wash): text selection and the selected race card's glow.

### Tertiary
- **Pending Amber** (signal-amber): pending and warning only. "Simulator offline", replay caveats, archive view, the API waking LED.
- **Loss Coral** (signal-coral): negative delta and failure only. Lost grid places, SHAP features pushing the wrong way, an errored chat message, API down.

### Neutral
- **Obsidian Deep** (obsidian-950): the season ribbon, inactive folder tabs, the stage floor of the gradient, 3D fog.
- **Obsidian Carbon** (obsidian-900): page ground and input fills.
- **Carbon Panel** (obsidian-800): framed surface fill and the carbon weave base.
- **Raised Carbon** (obsidian-700) and obsidian-600: user chat bubbles and skeleton rows.
- **Hairline** (line) and **Hairline Strong** (line-strong): default borders; the chamfered frame stroke; scrollbar thumb.
- **Silver Highlight** (silver-100): headings, driver codes, selected text.
- **Etched Silver** (silver-200): body text.
- **Brushed Silver** (silver-300): secondary UI text, pill text, inactive controls.
- **Dim Silver** (silver-400): meta, captions, placeholders, hover frame on race cards (7.2:1).
- **Graphite** (silver-500): decoration and disabled state only.

### Named Rules
**The One Light Rule.** Laser teal is the only hue the interface itself emits. It marks what is live, selected, focused, or actionable; it is never a fill for a whole panel.

**The Trim-Only Team Colour Rule.** Team colour (lib/teams.ts accents) appears only as a 2px accent line on a HUD callout or as small trim on a car, never as body paint. A teal body sheen was tried and rejected because it read as a livery; teal reaches the cars only as light. Unknown teams fall back to silver (#94A3B8), never a guessed colour.

**The Signal Means Something Rule.** Amber means pending or warning; coral means negative delta or failure. Neither is decoration.

**The Graphite Is Not Text Rule.** silver-500 (3.9:1) is for decoration and disabled controls only, never text a reader needs.

## Typography

**Display Font:** Chakra Petch (with Barlow, system sans)
**Body Font:** Barlow (with system sans)
**Label/Mono Font:** JetBrains Mono (with ui-monospace)

**Character:** Chakra Petch's squared, machined caps read as etched plate lettering; Barlow keeps prose calm and narrow; the mono carries every number so columns never jitter.

### Hierarchy
- **Display** (600, 1.5rem rising to 1.875rem at sm, uppercase, 0.04em): the Grand Prix name over the stage; the page's one h1. Driver codes on the podium callouts use the same face at 700, 1.5-1.875rem for P1 and 1.25-1.5rem for P2/P3.
- **Headline** (600, 1.125rem, uppercase, 0.06-0.08em): dossier sheet titles and empty or error state titles.
- **Title** (600, 0.9375rem, uppercase, 0.06em): race names on ribbon cards, document titles, driver selects.
- **Body** (400, 0.9375rem, 1.5): base UI text. Sheet prose and chat use body-small (0.875rem); explanatory copy caps at about 28rem (max-w-md).
- **Label** (600, 0.6875rem, uppercase, 0.14em): the HUD label. Buttons, tabs, chips, status chips, field labels, table headers.
- **Data** (JetBrains Mono 400, 0.6875rem): round numbers, dates, location lines, citation tags. Tabular numerals wherever figures align.
- **Micro** (JetBrains Mono 400, 0.625rem): telemetry pills and circuit plates.

### Named Rules
**The Three-Step Ramp Rule.** The tokenised ramp is micro / hud / base (0.625 / 0.6875 / 0.9375rem). Headlines and display step up through Tailwind's lg-3xl; nothing new sits between micro and hud.

**The Etched Caps Rule.** Chakra Petch is always uppercase and tracked (0.04em at display, 0.14em at label). Barlow is never uppercase.

**The No Kicker Rule.** No eyebrow or kicker label sits above a heading. A HUD label names a field, a control, or a tab; it never introduces a title.

## Layout

A single full-viewport surface, no routes. At lg and up the page locks to 100dvh with overflow hidden: ribbon (92px), stage (flex-1), dossier (clamp(240px, 34dvh, 380px) sheet plus 44px tabs). Below lg the page scrolls; the sheet grows to min(74dvh, 640px) and the stage holds a 540px minimum (440px at sm).

The season ribbon is a virtualised horizontal track of 176x72px race cards on an 8px gap with 20px end padding, edges faded by a 28px mask, the selected weekend snapped to centre. Below md the brand and season switcher take their own 48px row above the track.

The podium composition is shared by the 3D and lite stages: P1 centre at scale 1.0, P2 right at 0.84, P3 left at 0.72, each nosed toward the winner and revealed back to front (P3, then P2, then P1), each under its HUD callout. P1 carries the full callout (214px); P2 and P3 stay compact (172px); on phones all shrink to 116px.

Known gap (not a rule): on the phone lite stage at 390px the three callouts overlap and the P1 scale does not read.

Horizontal gutters are 16px, 24px at lg; the stage title block insets 32px at lg. The stage is 3D only where it will run well (WebGL, no reduced-motion or data-saver, over 2 GB RAM and 2 cores, wider than 639px); phones default to the lite CSS 3D stage. A visible 3D / Lite toggle overrides and persists.

## Elevation & Depth

Flat surfaces, lit space. Panels do not cast shadows; depth is conveyed by the stage lighting (a teal radial glow at 15% under the cars, a silver radial at 10% from above, obsidian-900 to obsidian-950 falloff), the faded HUD grid behind it, and the carbon weave inside frames. In the 3D scene the cars are carbon clearcoat, silver and titanium metals under cool white studio key lights, with a teal rim light, teal ring under each car, and fog at #05070a.

Known gap (not a rule): there is no legible floor reflection under the 3D cars yet.

### Shadow Vocabulary
- **Laser glow** (`drop-shadow(0 6px 14px var(--color-laser-wash))` on a wrapper, with a 2px lift): the selected race card only. It sits on a wrapper because clip-path would cut a shadow on the chamfered element itself.

### Named Rules
**The Light Not Shadow Rule.** Separation is a hairline or a change of obsidian step; emphasis is teal light. No box shadows on panels.

**The Material Under 4% Rule.** Weave and grid are material, not pattern: weave highlights at 3% white, grid lines at 7% silver, the grid masked to fade toward the edges.

## Shapes

Zero radius everywhere; corners are cut, not rounded. The laser-cut frame is a 1px hairline that follows 12px chamfers, drawn as the element's own background with an inset fill (the fill's cut is 0.41px tighter so the diagonal stays 1px). Default chamfers are top-right and bottom-left; variants cut all four corners (HUD callouts, decision blocks), the top two (folder tabs), or the bottom two (the dossier sheet). Small controls (buttons, chips, pills, inputs) are square with a plain 1px border. The only round forms are 6px status LEDs, compound swatches, and the ellipse plinth under each lite-stage car. Disabled placeholders for unavailable inputs use a dashed hairline.

## Components

### Buttons
Engraved and quiet; teal outline when it acts.
- **Shape:** square (0px), 1px border.
- **Laser:** laser-500 border, laser-300 HUD-label text, 6-8px by 12px; hover fills laser-900. Used for Retry, Try again, jump-to-race, send.
- **Ghost / icon:** no border, silver-300 or silver-400; hover lifts to silver-100 or laser. Disabled at 30-40% opacity.
- **Focus:** a 2px laser-400 outline, 2px offset, on every focusable element.

### Chips
- **Prompt chips:** line-strong border, silver-300 HUD label; hover turns border laser-500 and text laser-300.
- **Telemetry pills:** micro mono in a square hairline box. Laser tone (laser-500 border, laser-900 fill, laser-300 text) for odds and a podium hit; plain for result, grid delta, and tyre. The tyre pill appears only when a starting compound is known; the odds pill only once the API supplies probabilities.

### Cards / Containers
- **Corner Style:** laser-cut chamfer (12px), never radius.
- **Background:** obsidian-800, with carbon weave on the dossier sheet and active tab.
- **Shadow Strategy:** none (see Elevation & Depth).
- **Border:** 1px line-strong frame; laser-400 frame with laser-900 fill when selected or hovered.
- **Internal Padding:** 16px, 24px at lg; ribbon cards 8px by 12px.

### Inputs / Fields
- **Style:** square, 1px line-strong border, obsidian-900 fill, body-small silver-100 text, silver-400 placeholder, 40px minimum height.
- **Focus:** border shifts to laser-500 plus the global laser focus ring.
- **Disabled:** silver-500 HUD label, line border, not-allowed cursor. The strategy scenario controls are drawn but disabled under an amber "Simulator offline" tag; no control pretends to change a prediction.

### Navigation
- **Season ribbon:** obsidian-950 header, 92px at md. Brand wordmark in Chakra Petch 700 at 0.2em tracking with a teal interpunct; season stepper; draggable, wheel-scrollable race track; API LED and race step arrows at the end. Race cards show round in mono, circuit plate, a SPRINT tag on sprint weekends, race name, date and a status mark (check for completed, pulsing teal dot for live, hollow dot for upcoming). Selected card: laser frame, laser-900 fill, 2px lift and glow.
- **Dossier tabs:** four folder tabs (numbered 01-04 with a // separator in HUD label). The active tab is a shared-layout carbon folder with a laser frame that springs between tabs and joins the sheet; inactive tabs sit 6px lower on obsidian-950. Arrow keys, Home and End move between tabs; 1-4 and [ ] are page shortcuts.

### HUD Callout (signature)
A cut-all frame over 88% obsidian-900 pinned above each car: P-rank in laser-300 label, driver code in Chakra Petch, car number in mono, a 2px team accent line with the team name, a row of telemetry pills, and a "Why this call" link. Hover lights the frame laser and the car together.

### Source Chip (signature)
A HUD label with a status dot that states where the numbers come from. "Live forecast" (teal, pulsing) appears only when the forecast is under 36 hours old; older forecasts say "Latest forecast"; replays of decided races carry the amber caveat that a replay flatters the model.

### Motion
Entrances rise 14px and fade over 0.4s on an expo-out curve (cubic-bezier(0.16, 1, 0.3, 1)), sections staggered 50ms apart; the dossier sheet swaps with a 30px rise. Loading uses a scanning sheen across skeleton rows, never a spinner. Reduced motion sets transition-duration to 0s, not the common 0.01ms: the 0.01ms version still transitioned layout widths.

## Do's and Don'ts

### Do:
- **Do** frame surfaces with the laser-cut chamfer (12px cut, 1px line-strong hairline) instead of radius or shadow.
- **Do** spend laser teal only on live, selected, focused, or actionable things.
- **Do** keep team colour to a 2px accent line or small car trim; fall back to silver (#94A3B8) for an unmatched team.
- **Do** set every number in JetBrains Mono, tabular where figures align.
- **Do** draw a telemetry pill only when its data is real: tyre once a starting compound is known, odds once probabilities exist, result once the race is decided.
- **Do** say "Live forecast" only under 36 hours old, and mark replays of decided races with the amber caveat.
- **Do** default phones and constrained devices to the lite stage, with a visible 3D / Lite toggle.
- **Do** use transition-duration 0s under prefers-reduced-motion.

### Don't:
- **Don't** paint a car body in team colour or teal; a teal body sheen was tried and rejected as a livery. No team liveries.
- **Don't** use amber or coral for anything but pending/warning and negative delta/failure.
- **Don't** set text a reader needs in silver-500.
- **Don't** put a kicker or eyebrow label above a heading.
- **Don't** add box shadows, rounded cards, or a light theme.
- **Don't** wire scenario controls that change nothing; unavailable tools are drawn disabled and say why.
- **Don't** let weave or grid rise above material contrast (3% / 7%).
