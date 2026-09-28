# Model cards

What each model predicts, what it looks at, how accurate it is on races it had never seen, and where it falls short. Numbers are from the training run of 2026-09-28 (data through the 2026 Azerbaijan GP). The source of truth is `src/models/saved/*_metrics.json`, and the API serves them at `GET /model`.

## How to read the accuracy numbers

**MAE** (mean absolute error) is the average size of a miss. "3.26 places" means the predicted finishing position is, on average, 3.26 places from where the driver actually finished.

Every model is compared with a **naive baseline**: the simple guess a fan could make without any model. A model that doesn't beat its baseline isn't worth showing. The weekly retrain refuses to ship one.

The numbers are **honest**:
- Hyperparameters were chosen using only the earliest 60% of races (2022 to late 2024).
- The remaining 43 races (Nov 2024 to Sep 2026) were each predicted by a model trained only on the races before it, exactly as a live forecast would have been.
- None of those 43 races influenced the model that predicted them.
- The window includes the 2026 rule reset, the hardest period to predict.

**Stages.** One model answers at every point of a race weekend, with whatever is known:

| Stage | Known so far |
|---|---|
| Before practice | Form, team, circuit, weather forecast |
| After practice | + practice lap times |
| After qualifying | + grid and qualifying gaps |
| Race day | + starting tyres (rarely known in time) |

## Finishing position

**Predicts** where each driver finishes. The app shows the order, not the raw number (the model predicts 4.0 for a clear favourite; ranking is what matters).

| Stage | Model MAE | Baseline MAE | Baseline | Order agreement* |
|---|---|---|---|---|
| Before practice | **3.71** | 3.82 | driver's recent average finish | 0.57 |
| After practice | **3.66** | 3.82 | same | 0.58 |
| After qualifying | **3.26** | 3.43 | finish where you start | 0.65 |

\* Mean Spearman correlation between predicted and actual order per race (1 = perfect, 0 = random).

**Leans on most:** recent results, starting grid slot, team's recent results, team's qualifying pace, grid slot vs usual form, best practice lap.

**By season** (after-qualifying MAE, model vs finish-where-you-start): 2023 3.32 vs 3.82 · 2024 2.78 vs 2.91 · 2025 3.14 vs 3.34 · 2026 3.42 vs 3.50. It picked the winner in 38% of 2025 races and 60% of 2026 races.

**Limits.** Races are noisy: the grid alone explains most of the order, and the model adds about 5%. It can't see crashes, strategy calls or mid-race weather.

## Qualifying

**Predicts** each driver's gap to pole as a % of the pole lap. A percentage compares circuits fairly: 0.7 s is a big gap at Monaco and a small one at Spa. The app turns it into a lap time with the pole estimate below.

It sees only what is known before qualifying: no grid, no qualifying times, no race-day information, and no weather (the forecast is for Sunday's race, not Saturday).

| Stage | Model MAE | Baseline MAE | Baseline |
|---|---|---|---|
| Before practice | 0.715% | **0.703%** | team's recent qualifying gap |
| After practice | **0.642%** | 0.703% | same |

**Honest weak spot:** before any practice, it is no better than the team's recent qualifying pace. A "baseline plus correction" version was tried; it helped before practice (0.688%) but hurt after it (0.662%), which is when most qualifying forecasts are read, so it wasn't adopted. In 2026 alone (new cars), the baseline edges it after practice too (0.601% vs 0.568%).

**Leans on most:** team's qualifying pace, recent results, best practice lap.

## Places gained

**Predicts** grid slot minus finishing position (positive means places gained).

| Stage | Model MAE | Baseline MAE | Baseline |
|---|---|---|---|
| Before practice | 3.43 | 3.43 | no change from the grid |
| After practice | 3.43 | 3.43 | same |
| After qualifying | **3.18** | 3.43 | same |

Without a grid there's nothing to gain places from, so it only means something once qualifying has run. **Leans on most:** grid slot vs usual form (a fast driver starting low tends to recover), grid slot, and grid slot on a hard-to-pass track.

## Gap to the winner

**Predicts** each finisher's gap to the winner as a % of the winner's race time. Lapped cars count: a car one lap down is about one average lap behind. (FastF1's own time for a lapped car is only its gap inside the final lap; using it as-is had put lapped cars ahead of lead-lap ones.) Retirements have no gap and aren't predicted.

| Stage | Model MAE | Baseline MAE | Baseline |
|---|---|---|---|
| Before practice | **0.68%** | 0.85% | typical gap in past races |
| After qualifying | **0.64%** | 0.85% | same |

On a 90-minute race, 0.64% is about 35 seconds. The app shows gaps in seconds using the race-length estimate below, sorted to match the predicted order.

## Derived outputs (not extra models)

**Win / podium / points odds, likely range, retirement risk, beats-teammate, head-to-head** (`src/models/probabilities.py`)
- 10,000 simulated races, each a random finishing order drawn around the model's ranking. Each car also retires with a probability from its driver's and team's recent reliability.
- How spread out the order is ("temperature") was fitted on the walk-forward predictions, separately for each stage, so a Thursday forecast is appropriately less sure.
- Check, after qualifying:

| Event | Brier score | Guessing evenly | Better |
|---|---|---|---|
| Win | 0.033 | 0.047 | lower |
| Podium | 0.073 | 0.126 | lower |
| Points | 0.156 | 0.250 | lower |

- The likely range (P10 to P90) catches the actual finish 87% of the time, against an 80% target, so it is slightly wide.

**Pole / Q3 / out-in-Q1 odds**
- The same method on the qualifying model.
- Brier score for pole is 0.042; for reaching Q3 it is 0.141 (guessing evenly: 0.25).

**Predicted qualifying lap and race length** (`src/models/estimates.py`)
- Pole lap = fastest practice lap × this circuit's usual ratio of pole to fastest practice lap.
- Race length = laps × fastest practice lap × this circuit's usual ratio of winner's time to that product. Pit stops and typical safety cars are baked in.
- New venues use the all-circuit median. Before practice there are no lap times, so these show only once practice has run.

**Safety-car chance**
- The share of past races at the circuit with a safety car or VSC, shrunk toward the all-circuit rate when there are only a few.

**Tyre strategy** (`src/strategy/model.py`)
- Fitted per circuit on real race laps, controlling for each driver's own pace and fuel burn:
  - each compound's pace and wear rate
  - the real pit-lane time loss
  - the longest realistic stint per compound
- Circuits with little data borrow the all-circuit fit in proportion.
- Every legal one- and two-stop plan is costed. 2,000 races with random safety cars decide each plan's chance of being fastest (a stop under a safety car costs about half).
- Limits: tyre wear is linear (no cliff), tyres start new, and traffic and track position are ignored. It compares plans; it doesn't predict gaps.

**Championship odds** (`src/agent/scenarios.py`)
- The remaining races and sprints simulated 5,000 times with every driver's current forecast strength and real points.
- Assumes today's form holds. Ties are split, not decided on countback.

## What changed on 2026-09-28, and why it mattered

| Change | Effect |
|---|---|
| Training on every weekend stage | Before qualifying the old model scored 4.09 places MAE, worse than the naive 3.62. Now 3.71. |
| Forecast weather instead of measured race weather | Training now sees what a live forecast sees |
| Previous rules-era form weighted 0.1 | After-qualifying MAE 3.287 → 3.263; before-practice qualifying 0.735% → 0.715% |
| Long-run practice pace as a feature | Tried; no gain (3.749 vs 3.744), so it isn't a model input |
| A ranking objective (`rank:pairwise`) | Tried; no gain over regression (3.49 vs 3.47 places as ranks) |
| History view | Replaced the in-sample replay with these walk-forward predictions |
