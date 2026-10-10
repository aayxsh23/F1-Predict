# Model cards

What each model predicts, what it looks at, how accurate it is on races it had never seen, and where it falls short. Numbers are from the training run of 2026-10-10 (data from 2021 through the 2026 Bahrain GP at Kuala Lumpur; gap-to-winner from 2022). The source of truth is `src/models/saved/*_metrics.json` and `blend.json`; the API serves them at `GET /model`.

## How to read the accuracy numbers

**MAE** (mean absolute error) is the average size of a miss. "3.05 places" means the predicted finishing position is, on average, 3.16 places from where the driver actually finished.

Every model is compared with a **naive baseline**: the simple guess a fan could make without any model. The weekly retrain refuses to ship a model that doesn't beat its baseline at the most informed stage.

The numbers are **honest**:
- Hyperparameters, blend weights and every other setting were chosen using only the earliest 60% of races (2021 to mid-2024).
- The remaining 52 races (Jul 2024 to Oct 2026; 44 from Nov 2024 for gap-to-winner) were each predicted by a model trained only on the races before it, exactly as a live forecast would have been.
- Differences come with a 95% interval from resampling whole races. With ~50 races, a gap under about 0.05 places is usually noise; changes whose interval said "worse" were not shipped.
- The window includes the 2026 rule reset, the hardest period to predict.

**Stages.** One model answers at every point of a race weekend, with whatever is known:

| Stage | Known so far |
|---|---|
| Before practice | Form, team, circuit, weather forecast, pre-season testing pace, recent car upgrades |
| After practice | + practice lap times and this weekend's upgrades (published the Friday) |
| After Sprint Qualifying *(sprint weekends)* | + each driver's Sprint Qualifying gap to the fastest lap |
| After the Sprint *(sprint weekends)* | + Sprint result and Sprint race pace |
| After qualifying | + qualifying gaps; the **qualifying order stands in for the grid** |
| Official grid | + the FIA's starting grid with penalties applied (race morning) |

Until 2026-10-10 the "after qualifying" numbers were measured with the official grid, which a Saturday-night forecast doesn't have. They are now measured with the qualifying order, so they're slightly less flattering and true to what you see.

## What's shown: model plus blend

Three predictions are a weighted average of a model and a simple anchor, each weight chosen on the early races only (`src/models/blend.py`):

| What | Mix | Weight | Effect on the 44 held-out races |
|---|---|---|---|
| Finish, after qualifying | finish model and "grid minus predicted places gained" | 0.6 | MAE 3.19 → **3.05** (interval −0.19 to −0.09); winner picked 46% → **52%**; podium 1.94 → **2.08** of 3 |
| Finish, after the Sprint | finish model and Sprint result | 0.05 | MAE 3.73 → 3.69 (interval −0.06 to −0.01) |
| Qualifying, after Sprint Qualifying | qualifying model and the Sprint Qualifying gap | 0.25 | MAE 0.745% → **0.715%** (interval −0.052 to −0.005) |
| Qualifying, before practice | qualifying model and the team's recent qualifying gap | 0.45 | MAE 0.696% → **0.671%** (interval −0.041 to −0.010) |

The explanation behind each shown number is still an exact sum: the model's own contributions scaled by its weight, plus the anchor's.

## Finishing position

**Predicts** where each driver finishes among the cars that finish; the odds add retirements separately (below). It's trained on finishers only, because a retirement isn't a 20th-place pace.

| Stage | Shown MAE | Baseline MAE | Baseline | Order agreement* |
|---|---|---|---|---|
| Before practice | **3.61** | 3.68 | driver's recent average finish | 0.60 |
| After practice | **3.56** | 3.68 | same | 0.61 |
| After qualifying | **3.05** | 3.29 | finish where you qualified | 0.69 |
| Official grid | **3.07** | 3.38 | finish where you start | 0.69 |

Among cars that finished (the order the app shows), MAE is **2.68** before practice (baseline 2.99) and **2.14** after qualifying (baseline 2.69).

\* Mean Spearman correlation between predicted and actual order per race (1 = perfect, 0 = random).

**Honest weak spot:** after qualifying, "the pole-sitter wins" is still right more often (64% of these races) than the forecast's P1 (52%). A heavier grid weight would close that gap, but the early races chose 0.6 by average error, and choosing it from the later races would be grading our own exam.

**Leans on most:** recent results, grid slot, team's recent results, grid slot vs usual form, team's qualifying pace, best practice lap.

**Limits.** Races are noisy: the grid explains most of the order. It can't see crashes, strategy calls or mid-race weather.

## Qualifying

**Predicts** each driver's gap to pole as a % of the pole lap. A percentage compares circuits fairly: 0.7 s is a big gap at Monaco and a small one at Spa. It sees only what is known before qualifying: no grid, no qualifying times, and no weather (the forecast is for Sunday's race).

| Stage | Shown MAE | Baseline MAE | Baseline |
|---|---|---|---|
| Before practice | **0.671%** | 0.676% | team's recent qualifying gap |
| After practice | **0.637%** | 0.676% | same |
| After Sprint Qualifying | **0.715%** | 0.825% | the driver's Sprint Qualifying gap |

On laps that were real attempts (within 107% of pole), the after-practice error is **0.516%** (baseline 0.554%). About 4% of qualifying results are crashes or no-time runs; their chance enters the odds instead (below).

**Honest weak spot:** before any practice it's only level with the team's recent gap (interval −0.016 to +0.007). That baseline itself got better on 2026-10-10: it now counts only representative laps and includes pre-season testing.

**Leans on most:** team's qualifying pace, recent results, best practice lap.

## Places gained

**Predicts** grid slot minus finishing position (positive means places gained), trained on finishers.

| Stage | MAE | Baseline MAE | Baseline |
|---|---|---|---|
| After qualifying | **3.20** | 3.38 | no change from the grid |
| Official grid | **3.09** | 3.38 | same |

Without a grid there's nothing to gain places from, so before qualifying it only matches the baseline. Among finishers, after qualifying: 2.33 vs 2.79.

## Gap to the winner

**Predicts** each finisher's gap to the winner as a % of the winner's race time. Lapped cars count: a car one lap down is about one average lap behind. Retirements have no gap and aren't predicted.

| Stage | MAE | Baseline MAE | Baseline |
|---|---|---|---|
| Before practice | **0.66%** | 0.85% | typical gap in past races |
| After qualifying | **0.63%** | 0.85% | same |

On a 90-minute race, 0.63% is about 34 seconds. It learns from 2022 on: adding 2021 (a different car generation) made it clearly worse on the early races (0.536% → 0.646%).

## Sprint weekends

A sprint weekend runs one practice session, then Sprint Qualifying, the Sprint, Grand Prix qualifying and the race; the forecast updates after each. The models read a calendar flag (one practice session, not three), the Sprint Qualifying gap, the Sprint result and Sprint pace, blank on other weekends; the two sprint stages are trained and scored on sprint weekends only. A Sprint Qualifying lap more than 7% off the fastest counts as "no representative lap". The 2022 format (Friday qualifying set the Sprint grid) is left out.

On the 15 held-out sprint weekends:

| Stage | Finishing position MAE | Qualifying gap MAE |
|---|---|---|
| After practice (FP1) | 3.73 | 0.753% |
| After Sprint Qualifying | 3.73 | **0.715%** |
| After the Sprint | **3.69** | 0.715% |
| After qualifying | 3.24 | — |

The models alone barely use sprint columns (one weekend in five, shallow trees); the blends above are what turns the sprint sessions into real changes.

## Inputs added and tested on 2026-10-10

| Change | Held-out effect (95% interval) | Shipped |
|---|---|---|
| Pre-season testing pace, as one more qualifying session before round 1 | Qualifying before practice −0.012% (−0.021 to −0.005); rounds 1-4: 0.658% → 0.623% | yes |
| Team qualifying pace from representative laps only (≤107%) | Qualifying after practice −0.016% (−0.027 to −0.007) | yes |
| Car upgrades from the FIA's car-presentation documents (2024 on): this weekend's and the last three races' | Finish after qualifying −0.002 (−0.005 to 0.000): about zero so far | yes (honest data; may matter more with more seasons) |
| Team lineage (RB = AlphaTauri, Audi = Sauber) and venue aliases (Miami Gardens = Miami) | No measurable change; a correctness fix | yes |
| Pit-lane starts read as grid 0 (pole!) | Fixed: they now start at the back | yes |
| Starting tyres as inputs | Never known before a live race; identical accuracy without | removed |
| Wet-weather driver skill | Finish after qualifying +0.007 worse (+0.002 to +0.013) | no |
| 2021 season as training data (finish, places gained, qualifying) | Same 44 races: finish before practice −0.036 (−0.063 to −0.011), after practice −0.030; qualifying before practice −0.018 (−0.033 to −0.005); nothing significantly worse | yes |
| 2021 for gap to the winner | Early races 0.536% → 0.646% (worse) | no |
| 2018-2020 | Downloading (FastF1's courtesy limit, ~10 races an hour); to be tested the same way | pending |
| Dropping >107% qualifying results from training | Slightly worse even on representative laps | no |

## Derived outputs (not extra models)

**Win / podium / points odds, likely range, retirement risk, beats-teammate, head-to-head** (`src/models/probabilities.py`)
- 10,000 simulated races, each a random finishing order drawn around the shown ranking. Each car also retires with a probability from its driver's and team's recent reliability.
- How spread out the order is ("temperature") was fitted on the walk-forward predictions, separately for each stage.
- The likely range has its own temperature, fitted on the early races so the P10-P90 range holds the result 80% of the time. On the held-out races it holds 80% after qualifying and 74-80% at other stages (it was 87%: needlessly wide).
- Check, after qualifying:

| Event | Brier score | Before 2026-10-10 | Guessing evenly |
|---|---|---|---|
| Win | **0.030** | 0.033 | 0.047 |
| Podium | **0.068** | 0.074 | 0.126 |
| Points | **0.154** | 0.156 | 0.250 |

(Measured over 52 races now versus 43 before; on the identical 44 races the 2022-only version of tonight's model scored 0.028 for the win.)

**Pole / Q3 / out-in-Q1 odds**
- The same method on the qualifying model, plus each driver's chance of no representative lap (recent record, field rate about 4%).
- Brier score for pole is 0.041; for reaching Q3 it is 0.141 (guessing evenly: 0.25); after Sprint Qualifying 0.038 and 0.121.

**Predicted qualifying lap and race length** (`src/models/estimates.py`)
- Pole lap = fastest practice lap × this circuit's usual ratio of pole to fastest practice lap.
- Race length = laps × fastest practice lap × this circuit's usual ratio of winner's time to that product.

**Safety-car chance**: the share of past races at the circuit with a safety car or VSC, shrunk toward the all-circuit rate when there are only a few.

**Tyre strategy** (`src/strategy/model.py`): fitted per circuit on real race laps since 2022 (the 18-inch tyres); every legal one- and two-stop plan costed, 2,000 races with random safety cars. Limits: linear tyre wear, no traffic.

**Championship odds** (`src/agent/scenarios.py`): the remaining races and sprints simulated 5,000 times with every driver's current forecast strength and real points.

## Earlier changes (2026-09-28)

| Change | Effect |
|---|---|
| Training on every weekend stage | Before qualifying the old model scored 4.09 places MAE, worse than the naive 3.62. Then 3.71. |
| Forecast weather instead of measured race weather | Training sees what a live forecast sees |
| Previous rules-era form weighted 0.1 | After-qualifying MAE 3.287 → 3.263 |
| Long-run practice pace as a feature | Tried; no gain |
| A ranking objective (`rank:pairwise`) | Tried twice (also 2026-10-10: winner 50-52%); no better than the blend |
