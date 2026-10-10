# Model cards

What each model predicts, what it looks at, how accurate it is on races it had never seen, and where it falls short. Numbers are from the training run of 2026-10-11 (data through the 2026 Bahrain GP at Kuala Lumpur). Each model learns from the seasons that helped it on the tuning races: qualifying from 2018, finishing position and places gained from 2021, gap to the winner from 2022. The source of truth is `src/models/saved/*_metrics.json` and `blend.json`; the API serves them at `GET /model`.

## How to read the accuracy numbers

**MAE** (mean absolute error) is the average size of a miss. "3.05 places" means the predicted finishing position is, on average, 3.05 places from where the driver actually finished.

Every model is compared with a **naive baseline**: the simple guess a fan could make without any model. The weekly retrain refuses to ship a model that doesn't beat its baseline at the most informed stage, or that is clearly worse than the model it would replace on the latest 24 races (`src/models/regression_check.py`).

The numbers are **honest**:
- Hyperparameters, blend weights and every other setting were chosen using only the earliest 60% of each model's races.
- The remaining races were each predicted by a model trained only on the races before it, exactly as a live forecast would have been: 52 races from Jul 2024 for finishing position and places gained, 76 from Jul 2023 for qualifying, 44 from Nov 2024 for gap to the winner.
- Separately, every forecast the app actually published is scored once the race is run: the **live track record** (`src/models/live_record.py`, on the History page). That is the check on the live pipeline itself, not a simulation of it.
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

Four predictions are a weighted average of a model and a simple anchor, each weight chosen on the early races only (`src/models/blend.py`):

| What | Mix | Weight | Effect on the held-out races |
|---|---|---|---|
| Finish, after qualifying | finish model and "grid minus predicted places gained" | **0.7 for the front three grid slots**, 0.5 for the rest | MAE 3.18 → **3.05** (interval −0.17 to −0.09); winner picked 44% → **54%**; podium 1.98 → **2.06** of 3 |
| Finish, after the Sprint | finish model and Sprint result | 0.15 | MAE 3.73 → **3.64** (interval −0.16 to −0.01) |
| Qualifying, after Sprint Qualifying | qualifying model and the Sprint Qualifying gap | 0.05 | MAE 0.885% → **0.875%** (interval −0.015 to −0.005) |
| Qualifying, before practice | qualifying model and the team's recent qualifying gap | 0.35 | MAE 0.790% → **0.781%** (interval −0.017 to −0.002) |

The front three get their own weight because when the finish model disagreed with pole, it was nearly always wrong (once right in 11 such races): with one weight for everyone, the forecast picked 52% of winners; with the front's own weight, 54%.

The explanation behind each shown number is still an exact sum: the model's own contributions scaled by its weight, plus the anchor's.

## Finishing position

**Predicts** where each driver finishes among the cars that finish; the odds add retirements separately (below). It's trained on finishers only, because a retirement isn't a 20th-place pace.

| Stage | Shown MAE | Baseline MAE | Baseline | Order agreement* |
|---|---|---|---|---|
| Before practice | **3.60** | 3.68 | driver's recent average finish | 0.60 |
| After practice | **3.56** | 3.68 | same | 0.61 |
| After qualifying | **3.05** | 3.29 | finish where you qualified | 0.69 |
| Official grid | **3.07** | 3.38 | finish where you start | 0.69 |

Among cars that finished (the order the app shows), MAE is **2.68** before practice (baseline 2.99) and **2.13** after qualifying (baseline 2.69). Training on 2018-2020 as well made it worse on the tuning races (+0.05 places), so it learns from 2021 on.

\* Mean Spearman correlation between predicted and actual order per race (1 = perfect, 0 = random).

**Honest weak spot:** after qualifying, "the pole-sitter wins" is still right more often (64% of these races) than the forecast's P1 (54%). The early races chose the front-of-grid weight (0.7) by average error; choosing a heavier one because it scores better on the later races would be grading our own exam.

**Leans on most:** recent results, grid slot, team's recent results, grid slot vs usual form, team's qualifying pace, best practice lap.

**Limits.** Races are noisy: the grid explains most of the order. It can't see crashes, strategy calls or mid-race weather.

## Qualifying

**Predicts** each driver's gap to pole as a % of the pole lap. A percentage compares circuits fairly: 0.7 s is a big gap at Monaco and a small one at Spa. It sees only what is known before qualifying: no grid, no qualifying times, and no weather (the forecast is for Sunday's race).

| Stage | Shown MAE | Baseline MAE | Baseline |
|---|---|---|---|
| Before practice | **0.781%** | 0.796% | team's recent qualifying gap |
| After practice | **0.740%** | 0.796% | same |
| After Sprint Qualifying | **0.875%** | 0.949% | the driver's Sprint Qualifying gap |

These cover 76 races from mid-2023 (the averages include crashes and no-time runs). On laps that were real attempts (within 107% of pole), the after-practice error is **0.521%** (baseline 0.577%). It learns from 2018: on the same races, 2018-2020 cut the after-practice error by 0.013% (interval −0.022 to −0.005). About 4% of qualifying results are crashes or no-time runs; their chance enters the odds instead (below).

Since 2026-10-11 it also reads each driver's **own** recent qualifying pace and how they've been qualifying against their teammate (both from earlier races only). Pole is usually decided between teammates, and team pace alone can't split them. That made the before-practice forecast beat its baseline with an interval clear of zero (−0.026 to −0.002); before, it was only level. The same inputs made the finishing-position model worse, so only qualifying uses them.

**Honest weak spot:** calling the exact pole-sitter is still hard (about a third of races), even though the predicted order matches the real one closely.

**Leans on most:** team's qualifying pace, the driver's own qualifying pace, best practice lap.

## Places gained

**Predicts** grid slot minus finishing position (positive means places gained), trained on finishers.

| Stage | MAE | Baseline MAE | Baseline |
|---|---|---|---|
| After qualifying | **3.20** | 3.38 | no change from the grid |
| Official grid | **3.09** | 3.38 | same |

Without a grid there's nothing to gain places from, so before qualifying it only matches the baseline. Among finishers, after qualifying: 2.33 vs 2.79. Learns from 2021 on.

## Gap to the winner

**Predicts** each finisher's gap to the winner as a % of the winner's race time. Lapped cars count: a car one lap down is about one average lap behind. Retirements have no gap and aren't predicted.

| Stage | MAE | Baseline MAE | Baseline |
|---|---|---|---|
| Before practice | **0.66%** | 0.85% | typical gap in past races |
| After qualifying | **0.66%** | 0.85% | same |

On a 90-minute race, 0.66% is about 36 seconds. It learns from 2022 on: adding 2021 (a different car generation) made it clearly worse on the early races (0.536% → 0.646%).

**Honest weak spot: unstable tuning.** Its held-out error has drifted 0.627% → 0.641% → 0.657% over the last three retrains, each time because the hyperparameter search settled on a different combination after a small data change (the latest: 2018 Monza recovered). The no-regression check judged the latest one not clearly worse than its predecessor (−0.010 to +0.067 on the last 24 races), so it shipped; a steadier search is on the to-do list.

## Sprint weekends

A sprint weekend runs one practice session, then Sprint Qualifying, the Sprint, Grand Prix qualifying and the race; the forecast updates after each. The models read a calendar flag (one practice session, not three), the Sprint Qualifying gap, the Sprint result and Sprint pace, blank on other weekends; the two sprint stages are trained and scored on sprint weekends only. A Sprint Qualifying lap more than 7% off the fastest counts as "no representative lap". The 2022 format (Friday qualifying set the Sprint grid) is left out.

On the held-out sprint weekends (15 for finishing position, 20 for qualifying since mid-2023):

| Stage | Finishing position MAE | Qualifying gap MAE |
|---|---|---|
| After practice (FP1) | 3.73 | 0.895% |
| After Sprint Qualifying | 3.73 | **0.875%** |
| After the Sprint | **3.64** | 0.875% |
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
| 2018-2020 for qualifying | Tuning races: after practice −0.019% (−0.036 to −0.002); held-out −0.013% (−0.022 to −0.005) | yes |
| 2018-2020 for finishing position | Tuning races: +0.05 places worse (+0.017 to +0.099) | no |
| Dropping >107% qualifying results from training | Slightly worse even on representative laps | no |

## Derived outputs (not extra models)

**Win / podium / points odds, likely range, retirement risk, beats-teammate, head-to-head** (`src/models/probabilities.py`)
- 10,000 simulated races, each a random finishing order drawn around the shown ranking.
- **Retirements: one rate for the whole field**, the share of starters who retired over the last 48 races (16% now). Until 2026-10-11 each car got its driver's and team's recent rate, but that scored *worse* than one rate for everyone (Brier 0.1321 vs 0.1297): a car's ten or twenty recent races are mostly noise. So the app no longer pretends to know which car is fragile.
- **How spread out the order is ("temperature")** is chosen, separately per stage and on the early races only, for the probabilities it produces: one for win and podium, one for the top ten. Until 2026-10-11 the win temperature was fitted to explain the whole top three, which under-stated favourites: drivers given 40-60% won 87% of the time. On the same held-out predictions, the change cut the win error (Brier) after qualifying from 0.0300 to **0.0284**, podium 0.0673 → **0.0665**, top ten 0.1507 → **0.1481**; earlier stages are unchanged.
- What the win odds say vs what happened, after qualifying (52 held-out races): drivers given 5-10% won 4% of the time, 10-20% won 9%, 20-40% won 26%, 40% or more won 65%. Favourites are still slightly under-stated, much less than before.
- The likely range has its own temperature, fitted so the P10-P90 range holds the result 80% of the time: on the held-out races it holds 80-82% (it was 87% in early October: needlessly wide).

| After qualifying | Brier score | Guessing evenly |
|---|---|---|
| Win | **0.028** | 0.046 |
| Podium | **0.067** | 0.124 |
| Points | **0.148** | 0.250 |

**Pole / Q3 / out-in-Q1 odds**
- The same method on the qualifying model (temperature chosen for pole and Q3), plus each driver's chance of no representative lap (recent record, field rate about 4%).
- Brier score after practice: pole 0.038, reaching Q3 0.133 (guessing evenly: 0.25); on the same predictions the previous calibration scored 0.040 and 0.134.

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
