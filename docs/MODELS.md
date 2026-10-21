# Model cards

What each model predicts, what it looks at, how accurate it is on races it had never seen, and where it falls short. Numbers are from the training run of 2026-10-11, phase 10 (data through the 2026 Bahrain GP at Kuala Lumpur).

**Model families (since 2026-10-11).** Any kind of model may now serve a target if it wins the same contest (`src/models/compare.py`, one script per decision in `experiments/`, results in `src/models/saved/leaderboard.json`):

| Target | Model | Why |
|---|---|---|
| Finishing position | XGBoost 0.3 + LightGBM 0.7 (an average) | better on the tuning races at every stage; held-out before practice 3.57 vs 3.60 places, interval below zero |
| Qualifying | XGBoost 0.55 + LightGBM 0.45 | held-out after practice 0.731% vs 0.740%, interval below zero |
| Places gained | XGBoost | LightGBM added nothing |
| Gap to the winner | two parts: a classifier ("lapped or not?") weights two regressions by its probability | lapped cars were 26% of finishers but 45% of the error; better on the tuning races at every stage |

Every family explains itself as an exact sum (TreeSHAP per tree model; an average or a probability-weighted mix of exact sums is exact), checked by `tests/test_families.py` and `tests/test_blend_exact.py`. Each model learns from the seasons that helped it on the tuning races: qualifying from 2018, finishing position and places gained from 2021, gap to the winner from 2022. The source of truth is `src/models/saved/*_metrics.json` and `blend.json`; the API serves them at `GET /model`.

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

Five predictions are a weighted average of a model and a simple anchor, each weight chosen on the early races only (`src/models/blend.py`):

| What | Mix | Weight | Effect on the held-out races |
|---|---|---|---|
| Finish, after qualifying | finish model and "grid minus predicted places gained" | **0.75 for the front three grid slots**, 0.5 for the rest | MAE 3.16 → **3.05** (interval −0.15 to −0.08); winner picked 54% → **60%**; podium 2.00 → **2.04** of 3 |
| Finish, after the Sprint | finish model and Sprint result | 0.15 | MAE 3.67 → **3.60** (interval −0.15 to +0.01) |
| Qualifying, after Sprint Qualifying | qualifying model and the Sprint Qualifying gap | 0.05 | MAE 0.864% → **0.855%** (interval −0.013 to −0.005) |
| Qualifying, before practice | qualifying model and the team's recent qualifying gap | 0.35 | together with the teammate split below: MAE 0.784% → **0.776%** (interval −0.014 to −0.001) |
| Qualifying, before and after practice: **teammate split** | the model's average for each pair of teammates stays; the gap between them moves toward the drivers' ratings from past teammate gaps (`src/features/ratings.py`) | 0.4 | average error unchanged after practice (±0.002); it changes *who* is ahead within a team, which is what decides pole: per-season pole calls rose in 6 of 9 seasons (2026: 31% → 38%) |

The front three get their own weight because when the finish model disagreed with pole, it was nearly always wrong (once right in 11 such races): with one weight for everyone, the forecast picked 52% of winners; with the front's own weight, 54%.

The explanation behind each shown number is still an exact sum: the model's own contributions scaled by its weight, plus the anchor's.

## Finishing position

**Predicts** where each driver finishes among the cars that finish; the odds add retirements separately (below). It's trained on finishers only, because a retirement isn't a 20th-place pace.

| Stage | Shown MAE | Baseline MAE | Baseline | Order agreement* |
|---|---|---|---|---|
| Before practice | **3.58** | 3.68 | driver's recent average finish | 0.60 |
| After practice | **3.54** | 3.68 | same | 0.61 |
| After qualifying | **3.05** | 3.29 | finish where you qualified | 0.70 |
| Official grid | **3.06** | 3.38 | finish where you start | 0.69 |

Among cars that finished (the order the app shows), MAE is **2.65** before practice (baseline 2.99) and **2.12** after qualifying (baseline 2.69). Training on 2018-2020 as well made it worse on the tuning races (+0.05 places), so it learns from 2021 on.

\* Mean Spearman correlation between predicted and actual order per race (1 = perfect, 0 = random).

**Honest weak spots:** after qualifying, "the pole-sitter wins" is still right a little more often (64% of these races) than the forecast's P1 (60%). And in the first four rounds of a rules reset (2026, in the held-out window) it is worse than "finish where you qualified" (3.61 vs 3.38): form from the old cars still weighs too much (plan item 2.2). The early races chose the front-of-grid weight (0.7) by average error; choosing a heavier one because it scores better on the later races would be grading our own exam.

**Leans on most:** recent results, team's recent results, grid slot, team's qualifying pace, grid slot where results follow the grid, grid slot vs usual form. Its circuit facts (safety cars, retirements, how closely results follow the grid) are measured from earlier races at the circuit since 2026-10-11, not typed by hand.

**Limits.** Races are noisy: the grid explains most of the order. It can't see crashes, strategy calls or mid-race weather.

## Qualifying

**Predicts** each driver's gap to pole as a % of the pole lap. A percentage compares circuits fairly: 0.7 s is a big gap at Monaco and a small one at Spa. It sees only what is known before qualifying: no grid, no qualifying times, and no weather (the forecast is for Sunday's race).

| Stage | Shown MAE | Baseline MAE | Baseline |
|---|---|---|---|
| Before practice | **0.776%** | 0.796% | team's recent qualifying gap |
| After practice | **0.731%** | 0.796% | same |
| After Sprint Qualifying | **0.855%** | 0.949% | the driver's Sprint Qualifying gap |

These cover 76 races from mid-2023 (the averages include crashes and no-time runs). On laps that were real attempts (within 107% of pole), the after-practice error is **0.511%** (baseline 0.577%). It learns from 2018: on the same races, 2018-2020 cut the after-practice error by 0.013% (interval −0.022 to −0.005). About 4% of qualifying results are crashes or no-time runs; their chance enters the odds instead (below).

Since 2026-10-11 it also reads each driver's **own** recent qualifying pace and how they've been qualifying against their teammate (both from earlier races only). Pole is usually decided between teammates, and team pace alone can't split them. That made the before-practice forecast beat its baseline with an interval clear of zero (−0.026 to −0.002); before, it was only level. The same inputs made the finishing-position model worse, so only qualifying uses them.

**Honest weak spot:** calling the exact pole-sitter is still hard (about four races in ten), even though the predicted order matches the real one closely. Pole odds of 20-40% came true 46% of the time: too cautious, a calibration to-do.

**Leans on most:** team's qualifying pace, the driver's own qualifying pace, best practice lap.

## Places gained

**Predicts** grid slot minus finishing position (positive means places gained), trained on finishers.

| Stage | MAE | Baseline MAE | Baseline |
|---|---|---|---|
| After qualifying | **3.20** | 3.38 | no change from the grid |
| Official grid | **3.08** | 3.38 | same |

Without a grid there's nothing to gain places from, so before qualifying it only matches the baseline. Among finishers, after qualifying: 2.32 vs 2.79. Like finishing position, it reads circuit facts measured from earlier races. Learns from 2021 on.

## Gap to the winner

**Predicts** each finisher's gap to the winner as a % of the winner's race time. Lapped cars count: a car one lap down is about one average lap behind. Retirements have no gap and aren't predicted.

| Stage | MAE | Baseline MAE | Baseline |
|---|---|---|---|
| Before practice | **0.68%** | 0.85% | typical gap in past races |
| After qualifying | **0.64%** | 0.85% | same |

On a 90-minute race, 0.64% is about 35 seconds. Since 2026-10-11 it's a two-part model (see the top): a classifier gives each car's chance of finishing a lap or more down, and the answers of a lead-lap regression and a lapped-car regression are mixed by that chance. It learns from 2022 on: adding 2021 (a different car generation) made it clearly worse on the early races (0.536% → 0.646%).

**Fixed weak spot: unstable tuning.** Its held-out error drifted 0.627% → 0.641% → 0.657% over three retrains because the hyperparameter search ran every week and settled on different settings after small data changes. Since 2026-10-11 weekly retrains reuse the stored settings; the search runs monthly or on request.

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

## Changes tested on 2026-10-11 (phase 10)

Scripts in `experiments/`; "tuning" = the early races that choose, "held-out" = the later races that check.

| Change | Effect | Shipped |
|---|---|---|
| Finishing position: XGBoost + LightGBM average | Tuning better at every stage; held-out before practice 3.574 vs 3.605 (−0.058 to −0.001) | yes |
| Qualifying: XGBoost + LightGBM average | Held-out after practice 0.731% vs 0.740% (−0.015 to −0.004) | yes |
| Places gained: XGBoost + LightGBM | ±0.003 on the tuning races | no |
| Gap to the winner, two parts mixed by probability | Tuning 0.512% vs 0.527% after qualifying; held-out 0.650% vs 0.657% (n.s.) | yes |
| Gap to the winner, two parts with a hard switch | Worse on both windows (a wrong switch costs a lap) | no |
| Circuit facts measured from earlier races instead of typed | Finishing position and places gained better on the tuning races, held-out level; qualifying and gap to the winner worse | for the first two only |
| Forecasts at the lead time each stage really has (3 days before practice, 1 day on Saturday) | Held-out level or slightly better everywhere (finish after qualifying −0.003, interval below zero) | yes (a train/serve fix) |
| Practice pace per session (weighted, or latest session) | Qualifying worse on the tuning races with both | no |
| Teammate split blend (driver ratings) | See the blend table above | yes |
| Retirement chance leaning with the grid slot after qualifying | Brier 0.1302 → **0.1274** (−0.0039 to −0.0016) | yes |
| Title odds use each driver's pre-practice strength (not this weekend's grid) | Correctness fix: Singapore 2026, Russell's expected final points 255 → 299 after his grid penalty stopped counting in every remaining round | yes |

On every retrain each metrics file also gets a **segment table** (held-out, at the shown stage): era-start rounds, wet forecasts, street circuits, sprint weekends, rookies, drivers new to their team. Each is compared with the same naive baseline, and `saved/model_history.json` keeps every retrain's shown accuracy so drift is visible.

## Derived outputs (not extra models)

**Win / podium / points odds, likely range, retirement risk, beats-teammate, head-to-head** (`src/models/probabilities.py`)
- 10,000 simulated races, each a random finishing order drawn around the shown ranking.
- **Retirements: one rate for the whole field**, the share of starters who retired over the last 48 races (16% now), **leaning with the grid slot once there is a grid** (a logistic curve with one slope: about 11% at the front to 20% at the back; held-out Brier 0.1274 vs 0.1302 for the flat rate). Until 2026-10-11 each car got its driver's and team's recent rate, but that scored *worse* than one rate for everyone (Brier 0.1321 vs 0.1297): a car's ten or twenty recent races are mostly noise. So the app no longer pretends to know which car is fragile.
- **How spread out the order is ("temperature")** is chosen, separately per stage and on the early races only, for the probabilities it produces: one for win and podium, one for the top ten. Until 2026-10-11 the win temperature was fitted to explain the whole top three, which under-stated favourites: drivers given 40-60% won 87% of the time. On the same held-out predictions, the change cut the win error (Brier) after qualifying from 0.0300 to **0.0284**, podium 0.0673 → **0.0665**, top ten 0.1507 → **0.1481**; earlier stages are unchanged.
- What the win odds say vs what happened, after qualifying (53 held-out races, shown on the History page since 2026-10-21): drivers given 5-10% (7% on average) won 5% of the time, 10-20% (14%) won 7%, 20-40% (28%) won 28%, 40% or more (50%) won 63%. Favourites are still slightly under-stated, much less than before; the 10-20% group is 44 drivers, so its gap is about three wins.
- **Each finishing place's chance** (since 2026-10-21, the driver page): the share of the simulated races in which the car finished in each place. A retirement counts as last, which is why the bars rise again at the back.
- **What if they started elsewhere?** (`src/models/whatif.py`, `GET /whatif`): the refresh job predicts each driver's finish from every grid slot (the other cars unchanged), and the API re-runs only the odds simulation for the chosen slot. Before qualifying, the other qualifying inputs are the model's own predictions. There is no rain what-if: the models barely react to the rain forecast, so it would show almost nothing.
- The likely range is where the car finishes **if it finishes** (its retirement chance is shown separately): with every car's retirement chance at 16%, a range that counted retirements would end at last place for everyone. Its own temperature is fitted so the P10-P90 range holds a finisher's result 80% of the time; on the held-out races it holds 80-82%.

| After qualifying | Brier score | Guessing evenly |
|---|---|---|
| Win | **0.028** | 0.046 |
| Podium | **0.067** | 0.124 |
| Points | **0.148** | 0.250 |

(After the phase-10 retrain: win 0.0277, podium 0.0667, points 0.1479.)

**The whole finishing order, not just three yes/no events** (since 2026-10-21, every retrain; `checks` in `pl_calibration.json`):

| Held-out races | After practice | After qualifying | Everyone equal |
|---|---|---|---|
| Ranked probability score (finishing order; lower is better, and close misses cost less than far ones) | 0.124 | **0.111** | 0.175 |
| Log loss, win / podium / points | 0.129 / 0.266 / 0.507 | **0.091 / 0.222 / 0.455** | 0.194 / 0.415 / 0.693 |
| Calibration error, win / podium / points (average gap between stated and actual rates) | 1.2 / 2.8 / 3.5 points | 1.4 / 2.7 / 2.6 points | |

Qualifying after practice: log loss pole 0.134, Q3 0.415; calibration error 2.1 and 2.7 points.

**Pole / Q3 / out-in-Q1 odds**
- The same method on the qualifying model (temperature chosen for pole and Q3), plus each driver's chance of no representative lap (recent record, field rate about 4%).
- Brier score after practice: pole 0.039, reaching Q3 0.131 (guessing evenly: 0.25).

**Predicted qualifying lap and race length** (`src/models/estimates.py`)
- Pole lap = fastest practice lap × this circuit's usual ratio of pole to fastest practice lap.
- Race length = laps × fastest practice lap × this circuit's usual ratio of winner's time to that product.

**Safety-car chance**: the share of past races at the circuit with a safety car or VSC, shrunk toward the all-circuit rate when there are only a few.

**Tyre strategy** (`src/strategy/model.py`): fitted per circuit on real race laps since 2022 (the 18-inch tyres); every legal one- and two-stop plan costed, 2,000 races with random safety cars. Limits: linear tyre wear, no traffic.

**Championship odds** (`src/agent/scenarios.py`): the remaining races and sprints simulated 5,000 times with real points and every driver's strength from this weekend's forecast **before practice** (form, team, circuit). Until 2026-10-11 it used the shown forecast, which after qualifying includes the grid, so one weekend's pole or penalty counted in every remaining round. Precomputed by the refresh job (`data/predictions/championship.json`). Ties on points go to FIA countback (more wins, then more second places, down to fifths; since 2026-10-21).
- **Tested and not switched on** (`experiments/title_odds_backtest.py`, the odds re-run after every round of 2023-2025 with nothing from later): each driver's strength at each remaining venue instead of one for the season, and team form drifting round to round. Each won one test window and lost the other: venue strengths scored better in 2023-24 and worse in 2025, drift halved the constructors' error in 2023-24 (McLaren's 2024 rise) and hurt in 2025. Three seasons can't separate a real gain from one season's story.

**Driver ratings** (`src/features/ratings.py`, `GET /ratings`, the Title race page): each driver's one-lap pace against an average driver in % of a lap, from every past teammate qualifying gap fitted at once, recent races counting more.

## The chat (Pit Radio)

Two models, neither of which ever produces a number itself.

**The question classifier** (`src/agent/router.py`, weights in `src/agent/router_model.json`)
- **What it does:** sorts a question into one of 14 fixed kinds, each answered by a plain data lookup, or "open".
- **How it's built:** TF-IDF over words and word pairs (3,000 features) plus logistic regression. Trained on 22,027 questions labelled by the first lookup the teacher model chose for them; history, records and background count as "open".
- **Accuracy:** measured on a fifth of the questions held out by conversation (`experiments/router_classifier.py`).
  - On free-form questions, the keyword rules it replaced were right 53% of the time; the classifier is right **91.5%** of the time.
  - Confidently wrong answers (a fixed answer to the wrong kind of question) fall from **44%** to **2.8%**.
  - The rules' typical failure was a history question containing "win" getting this weekend's forecast.
- **Unusual wording** (since 2026-10-21, `experiments/router_paraphrases.py`): the generated questions were tidy, so a model rewrote 2,976 of them three ways each (casual or misspelled, indirect, second-language or slang; 8,933 in all, $0.31) and they joined the training data, split by source question. On reworded test questions, fixed questions recognised **18% → 65%** (accuracy 51% → 78%); on the original free-form questions accuracy held (91.4% → 91.7%), confidently wrong 2.5% → 3.8%.
- **Without a local model:** it takes the likeliest fixed kind when it's above 30% likely (re-tuned after the reworded questions), since an open question then only gets a generic reply.
- **Follow-ups:** a short "and Leclerc?" keeps the previous question's kind.
- **Overrides:** two exact rules stay. An article number goes to the rules, and "why" with "predicted" goes to the prediction breakdown (124 of 129 such questions wanted it).
- **Limit:** a third of reworded fixed questions still land in "open". That's harmless when the local model runs; on the serverless app they get the generic reply.

**Search** (`src/rag/corpus.py`, BM25 over Wikipedia and over the FIA rules, rulings and circuit notes)
- **Accuracy:** for questions naming a driver, team or circuit, that entity's own Wikipedia page is among the 5 hits the model reads **80%** of the time, and first **70-73%** of the time.
  - That holds for the raw question and for the queries the teacher model wrote (`experiments/retrieval_eval.py`: 79 held-out questions, 928 teacher queries).
  - Misses are mostly broad questions where a related page outranks the entity's own.
- **Rules search** (since 2026-10-21; only 23 questions, labelled by the article a checked answer cited): the teacher's keyword query finds it among the 5 hits **83%** of the time, the user's raw question 61% (casual words like "ballast" for "minimum mass"). The chat model writes keyword queries, so it is fine; without the model the app searches the raw question and is weaker.
- **Decision:** no embeddings yet; the plan adds them only if this falls short.

**The chat model, `f1-analyst`** (`training/README.md` for how it's built)
- **Base:** Qwen3.5 9B (Apache 2.0).
- **Training:**
  - LoRA rank 32 on every attention and MLP projection, bf16, 2 passes.
  - Data: 24,997 conversations whose lookups ran on the app's real data. The answers were written by gpt-oss-120b and kept only if a code check and a judge model passed them.
  - Hardware: one L40S, 16.1 h, about $34.
  - It learned only its own turns (lookups and answers). Validation loss fell from 0.365 to 0.294.
- **Served:** Q4_K_M in Ollama (5.8 GB) on the owner's PC (4 GB GPU, 16 GB RAM), at about 17 s per answer once loaded. It gets up to 4 lookups, then is told to answer with what it has.
- **Held-out results:** the same 200 questions the training data never saw, played through the app's own chat loop. Scored by the training data's code checks and one judge from another model family (Kimi K2.5); `training/eval_chat.py`.

| | Untuned Qwen3.5 9B | f1-analyst |
|---|---|---|
| Right first lookup (97 questions with a known one) | 79 | **97** |
| Passed the code checks and the judge | 89 | **145** |
| Trick questions handled (40: false premises, invented facts, news, off-topic) | 16 | **32** |
| Answer shown as written: passed the app's fact check | 137 | **179** |
| An invented number or name (the app shows the plain answer instead) | 33 | 20 |
| Talked about "tools" or "functions" in the answer | 17 | 0 |
| Answered a factual question without looking anything up | 27 | 1 |
| No answer | 6 | 2 |
| Median time | 16.7 s | 16.8 s |

Two notes on fairness:
- **The "answer now" turn:** the untuned model ran without it; it was added after the fine-tuned model's first run, in which 31 of 200 answers stopped at 4 lookups, still refining searches.
- **A scoring fix:** until 2026-10-20 the code checks also failed correct refusals ("I only cover F1") of the evaluation set's off-topic, news and future questions, for every model. Both columns are re-scored with the fix.

**All 571 held-out questions** (2026-10-21, f1-analyst only, judge $0.26). The 200 above were the first 200 of the set: every trick question and most of the fixed-kind ones (97 of 172), but only 63 of the 359 open ones. The full set is mostly open questions, which are harder, so it scores lower:

| | First 200 | Other 371 | All 571 |
|---|---|---|---|
| Passed the code checks and the judge | 145 (72.5%) | 241 (65.0%) | **386 (67.6%)** |
| Shown as written (passed the app's fact check) | 179 | 325 | **504 (88.3%)** |
| An invented number or name | 20 | 42 | 62 (10.9%) |
| Answered without a lookup | 1 | 6 | 7 |
| No answer | 2 | 3 | 5 |

- **By kind:** right first lookup on 170 of 172 fixed-kind questions; passed 141/172 fixed (82%), 32/40 tricks (80%), 213/359 open (59%).
- **Weakest open kinds** all lean on the knowledge search: technical explanations 6/26 (downforce, ground effect, the hybrid unit; the corpus has race, driver and team pages but few engineering ones, so the model fills in from memory and the judge fails it as unsourced, sometimes wrong), team histories 8/26 (lineage like Brawn → Mercedes isn't in the team lookup, so a correct "Mercedes" fails the fact check), rules explained 13/29, comparisons and biographies 50%.
- **Time:** median 20.9 s (16.8 s on the first 200; the PC was also running builds and tests during the other 371).

- **Limits:**
  - Style: it sometimes cites sources in the teacher's bracket style ("【source: ...】"); since 2026-10-21 the app rewrites those as plain words.
  - Trick questions: 8 of 40 still go wrong. Two each are a statistic the data doesn't hold, a question about the future, and a rules question; one is a false premise answered rather than corrected; one needs arithmetic.
  - Fact check: about 1 answer in 9 contains a number or name the lookups didn't return. The app catches these and shows the plain data answer.
  - Knowledge questions: open questions pass 59% of the time, technical explanations 23% (see "All 571" above).
  - Freshness: it knows only what the tools return. News, rumours and anything after the last data refresh get "I don't have that".

## Earlier changes (2026-09-28)

| Change | Effect |
|---|---|
| Training on every weekend stage | Before qualifying the old model scored 4.09 places MAE, worse than the naive 3.62. Then 3.71. |
| Forecast weather instead of measured race weather | Training sees what a live forecast sees |
| Previous rules-era form weighted 0.1 | After-qualifying MAE 3.287 → 3.263 |
| Long-run practice pace as a feature | Tried; no gain |
| A ranking objective (`rank:pairwise`) | Tried twice (also 2026-10-10: winner 50-52%); no better than the blend |
