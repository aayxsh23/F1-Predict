# F1 Predict

**Forecasts every Formula 1 race weekend, explains why, and shows how accurate it has really been.**

![The "Why this prediction" tab: a plain-language summary, the inputs that pushed the prediction up or down, and the teammate comparison](docs/why-tab.png)

## What it does

For the next race, and for every race since 2022, the app shows:

| Question | What you see |
|---|---|
| Who will be on pole? | Predicted qualifying order and lap times, with each driver's chance of pole, of reaching Q3, and of going out in Q1 |
| Who will win? | Predicted finishing order, chances to win / finish on the podium / score points, a likely finishing range, retirement risk, and gap to the winner |
| Why? | The inputs that pushed each prediction up or down (recent results, grid slot, practice pace, circuit, weather forecast...), in plain words, plus a written explanation |
| What's the best strategy? | The fastest tyre plans, pit windows, and what a safety car on a given lap changes |
| Who wins the title? | Live standings with each driver's and team's chance of the championship |
| Anything else? | An analyst chat that answers from the app's own numbers and cites the FIA rules it uses |
| Can I trust it? | For every past race: what it predicted beforehand next to what happened, and its average error against simple guesses |

Forecasts refresh automatically through the weekend, getting sharper after each practice session and after qualifying.

## How it works

```
FastF1 (timing data) --+
Open-Meteo (forecasts) +--> per-race tables --> features --> 4 XGBoost models --+--> odds, ranges, lap times, race length
Jolpica (standings) ---+    (form, team, circuit,           (qualifying,       |--> strategy simulator (fitted on real stints)
                            practice, grid, weather)         finish, places     |--> championship simulation
                                                             gained, gap)       v
                             GitHub Actions: refresh every 30 min on race weekends,   JSON in data/predictions/
                             retrain weekly (only if the models still beat their      |
                             baselines)                                               v
                                                              FastAPI on Vercel --> React app
                                                              + Gemini analyst (LangGraph agent with 13 tools)
```

1. **Data.** One row per driver per race since 2022: grid, qualifying and practice times, the race-start weather forecast, result, gap to the winner, plus lap-by-lap tyre stints.
2. **Features.** Each driver's recent form, their history at this circuit, the team's form and pace, the teammate comparison and the circuit's characteristics. Every one is computed only from earlier races, so nothing leaks from the future.
3. **Four models, one per question.** XGBoost gradient-boosted trees for qualifying gap, finishing position, places gained and gap to the winner. Each serves every weekend stage: it was trained on copies of past races with the not-yet-known information hidden.
4. **Derived outputs.**
   - Odds and ranges come from simulating each race 10,000 times around the model's order.
   - Lap times and race length come from circuit history.
   - Strategy comes from a tyre-wear model fitted on real stints.
5. **Explanations.** SHAP values show how much each input moved each prediction. The analyst chat and the written explanations use Google Gemini with the app's own data as tools, and cite the regulation articles they use.

## Accuracy

Measured on the latest 43 races (Nov 2024 to Sep 2026). Each was predicted by a model trained only on earlier races, the way a live forecast is.

| Prediction | After qualifying | Simple guess |
|---|---|---|
| Finishing position | off by **3.26** places on average | 3.43 (finish where you start) |
| Qualifying gap (after practice) | **0.64%** of a lap (~0.6 s) | 0.70% (team's usual gap) |
| Gap to winner | **0.64%** of race time (~35 s) | 0.85% |
| Win odds (Brier, lower is better) | **0.033** | 0.047 (everyone equal) |

The full model cards, including where the models are weak, are in [docs/MODELS.md](docs/MODELS.md).

## Run it locally

Requires Python 3.12+ and Node 24.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt     # macOS/Linux: .venv/bin/pip
cp .env.example .env                                  # then add your GEMINI_API_KEY
.venv/Scripts/python -m uvicorn src.api.main:app --reload --env-file .env

cd frontend && npm ci && npm run dev                  # http://localhost:5173
```

Everything except the chat and written explanations works without a Gemini key.

| Task | Command |
|---|---|
| Forecast the next race now | `python -m src.models.refresh_job` |
| Ingest newly finished races | `python -m src.data.ingest --seasons 2026` |
| Rebuild the feature table | `python -m src.features.build_dataset` |
| Retrain and evaluate all four models | `python -m src.models.train` (about 10 min) |
| Refit odds, history view, strategy | `python -m src.models.probabilities`, `python -m src.models.backtest_export`, `python -m src.strategy.model` |
| Rebuild the regulations index | `python -m src.rag.corpus` |
| Chat in the terminal | `python -m src.agent.cli` |
| Tests | `python -m src.features.build_dataset` once, then `pytest`; `npm run lint && npm run build` in `frontend/` |

## Deploy to Vercel

One Vercel project serves both the app and the API (`vercel.json`: the frontend builds to static files, `api/index.py` runs FastAPI as a Python function).

1. Import the GitHub repo in Vercel, keeping the root directory as the repo root.
2. Add the environment variable `GEMINI_API_KEY`. Optionally add `GEMINI_MODEL`.
3. Deploy.

The scheduled GitHub Actions commit fresh forecasts to the repo, and each commit redeploys. To serve new forecasts without redeploying, set `PREDICTIONS_DATA_SOURCE` to the raw GitHub URL of `data/predictions` (see `.env.example`).

The API deliberately imports no pandas, XGBoost or torch. Everything heavy is precomputed by the scheduled jobs, which keeps the function around 180 MB (Vercel's cap is 250 MB) and its cold starts short.

## API

| Route | Returns |
|---|---|
| `GET /predictions/latest`, `/predictions/{season}/{round}` | The forecast with odds, ranges, qualifying odds and lap times |
| `GET /predictions/{season}/{round}/explain?driver=&target=` | The SHAP breakdown behind one prediction |
| `GET /predictions/{season}/{round}/timeline` | How the forecast moved through the weekend |
| `GET /strategy/{season}/{round}?sc_lap=` | Tyre strategies, pit windows, safety-car scenario |
| `GET /championship` | Standings with simulated title odds |
| `GET /backtest/races`, `/backtest/{season}/{round}`, `/backtest/summary` | Past races, predicted vs actual, and accuracy by season |
| `GET /model` | Held-out accuracy of each model |
| `GET /races?season=` | The calendar |
| `GET /regulations`, `/regulations/search?query=`, `/regulations/{file}` | FIA regulations and steward decisions |
| `POST /explain` | A written explanation of one prediction (Gemini) |
| `POST /chat` | The analyst, streamed as server-sent events |

## Documentation map

| Document | What's in it |
|---|---|
| [docs/MODELS.md](docs/MODELS.md) | Model cards: targets, inputs, accuracy, limits |
| [PRODUCT.md](PRODUCT.md) | Who it's for and the product principles |
| [frontend/DESIGN.md](frontend/DESIGN.md) | The visual design system |
| [AGENTS.md](AGENTS.md) | Rules for anyone (or any AI agent) changing the code |
| [LEARNING.md](LEARNING.md) | A tutorial on every concept and decision, written to learn from |
| [PROGRESS.md](PROGRESS.md) | The build log |
| [f1-race-predictor-explainer-project-plan.md](f1-race-predictor-explainer-project-plan.md) | The original plan and phase history |

Not affiliated with Formula 1, the FIA or any team. Timing data from FastF1, standings from Jolpica-F1, weather from Open-Meteo.
