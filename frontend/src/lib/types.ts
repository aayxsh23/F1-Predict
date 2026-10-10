export type Target = 'qualifying' | 'finish_position' | 'quali_delta' | 'race_time'

export const TARGETS: Target[] = ['finish_position', 'qualifying', 'quali_delta', 'race_time']

export const TARGET_LABELS: Record<Target, string> = {
  qualifying: 'Qualifying gap',
  finish_position: 'Finishing position',
  quali_delta: 'Places gained',
  race_time: 'Gap to winner',
}

/** What the number means, in a sentence, for the "Why" tab's scale. */
export const TARGET_MEANING: Record<Target, string> = {
  qualifying: 'Gap to pole as a % of the pole lap (0.5% is about half a second)',
  finish_position: 'Finishing position',
  quali_delta: 'Places gained from the starting grid',
  race_time: "Gap to the winner as a % of the winner's race time",
}

export const TARGET_UNITS: Record<Target, string> = {
  qualifying: '%',
  finish_position: '',
  quali_delta: '',
  race_time: '%',
}

// true when a HIGHER value is the better outcome (places gained); for the other
// three a lower number is better. Governs whether a contribution reads as good.
export const TARGET_HIGHER_IS_BETTER: Record<Target, boolean> = {
  qualifying: false,
  finish_position: false,
  quali_delta: true,
  race_time: false,
}

export type Stage = 'pre_weekend' | 'post_practice' | 'post_sprint_quali' | 'post_sprint' | 'post_quali' | 'race_day'

export interface KnownSessions {
  practice: boolean
  /** present only on sprint weekends */
  sprint_qualifying?: boolean
  sprint?: boolean
  qualifying: boolean
  /** the FIA's official grid (penalties applied) is in; before that the qualifying order stands in */
  grid: boolean
  /** older forecasts only: starting tyres stopped being an input on 2026-10-10 */
  compound?: boolean
}

export interface DriverPrediction {
  driver: string
  team: string
  driver_number?: number | null
  grid_position: number | null
  quali_gap_to_pole: number | null // seconds, once qualifying has happened
  quali_gap_pct?: number | null
  practice_pace: number | null
  practice_long_run_pace?: number | null
  predicted_qualifying_gap_pct?: number | null
  predicted_qualifying_gap: number | null // seconds behind the predicted pole
  predicted_quali_lap_s?: number | null
  predicted_finish_position: number | null
  predicted_quali_to_race_delta: number | null
  predicted_race_gap_pct?: number | null
  predicted_race_time_gap: number | null // seconds behind the predicted winner
  feature_row: Record<string, number | string | null>
  // added by the API when served (src/api/enrich.py)
  probabilities?: { win: number; podium: number; top10: number }
  position_band?: [number, number]
  expected_position?: number
  retire_risk?: number
  beats_teammate?: number | null
  quali_odds?: { pole: number; q3: number; q1_out: number }
}

export interface RaceInfo {
  laps: number | null
  expected_duration_s: number | null
  pole_time_estimate_s: number | null
  safety_car_probability: number | null
}

export interface PredictionPayload {
  season: number
  round: number
  location: string
  event_name?: string
  race_start_utc?: string | null
  generated_at: string
  stage?: Stage
  session_label?: string
  sprint_weekend?: boolean
  known_sessions: KnownSessions
  model_trained_at?: string | null
  race?: RaceInfo
  drivers: DriverPrediction[]
}

export interface ShapContribution {
  feature: string
  label?: string
  value: number | string | null
  shap: number
}

export interface ShapExplanation {
  driver: string
  target: Target
  predicted_value: number
  base_value: number
  top_contributions: ShapContribution[]
}

export interface ExplainResult {
  prediction: number
  target: Target
  circuit: string
  top_features: Array<{ feature: string; shap_value: number; phrase: string }>
  retrieval_query: string
  sources: string[]
  explanation: string
}

export interface TimelineSnapshot {
  label: string
  stage: Stage
  generated_at: string
  drivers: Array<{ driver: string; predicted_finish_position: number | null }>
}

export interface BacktestIndexEntry {
  season: number
  round: number
  location: string
}

export interface BacktestTargetResult {
  actual: number | null
  predicted: number | null
}

export interface BacktestDriverEntry {
  driver: string
  team: string
  qualifying: BacktestTargetResult // % of the pole lap
  finish_position: BacktestTargetResult
  quali_delta: BacktestTargetResult
  race_time: BacktestTargetResult // % of the winner's race time
  driver_number?: number | null
}

export interface BacktestRacePayload {
  season: number
  round: number
  location: string
  method?: string
  pole_time_s?: number | null
  race_duration_s?: number | null
  drivers: BacktestDriverEntry[]
}

export interface StageAccuracy {
  mae: number
  baseline_mae: number
  n: number
  mean_race_spearman?: number
}

export interface ModelCardEntry {
  unit: string
  baseline: string
  evaluation: string
  stages: Partial<Record<Stage, StageAccuracy>>
  data_through: string
  trained_at: string
  top_features: Array<{ feature: string; label: string; weight: number }>
}

export type ModelCard = Record<Target, ModelCardEntry>

export interface LiveSnapshotScore {
  label: string
  stage: Stage | null
  generated_at: string | null
  cars: number
  finish_mae: number | null
  finish_mae_finishers: number | null
  order_agreement: number | null
  winner_called?: boolean
  podium_called?: number
  grid_baseline_mae?: number | null
  quali_mae?: number | null
  pole_called?: boolean
}

/** GET /live-record: what the app actually published, scored once the race was run. */
export interface LiveRecord {
  method: string
  generated_at: string
  races: Array<{ season: number; round: number; location: string; snapshots: LiveSnapshotScore[] }>
  by_stage: Array<{ label: string; races: number; finish_mae: number | null; finish_mae_finishers: number | null; winner_called_share: number | null; grid_baseline_mae?: number | null; quali_mae?: number | null }>
}

export type Compound = 'SOFT' | 'MEDIUM' | 'HARD'

export interface Stint {
  compound: Compound
  from_lap: number
  to_lap: number
  laps: number
}

export interface StrategyPlan {
  name: string
  stops: number
  pit_laps: number[]
  pit_windows: Array<[number, number]>
  stints: Stint[]
  time_vs_best_s: number
  chance_fastest: number
}

export interface StrategyResult {
  season: number
  round: number
  circuit: string
  laps: number
  pit_loss_s: number
  safety_car_probability: number
  races_of_data: number
  compounds: Record<Compound, { pace_vs_medium_s: number; deg_s_per_lap: number; max_stint_laps: number }>
  strategies: StrategyPlan[]
  scenario: null | {
    sc_lap: number
    best: string
    pit_laps: number[]
    stints: Stint[]
    pits_under_safety_car: boolean
    gain_vs_sticking_to_plan_s: number
  }
}

export interface ChampionshipDriver {
  code: string
  given_name: string
  family_name: string
  team: string | null
  points: number
  wins: number
  position: number
  title_chance: number
  expected_points: number
}

export interface ChampionshipTeam {
  name: string
  points: number
  wins: number
  position: number
  title_chance: number
}

export interface Championship {
  remaining: { races: number; sprints: number }
  drivers: ChampionshipDriver[]
  constructors: ChampionshipTeam[]
}

// --- chat ---

export interface ChatSource {
  filename: string
  title: string
  article: string | null
  doc_type: string
}

export interface ChatStep {
  id: string
  name: string
  done: boolean
  ok: boolean
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
  steps?: ChatStep[]
  sources?: ChatSource[]
  streaming?: boolean
  error?: boolean
}

export type ChatEvent =
  | { type: 'token'; text: string }
  | { type: 'tool_start'; id: string; name: string; args: Record<string, unknown> }
  | { type: 'tool_end'; id: string; name: string; ok: boolean }
  | { type: 'sources'; items: ChatSource[] }
  | { type: 'done' }
  | { type: 'error'; message: string }

export interface ChatContext {
  season?: number
  round?: number
  race_name?: string
  driver?: string
}

// one row of GET /races?season=, the season calendar
export interface RaceSummary {
  RoundNumber: number
  EventName: string
  Location: string
  Country: string
  EventFormat: string
  Session1DateUtc: string | null
  EventDate: string | null
  RaceStartUtc?: string | null
}

export type RegulationDocType = 'regulation' | 'steward_decision'

export interface RegulationDocument {
  filename: string
  doc_type: RegulationDocType
  title: string
  size_bytes: number
  modified_at: string
  grand_prix?: string
  car_number?: number
}

export interface RegulationDetail {
  filename: string
  text: string
}

export interface RegulationSearchHit {
  filename: string
  doc_type: RegulationDocType
  article: string | null
  snippet: string
}
