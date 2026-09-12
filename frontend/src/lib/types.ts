export type Target = 'qualifying' | 'finish_position' | 'quali_delta' | 'race_time'

export const TARGETS: Target[] = ['qualifying', 'finish_position', 'quali_delta', 'race_time']

export const TARGET_LABELS: Record<Target, string> = {
  qualifying: 'Qualifying gap to pole',
  finish_position: 'Finishing position',
  quali_delta: 'Grid-to-finish change',
  race_time: 'Race time gap to winner',
}

export const TARGET_UNITS: Record<Target, string> = {
  qualifying: 's',
  finish_position: '',
  quali_delta: '',
  race_time: 's',
}

// true when a HIGHER predicted value is the better outcome for this target
// (quali_delta: grid minus finish, so a bigger positive number means more
// positions gained). The other three are all "lower is better" (a smaller
// gap-to-pole, finishing position, or gap-to-winner). This governs whether a
// positive or negative SHAP contribution reads as "good" in the UI.
export const TARGET_HIGHER_IS_BETTER: Record<Target, boolean> = {
  qualifying: false,
  finish_position: false,
  quali_delta: true,
  race_time: false,
}

export interface KnownSessions {
  practice: boolean
  qualifying: boolean
  grid: boolean
  compound: boolean
}

export interface DriverPrediction {
  driver: string
  team: string
  grid_position: number | null
  quali_gap_to_pole: number | null
  practice_pace: number | null
  predicted_qualifying_gap: number | null
  predicted_finish_position: number | null
  predicted_quali_to_race_delta: number | null
  predicted_race_time_gap: number | null
  feature_row: Record<string, number | string | null>
}

export interface PredictionPayload {
  season: number
  round: number
  location: string
  generated_at: string
  known_sessions: KnownSessions
  drivers: DriverPrediction[]
}

export interface ShapContribution {
  feature: string
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

export interface TopFeature {
  feature: string
  shap_value: number
  phrase: string
}

export interface ExplainResult {
  prediction: number
  target: Target
  circuit: string
  top_features: TopFeature[]
  retrieval_query: string
  sources: string[]
  explanation: string
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
  qualifying: BacktestTargetResult
  finish_position: BacktestTargetResult
  quali_delta: BacktestTargetResult
  race_time: BacktestTargetResult
}

export interface BacktestRacePayload {
  season: number
  round: number
  location: string
  drivers: BacktestDriverEntry[]
}

export interface AgentResponse {
  reply: string
  conversation_id: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
}
