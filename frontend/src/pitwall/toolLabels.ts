/** What each Pit Radio lookup is doing, in words a fan reads while waiting. */
export const TOOL_LABELS: Record<string, string> = {
  race_forecast: 'Reading the forecast',
  explain_prediction: 'Breaking down the prediction',
  head_to_head: 'Simulating the head-to-head',
  championship: 'Simulating the title race',
  title_scenario_for: 'Doing the title maths',
  race_strategy: 'Running the strategy model',
  forecast_timeline: 'Checking how the forecast moved',
  search_rules: 'Searching the rulebook',
  circuit_guide: 'Reading the circuit guide',
  driver_history: 'Looking up past results',
  past_race_prediction: 'Checking a past prediction',
  model_track_record: "Checking the model's track record",
  season_schedule: 'Checking the calendar',
}
