/**
 * Shapes for the generated data. These mirror what
 * tools/make_real_fixtures.py emits, which in turn mirrors what the Python
 * pipeline writes.
 *
 * `provenance` follows the four tags the simulator itself uses for every
 * parameter, so a figure on screen can always say where it came from:
 *   fitted   measured from real data
 *   swept    unmeasurable, reported across a range
 *   cited    published
 *   free     a design choice
 *   derived  computed here from measured values (arithmetic shown)
 */
export type Provenance = 'fitted' | 'swept' | 'cited' | 'free' | 'derived'

export interface StageTiming {
  stage: string
  status: 'ok' | 'fail'
  seconds: number
}

export interface Meta {
  profile: string | null
  population: number | null
  started: string | null
  finished: string | null
  stages: StageTiming[]
  total_seconds: number | null
  models: string
}

export interface Metric {
  pr_auc: number
  roc_auc: number
  recall_at_0p1: number
  recall_at_1: number
  precision_at_budget: number
  n_positives?: number
  n_total?: number
}

export interface DetectorConfig {
  id: string
  label: string
  family: 'rule' | 'flat' | 'mixture'
  note: string | null
  metrics: Metric
}

export interface ExpertWeight {
  name: string
  weight: number
  normalized_weight: number
}

export interface FeatureGain {
  name: string
  gain: number
  per_entity: boolean
}

export interface OperatingPoint {
  provenance: Provenance
  basis: string
  alert_budget: number
  precision: number
  recall: number
  f1: number
  true_positives: number
  n_positives: number
  note: string
}

export interface Bands {
  step_up: number
  hold: number
  decline: number
  block: number
}

export interface DetectorMetrics {
  source: string
  base_rate: number
  alert_budget: number
  train_rows: number | null
  train_fraud: number | null
  test_rows: number | null
  test_fraud: number | null
  configs: DetectorConfig[]
  experts: ExpertWeight[]
  feature_gains: FeatureGain[]
  operating_point: OperatingPoint | null
  per_entity_lift: number | null
  fitted_bands: Bands | null
}

/** One run-length-encoded segment of an action chain. */
export interface ActionRun {
  action: string
  times: number
}

export interface StrategySample {
  update: number
  count: number
  runs: ActionRun[]
  /** The log truncates long chains, so the tape shown may be shorter than the real one. */
  truncated: boolean
}

export interface CoadaptPoint {
  update: number
  extracted: number
  policyReturn: number
  entropy: number
  refit: boolean
}

export interface PosteriorLevel {
  label: string
  coef: number
  reference: boolean
}

export interface CoadaptMetrics {
  source: string
  rows: Array<[number, number, number, number]>
  refit_updates: number[]
  checksum_extracted: number
  warm_start: {
    initial_defender_fraud: number | null
    bc_final_loss: number | null
    critic_final_loss: number | null
  }
  reads: {
    extracted_first: number | null
    extracted_last: number | null
    extracted_max: number | null
    zeros: number
    entropy_start: number | null
    entropy_peak: number | null
    entropy_end: number | null
  }
  strategies: StrategySample[]
  final_sequences: Array<{ count: number; runs: ActionRun[]; truncated: boolean }>
  selection: {
    observations: number | null
    selecting: boolean | null
    groups: Array<{ name: string; levels: PosteriorLevel[] }>
    /** Posteriors the run produced that are not stable enough across runs to publish. */
    withheld: string[]
  }
}

export interface RunReport {
  source: string
  warm_start: {
    entities: number | null
    events: number | null
    dormant_share: number | null
    cards_with_a_median: number | null
    history_days: number | null
  }
  hard_negatives: Record<string, number>
  rule_trigger_rates: Record<string, number>
  rule_target: number | null
  rule_verdict: string | null
  rule_events: number | null
  episodes: number | null
  reached_monetized: number | null
  benign_auths: number | null
  fraud_auths: number | null
  fraud_auth_share: number | null
  per_vertical: Record<string, number>
  top_sequences: Array<{ count: number; chain: string[] }>
}

export interface FidelityComparison {
  name: string
  observed: number | null
  target: number | null
  gap: number | null
  noise_floor: number | null
  /** gap / noise_floor. Near 1.0 means synthetic differs from real about as much as real differs from itself. */
  ratio: number | null
  unit: string | null
}

export interface Fidelity {
  source: string
  created_utc: string | null
  verdict_ladder: { indistinguishable: number; close: number; structural_gap: number }
  split: {
    left_entities: number
    left_rows: number
    right_entities: number
    right_rows: number
    fingerprint: string
  }
  comparisons: FidelityComparison[]
  amount: Record<string, number>
  amount_heterogeneity: Record<string, number> | null
  circadian: Record<string, number | number[]> | null
  category_mix: Record<string, number> | null
  fanout_targets: Record<string, number>
  rejected: Record<string, unknown>
  all_floors: Record<string, number>
  /** Which parameter groups were measured, and which could not be. */
  provenance: {
    fitted: string[]
    swept: string[]
    n_noise_floors: number
    n_targets: number
  }
}

export interface GraphFacts {
  source: string
  fanout_observed_mean: number | null
  fanout_target_mean: number | null
  invariants_hold: boolean
  targets: Record<string, number>
  variance_to_mean_note: string
}

export interface TextSample {
  vertical: string
  tier: number
  fraudulent: boolean
  persona: string | null
  text: string | null
  facts: Record<string, string | number> | null
}

export interface TextSamples {
  source: string
  generator: string | null
  embed_model: string | null
  embed_dim: number | null
  fingerprint: string | null
  n_entries: number
  /** Both classes are generated; the label separates intent, not provenance. */
  note: string
  tier_note: string
  samples: TextSample[]
}
