/**
 * A deliberately small feature space, named after the features the real
 * detector actually leads on by gain (device_age_days,
 * seconds_since_last_auth, entry_mode, amount, device_new_to_card,
 * is_first_txn_this_merchant), plus a text score that only text-bearing events
 * carry.
 */
export const FEATURES = [
  'device_age',
  'velocity',
  'card_not_present',
  'amount_vs_median',
  'off_hours',
  'first_at_merchant',
  'text_score',
] as const

export const N_FEATURES = FEATURES.length

export type ExpertName = 'transaction' | 'binding' | 'identity' | 'network' | 'text'

/**
 * Routing is a schema fact, not a learned gate. Each expert reads only the
 * columns its event types carry, mirroring EXPERT_EVENT_TYPES and the column
 * subsets in fraudsim/defender/experts.py.
 */
export const EXPERT_COLUMNS: Record<ExpertName, number[]> = {
  transaction: [1, 3, 2], // velocity, amount_vs_median, card_not_present
  binding: [0, 5], // device_age, first_at_merchant
  identity: [4, 0], // off_hours, device_age
  network: [1, 5, 0], // velocity, first_at_merchant, device_age
  text: [6, 3], // text_score, amount_vs_median
}

export const EXPERT_ORDER: ExpertName[] = [
  'network',
  'transaction',
  'binding',
  'identity',
  'text',
]

/** Mean feature vector for ordinary traffic. */
export const BENIGN_MEAN = [0.62, 0.18, 0.28, 0.22, 0.16, 0.24, 0.0]

export function slice(x: Float64Array, cols: number[]): Float64Array {
  const out = new Float64Array(cols.length)
  for (let i = 0; i < cols.length; i++) out[i] = x[cols[i]]
  return out
}
