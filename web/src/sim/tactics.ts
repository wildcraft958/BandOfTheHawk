import { ACTIONS } from '../data/taxonomy'

const COST = new Map(ACTIONS.map((a) => [a.id, a.cost]))

/** Sum the real ACTION_SPECS costs for a chain. */
function chainCost(chain: string[]): number {
  return chain.reduce((acc, id) => acc + (COST.get(id) ?? 1), 0)
}

export interface Tactic {
  id: string
  label: string
  /** Drawn from the real stage machine: every chain is a legal path. */
  chain: string[]
  cost: number
  /** Value a single unblocked monetizing event yields. */
  yieldPerEvent: number
  /**
   * Mean feature vector for the events this tactic emits, in FEATURES order.
   * A tactic is detectable exactly insofar as these differ from BENIGN_MEAN on
   * columns some expert reads.
   */
  mean: number[]
  /** Whether its events carry text, which only the text expert reads. */
  text: boolean
}

/**
 * Seven chains over the real 20-action space, with costs summed from the real
 * ACTION_SPECS. The last one is the refund loop the trained policy actually
 * converged on: expensive, many small events, and loading almost entirely on
 * text.
 *
 * Two properties here are deliberate, and the engine does not close its loop
 * without either.
 *
 * Each tactic is identified by a distinct PAIR of features, and each pair sits
 * inside a single expert's columns so that one expert can learn the conjunction.
 * Two earlier versions were wrong here in opposite directions. The first had
 * device_age among the three largest deviations of six of the seven tactics, and
 * three of the five experts read device_age, so learning one channel covered
 * nearly all of them at once: a transitive game has a best tactic, so the
 * attacker picks it on update 0 and never moves again. The second gave each
 * tactic a single extreme feature, which made them independent but left the
 * defender useless, because among sixteen thousand ordinary events a few hundred
 * are extreme on some single feature by chance and they swamp forty fraud events
 * at the top of the queue. A conjunction of two features is what a benign event
 * almost never produces by accident, which is the same reason the real detector
 * leads on per-entity features rather than raw amounts.
 *
 * Gross value per episode is held within about 1.3x across all seven. The first
 * version spread it 13x, because the refund chain emits twelve monetizing events
 * at 300 each while card testing emits three at 90. At that spread detection is
 * irrelevant: the refund loop wins on arithmetic before the defender exists.
 * Equalising it means the attacker is choosing on what is currently being
 * caught, which is the only thing that makes the loop a loop.
 *
 * yieldPerEvent and mean are tuned. The chains, the action costs, and which
 * expert reads which column are not.
 */
export const TACTICS: Tactic[] = [
  {
    id: 'card_testing',
    label: 'Card testing',
    chain: ['buy_creds', 'add_device_selfserve', 'attempt_auth', 'attempt_auth', 'attempt_auth'],
    cost: 0,
    yieldPerEvent: 380,
    // Signature: velocity with card_not_present. Both read by the transaction expert.
    mean: [0.58, 0.88, 0.86, 0.26, 0.20, 0.30, 0.0],
    text: false,
  },
  {
    id: 'voice_clone',
    label: 'Voice clone',
    chain: ['harvest_voice', 'buy_creds', 'call_ivr_provision', 'attempt_auth', 'attempt_auth'],
    cost: 0,
    yieldPerEvent: 550,
    // Signature: off_hours on a young device. Both read by the identity expert.
    mean: [0.18, 0.22, 0.32, 0.28, 0.84, 0.28, 0.0],
    text: false,
  },
  {
    id: 'deepfake_onboarding',
    label: 'Deepfake onboarding',
    chain: ['make_synth_id', 'harvest_face', 'submit_kyc', 'attempt_auth', 'attempt_auth'],
    cost: 0,
    yieldPerEvent: 600,
    // Signature: a device with no history, first time at the merchant. Both read by the binding expert.
    mean: [0.05, 0.22, 0.32, 0.26, 0.20, 0.86, 0.0],
    text: false,
  },
  {
    id: 'phishing_ato',
    label: 'Agentic phishing',
    chain: ['phish_holder', 'reset_password', 'attempt_auth', 'attempt_auth', 'attempt_auth'],
    cost: 0,
    yieldPerEvent: 340,
    // Signature: velocity with first_at_merchant. Both read by the network expert.
    mean: [0.56, 0.84, 0.32, 0.26, 0.22, 0.84, 0.0],
    text: false,
  },
  {
    id: 'mule_layering',
    label: 'Mule layering',
    chain: ['buy_creds', 'add_device_selfserve', 'add_payee', 'transfer_p2p', 'transfer_p2p'],
    cost: 0,
    yieldPerEvent: 520,
    // Signature: velocity with amount_vs_median. Both read by the transaction expert.
    mean: [0.58, 0.80, 0.30, 0.88, 0.20, 0.28, 0.0],
    text: false,
  },
  {
    id: 'friendly_fraud',
    label: 'Friendly fraud',
    chain: ['buy_creds', 'add_device_selfserve', 'attempt_auth', 'file_dispute'],
    cost: 0,
    yieldPerEvent: 470,
    // Signature: card_not_present with amount_vs_median. Both read by the transaction expert. Carries text, but faintly.
    mean: [0.60, 0.20, 0.86, 0.84, 0.18, 0.26, 0.22],
    text: true,
  },
  {
    id: 'refund_loop',
    label: 'Refund loop',
    chain: [
      'buy_creds',
      'reset_password',
      ...Array.from({ length: 12 }, () => 'request_refund'),
    ],
    cost: 0,
    yieldPerEvent: 100,
    // Signature: text_score with amount_vs_median. Both read by the text expert, the thinnest channel the defender has. This is the tactic the real trained policy converged on.
    mean: [0.60, 0.20, 0.30, 0.80, 0.18, 0.26, 0.86],
    text: true,
  },
]

for (const t of TACTICS) {
  t.cost = chainCost(t.chain)
}

/** Events a single episode of this tactic emits. */
export function eventsPerEpisode(t: Tactic): number {
  return t.chain.filter(
    (a) => a === 'attempt_auth' || a === 'transfer_p2p' || a === 'request_refund' || a === 'file_dispute',
  ).length
}
