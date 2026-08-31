/**
 * The attack surface, transcribed from the simulator rather than measured.
 *
 * Sources, all verified against the code at the paths named:
 *   actions      src/fraudsim/engine/actions.py   (ActionName, ACTION_SPECS, NEEDS_ARTIFACT)
 *   stages       src/fraudsim/engine/stages.py    (Stage, LEGAL_ACTIONS, ADVANCES)
 *   verticals    src/fraudsim/attacker/scripted.py (VERTICALS, ZERO_SHOT_HOLDOUTS, docstrings)
 *   postures     src/fraudsim/attacker/nets.py    (STEALTH_NAMES and the comment block)
 *   entry stage + generative capability: the paper's taxonomy table (Section 4)
 *
 * Episode counts are NOT here: those are measured, and come from run_report.json.
 */

export type Stage = 'none' | 'acquired' | 'bound' | 'monetized' | 'terminal'
export type Phase = 'acquire' | 'bind' | 'spend' | 'extract'

export interface Action {
  id: string
  phase: Phase
  cost: number
  emitsEvent: boolean
  description: string
  /** The generative artifact this action requires, from NEEDS_ARTIFACT. This is where the GenAI enters. */
  genaiTool: string | null
}

/** All 20 actions, in declaration order, with the real costs. */
export const ACTIONS: Action[] = [
  { id: 'phish_holder', phase: 'acquire', cost: 1.0, emitsEvent: true, description: 'contact a holder under a pretext', genaiTool: 'write_phish' },
  { id: 'buy_creds', phase: 'acquire', cost: 2.0, emitsEvent: false, description: 'obtain card details', genaiTool: null },
  { id: 'make_synth_id', phase: 'acquire', cost: 3.0, emitsEvent: false, description: 'assemble an identity', genaiTool: null },
  { id: 'harvest_voice', phase: 'acquire', cost: 1.5, emitsEvent: false, description: 'obtain a voice sample', genaiTool: 'clone_voice' },
  { id: 'harvest_face', phase: 'acquire', cost: 1.5, emitsEvent: false, description: 'obtain a face sample', genaiTool: 'deepfake_selfie' },

  { id: 'call_ivr_provision', phase: 'bind', cost: 3.0, emitsEvent: true, description: 'provision a card by phone', genaiTool: null },
  { id: 'submit_kyc', phase: 'bind', cost: 3.0, emitsEvent: true, description: 'submit identity documents', genaiTool: null },
  { id: 'add_device_selfserve', phase: 'bind', cost: 1.0, emitsEvent: true, description: 'bind a device in the app', genaiTool: null },
  { id: 'sim_swap', phase: 'bind', cost: 5.0, emitsEvent: true, description: 'move a number to a new carrier', genaiTool: 'pretext' },
  { id: 'reset_password', phase: 'bind', cost: 1.0, emitsEvent: true, description: 'reset account credentials', genaiTool: null },
  { id: 'add_payee', phase: 'bind', cost: 1.0, emitsEvent: true, description: 'register a transfer destination', genaiTool: null },
  { id: 'open_ticket', phase: 'bind', cost: 2.0, emitsEvent: true, description: 'raise a support request', genaiTool: 'write_ticket' },
  { id: 'escalate_limit', phase: 'bind', cost: 2.0, emitsEvent: true, description: 'request a higher limit', genaiTool: null },

  { id: 'attempt_auth', phase: 'spend', cost: 0.5, emitsEvent: true, description: 'authorise a purchase', genaiTool: null },
  { id: 'complete_3ds', phase: 'spend', cost: 1.0, emitsEvent: true, description: 'answer a step-up challenge', genaiTool: null },
  { id: 'transfer_p2p', phase: 'spend', cost: 0.5, emitsEvent: true, description: 'send money to a payee', genaiTool: null },
  { id: 'request_refund', phase: 'spend', cost: 2.0, emitsEvent: true, description: 'ask a merchant for a refund', genaiTool: 'write_refund_claim' },

  { id: 'file_dispute', phase: 'extract', cost: 2.0, emitsEvent: true, description: 'dispute a settled transaction', genaiTool: 'write_dispute' },
  { id: 'cash_out', phase: 'extract', cost: 1.0, emitsEvent: true, description: 'convert to an untraceable form', genaiTool: null },
  { id: 'launder_chain', phase: 'extract', cost: 2.0, emitsEvent: true, description: 'move funds through intermediaries', genaiTool: null },
]

export const PHASE_LABEL: Record<Phase, string> = {
  acquire: 'acquiring the means',
  bind: 'binding them to something usable',
  spend: 'spending',
  extract: 'extracting',
}

/**
 * Stage gating is structural, not learned. Spending a card nobody has bound is
 * not a strategy that fails; it is an event that cannot occur. The mask is
 * applied to the policy's logits before the softmax, so no probability is ever
 * placed on an impossible action.
 */
export const LEGAL_ACTIONS: Record<Stage, string[]> = {
  none: ['phish_holder', 'buy_creds', 'make_synth_id', 'harvest_voice', 'harvest_face'],
  acquired: [
    'call_ivr_provision', 'submit_kyc', 'add_device_selfserve', 'add_payee',
    'sim_swap', 'reset_password', 'harvest_voice', 'harvest_face',
  ],
  bound: [
    'attempt_auth', 'complete_3ds', 'transfer_p2p', 'request_refund',
    'add_payee', 'escalate_limit', 'open_ticket',
  ],
  monetized: ['cash_out', 'launder_chain', 'file_dispute', 'attempt_auth'],
  terminal: [],
}

/** Only these advance the stage, and only on success. */
export const ADVANCES: Array<{ from: Stage; to: Stage; via: string[] }> = [
  { from: 'none', to: 'acquired', via: ['phish_holder', 'buy_creds', 'make_synth_id'] },
  {
    from: 'acquired',
    to: 'bound',
    via: ['call_ivr_provision', 'submit_kyc', 'add_device_selfserve', 'sim_swap', 'reset_password'],
  },
  { from: 'bound', to: 'monetized', via: ['attempt_auth', 'transfer_p2p'] },
]

export const STAGE_ORDER: Stage[] = ['none', 'acquired', 'bound', 'monetized', 'terminal']

export const STAGE_BLURB: Record<Stage, string> = {
  none: 'holds nothing; may only obtain a means',
  acquired: 'holds a means it cannot yet spend',
  bound: 'value can move',
  monetized: 'extraction and laundering',
  terminal: 'the episode is over',
}

export interface Vertical {
  id: string
  label: string
  entryStage: Stage
  /** The step where a generative model changes the attacker's cost. */
  capability: string
  blurb: string
  heldOut: boolean
  simulated: boolean
  /** Why it is not simulated, when it is not. */
  exclusion?: string
}

/**
 * Eleven identified, nine simulated. The two exclusions are stated rather than
 * hidden: describing an excluded vertical is more useful than simulating one on
 * fabricated numbers.
 */
export const VERTICALS: Vertical[] = [
  {
    id: 'card_testing', label: 'Card testing', entryStage: 'bound',
    capability: 'none, volume and velocity',
    blurb: 'Buy a batch of stolen cards, bind one, probe with small authorisations.',
    heldOut: false, simulated: true,
  },
  {
    id: 'voice_clone', label: 'Voice clone', entryStage: 'acquired',
    capability: 'speech synthesis for IVR provisioning',
    blurb: 'Harvest a voice sample, provision the card by phone, then spend.',
    heldOut: false, simulated: true,
  },
  {
    id: 'deepfake_onboarding', label: 'Deepfake onboarding', entryStage: 'acquired',
    capability: 'face and document synthesis for KYC',
    blurb: 'Assemble a synthetic identity, clear liveness with a deepfake, pass KYC.',
    heldOut: false, simulated: true,
  },
  {
    id: 'phishing_ato', label: 'Agentic phishing', entryStage: 'none',
    capability: 'personalised outreach at scale',
    blurb: 'Phish a holder, reset the password, rebind a device, take over.',
    heldOut: false, simulated: true,
  },
  {
    id: 'friendly_fraud', label: 'Friendly fraud', entryStage: 'monetized',
    capability: 'dispute narrative writing',
    blurb: 'Spend normally, then dispute the settled charge as unauthorised.',
    heldOut: false, simulated: true,
  },
  {
    id: 'support_se', label: 'Support engineering', entryStage: 'acquired',
    capability: 'pretext and ticket writing',
    blurb: 'Open a support ticket under a pretext, escalate a limit, spend.',
    heldOut: false, simulated: true,
  },
  {
    id: 'mule_layering', label: 'Mule layering', entryStage: 'monetized',
    capability: 'none, transfer topology',
    blurb: 'Add a payee, wait out the cooling-off, transfer, then launder and cash out.',
    heldOut: false, simulated: true,
  },
  {
    id: 'sim_swap', label: 'SIM swap', entryStage: 'acquired',
    capability: 'carrier pretext writing',
    blurb: 'Move the number to a new SIM, intercept the OTP, clear the step-up.',
    heldOut: true, simulated: true,
  },
  {
    id: 'refund_abuse', label: 'Refund abuse', entryStage: 'bound',
    capability: 'refund claim writing',
    blurb: 'Buy, then claim the item never arrived and request a refund.',
    heldOut: true, simulated: true,
  },
  {
    id: 'merchant_collusion', label: 'Merchant collusion', entryStage: 'monetized',
    capability: 'not modelled',
    blurb: 'A merchant of record cooperating with the fraud.',
    heldOut: false, simulated: false,
    exclusion:
      'Needs a settlement and clawback layer the money model does not implement, and no available source carries merchant onboarding or settlement data, so every parameter would be invented.',
  },
  {
    id: 'bust_out', label: 'Bust-out', entryStage: 'bound',
    capability: 'not modelled',
    blurb: 'Build good standing over months, then draw everything at once and vanish.',
    heldOut: false, simulated: false,
    exclusion:
      'Requires a credit-limit and repayment history model over a far longer horizon than the simulated window.',
  },
]

/**
 * The posture head. Four ordinal choices, not a continuous dial: the policy
 * names an intent and the environment resolves it against the entity graph.
 */
export const POSTURES = [
  { id: 'loud', label: 'Loud', blurb: 'whatever binding the world prefers (the newest), online entry' },
  { id: 'aged', label: 'Aged', blurb: "route through the card's oldest surviving binding" },
  { id: 'aged_cool', label: 'Aged + cool', blurb: 'as aged, and wait out a floor before acting' },
  { id: 'rotate', label: 'Rotate', blurb: 'move to another card in the dump before acting' },
] as const
