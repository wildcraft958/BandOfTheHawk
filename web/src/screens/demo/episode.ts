import { ACTIONS, ADVANCES, LEGAL_ACTIONS, STAGE_ORDER, type Stage } from '../../data/taxonomy'
import { BANDS } from '../../data/paper'
import { mulberry32 } from '../../lib/rng'

/**
 * One attack episode, walked through the real stage machine.
 *
 * What is real: the five stages, the actions each stage admits, the three
 * transitions that advance it, the per-action costs, the five decisions the
 * defender's boundaries separate, and the per-episode threshold jitter.
 *
 * What is not: the score. It is a transparent stand-in computed from the
 * action's own cost and how loud it is, not the trained ensemble. This
 * demonstrates the gating and the decision ladder. It does not predict how often
 * a vertical succeeds, and the paper reports no such figure.
 *
 * Pure and seeded, so the same inputs always produce the same tape.
 */

export type Decision = 'approve' | 'step up' | 'hold' | 'decline' | 'block'

export const MITIGATION: Record<Decision, string> = {
  approve: 'nothing, the event passes',
  'step up': 'challenge the cardholder',
  hold: 'freeze the card for 24h',
  decline: 'freeze for 72h',
  block: 'unbind the device and blocklist it',
}

/** Only these end an episode. A step-up is friction, not a refusal. */
const STOPS: Decision[] = ['decline', 'block']

export interface Step {
  n: number
  action: string
  stage: Stage
  advancedTo: Stage | null
  scored: boolean
  score: number | null
  decision: Decision | null
  cost: number
}

export interface Episode {
  steps: Step[]
  finalStage: Stage
  stopped: boolean
  stoppedBy: Decision | null
  extracted: number
  totalCost: number
  bands: { stepUp: number; hold: number; decline: number; block: number }
  jitter: number
}

export interface Options {
  seed: number
  entryStage: Stage
  /** 0 to 3, from Table 8. Ordinal only: a higher tier buys a richer artifact, never a stated success rate. */
  tier: number
  maxSteps?: number
}

const COST = new Map(ACTIONS.map((a) => [a.id, a.cost]))
const EMITS = new Map(ACTIONS.map((a) => [a.id, a.emitsEvent]))
const PHASE = new Map(ACTIONS.map((a) => [a.id, a.phase]))
const NEEDS_ARTIFACT = new Set(ACTIONS.filter((a) => a.genaiTool).map((a) => a.id))

/**
 * What makes an event look like fraud here, and why each term is defensible.
 *
 * A first attempt scored on the action's cost alone. That was arbitrary and it
 * broke the game: bound-stage actions cost 0.5 to 2.0, so the score topped out
 * around 0.33 against a decline boundary of 0.74, and the defender could not
 * stop anything. Every episode from the bound stage succeeded.
 *
 * The terms below follow the feature families the run actually leans on. The
 * paper reports that device age dominated the detector's feature importances,
 * and the feature table carries eighteen compound window features over 1h, 24h
 * and 7d, so binding a new credential and stacking events quickly are the two
 * things that should move a score.
 */
const PHASE_WEIGHT: Record<string, number> = {
  /** Obtaining a means. Mostly quiet: much of it emits no event at all. */
  acquire: 0.3,
  /** Binding it to something usable. The loudest, because it is a new binding
   *  on a device with no history, which is the signal the detector leans on most. */
  bind: 0.52,
  /** Spending. Moves value, so it draws scrutiny, but a single small
   *  authorisation on an established card is unremarkable. */
  spend: 0.46,
  /** Extracting and laundering. The most scrutinised, and the point of it all. */
  extract: 0.58,
}

/**
 * Each scored event raises the next one, as a velocity window does, but the term
 * saturates. An unbounded one made every episode longer than about six steps a
 * certain block, which is the mirror of the first bug rather than a fix.
 */
const VELOCITY_CEILING = 0.15
const VELOCITY_SCALE = 2.5
const velocityTerm = (scored: number) =>
  VELOCITY_CEILING * (1 - Math.exp(-scored / VELOCITY_SCALE))

/** Once value is moving, an episode cashes out rather than running indefinitely. */
const EXTRACT_STEPS = 3

/** How much a generative artifact quiets an action that needs one, per tier. */
const TIER_QUIETING = 0.05

function decide(score: number, b: Episode['bands']): Decision {
  if (score >= b.block) return 'block'
  if (score >= b.decline) return 'decline'
  if (score >= b.hold) return 'hold'
  if (score >= b.stepUp) return 'step up'
  return 'approve'
}

function advance(stage: Stage, action: string): Stage | null {
  const rule = ADVANCES.find((a) => a.from === stage && a.via.includes(action))
  return rule ? rule.to : null
}

export function runEpisode({ seed, entryStage, tier, maxSteps = 10 }: Options): Episode {
  const rand = mulberry32(seed)

  // Drawn once per episode, as the simulator does, so the same vector does not
  // always meet the same boundaries.
  const jitter = (rand() - 0.5) * 0.06
  const bands = {
    stepUp: BANDS.stepUp + jitter,
    hold: BANDS.hold + jitter,
    decline: BANDS.decline + jitter,
    block: BANDS.block + jitter,
  }

  const steps: Step[] = []
  let stage: Stage = entryStage
  let extracted = 0
  let totalCost = 0
  let stoppedBy: Decision | null = null
  let scoredSoFar = 0
  let monetizedSteps = 0

  for (let n = 1; n <= maxSteps && stage !== 'terminal' && !stoppedBy; n += 1) {
    const legal = LEGAL_ACTIONS[stage]
    if (legal.length === 0) break

    // Prefer an action that advances the stage, so the episode makes progress
    // rather than milling about, but not always.
    const advancing = legal.filter((a) => advance(stage, a))
    const pool = advancing.length > 0 && rand() < 0.72 ? advancing : legal
    const action = pool[Math.floor(rand() * pool.length)]

    const cost = COST.get(action) ?? 1
    totalCost += cost
    const emits = EMITS.get(action) ?? true

    // A higher capability tier buys a quieter artifact, but only for the actions
    // that need one. That is where a generative model changes the attacker's cost.
    const quieting = NEEDS_ARTIFACT.has(action) ? Math.min(tier, 3) * TIER_QUIETING : 0
    const velocity = velocityTerm(scoredSoFar)
    const base = PHASE_WEIGHT[PHASE.get(action) ?? 'spend'] ?? 0.46
    const score = emits
      ? Math.max(0, Math.min(1, base + velocity + (rand() - 0.5) * 0.26 - quieting))
      : null
    if (emits) scoredSoFar += 1
    const decision = score == null ? null : decide(score, bands)
    if (decision && STOPS.includes(decision)) stoppedBy = decision

    const advancedTo = stoppedBy ? null : advance(stage, action)
    if (advancedTo) stage = advancedTo
    if (stage === 'monetized' && !stoppedBy) extracted += cost * 120 * (1 + rand())

    steps.push({ n, action, stage, advancedTo, scored: emits, score, decision, cost })

    if (stage === 'monetized') {
      monetizedSteps += 1
      if (monetizedSteps >= EXTRACT_STEPS) break
    }
  }

  return {
    steps,
    finalStage: stage,
    stopped: stoppedBy != null,
    stoppedBy,
    extracted: Math.round(extracted),
    totalCost: Math.round(totalCost * 10) / 10,
    bands,
    jitter,
  }
}

/** Every action must have been legal at the stage it was taken in. */
export function legalityHolds(ep: Episode, entryStage: Stage): boolean {
  let stage: Stage = entryStage
  for (const s of ep.steps) {
    if (!LEGAL_ACTIONS[stage].includes(s.action)) return false
    if (s.advancedTo) stage = s.advancedTo
  }
  return true
}

export const STAGE_INDEX = new Map(STAGE_ORDER.map((s, i) => [s, i]))
