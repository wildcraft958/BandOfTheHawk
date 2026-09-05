import { mulberry32, gaussian } from '../lib/rng'
import { Logistic } from './logistic'
import {
  BENIGN_MEAN,
  EXPERT_COLUMNS,
  EXPERT_ORDER,
  N_FEATURES,
  slice,
  type ExpertName,
} from './features'
import { TACTICS, eventsPerEpisode } from './tactics'

export interface SimConfig {
  seed: number
  /** Updates between defender refits. The real run used 12. */
  refitEvery: number
  /** Updates a label waits before the defender may train on it. The real run used 4320 minutes. */
  labelLatency: number
  /**
   * Share of traffic the defender can review and action per update, highest
   * risk first.
   *
   * A real defender works to a review capacity, not an absolute score cutoff.
   * fraudsim/defender/metrics.py takes alert_budget=100 for exactly this reason,
   * which is where precision_at_budget comes from. Ranking under a capacity also
   * means the defender can never block everything, so the arms race stays an
   * arms race.
   *
   * A share rather than a count, because a fixed count against a stream this
   * size caps the defender at blocking about half of everything no matter how
   * well it learns, and a capped defender can never clear a channel. Clearing a
   * channel is what forces the attacker to move, so with a fixed count the loop
   * has no reason to close.
   *
   * Too much capacity is its own failure. If the defender can cover every
   * channel at once the attacker loses its gradient and stops learning, which is
   * disengagement, not a win. The capacity wants to be enough for one channel
   * and not enough for all of them.
   */
  alertRate: number
  /**
   * One benign event in this many is retained for training.
   *
   * At a payment system's base rate almost every event is ordinary, and keeping
   * all of them buys nothing: the signal is in the positives. Keeping every
   * fraud row and sampling the benign ones is the standard case-control design.
   * It also means the training set carries a different class prior from the
   * population, which is what calibrated() corrects for display.
   */
  benignSampleRate: number
  /** How sharply the attacker reallocates toward whatever is paying. */
  eta: number
  /** Floor on any tactic's probability, so nothing is ever fully abandoned. */
  epsilon: number
  episodesPerUpdate: number
  benignPerUpdate: number
  /**
   * Updates of benign history a refit may train on. Confirmed fraud is kept
   * three times longer, which is the asymmetric retention the paper describes:
   * a confirmed fraud is worth far more than a confirmed benign.
   *
   * Retention is what makes this a loop rather than an equilibrium. A defender
   * that never forgets accumulates a model good against every tactic at once and
   * the arms race stops.
   */
  retention: number
}

/**
 * Defaults chosen for a limit cycle, not for a maximum knockback.
 *
 * Swept over capacity and learning rate. Too little capacity and both sides
 * settle into a mediocre stable state: one tactic dominant for the whole run and
 * a flat line from update 40. Too much and the defender clears every channel at
 * once, extraction goes to zero and the attacker has nothing left to learn from,
 * which is disengagement rather than a win. This region keeps both engaged.
 *
 * alertRate is set so review capacity lands near the number of fraud events per
 * update, which is the regime a real alert budget sits in: the run reports a
 * budget of 100 against 183 positives. Raising it trades precision for coverage
 * and lowering it does the reverse, which is the operational tradeoff the two
 * controls on screen exist to show.
 */
export const DEFAULT_CONFIG: SimConfig = {
  seed: 1,
  refitEvery: 12,
  labelLatency: 3,
  alertRate: 0.0025,
  benignSampleRate: 40,
  eta: 1.0,
  epsilon: 0.05,
  episodesPerUpdate: 10,
  benignPerUpdate: 16000,
  retention: 18,
}

export interface Frame {
  t: number
  extracted: number
  blocked: number
  /** Share of actioned events that were actually fraud. Comparable to the real 0.99. */
  precisionAtBudget: number
  /** Events the defender had capacity to review and action this update. */
  reviewed: number
  /**
   * Share of each tactic's extractable value the defender stopped this update,
   * in TACTICS order. One population per tactic, so these are comparable with
   * each other, and this is the number that explains the bar beside it: a
   * channel the defender closes is a channel the attacker leaves.
   */
  blockedShare: number[]
  /**
   * Mean risk score over all traffic, put back on the population's scale by
   * calibrated(). At a 0.5% base rate a calibrated mean sits near 0.005.
   */
  meanCalibratedScore: number
  fraudEvents: number
  baseRate: number
  entropy: number
  refit: boolean
  weights: number[]
  expertAccuracy: Record<ExpertName, number>
  combinerWeights: number[]
  topTactic: string
}

interface LabelledEvent {
  t: number
  x: Float64Array
  y: number
}

export interface SimState {
  t: number
  weights: number[]
  experts: Record<ExpertName, Logistic>
  combiner: Logistic
  buffer: LabelledEvent[]
  frames: Frame[]
  expertAccuracy: Record<ExpertName, number>
}

const MAX_UPDATES = 150
const BUFFER_CAP = 40000

function newExperts(): Record<ExpertName, Logistic> {
  return Object.fromEntries(
    EXPERT_ORDER.map((name) => [name, new Logistic(EXPERT_COLUMNS[name].length)]),
  ) as Record<ExpertName, Logistic>
}

export function initSim(): SimState {
  return {
    t: 0,
    weights: TACTICS.map(() => 1 / TACTICS.length),
    experts: newExperts(),
    combiner: new Logistic(EXPERT_ORDER.length),
    buffer: [],
    frames: [],
    expertAccuracy: Object.fromEntries(EXPERT_ORDER.map((n) => [n, 0])) as Record<
      ExpertName,
      number
    >,
  }
}

function drawEvent(
  rand: () => number,
  mean: number[],
  spread: number,
  carriesText: boolean,
): Float64Array {
  const x = new Float64Array(N_FEATURES)
  for (let i = 0; i < N_FEATURES; i++) {
    x[i] = Math.max(0, Math.min(1, mean[i] + gaussian(rand) * spread))
  }
  // An event either carries a text artifact or it does not; there is no such
  // thing as a faint one. Left to the Gaussian, a mean of zero clamped at zero
  // still puts about half of every tactic's events above the threshold the text
  // expert uses to decide what it can read, so the text expert ends up training
  // on most of the traffic and scoring as the strongest of the five. That is the
  // reverse of the run's own finding, where text is the thinnest channel and the
  // gap the attacker eventually found.
  if (!carriesText) x[6] = 0
  return x
}

/** The five expert scores for one event, in EXPERT_ORDER. */
function expertScores(state: SimState, x: Float64Array): Float64Array {
  const out = new Float64Array(EXPERT_ORDER.length)
  for (let i = 0; i < EXPERT_ORDER.length; i++) {
    const name = EXPERT_ORDER[i]
    const expert = state.experts[name]
    // An unfitted expert abstains at 0.5 rather than asserting anything.
    out[i] = expert.fitted ? expert.predict(slice(x, EXPERT_COLUMNS[name])) : 0.5
  }
  return out
}

/**
 * King and Zeng's prior correction, as an intercept offset.
 *
 * Every fraud row is kept and benign rows are sampled at 1 in benignSampleRate,
 * and the combiner is class weighted on top of that, so its fitted intercept
 * carries a roughly even class prior rather than the population's. Left
 * uncorrected, a score shown next to the fitted bands (0.53 / 0.63 / 0.74 /
 * 0.84) would be on a scale of its own.
 *
 * This deliberately changes nothing about the loop. Ranking is invariant to a
 * constant shift of the intercept, so no event changes place in the queue and no
 * blocking decision moves. It exists so a number on screen means what a reader
 * will take it to mean.
 */
export function calibrated(p: number, trueRate: number): number {
  const eps = 1e-9
  const q = Math.min(1 - eps, Math.max(eps, p))
  const logit = Math.log(q / (1 - q))
  const offset = Math.log((1 - trueRate) / Math.max(trueRate, eps))
  return 1 / (1 + Math.exp(-(logit - offset)))
}

function risk(state: SimState, x: Float64Array): number {
  const s = expertScores(state, x)
  return state.combiner.fitted ? state.combiner.predict(s) : 0.5
}

function entropyOf(weights: number[]): number {
  let h = 0
  for (const w of weights) if (w > 0) h -= w * Math.log(w)
  return h
}

/**
 * One co-adaptation update.
 *
 * The attacker allocates episodes across tactics by its current weights, earns
 * on events the defender does not block, pays the real action costs, and
 * reallocates by exponential weights on the realised net. The defender refits
 * on cadence over the labelled events old enough to have cleared the label
 * latency.
 *
 * Nothing about the trajectory is scripted. It escalates, collapses at a refit,
 * and migrates to whichever channel the defender has least evidence on, because
 * that is what these two rules do when run against each other.
 */
export function stepSim(state: SimState, config: SimConfig): SimState {
  const rand = mulberry32(config.seed * 7919 + state.t * 104729)
  const t = state.t

  const net = new Array(TACTICS.length).fill(0)

  interface Scored {
    x: Float64Array
    y: number
    tactic: number
    value: number
  }
  const pool: Scored[] = []

  // Benign traffic, so the defender faces a real discrimination problem. Text
  // bearing events are rare among benign traffic, which is why the text expert
  // ends up with the least evidence to learn from. Nothing hardcodes that.
  for (let i = 0; i < config.benignPerUpdate; i++) {
    const carriesText = rand() < 0.06
    const mean = BENIGN_MEAN.slice()
    if (carriesText) mean[6] = 0.3 + rand() * 0.2
    pool.push({ x: drawEvent(rand, mean, 0.2, carriesText), y: 0, tactic: -1, value: 0 })
  }

  const spent = new Array(TACTICS.length).fill(0)
  const episodesRun = new Array(TACTICS.length).fill(0)

  for (let ti = 0; ti < TACTICS.length; ti++) {
    const tactic = TACTICS[ti]
    const episodes = Math.max(1, Math.round(state.weights[ti] * config.episodesPerUpdate))
    episodesRun[ti] = episodes
    const perEpisode = eventsPerEpisode(tactic)
    for (let e = 0; e < episodes; e++) {
      spent[ti] += tactic.cost
      for (let k = 0; k < perEpisode; k++) {
        pool.push({
          x: drawEvent(rand, tactic.mean, 0.2, tactic.text),
          y: 1,
          tactic: ti,
          value: tactic.yieldPerEvent,
        })
      }
    }
  }

  // Rank the whole update by risk and action the top of the queue.
  const scored = pool.map((ev) => ({ ev, score: risk(state, ev.x) }))
  scored.sort((a, b) => b.score - a.score)
  const budget = Math.min(Math.round(config.alertRate * scored.length), scored.length)

  let extracted = 0
  let blocked = 0
  let truePositives = 0
  const gained = new Array(TACTICS.length).fill(0)
  const stopped = new Array(TACTICS.length).fill(0)

  for (let i = 0; i < scored.length; i++) {
    const { ev } = scored[i]
    const actioned = i < budget
    if (actioned && ev.y === 1) truePositives++
    if (ev.y === 1) {
      if (actioned) {
        blocked += ev.value
        stopped[ev.tactic] += ev.value
      } else {
        extracted += ev.value
        gained[ev.tactic] += ev.value
      }
    }
  }

  // Keep every fraud row; sample the benign ones. Bernoulli off the seeded RNG,
  // so the sample is unbiased across the score range and still reproducible.
  const keepBenign = 1 / Math.max(1, config.benignSampleRate)
  for (const { ev } of scored) {
    if (ev.y === 1 || rand() < keepBenign) {
      state.buffer.push({ t, x: ev.x, y: ev.y })
    }
  }

  for (let ti = 0; ti < TACTICS.length; ti++) {
    net[ti] = (gained[ti] - spent[ti]) / Math.max(episodesRun[ti], 1)
  }

  const fraudEvents = pool.reduce((a, ev) => a + ev.y, 0)
  const precisionAtBudget = budget > 0 ? truePositives / budget : 0
  const baseRate = pool.length > 0 ? fraudEvents / pool.length : 0

  // Exponential weights (Hedge). Named as such: this is not policy gradient and
  // it is not PPO.
  const mean = net.reduce((a, v) => a + v, 0) / net.length
  const adv = net.map((v) => v - mean)
  const scale = Math.max(...adv.map(Math.abs), 1e-9)
  let weights = state.weights.map((w, i) => w * Math.exp((config.eta * adv[i]) / scale))
  const total = weights.reduce((a, v) => a + v, 0) || 1
  weights = weights.map((w) => w / total)
  weights = weights.map((w) => (1 - config.epsilon) * w + config.epsilon / TACTICS.length)

  // Defender refit, over labels that have cleared the latency window.
  let refit = false
  let expertAccuracy = state.expertAccuracy
  if ((t + 1) % config.refitEvery === 0) {
    const benignFloor = t - config.labelLatency - config.retention
    const fraudFloor = t - config.labelLatency - config.retention * 3
    const visible = state.buffer.filter(
      (ev) =>
        ev.t <= t - config.labelLatency && ev.t >= (ev.y === 1 ? fraudFloor : benignFloor),
    )
    const positives = visible.reduce((a, ev) => a + ev.y, 0)

    if (visible.length > 40 && positives > 4 && positives < visible.length) {
      const labels = Uint8Array.from(visible.map((ev) => ev.y))
      const nextAccuracy = { ...state.expertAccuracy }
      // Each refit is a retrain from scratch on the retained window, not a
      // continuation. That is what lets coverage of an abandoned tactic decay.
      state.experts = newExperts()
      state.combiner = new Logistic(EXPERT_ORDER.length)

      for (const name of EXPERT_ORDER) {
        const cols = EXPERT_COLUMNS[name]
        const rows = visible.map((ev) => slice(ev.x, cols))
        // Text-bearing events are the only ones the text expert can learn from.
        const mask =
          name === 'text' ? visible.map((ev) => ev.x[6] > 0.01) : visible.map(() => true)
        const keptRows: Float64Array[] = []
        const keptLabels: number[] = []
        for (let i = 0; i < rows.length; i++) {
          if (mask[i]) {
            keptRows.push(rows[i])
            keptLabels.push(labels[i])
          }
        }
        const expert = state.experts[name]
        // Longer and gentler than it looks like it needs to be. Each expert sees
        // every fraud row but can only distinguish the ones that deviate on its
        // own columns, so most positives look ordinary to it. At a short, fast
        // fit the class weighted objective saturates to predicting a single
        // class, which scores exactly 0.500 balanced accuracy and reads on
        // screen as an expert that learned nothing. Fourteen epochs at a third
        // of the step size finds a real boundary instead.
        expert.fit(keptRows, Uint8Array.from(keptLabels), 14, 0.03)
        nextAccuracy[name] = expert.balancedAccuracy(keptRows, Uint8Array.from(keptLabels))
      }

      const stacked = visible.map((ev) => expertScores(state, ev.x))
      state.combiner.fit(stacked, labels, 10, 0.06)

      expertAccuracy = nextAccuracy
      refit = true
    }

  }

  // Prune on the same axis the refit filter uses. A count based cap is what
  // silently starved the defender: at a realistic base rate this pushes tens of
  // thousands of events per update, so keeping the last 9,000 rows left only
  // rows stamped with the current update, while the refit filter needs rows at
  // least labelLatency updates old. `visible` was provably always empty, the
  // defender never trained once, and every symptom showed up on the attacker's
  // side of the chart.
  const oldestUseful = t - config.labelLatency - config.retention * 3
  state.buffer = state.buffer.filter((ev) => ev.t >= oldestUseful)
  if (state.buffer.length > BUFFER_CAP) {
    state.buffer = state.buffer.slice(state.buffer.length - BUFFER_CAP)
  }

  const topIndex = weights.indexOf(Math.max(...weights))
  const frame: Frame = {
    t,
    extracted,
    blocked,
    precisionAtBudget,
    reviewed: budget,
    blockedShare: TACTICS.map((_, i) => {
      const total = stopped[i] + gained[i]
      return total > 0 ? stopped[i] / total : 0
    }),
    meanCalibratedScore:
      scored.length > 0
        ? scored.reduce((a, r) => a + calibrated(r.score, Math.max(baseRate, 1e-6)), 0) /
          scored.length
        : 0,
    fraudEvents,
    baseRate,
    entropy: entropyOf(weights),
    refit,
    weights,
    expertAccuracy,
    combinerWeights: Array.from(state.combiner.w),
    topTactic: TACTICS[topIndex].label,
  }

  return {
    ...state,
    t: t + 1,
    weights,
    expertAccuracy,
    frames: [...state.frames, frame].slice(-MAX_UPDATES),
  }
}

export function runToEnd(config: SimConfig, updates = MAX_UPDATES): SimState {
  let state = initSim()
  for (let i = 0; i < updates; i++) state = stepSim(state, config)
  return state
}

export { MAX_UPDATES }
