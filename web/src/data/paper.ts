/**
 * Facts transcribed from the locked solution document, Final_Locked.pdf.
 *
 * These are NOT parsed from a run log. The co-adaptation log this site used
 * previously described a different, larger run that the locked document
 * supersedes, and that log is no longer in the repository. Everything here
 * carries the table or section it came from so any figure on screen can be
 * checked against the paper line by line.
 *
 * Two regimes appear in the document and they are not interchangeable:
 *   static detection  12,000 holders, fixed scripted red team, raised prevalence
 *   co-adaptation     600 holders, 24 updates, refit every 6, 2% prevalence,
 *                     four paired seeds, so eight runs
 */

export const SOURCE = 'Final_Locked.pdf'

/** Section 8, opening. The regime any co-adaptation figure must be read against. */
export const COADAPT_SETUP = {
  holders: 600,
  updates: 24,
  refitEvery: 6,
  prevalence: 0.02,
  seeds: 4,
  runs: 8,
  cite: 'Table 5 caption',
}

/** Section 8.1, Table 4. Static detection, and the one regime the site already matched. */
export const STATIC_SETUP = {
  holders: 12000,
  adversary: 'fixed scripted red team',
  prevalenceNote: 'a raised prevalence, so the ablation delta is stable',
  cite: 'Table 4 caption',
}

export interface Block {
  arm: 'Full' | 'Ablated'
  seed: number
  pre: number
  r1: number
  r2: number
  r3: number
  peak: number
}

/** Table 5. Mean value extracted per episode within each inter-refit block. */
export const BLOCKS: Block[] = [
  { arm: 'Full', seed: 0, pre: 2687, r1: 7608, r2: 2601, r3: 2155, peak: 10065 },
  { arm: 'Full', seed: 1, pre: 2536, r1: 2784, r2: 2880, r3: 10398, peak: 12135 },
  { arm: 'Full', seed: 2, pre: 2404, r1: 5116, r2: 9351, r3: 4075, peak: 11364 },
  { arm: 'Full', seed: 3, pre: 2358, r1: 2054, r2: 4372, r3: 5062, peak: 6773 },
  { arm: 'Ablated', seed: 0, pre: 3295, r1: 1377, r2: 1010, r3: 722, peak: 4788 },
  { arm: 'Ablated', seed: 1, pre: 1868, r1: 3804, r2: 1805, r3: 5050, peak: 6987 },
  { arm: 'Ablated', seed: 2, pre: 2743, r1: 4764, r2: 3845, r3: 11129, peak: 12965 },
  { arm: 'Ablated', seed: 3, pre: 3191, r1: 2834, r2: 1307, r3: 1135, peak: 4548 },
]

/**
 * Section 8.2. The rule is stated in the paper: some post-refit block mean
 * exceeds the pre-refit block mean, and at least one refit is followed by a drop.
 */
export const COEVOLUTION = {
  runsWithFullPattern: 5,
  ofRuns: 8,
  peakOverOpeningMin: 1.4,
  peakOverOpeningMax: 4.8,
  cite: 'Section 8.2',
  exceptions: [
    {
      which: 'Ablated seeds 0 and 3',
      what: 'the defender takes control at the first refit and never cedes it, which is the loop working in the defender’s favour',
    },
    {
      which: 'Full seed 1',
      what: 'the attacker climbs monotonically and no refit reverses it, which is the loop working in the attacker’s favour, and it reaches the highest final block mean in the full arm',
    },
  ],
}

export interface PairedSeed {
  seed: number
  full: number
  ablated: number
  difference: number
}

/** Table 6. Paired comparison of mean post-refit extraction. */
export const ABLATION: {
  seeds: PairedSeed[]
  meanFull: number
  meanAblated: number
  meanDifference: number
  ci: [number, number]
  resamples: number
  cite: string
} = {
  seeds: [
    { seed: 0, full: 4121, ablated: 1036, difference: 3085 },
    { seed: 1, full: 5354, ablated: 3553, difference: 1801 },
    { seed: 2, full: 6180, ablated: 6579, difference: -399 },
    { seed: 3, full: 3829, ablated: 1759, difference: 2071 },
  ],
  meanFull: 4871,
  meanAblated: 3232,
  meanDifference: 1639,
  ci: [219, 2764],
  resamples: 20000,
  cite: 'Table 6 and Section 8.3',
}

/** Section 8.4. Distinct converged sequences, averaged over seeds. */
export const DIVERSITY = {
  full: { before: 1.2, after: 7.0 },
  ablated: { before: 1.2, after: 5.8, peak: 7.0, peakAt: 'the second refit' },
  cite: 'Section 8.4',
}

/**
 * Section 8.5. The other side of the ledger, and the paper's own framing of the
 * defender's limit: a detector that tightens without limit wins any contest
 * measured on recall alone.
 */
export const FRICTION = {
  full: 0.0067,
  ablated: 0.0019,
  pooled: 0.0043,
  refits: 16,
  range: [0, 0.03] as [number, number],
  readThisOne: 'full',
  cite: 'Section 8.5',
}

/**
 * Section 10. The claim this site used to lead with, and why it is gone.
 *
 * The site previously showed zero-shot recall of 1.000 as a headline. The locked
 * document withholds that measurement: the SIM swap action is legal in the
 * learned attacker's action space and the policy uses it, so the defender trains
 * on SIM swap traffic and is then asked whether it generalises to SIM swap.
 */
export const WITHHELD = {
  what: 'zero-shot recall',
  why:
    'SIM swap is designated a held-out vertical, but the SIM swap action is also legal in the learned attacker’s action space and the policy uses it. The defender therefore trains on SIM swap traffic and is then asked whether it generalises to SIM swap.',
  consequence:
    'High recall could not be credited to generalisation and low recall could not be blamed on it, so the measurement is withheld rather than explained away.',
  fix: 'Removing the held-out actions from the learned attacker’s legality mask would restore it.',
  cite: 'Section 10, Limitations',
}

/** Section 10. Stated openly in the paper, so it is stated openly here. */
export const PREVALENCE_CAVEAT = {
  measuredAt: 0.02,
  deployedNearer: 0.005,
  why:
    'Reaching 0.5% honestly would need roughly six hundred thousand additional benign rows per refit window, which is not tractable at the run counts this comparison needs.',
  cite: 'Section 10, Limitations',
}

/**
 * Section 5. The four boundaries separating approve, step up, hold, decline and
 * block.
 *
 * The paper's position is that these sit on a cost curve rather than at round
 * numbers: they begin FREE and become CITED by grid search against a published
 * cost curve, they are re-searched at every refit, and threshold jitter
 * perturbs them by a small amount drawn once per episode so a policy cannot
 * locate a fixed threshold and sit beneath it.
 *
 * The values below are one refit's searched output, carried from the run whose
 * log is no longer in the repository. They are shown as an example of what the
 * search produces, never as the thresholds.
 */
export const BANDS = {
  stepUp: 0.53,
  hold: 0.63,
  decline: 0.74,
  block: 0.84,
  /** The five actions the four boundaries separate. */
  ladder: ['approve', 'step up', 'hold', 'decline', 'block'] as const,
  note: 'one refit’s grid-search output, re-searched every refit and jittered per episode',
  short: 'one refit’s search',
  cite: 'Section 5',
}

/**
 * Table 8, the capability ladder. Length and specificity grow together and the
 * instructions never become more adversarial: the tier is richness, not intent.
 *
 * Ordinal only. The ladder is anchored to published benchmark ladders that are
 * not cardinally comparable across editions, and the benchmark organisers say
 * so, so the paper claims only that success rises with capability across the
 * swept range. No absolute per-tier success rate is claimed, and none is shown.
 */
export const TIERS = [
  { tier: 0, length: '2 to 3 terse sentences', extra: null },
  { tier: 1, length: 'a short paragraph', extra: null },
  { tier: 2, length: '3 to 4 paragraphs with a clear order of events', extra: null },
  {
    tier: 3,
    length: '4 to 6 paragraphs with a clear timeline and specific references',
    extra: 'one further instruction, which differs by vertical',
  },
] as const

export const TIER_NOTE = {
  ordinalOnly: true,
  claim: 'success rises with capability across the swept range',
  notClaimed: 'any absolute per-tier success rate',
  cite: 'Table 8 and Section 10',
}
