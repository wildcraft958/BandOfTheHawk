import raw from './ablation_runs.json'

/**
 * The committed ablation runs, per update.
 *
 * This is the one part of the co-adaptation story that is DERIVED rather than
 * transcribed: tools/make_ablation_series.py reads the JSON files tracked at
 * artifacts/ablation/ and copies out only the series the site draws, so anyone
 * can open the same file and read the same number.
 *
 * These are not the runs the solution document reports. The direction agrees and
 * the magnitude does not, so the document's figures in data/paper.ts remain the
 * ones cited for any headline, and the two are never mixed.
 */
export interface AblationRun {
  arm: 'full' | 'ablated'
  seed: number
  /** The file this came from, so a reader can check it. */
  file: string
  /** Value extracted per episode, one entry per update. */
  extraction: number[]
  /** Policy entropy, one entry per update. */
  entropy: number[]
  /**
   * Update indices at which the defender refits, exactly as the pipeline
   * recorded them: zero based, and the refit happens AFTER that update's
   * episodes have been played and scored. Use refitUpdates() to read them in
   * the one based numbering the screen shows.
   */
  refits: number[]
  /** Share of genuine authorisations refused, one entry per refit. */
  friction: number[]
}

export interface AblationSeries {
  source: string
  note: string
  updates: number
  refits: number[]
  seeds: number
  runs: AblationRun[]
}

export const ablation = raw as AblationSeries

export const runFor = (arm: 'full' | 'ablated', seed: number): AblationRun | undefined =>
  ablation.runs.find((r) => r.arm === arm && r.seed === seed)

/**
 * The refits in the numbering a reader sees.
 *
 * The pipeline appends the loop counter, which starts at zero, once the refit
 * has already happened for that update. Update 1 on screen is therefore index
 * 0, and a refit recorded at index 5 fired at the end of update 6. Getting this
 * wrong puts the marker a block early and makes the chart claim a knockback one
 * update before the data shows one.
 */
export const refitUpdates = (run: AblationRun): number[] => run.refits.map((i) => i + 1)

if (import.meta.env.DEV) {
  console.assert(ablation.runs.length === 8, `expected 8 runs, got ${ablation.runs.length}`)
  console.assert(
    ablation.runs.every((r) => r.extraction.length === ablation.updates),
    'a run has the wrong number of updates',
  )
  console.assert(
    !JSON.stringify(raw).includes('zero_shot'),
    'the withheld measurement leaked into the bundled data',
  )
}
