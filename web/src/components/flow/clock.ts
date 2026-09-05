/**
 * When each stage of the architecture flow wakes.
 *
 * Data, not components, so the nodes and the darts travelling the edges read
 * one schedule. Nothing schedules itself: every part takes a delay from here.
 * Two animations of the same graph on two different clocks is how a dart ends
 * up passing through a node long before that node lights up.
 */

/** One hop. Also a dart's duration, so a dart lands on the beat its node wakes. */
export const STEP = 620

/**
 * The two sides share a slot wherever they genuinely run concurrently: attack
 * and benign traffic are generated at the same time, and the flat detector and
 * the expert mixture score the same rows.
 */
export const AT = {
  world: 0,
  sides: STEP * 1.3,
  builder: STEP * 2.8,
  log: STEP * 3.9,
  detectors: STEP * 5.2,
  bands: STEP * 6.7,
  refit: STEP * 7.8,
}

/** The full pass, then a beat before it runs again. */
export const RUN = AT.refit + 2800

/** The edge feeding a stage starts one hop earlier, so its dart arrives on time. */
export const feed = (to: number): number => Math.max(0, to - STEP)
