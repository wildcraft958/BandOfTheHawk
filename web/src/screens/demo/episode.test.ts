/**
 * Tests for the episode engine. Run with: npm run test:episode
 *
 * Two of these exist because the first two versions of the scoring were wrong in
 * opposite directions, and only a spread check caught either. Scoring on the
 * action's cost alone capped the score around 0.33 against a decline boundary of
 * 0.74, so nothing was ever stopped. Replacing it with an unbounded velocity
 * term made every episode past about six steps a certain block. The engine is
 * only interesting if both outcomes are common, so that is asserted rather than
 * eyeballed.
 */
import { runEpisode, legalityHolds } from './episode'
import { VERTICALS } from '../../data/taxonomy'

let fails = 0
const check = (name: string, ok: boolean, extra = '') => {
  console.log(`  ${ok ? 'pass' : 'FAIL'}  ${name}${extra ? '  ' + extra : ''}`)
  if (!ok) fails += 1
}

// 1. Determinism: same seed, same tape.
const a = runEpisode({ seed: 42, entryStage: 'none', tier: 2 })
const b = runEpisode({ seed: 42, entryStage: 'none', tier: 2 })
check('same seed gives an identical tape', JSON.stringify(a) === JSON.stringify(b))

// 2. Different seeds actually differ.
const c = runEpisode({ seed: 43, entryStage: 'none', tier: 2 })
check('a different seed gives a different tape', JSON.stringify(a) !== JSON.stringify(c))

// 3. Legality holds for every vertical across many seeds.
let illegal = 0
for (const v of VERTICALS.filter((v) => v.simulated)) {
  for (let seed = 0; seed < 300; seed += 1) {
    const ep = runEpisode({ seed, entryStage: v.entryStage, tier: 3 })
    if (!legalityHolds(ep, v.entryStage)) illegal += 1
  }
}
check('no illegal action at any stage', illegal === 0, `${illegal} violations over ${9 * 300} episodes`)

// 4. A stopped episode stops: no steps after the stopping decision.
let trailing = 0
for (let seed = 0; seed < 500; seed += 1) {
  const ep = runEpisode({ seed, entryStage: 'none', tier: 1 })
  if (!ep.stopped) continue
  const at = ep.steps.findIndex((s) => s.decision === 'decline' || s.decision === 'block')
  if (at !== -1 && at !== ep.steps.length - 1) trailing += 1
}
check('a decline or block ends the episode', trailing === 0, `${trailing} tapes continued past a stop`)

// 5. Extraction only ever accrues at monetized.
let early = 0
for (let seed = 0; seed < 500; seed += 1) {
  const ep = runEpisode({ seed, entryStage: 'none', tier: 3 })
  if (ep.extracted > 0 && !['monetized', 'terminal'].includes(ep.finalStage)) early += 1
}
check('extraction requires reaching monetized', early === 0, `${early} episodes extracted without it`)

// 6. A real spread: both outcomes must be common, or it is not a game.
let stopped = 0
let succeeded = 0
const decisions = new Set<string>()
for (let seed = 0; seed < 400; seed += 1) {
  const ep = runEpisode({ seed, entryStage: 'none', tier: 2 })
  if (ep.stopped) stopped += 1
  if (ep.finalStage === 'monetized' && !ep.stopped) succeeded += 1
  for (const st of ep.steps) if (st.decision) decisions.add(st.decision)
}
const pctStopped = (stopped / 400) * 100
const pctWon = (succeeded / 400) * 100
check('the defender stops a real share', pctStopped >= 15, pctStopped.toFixed(0) + '% stopped')
check('the attacker succeeds a real share', pctWon >= 15, pctWon.toFixed(0) + '% reached monetized')
check('all five decisions are reachable', decisions.size === 5, [...decisions].sort().join(', '))

// 7. A higher capability tier should help, not hurt.
const winsAt = (tier: number) => {
  let n = 0
  for (let seed = 0; seed < 400; seed += 1) {
    const ep = runEpisode({ seed, entryStage: 'none', tier })
    if (!ep.stopped) n += 1
  }
  return n
}
const low = winsAt(0)
const high = winsAt(3)
check('a higher tier survives more often', high >= low, `tier 0: ${low}, tier 3: ${high}`)

if (fails > 0) {
  // Throwing gives tsx a non-zero exit without pulling node types into the
  // app's typecheck, which this file shares.
  throw new Error(`${fails} engine test${fails === 1 ? '' : 's'} failed`)
}
console.log('\n  all engine tests pass')
