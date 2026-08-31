/**
 * The only module that imports JSON.
 *
 * Vite turns each `import x from './y.json'` into a bundled literal at build
 * time, so the page never issues a network request for data. That is what keeps
 * the built single file genuinely self-contained rather than merely intended to
 * be.
 */
import coadaptRaw from './coadapt_metrics.json'
import detectorRaw from './detector_metrics.json'
import fidelityRaw from './fidelity.json'
import graphRaw from './graph.json'
import metaRaw from './meta.json'
import runReportRaw from './run_report.json'

import type {
  CoadaptMetrics,
  CoadaptPoint,
  DetectorMetrics,
  Fidelity,
  GraphFacts,
  Meta,
  RunReport,
} from './types'

export const meta = metaRaw as Meta
export const detectors = detectorRaw as unknown as DetectorMetrics
export const runReport = runReportRaw as unknown as RunReport
export const fidelity = fidelityRaw as unknown as Fidelity
export const graph = graphRaw as unknown as GraphFacts

const coadaptSource = coadaptRaw as unknown as CoadaptMetrics
const refitSet = new Set(coadaptSource.refit_updates)

/** The 150 co-adaptation updates, with the refit flag resolved onto each row. */
export const points: CoadaptPoint[] = coadaptSource.rows.map(
  ([update, extracted, policyReturn, entropy]) => ({
    update,
    extracted,
    policyReturn,
    entropy,
    refit: refitSet.has(update),
  }),
)

export const coadapt = coadaptSource

/**
 * Any data file still carrying the `_fixture` flag holds placeholder values, not
 * a measured result. The prototype shows a banner for as long as one is present,
 * so a placeholder cannot quietly reach a judge. Real generated output has no
 * such flag, so this is normally empty.
 */
export const fixtureFiles: string[] = Object.entries({
  'coadapt_metrics.json': coadaptRaw,
  'detector_metrics.json': detectorRaw,
  'fidelity.json': fidelityRaw,
  'graph.json': graphRaw,
  'meta.json': metaRaw,
  'run_report.json': runReportRaw,
})
  .filter(([, payload]) => (payload as Record<string, unknown>)._fixture === true)
  .map(([name]) => name)

if (import.meta.env.DEV) {
  // Tree-shaken from the production bundle. Catches a drifting extractor the
  // moment it drifts, rather than in front of a judge.
  console.assert(points.length === 150, `expected 150 updates, got ${points.length}`)
  console.assert(
    coadapt.refit_updates.join() === '11,23,35,47,59,71,83,95,107,119,131,143',
    `unexpected refit indices: ${coadapt.refit_updates.join()}`,
  )
  const sum = points.reduce((acc, p) => acc + p.extracted, 0)
  console.assert(
    Math.abs(sum - coadapt.checksum_extracted) < 0.5,
    `extraction checksum drift: ${sum.toFixed(1)} vs ${coadapt.checksum_extracted}`,
  )
  console.assert(detectors.configs.length === 5, 'expected 5 detector configurations')
  console.assert(fixtureFiles.length === 0, `synthetic fixtures bundled: ${fixtureFiles.join(', ')}`)
}
