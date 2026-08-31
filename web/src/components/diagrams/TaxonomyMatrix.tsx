import { Badge } from '../ui/Badge'
import { VERTICALS } from '../../data/taxonomy'
import { runReport } from '../../data/run'

const STAGE_TONE = {
  none: 'text-ink-3',
  acquired: 'text-ink-2',
  bound: 'text-ink',
  monetized: 'text-value',
  terminal: 'text-ink-3',
} as const

/**
 * Eleven identified verticals across two axes: where in the account lifecycle
 * the attack enters, and which generative capability changes the attacker's
 * cost. Two axes rather than a flat list, because the claim being graded is
 * that the surface is spanned rather than sampled.
 */
export function TaxonomyMatrix() {
  const episodes = runReport.per_vertical
  const simulated = VERTICALS.filter((v) => v.simulated)
  const excluded = VERTICALS.filter((v) => !v.simulated)

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[52rem] border-collapse text-left">
        <caption className="sr-only">
          Identified GenAI-enabled payment fraud verticals, their entry stage, the generative
          capability each requires, and how many episodes were simulated.
        </caption>
        <thead>
          <tr className="border-b border-rule-strong font-mono text-[0.625rem] uppercase tracking-[0.12em] text-ink-3">
            <th scope="col" className="py-2 pr-4 font-medium">Vertical</th>
            <th scope="col" className="py-2 pr-4 font-medium">Enters at</th>
            <th scope="col" className="py-2 pr-4 font-medium">Generative capability</th>
            <th scope="col" className="py-2 pr-4 text-right font-medium">Episodes</th>
          </tr>
        </thead>
        <tbody>
          {simulated.map((v) => (
            <tr key={v.id} className="border-b border-rule align-top">
              <td className="py-3 pr-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-ink">{v.label}</span>
                  {v.heldOut && <Badge tone="holdout">held out</Badge>}
                </div>
                <div className="mt-1 max-w-md text-[0.8125rem] text-ink-3">{v.blurb}</div>
              </td>
              <td className={`py-3 pr-4 font-mono text-[0.6875rem] uppercase ${STAGE_TONE[v.entryStage]}`}>
                {v.entryStage}
              </td>
              <td className="py-3 pr-4 text-[0.8125rem] text-ink-2">
                {v.capability.startsWith('none') ? (
                  <span className="text-ink-3">{v.capability}</span>
                ) : (
                  v.capability
                )}
              </td>
              <td className="num py-3 pr-4 text-right text-ink">
                {v.heldOut ? (
                  <span className="font-mono text-[0.6875rem] text-holdout">0 — zero-shot</span>
                ) : (
                  (episodes[v.id] ?? '—')
                )}
              </td>
            </tr>
          ))}

          {excluded.map((v) => (
            <tr key={v.id} className="border-b border-rule align-top opacity-55">
              <td className="py-3 pr-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-ink-2 line-through decoration-ink-3/50">{v.label}</span>
                  <Badge tone="neutral">not simulated</Badge>
                </div>
                <div className="mt-1 max-w-md text-[0.8125rem] text-ink-3">{v.exclusion}</div>
              </td>
              <td className="py-3 pr-4 font-mono text-[0.6875rem] uppercase text-ink-3">
                {v.entryStage}
              </td>
              <td className="py-3 pr-4 text-[0.8125rem] text-ink-3">{v.capability}</td>
              <td className="py-3 pr-4 text-right font-mono text-[0.6875rem] text-ink-3">—</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
