import { Badge } from '../ui/Badge'
import { detectors } from '../../data/run'
import { fixed, int } from '../../lib/format'
import type { Metric } from '../../data/types'
import { Note } from '../ui/Note'

/**
 * The per-entity feature ablation, as a head to head rather than one scalar.
 *
 * Both columns are real configurations from the same run, so the difference is
 * the feature block and nothing else. The whole row is shown because a single
 * scalar hides which metrics moved at all, but the result is a null one and is
 * reported that way: the one non-zero cell is worth three fraud cases out of
 * 183, which is why the solution document calls the ablation null.
 */
const ROWS: Array<{ key: keyof Metric; label: string; note: string }> = [
  { key: 'pr_auc', label: 'PR-AUC', note: 'the primary metric for rare positives' },
  { key: 'roc_auc', label: 'ROC-AUC', note: 'flattered by true negatives' },
  { key: 'recall_at_0p1', label: 'recall at 0.1% FPR', note: 'tight budget' },
  { key: 'recall_at_1', label: 'recall at 1% FPR', note: 'looser budget' },
  { key: 'precision_at_budget', label: 'precision at budget', note: 'reviewer queue purity' },
]

export function Ablation() {
  const withFeatures = detectors.configs.find((c) => c.id === 'gbdt_full')
  const without = detectors.configs.find((c) => c.id === 'gbdt_no_per_entity')
  if (!withFeatures || !without) return null

  const tightDelta =
    withFeatures.metrics.recall_at_0p1 - without.metrics.recall_at_0p1

  return (
    <div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[30rem] border-collapse text-[0.9375rem]">
          <thead>
            <tr className="border-b border-rule text-left">
              <th className="pb-2 pr-4 font-normal text-[0.75rem] uppercase tracking-[0.09em] text-ink-3">
                metric
              </th>
              <th className="pb-2 pr-4 text-right font-normal text-[0.75rem] uppercase tracking-[0.09em] text-ink-3">
                with
              </th>
              <th className="pb-2 pr-4 text-right font-normal text-[0.75rem] uppercase tracking-[0.09em] text-ink-3">
                without
              </th>
              <th className="pb-2 text-right font-normal text-[0.75rem] uppercase tracking-[0.09em] text-ink-3">
                delta
              </th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row) => {
              const a = withFeatures.metrics[row.key] as number
              const b = without.metrics[row.key] as number
              const delta = a - b
              const tone =
                Math.abs(delta) < 0.001 ? 'text-ink-3' : delta > 0 ? 'text-pass' : 'text-atk'
              return (
                <tr key={row.key} className="border-b border-rule-subtle">
                  <td className="py-2 pr-4">
                    <span className="text-ink">{row.label}</span>
                    <span className="ml-2 text-[0.8125rem] text-ink-3">{row.note}</span>
                  </td>
                  <td className="num py-2 pr-4 text-right text-ink">{fixed(a, 4)}</td>
                  <td className="num py-2 pr-4 text-right text-ink-2">{fixed(b, 4)}</td>
                  <td className={`num py-2 text-right ${tone}`}>
                    {delta >= 0 ? '+' : ''}
                    {fixed(delta, 4)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-3">
        <Badge tone="neutral">null result</Badge>
        <span className="num text-[0.8125rem] text-ink-3">
          PR-AUC lift {fixed(detectors.per_entity_lift, 4)}
        </span>
      </div>
      <Note
        label="why this is a table and not a single number"
      >
        <p className="prose-sans text-[0.875rem] leading-relaxed text-ink-2">
        Same model, same split, one feature block removed. The per-entity features are the ones that
        ask whether an event is unusual for this particular card. They are worth{' '}
        {fixed(detectors.per_entity_lift, 4)} PR-AUC, which is nothing, and the ablation is reported
        as the null result it is. The whole row is shown so the reader can see that four of the five
        metrics do not move at all rather than take one scalar on trust.
      </p>
        <p className="prose-sans mt-2 text-[0.875rem] leading-relaxed text-ink-2">
        The one cell that moves is recall at the 0.1% budget, by{' '}
        <span className="num">{fixed(tightDelta, 4)}</span>. Read against{' '}
        {int(withFeatures.metrics.n_positives)} positives that is{' '}
        {Math.round(tightDelta * (withFeatures.metrics.n_positives ?? 0))} fraud cases, so it does
        not rescue the block either.
      </p>
        <p className="prose-sans mt-2 text-[0.875rem] leading-relaxed text-ink-3">
        These features are not inert: they appear in the top twelve by gain. They are redundant
        against device age, time since last authorisation and the entry-mode aggregates, which rank
        far above them. The per-entity work earned its place in benign fidelity, which is what it was
        for. A detector facing this red team does not need it.
      </p>
      </Note>
    </div>
  )
}
