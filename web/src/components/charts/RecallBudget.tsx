import { Badge } from '../ui/Badge'
import { detectors } from '../../data/run'
import { fixed, int } from '../../lib/format'
import { Note } from '../ui/Note'

/**
 * Why this page shows more than one recall.
 *
 * The detector table reports recall at two false-positive rates and the
 * operating point reports recall at an alert budget, and the three numbers are
 * far apart: 0.9945, 0.9727 and 0.5410. Read without the alert counts that looks
 * like a contradiction, which is exactly how it read to a reader who asked.
 *
 * Recall is a function of how many alerts a team will review. Every row below is
 * the same detector on the same test set, and the only thing that changes is the
 * size of the queue. The alert counts are derived from the reported rates and the
 * test split, and the arithmetic is on screen so it can be checked.
 */
export function RecallBudget() {
  const full = detectors.configs.find((c) => c.id === 'gbdt_full')
  const op = detectors.operating_point
  const positives = detectors.test_fraud
  const total = detectors.test_rows
  if (!full || !op || positives == null || total == null) return null

  const negatives = total - positives


  const at = (fpr: number, recall: number) => {
    const falseAlerts = fpr * negatives
    const truePositives = recall * positives
    return { falseAlerts, truePositives, alerts: falseAlerts + truePositives, recall }
  }

  const tight = at(0.001, full.metrics.recall_at_0p1)
  const loose = at(0.01, full.metrics.recall_at_1)

  const table = [
    {
      basis: `alert budget of ${op.alert_budget}`,
      recall: op.recall,
      truePositives: op.true_positives,
      alerts: op.alert_budget,
      derived: true,
      note: 'a queue a real team can staff',
    },
    {
      basis: '0.1% false-positive rate',
      recall: tight.recall,
      truePositives: tight.truePositives,
      alerts: tight.alerts,
      derived: false,
      note: 'about twice that queue',
    },
    {
      basis: '1% false-positive rate',
      recall: loose.recall,
      truePositives: loose.truePositives,
      alerts: loose.alerts,
      derived: false,
      note: 'more than three times it',
    },
  ]

  return (
    <div>
      <p className="prose-sans max-w-3xl text-[0.9375rem] leading-relaxed text-ink-2">
        Three recalls appear on this page and they look like they disagree. They do not: recall is a
        function of how many alerts you are willing to review. Same detector, same{' '}
        {int(total)} test rows, same {int(positives)} fraud cases. Only the size of the queue
        changes.
      </p>

      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[34rem] border-collapse text-[0.9375rem]">
          <thead>
            <tr className="border-b border-rule text-left">
              {['operating point', 'alerts to review', 'fraud caught', 'recall'].map((h, i) => (
                <th
                  key={h}
                  className={`pb-2 text-[0.75rem] font-normal uppercase tracking-[0.09em] text-ink-3 ${
                    i === 0 ? 'pr-4' : 'pr-4 text-right'
                  }`}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.map((r) => (
              <tr key={r.basis} className="border-b border-rule-subtle">
                <td className="py-2.5 pr-4">
                  <span className="text-ink">{r.basis}</span>
                  {r.derived && (
                    <span className="ml-2 align-middle">
                      <Badge tone="neutral">derived</Badge>
                    </span>
                  )}
                  <span className="ml-2 text-[0.8125rem] text-ink-3">{r.note}</span>
                </td>
                <td className="num py-2.5 pr-4 text-right text-ink-2">{Math.round(r.alerts)}</td>
                <td className="num py-2.5 pr-4 text-right text-ink-2">
                  {Math.round(r.truePositives)} of {int(positives)}
                </td>
                <td className="num py-2.5 text-right text-ink">{fixed(r.recall, 4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Note label="the arithmetic behind these three rows">
        <p className="prose-sans max-w-3xl text-[0.875rem] leading-relaxed text-ink-2">
          {int(negatives)} negatives at a 1% false-positive rate is{' '}
          {Math.round(loose.falseAlerts)} false alerts, plus {Math.round(loose.truePositives)} true
          ones, so reaching {fixed(loose.recall, 4)} costs about {Math.round(loose.alerts)} alerts.
          Cut the queue to {op.alert_budget} and you keep {op.true_positives} of the fraud, which is
          recall {fixed(op.recall, 4)} at precision {fixed(op.precision, 2)}. The high recalls are
          real and so is the cost of collecting them.
        </p>
        <p className="prose-sans mt-2 max-w-3xl text-[0.875rem] leading-relaxed text-ink-3">
          The two false-positive rows are computed here from the reported rates and the test split,
          not read from the run. The budget row is the run&rsquo;s own reported precision at its
          budget, with recall and F1 derived from it.
        </p>
      </Note>
    </div>
  )
}
