import { Panel } from '../components/ui/Panel'
import { KpiStrip } from '../components/ui/KpiStrip'
import { Badge } from '../components/ui/Badge'
import { Meter } from '../components/ui/Meter'
import { Chip } from '../components/ui/Chip'
import { Ablation } from '../components/charts/Ablation'
import { RecallBudget } from '../components/charts/RecallBudget'
import { WithheldNote } from '../components/diagrams/WithheldNote'
import { FRICTION , PREVALENCE_CAVEAT , BANDS } from '../data/paper'
import { coadapt, detectors, runReport } from '../data/run'
import { VERTICALS } from '../data/taxonomy'
import { fixed, int, pct } from '../lib/format'
import { PageHead } from '../components/ui/PageHead'
import { Note } from '../components/ui/Note'

const VERTICAL_LABEL = new Map(VERTICALS.map((v) => [v.id, v.label]))

export function Dashboard() {
  const full = detectors.configs.find((c) => c.id === 'gbdt_full')
  const op = detectors.operating_point
  const maxWeight = Math.max(...detectors.experts.map((e) => e.weight))
  // Learned combiner against fixed average: the second of the paper's two
  // ablations in this table, computed from the rows themselves.
  const learnedPr = detectors.configs.find((c) => c.id === 'experts_learned')?.metrics.pr_auc
  const fixedPr = detectors.configs.find((c) => c.id === 'experts_fixed')?.metrics.pr_auc
  const combinerLift =
    learnedPr != null && fixedPr != null ? learnedPr - fixedPr : null
  const maxGain = Math.max(...detectors.feature_gains.map((f) => f.gain))
  const maxEpisodes = Math.max(...Object.values(runReport.per_vertical))
  const bands = BANDS

  return (
    <div className="space-y-4 px-4 py-5 sm:px-6">
      <PageHead
        title="Detection"
        blurb="Five detector configurations, what each expert is weighted at, and what recall actually costs at a given alert budget."
      />

      <KpiStrip
        items={[
          {
            label: 'fraud episodes',
            value: int(runReport.episodes),
            detail: `${int(runReport.reached_monetized)} reached monetized`,
          },
          {
            label: 'fraud auths',
            value: int(runReport.fraud_auths),
            detail: `${pct(runReport.fraud_auth_share, 1)} of authorisations in this run`,
            tone: 'atk',
          },
          {
            label: 'pr-auc',
            value: fixed(full?.metrics.pr_auc, 4),
            detail: `recall ${pct(full?.metrics.recall_at_0p1)} at 0.1% FPR`,
            tone: 'pass',
          },
          {
            label: 'genuine refused',
            value: pct(FRICTION.full, 2),
            detail: `mean over ${FRICTION.refits} refits, up to ${pct(FRICTION.range[1], 1)}`,
            tone: 'def',
          },
          {
            label: 'peak extraction',
            value: int(coadapt.reads.extracted_max),
            detail: 'per episode, model units',
            tone: 'value',
          },
        ]}
      />

      <Panel
        name="detector comparison"
        tone="pass"
        aside={
          <span className="hidden text-[0.75rem] uppercase tracking-[0.09em] text-ink-3 sm:inline">
            {int(detectors.train_rows)} train / {int(detectors.test_rows)} test &middot;{' '}
            {int(detectors.train_fraud)} / {int(detectors.test_fraud)} fraud
          </span>
        }
        bodyClassName="overflow-x-auto"
      >
        <table className="w-full min-w-[46rem] border-collapse text-left">
          <caption className="sr-only">
            Five detector configurations compared on PR-AUC, ROC-AUC, recall at fixed false
            positive rates, and precision at an alert budget of 100.
          </caption>
          <thead>
            <tr className="border-b border-rule text-[0.75rem] uppercase tracking-[0.09em] text-ink-3">
              <th scope="col" className="px-4 py-2.5 font-semibold">Configuration</th>
              <th scope="col" className="px-3 py-2.5 text-right font-semibold">PR-AUC</th>
              <th scope="col" className="px-3 py-2.5 text-right font-semibold">ROC-AUC</th>
              <th scope="col" className="px-3 py-2.5 text-right font-semibold">R@0.1%</th>
              <th scope="col" className="px-3 py-2.5 text-right font-semibold">R@1%</th>
              <th scope="col" className="px-3 py-2.5 text-right font-semibold">P@budget</th>
            </tr>
          </thead>
          <tbody>
            {detectors.configs.map((c) => {
              const best = c.id === 'gbdt_full'
              return (
                <tr key={c.id} className="border-b border-rule-subtle align-top hover:bg-surface-hover">
                  <td className="px-4 py-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className={best ? 'text-ink' : 'text-ink-2'}>{c.label}</span>
                      {best && <Badge tone="pass">best</Badge>}
                      {c.family === 'rule' && <Badge tone="atk">baseline</Badge>}
                    </div>
                    {c.note && (
                      <p className="prose-sans mt-1.5 max-w-xl text-[0.875rem] leading-snug text-ink-3">
                        {c.note}
                        {/* The paper reports two ablations in this table and the note
                            above carries only the losing one. The other one is the
                            test of whether learning the weights was worth it, and it
                            passed. Subtracted from the two rows above rather than
                            asserted, so it cannot drift from them. */}
                        {c.id === 'experts_learned' && combinerLift != null && (
                          <>
                            {' '}
                            It does beat the fixed average by{' '}
                            <span className="num text-pass">{fixed(combinerLift, 4)}</span> PR-AUC,
                            so combining the expert opinions is worth learning even where the
                            decomposition itself costs accuracy.
                          </>
                        )}
                      </p>
                    )}
                  </td>
                  <td className="num px-3 py-3 text-right text-ink">{fixed(c.metrics.pr_auc, 4)}</td>
                  <td
                    className={`num px-3 py-3 text-right ${c.metrics.roc_auc < 0.5 ? 'text-atk' : 'text-ink-2'}`}
                  >
                    {fixed(c.metrics.roc_auc, 4)}
                  </td>
                  <td className="num px-3 py-3 text-right text-ink-2">
                    {fixed(c.metrics.recall_at_0p1, 4)}
                  </td>
                  <td className="num px-3 py-3 text-right text-ink-2">
                    {fixed(c.metrics.recall_at_1, 4)}
                  </td>
                  <td className="num px-3 py-3 text-right text-ink-2">
                    {fixed(c.metrics.precision_at_budget, 4)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>

        <div className="border-t border-rule px-4 py-4">
          <Note label="why PR-AUC is the primary metric here" className="mt-0 border-t-0 pt-0">
            <p className="prose-sans max-w-3xl text-[0.9375rem] text-ink-2">
              PR-AUC is the primary metric, not accuracy and not plain ROC-AUC. At a{' '}
              {pct(PREVALENCE_CAVEAT.measuredAt, 0)} fraud share a classifier that always answers
              &ldquo;not fraud&rdquo; scores {pct(1 - PREVALENCE_CAVEAT.measuredAt, 0)} accuracy
              while catching nothing, and at the {pct(PREVALENCE_CAVEAT.deployedNearer, 1)} a real
              portfolio carries it scores {pct(1 - PREVALENCE_CAVEAT.deployedNearer, 1)}.
            </p>
        </Note>
          {op && (
            <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-rule-subtle pt-3">
              <Badge tone="neutral">derived</Badge>
              <span className="text-[0.875rem] text-ink-2">
                At an alert budget of {op.alert_budget}: precision{' '}
                <span className="num text-ink">{fixed(op.precision, 4)}</span>, recall{' '}
                <span className="num text-ink">{fixed(op.recall, 4)}</span>, F1{' '}
                <span className="num text-ink">{fixed(op.f1, 4)}</span>
              </span>
            </div>
          )}
        </div>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel name="defender expert weights" tone="def">
          <div className="space-y-3.5">
            {detectors.experts.map((e) => (
              <Meter
                key={e.name}
                label={e.name}
                value={e.weight}
                max={maxWeight}
                display={`${pct(e.normalized_weight)}`}
                tone={e.name === 'text' ? 'atk' : 'def'}
                note={
                  e.name === 'text'
                    ? 'the thinnest channel, and the one the attacker converged on'
                    : undefined
                }
              />
            ))}
          </div>
        </Panel>

        <Panel name="risk bands → mitigation" tone="value">
          {bands ? (
            <ul className="space-y-2.5">
              {(
                [
                  ['step_up', bands.stepUp, 'challenge the cardholder', 'def'],
                  ['hold', bands.hold, 'freeze the card 24h', 'value'],
                  ['decline', bands.decline, 'freeze 72h', 'value'],
                  ['block', bands.block, 'unbind device, add to blocklist', 'atk'],
                ] as const
              ).map(([name, threshold, mitigation, tone]) => (
                <li key={name} className="flex items-baseline gap-3">
                  <span className="num w-12 shrink-0 text-[0.9375rem] text-ink">
                    {threshold.toFixed(2)}
                  </span>
                  <Chip tone={tone}>{name}</Chip>
                  <span className="text-[0.8125rem] text-ink-2">{mitigation}</span>
                </li>
              ))}
            </ul>
          ) : null}
          <Note
            label="why these four numbers move"
          >
            <p className=" text-[0.8125rem] leading-relaxed text-ink-3">
            One refit&rsquo;s grid-search output, not fixed thresholds. The four boundaries are
            re-searched against the cost curve at every refit, and jitter moves them slightly on
            every episode so a policy cannot find a threshold and sit under it. Axis is relative
            cost, never currency: amounts are unit where event value is not to hand.
          </p>
        </Note>
        </Panel>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel name="top features by gain" tone="pass">
          <div className="space-y-3">
            {detectors.feature_gains.slice(0, 8).map((f) => (
              <Meter
                key={f.name}
                label={f.name}
                value={f.gain}
                max={maxGain}
                display={f.gain.toFixed(1)}
                tone={f.per_entity ? 'holdout' : 'pass'}
                note={f.per_entity ? 'per-entity feature' : undefined}
              />
            ))}
          </div>
          <p className="prose-sans mt-4 border-t border-rule pt-3 text-[0.875rem] text-ink-3">
            Per-entity features are marked. The ablation below measures what they are worth.
          </p>
        </Panel>

        <Panel name="episodes per vertical" tone="atk">
          <div className="space-y-3">
            {Object.entries(runReport.per_vertical)
              .sort((a, b) => b[1] - a[1])
              .map(([id, count]) => (
                <Meter
                  key={id}
                  label={VERTICAL_LABEL.get(id) ?? id}
                  value={count}
                  max={maxEpisodes}
                  display={String(count)}
                  tone="atk"
                />
              ))}
          </div>
          <Note
            label="the two verticals absent from this table"
          >
            <p className="prose-sans text-[0.875rem] leading-relaxed text-ink-2">
            Two further verticals, SIM swap and refund abuse, are designated held out and so appear
            in no row above. What the defender scores on them is not reported, for the reason below.
          </p>
        </Note>
          <div className="mt-3">
            <WithheldNote compact />
          </div>
        </Panel>
      </div>

      <Panel name="why this page shows more than one recall" tone="def">
        <RecallBudget />
      </Panel>

      <Panel name="ablation: what the per-entity features are worth" tone="holdout">
        <Ablation />
      </Panel>

    </div>
  )
}
