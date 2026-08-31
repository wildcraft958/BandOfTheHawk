import { Panel } from '../components/ui/Panel'
import { KpiStrip } from '../components/ui/KpiStrip'
import { Badge } from '../components/ui/Badge'
import { Meter } from '../components/ui/Meter'
import { Chip } from '../components/ui/Chip'
import { Seismograph } from '../components/charts/Seismograph'
import { ClosedLoop } from '../components/diagrams/ClosedLoop'
import { coadapt, detectors, meta, points, runReport } from '../data/run'
import { VERTICALS } from '../data/taxonomy'
import { fixed, int, pct } from '../lib/format'

const VERTICAL_LABEL = new Map(VERTICALS.map((v) => [v.id, v.label]))

function tapeText(runs: Array<{ action: string; times: number }>): string {
  return runs.map((r) => (r.times > 1 ? `${r.action} x${r.times}` : r.action)).join(' > ')
}

export function Dashboard() {
  const full = detectors.configs.find((c) => c.id === 'gbdt_full')
  const op = detectors.operating_point
  const maxWeight = Math.max(...detectors.experts.map((e) => e.weight))
  const maxGain = Math.max(...detectors.feature_gains.map((f) => f.gain))
  const maxEpisodes = Math.max(...Object.values(runReport.per_vertical))
  const bands = detectors.fitted_bands

  return (
    <div className="space-y-4 px-4 py-5 sm:px-6">
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
            detail: `${pct(runReport.fraud_auth_share, 2)} of ${int(runReport.benign_auths)} benign`,
            tone: 'atk',
          },
          {
            label: 'pr-auc',
            value: fixed(full?.metrics.pr_auc, 4),
            detail: `recall ${pct(full?.metrics.recall_at_0p1)} at 0.1% FPR`,
            tone: 'pass',
          },
          {
            label: 'zero-shot recall',
            value: fixed(coadapt.zero_shot[0]?.recall, 3),
            detail: `${coadapt.zero_shot.length} verticals never trained on`,
            tone: 'holdout',
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
        name="arms race: value extracted per update"
        live
        aside={
          <span className="hidden text-[0.625rem] uppercase tracking-[0.09em] text-ink-3 sm:inline">
            {points.length} updates &middot; {coadapt.refit_updates.length} refits &middot; symlog
          </span>
        }
        bodyClassName="px-4 pb-3 pt-5"
      >
        <Seismograph points={points} refits={coadapt.refit_updates} />
        <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1 text-[0.625rem] text-ink-3">
          <span>
            <span className="text-value">&#9473;</span> value extracted
          </span>
          <span>
            <span className="text-def">&#9474;</span> defender refit
          </span>
          <span>
            <span className="text-def">&#9472;</span> zero line, {coadapt.reads.zeros}{' '}
            updates at exactly 0.0
          </span>
          <span className="text-ink-2">
            entropy {fixed(coadapt.reads.entropy_start, 3)} &rarr; peak{' '}
            {fixed(coadapt.reads.entropy_peak, 3)} &rarr; {fixed(coadapt.reads.entropy_end, 3)}
          </span>
        </div>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-[1.55fr_1fr]">
        <Panel
          name="attacker strategy stream"
          live
          aside={
            <span className="text-[0.625rem] uppercase tracking-[0.09em] text-ink-3">
              sampled at each refit
            </span>
          }
          bodyClassName="max-h-[26rem] overflow-y-auto"
        >
          <ul className="divide-y divide-rule-subtle">
            {coadapt.strategies.map((s) => {
              const isRefund = s.runs.some((r) => r.action === 'request_refund')
              const isPhish = s.runs.some((r) => r.action === 'phish_holder')
              return (
                <li
                  key={s.update}
                  className="flex items-start gap-3 px-4 py-2.5 hover:bg-surface-hover"
                >
                  <span className="num w-16 shrink-0 text-[0.6875rem] text-ink-3">
                    u{String(s.update).padStart(3, '0')}
                  </span>
                  <span className="min-w-0 flex-1 break-words text-[0.6875rem] text-ink">
                    {tapeText(s.runs)}
                    {s.truncated && <span className="text-ink-3"> &hellip;</span>}
                  </span>
                  <Chip tone={isRefund ? 'holdout' : isPhish ? 'atk' : 'def'}>
                    {isRefund ? 'refund' : isPhish ? 'phish' : 'ivr'}
                  </Chip>
                  <span className="num w-10 shrink-0 text-right text-[0.6875rem] text-ink-2">
                    {s.count}x
                  </span>
                </li>
              )
            })}
          </ul>
          <div className="border-t border-rule px-4 py-3 text-[0.6875rem] text-ink-2">
            Chains are truncated by the run log. The untruncated final strategy is below.
          </div>
        </Panel>

        <div className="space-y-4">
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
                    ['step_up', bands.step_up, 'challenge the cardholder', 'def'],
                    ['hold', bands.hold, 'freeze the card 24h', 'value'],
                    ['decline', bands.decline, 'freeze 72h', 'value'],
                    ['block', bands.block, 'unbind device, add to blocklist', 'atk'],
                  ] as const
                ).map(([name, threshold, mitigation, tone]) => (
                  <li key={name} className="flex items-baseline gap-3">
                    <span className="num w-12 shrink-0 text-[0.8125rem] text-ink">
                      {threshold.toFixed(2)}
                    </span>
                    <Chip tone={tone}>{name}</Chip>
                    <span className="text-[0.6875rem] text-ink-2">{mitigation}</span>
                  </li>
                ))}
              </ul>
            ) : null}
            <p className="mt-4 border-t border-rule pt-3 text-[0.6875rem] text-ink-3">
              Fitted from the cost curve. Axis is relative cost, never currency: amounts are unit
              where event value is not to hand.
            </p>
          </Panel>
        </div>
      </div>

      <Panel
        name="detector comparison"
        tone="pass"
        aside={
          <span className="hidden text-[0.625rem] uppercase tracking-[0.09em] text-ink-3 sm:inline">
            {int(detectors.train_rows)} train / {int(detectors.test_rows)} test &middot;{' '}
            {pct(detectors.base_rate, 2)} base rate
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
            <tr className="border-b border-rule text-[0.625rem] uppercase tracking-[0.09em] text-ink-3">
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
                      <p className="prose-sans mt-1.5 max-w-xl text-[0.75rem] leading-snug text-ink-3">
                        {c.note}
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
          <p className="prose-sans max-w-3xl text-[0.8125rem] text-ink-2">
            PR-AUC is the primary metric, not accuracy and not plain ROC-AUC. At a{' '}
            {pct(detectors.base_rate, 2)} base rate a classifier that always answers &ldquo;not
            fraud&rdquo; scores {pct(1 - detectors.base_rate, 1)} accuracy while catching nothing.
          </p>
          {op && (
            <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-rule-subtle pt-3">
              <Badge tone="neutral">derived</Badge>
              <span className="text-[0.75rem] text-ink-2">
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
          <p className="prose-sans mt-4 border-t border-rule pt-3 text-[0.75rem] text-ink-3">
            Per-entity features added {fixed(detectors.per_entity_lift, 4)} PR-AUC. The run&rsquo;s
            own verdict: they add little here.
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
            {coadapt.zero_shot.map((z) => (
              <Meter
                key={z.vertical}
                label={VERTICAL_LABEL.get(z.vertical) ?? z.vertical}
                value={0}
                max={maxEpisodes}
                display="0"
                tone="holdout"
                note={`held out of training, still caught at ${fixed(z.recall, 3)} recall`}
              />
            ))}
          </div>
        </Panel>
      </div>

      <Panel name="the closed loop" tone="value">
        <ClosedLoop stages={meta.stages} total={meta.total_seconds ?? 1} />
      </Panel>
    </div>
  )
}
