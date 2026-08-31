import { useState } from 'react'
import { Panel } from '../components/ui/Panel'
import { KpiStrip } from '../components/ui/KpiStrip'
import { Badge } from '../components/ui/Badge'
import { Chip } from '../components/ui/Chip'
import { Label } from '../components/ui/Label'
import { Tape } from '../components/ui/Tape'
import { ArmsRaceChart } from '../components/charts/ArmsRaceChart'
import { EntropyTrack } from '../components/charts/EntropyTrack'
import { PosteriorChart } from '../components/charts/PosteriorChart'
import { coadapt, detectors, points } from '../data/run'
import { VERTICALS } from '../data/taxonomy'
import { fixed, int } from '../lib/format'

const HELD_OUT = new Set(VERTICALS.filter((v) => v.heldOut).map((v) => v.id))
const LABEL = new Map(VERTICALS.map((v) => [v.id, v.label]))

export function Loop() {
  const [selected, setSelected] = useState<number | null>(11)

  const sample = coadapt.strategies.find((s) => s.update === selected)
  const final = coadapt.final_sequences[0]
  const textExpert = detectors.experts.find((e) => e.name === 'text')
  const reads = coadapt.reads

  return (
    <div className="space-y-4 px-4 py-5 sm:px-6">
      <KpiStrip
        items={[
          { label: 'updates', value: points.length, detail: 'live co-adaptation phase' },
          {
            label: 'defender refits',
            value: coadapt.refit_updates.length,
            detail: 'every 12 updates',
            tone: 'def',
          },
          {
            label: 'updates at zero',
            value: reads.zeros,
            detail: 'attacker fully shut out',
            tone: 'pass',
          },
          {
            label: 'peak extraction',
            value: int(reads.extracted_max),
            detail: `started ${int(reads.extracted_first)}, ended ${int(reads.extracted_last)}`,
            tone: 'value',
          },
          {
            label: 'entropy',
            value: fixed(reads.entropy_end, 3),
            detail: `from ${fixed(reads.entropy_start, 3)}, peaked ${fixed(reads.entropy_peak, 3)}`,
            tone: 'holdout',
          },
        ]}
      />

      <Panel
        name="value extracted, with every defender refit"
        live
        aside={
          <span className="hidden text-[0.625rem] uppercase tracking-[0.09em] text-ink-3 md:inline">
            pick a marker to see what the attacker was running
          </span>
        }
      >
        <ArmsRaceChart
          points={points}
          refits={coadapt.refit_updates}
          selected={selected}
          onSelect={setSelected}
        />

        <div className="mt-4 border-t border-rule pt-4">
          {sample ? (
            <>
              <div className="flex flex-wrap items-center gap-3">
                <Label>strategy sampled before the refit at update {sample.update}</Label>
                <Badge tone="atk">{sample.count} occurrences</Badge>
                {sample.truncated && <Badge tone="neutral">truncated by the log</Badge>}
              </div>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {sample.runs.map((r, i) => (
                  <Chip key={i} tone={r.action === 'request_refund' ? 'holdout' : 'atk'}>
                    {r.action}
                    {r.times > 1 ? ` x${r.times}` : ''}
                  </Chip>
                ))}
              </div>
            </>
          ) : (
            <p className="text-[0.75rem] text-ink-3">
              No refit selected. Pick one of the twelve markers above.
            </p>
          )}
        </div>
      </Panel>

      <Panel name="policy entropy: the refit forces re-exploration" tone="def">
        <EntropyTrack points={points} refits={coadapt.refit_updates} />
        <p className="prose-sans mt-3 max-w-3xl text-[0.8125rem] leading-relaxed text-ink-2">
          Entropy does not fall steadily. It holds near{' '}
          <span className="num text-ink">{fixed(reads.entropy_start, 3)}</span> through the first
          refit, then jumps above{' '}
          <span className="num text-def">{fixed(reads.entropy_peak, 3)}</span> for the twenty
          updates that follow, and only then decays to{' '}
          <span className="num text-ink">{fixed(reads.entropy_end, 3)}</span>. The refit did not
          only crush the attacker&rsquo;s income. It destroyed a converged policy and threw it
          back into exploration, and what it converged on next was tighter than where it started.
        </p>
      </Panel>

      <Panel name="the result worth reading twice" tone="holdout">
        <div className="grid gap-6 lg:grid-cols-[1.1fr_1fr]">
          <div>
            <p className="prose-sans text-[0.9375rem] leading-relaxed text-ink">
              The attacker converged on buy credentials, reset the password, then request refunds.
              That chain appears in{' '}
              <span className="text-holdout">none of the seven trained verticals</span>. Refund
              abuse was one of the two verticals deliberately held out of training, and the policy
              rediscovered it on its own.
            </p>
            <p className="prose-sans mt-4 text-[0.9375rem] leading-relaxed text-ink">
              It found the route through the defender&rsquo;s thinnest channel. The text expert
              carries{' '}
              <span className="num text-def">
                {fixed(textExpert?.weight, 3)} of {fixed(
                  detectors.experts.reduce((a, e) => a + e.weight, 0),
                  3,
                )}
              </span>{' '}
              combiner weight, or{' '}
              <span className="text-def">
                {textExpert ? (textExpert.normalized_weight * 100).toFixed(1) : '4.7'}%
              </span>
              . The refund loop loads almost entirely on text and binding.
            </p>
            <p className="prose-sans mt-4 text-[0.9375rem] leading-relaxed text-ink">
              The defender caught it anyway.
            </p>

            <ul className="mt-5 space-y-2">
              {coadapt.zero_shot.map((z) => (
                <li
                  key={z.vertical}
                  className="flex items-center justify-between gap-4 rounded-panel border border-rule bg-surface px-3 py-2.5"
                >
                  <span className="flex items-center gap-2.5">
                    <Badge tone="holdout">held out</Badge>
                    <span className="text-[0.8125rem] text-ink">
                      {LABEL.get(z.vertical) ?? z.vertical}
                    </span>
                  </span>
                  <span className="num text-[0.875rem] text-pass">
                    {fixed(z.recall, 3)} recall
                  </span>
                </li>
              ))}
            </ul>
            <p className="prose-sans mt-3 text-[0.75rem] text-ink-3">
              Recall is measured at a 0.5 threshold on a small positive count. Both held-out
              verticals scored 1.000, which is a strong result on a narrow base and is reported as
              such.
            </p>
          </div>

          <div>
            <Label>how the strategy mutated, sampled at each refit</Label>
            <ol className="mt-3 space-y-1.5">
              {coadapt.strategies.map((s) => {
                const isRefund = s.runs.some((r) => r.action === 'request_refund')
                const active = selected === s.update
                return (
                  <li key={s.update}>
                    <button
                      type="button"
                      onClick={() => setSelected(s.update)}
                      className={`flex w-full items-center gap-3 rounded-panel border px-2.5 py-1.5 text-left transition-colors duration-150 ${
                        active
                          ? 'border-atk/60 bg-atk/10'
                          : 'border-rule hover:border-ink-3/50 hover:bg-surface-hover'
                      }`}
                    >
                      <span className="num w-10 shrink-0 text-[0.625rem] text-ink-3">
                        u{s.update}
                      </span>
                      <span className="min-w-0 flex-1 truncate text-[0.625rem] text-ink-2">
                        {s.runs.map((r) => r.action).join(' > ')}
                      </span>
                      <Chip tone={isRefund ? 'holdout' : 'atk'}>
                        {isRefund ? 'refund' : s.runs[0]?.action === 'phish_holder' ? 'phish' : 'ivr'}
                      </Chip>
                    </button>
                  </li>
                )
              })}
            </ol>
            <p className="prose-sans mt-3 text-[0.75rem] text-ink-2">
              IVR provisioning at update 11, phish and drain at 23 and 35, then the refund loop
              from 47 onward. It never left.
            </p>
          </div>
        </div>
      </Panel>

      {final && (
        <Panel
          name="the final trained policy, at full length"
          tone="atk"
          aside={<Badge tone="atk">{final.count} occurrences</Badge>}
        >
          <Tape runs={final.runs} />
          <p className="prose-sans mt-5 max-w-3xl border-t border-rule pt-4 text-[0.8125rem] leading-relaxed text-ink-2">
            Two setup actions and then one action repeated, which exactly saturates the forty-action
            episode budget set in <span className="text-ink">configs/simulation.yaml</span>. This is
            a degenerate exploit, and stating that is part of the result: a learned attacker will
            find the cheapest repeatable path, not an interesting one. Nobody scripted this chain.
          </p>
        </Panel>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel name="what victim selection learned" tone="def">
          <PosteriorChart groups={coadapt.selection.groups} />
          <p className="prose-sans mt-5 border-t border-rule pt-3 text-[0.8125rem] text-ink-2">
            A contextual bandit over {int(coadapt.selection.observations)} candidates, actively
            selecting. It learned to prefer one BIN tier strongly and to avoid another, and to
            prefer cards older than three years. Nothing told it which cards were worth attacking.
          </p>
        </Panel>

        <Panel name="how the loop was bootstrapped" tone="pass">
          <dl className="grid grid-cols-2 gap-x-6 gap-y-3">
            {(
              [
                ['initial defender positives', int(coadapt.warm_start.initial_defender_fraud)],
                ['behaviour-clone final loss', fixed(coadapt.warm_start.bc_final_loss, 4)],
                ['critic final loss', fixed(coadapt.warm_start.critic_final_loss, 1)],
                ['refit cadence', 'every 12 updates'],
                ['episodes per update', '80'],
                ['label latency', '4320 min'],
              ] as const
            ).map(([k, v]) => (
              <div key={k}>
                <Label>{k}</Label>
                <dd className="num mt-0.5 text-[0.875rem] text-ink">{v}</dd>
              </div>
            ))}
          </dl>
          <p className="prose-sans mt-5 border-t border-rule pt-3 text-[0.8125rem] text-ink-2">
            The defender always trains on stale labels: a 4320 minute latency is three days, which
            is what a real chargeback cycle looks like. The critic loss is large because the reward
            scale is large, not because the critic failed.
          </p>
          <p className="prose-sans mt-3 text-[0.75rem] text-ink-3">
            Held out of training entirely:{' '}
            {[...HELD_OUT].map((id) => LABEL.get(id) ?? id).join(' and ')}.
          </p>
        </Panel>
      </div>
    </div>
  )
}
