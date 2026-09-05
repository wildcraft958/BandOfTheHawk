import { useState } from 'react'
import { Panel } from '../components/ui/Panel'
import { KpiStrip } from '../components/ui/KpiStrip'
import { Badge } from '../components/ui/Badge'
import { Chip } from '../components/ui/Chip'
import { Label } from '../components/ui/Label'
import { Tape } from '../components/ui/Tape'
import { WithheldNote } from '../components/diagrams/WithheldNote'
import { ABLATION, COADAPT_SETUP, COEVOLUTION, DIVERSITY, FRICTION } from '../data/paper'
import { LoopStepper } from '../components/charts/LoopStepper'
import { LoopRunning } from '../components/charts/LoopRunning'
import { CoEvolution } from '../components/charts/CoEvolution'
import { StealthAblation } from '../components/charts/StealthAblation'
import { PosteriorChart } from '../components/charts/PosteriorChart'
import { coadapt, meta } from '../data/run'
import { VERTICALS } from '../data/taxonomy'
import { fixed, int, pct } from '../lib/format'
import { PageHead } from '../components/ui/PageHead'
import { Note } from '../components/ui/Note'

const HELD_OUT = new Set(VERTICALS.filter((v) => v.heldOut).map((v) => v.id))
const LABEL = new Map(VERTICALS.map((v) => [v.id, v.label]))

function tapeText(runs: Array<{ action: string; times: number }>): string {
  return runs.map((r) => (r.times > 1 ? `${r.action} x${r.times}` : r.action)).join(' > ')
}

export function Loop() {
  const [selected, setSelected] = useState<number | null>(11)

  const final = coadapt.final_sequences[0]

  return (
    <div className="space-y-4 px-4 py-5 sm:px-6">
      <PageHead
        title="Co-evolution"
        blurb="What happened when the attacker and the defender adapted against each other, across four paired seeds."
      />

      <KpiStrip
        items={[
          {
            label: 'co-evolution',
            value: `${COEVOLUTION.runsWithFullPattern} of ${COEVOLUTION.ofRuns}`,
            detail: 'runs showing the full pattern',
            tone: 'value',
          },
          {
            label: 'peak over opening',
            value: `${COEVOLUTION.peakOverOpeningMin} to ${COEVOLUTION.peakOverOpeningMax}x`,
            detail: 'in all eight runs',
            tone: 'atk',
          },
          {
            label: 'stealth uplift',
            value: `+${int(ABLATION.meanDifference)}`,
            detail: `95% interval [+${ABLATION.ci[0]}, +${ABLATION.ci[1]}]`,
            tone: 'holdout',
          },
          {
            label: 'genuine refused',
            value: pct(FRICTION.full, 2),
            detail: `mean over ${FRICTION.refits} refits`,
            tone: 'def',
          },
          {
            label: 'scale',
            value: int(COADAPT_SETUP.holders),
            detail: `holders, ${COADAPT_SETUP.updates} updates, refit every ${COADAPT_SETUP.refitEvery}`,
          },
        ]}
      />

      <Panel
        name="run the loop yourself"
        tone="def"
        aside={
          <span className="hidden text-[0.75rem] uppercase tracking-[0.09em] text-ink-3 md:inline">
            both sides learning live, in this browser
          </span>
        }
      >
        <LoopRunning />
      </Panel>

      <Panel
        name="the loop closing, one update at a time"
        tone="atk"
        aside={
          <span className="hidden text-[0.75rem] uppercase tracking-[0.09em] text-ink-3 md:inline">
            play it, or step through update by update
          </span>
        }
      >
        <LoopStepper />
      </Panel>

      <Panel name="co-evolution across the paired seeds" tone="atk">
        <CoEvolution />
      </Panel>

      <Panel name="the stealth ablation" tone="holdout">
        <StealthAblation />
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel name="strategy diversity rises under pressure" tone="value">
          <div className="space-y-4">
            {(
              [
                ['full attacker', DIVERSITY.full.before, DIVERSITY.full.after, 'atk'],
                ['stealth ablated', DIVERSITY.ablated.before, DIVERSITY.ablated.after, 'neutral'],
              ] as const
            ).map(([label, before, after, tone]) => (
              <div key={label}>
                <div className="flex items-baseline justify-between gap-3">
                  <span className="text-[0.9375rem] text-ink">{label}</span>
                  <span className="num text-[0.875rem] text-ink-2">
                    {before} to {after} sequences
                  </span>
                </div>
                <div className="mt-1.5 h-2 w-full rounded-full bg-rule">
                  <div
                    className={`h-full rounded-full ${tone === 'atk' ? 'bg-atk' : 'bg-ink-3'}`}
                    style={{ width: `${(after / 8) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
          <Note
            label="how diversity was counted"
          >
            <p className="prose-sans text-[0.875rem] leading-relaxed text-ink-2">
            Distinct converged sequences, averaged over seeds, before the first refit against by the
            last. The ablated arm peaks at {DIVERSITY.ablated.peak} at {DIVERSITY.ablated.peakAt}.
            Both arms diversify under pressure, which is consistent with the co-evolution appearing
            in both.
          </p>
          </Note>
        </Panel>

        <Panel name="what the defender costs genuine customers" tone="def">
          <div className="space-y-3">
            {(
              [
                ['against the full attacker', FRICTION.full, 'def'],
                ['against the ablated attacker', FRICTION.ablated, 'neutral'],
                ['pooled across both arms', FRICTION.pooled, 'neutral'],
              ] as const
            ).map(([label, value, tone]) => (
              <div key={label}>
                <div className="flex items-baseline justify-between gap-3">
                  <span className="text-[0.9375rem] text-ink-2">{label}</span>
                  <span className={`num text-[0.9375rem] ${tone === 'def' ? 'text-def' : 'text-ink-2'}`}>
                    {pct(value, 2)}
                  </span>
                </div>
                <div className="mt-1.5 h-2 w-full rounded-full bg-rule">
                  <div
                    className={`h-full rounded-full ${tone === 'def' ? 'bg-def' : 'bg-ink-3'}`}
                    style={{ width: `${(value / FRICTION.range[1]) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
          <Note
            label="why this cost is reported at all"
          >
            <p className="prose-sans text-[0.875rem] leading-relaxed text-ink-2">
            A detector that tightens without limit wins any contest measured on recall alone. Read
            the {pct(FRICTION.full, 2)} figure: it is the configuration the co-adaptation results
            come from, and it ranges from zero to {pct(FRICTION.range[1], 1)} across the{' '}
            {FRICTION.refits} refits. The weaker adversary lets the cost curve settle looser, which
            is why the ablated number is lower rather than better.
          </p>
          </Note>
        </Panel>
      </div>

      <Panel
        name="the result worth reading twice"
        tone="holdout"
        aside={
          <span className="hidden text-[0.75rem] uppercase tracking-[0.09em] text-ink-3 md:inline">
            {int(meta.population)} holder run, before the per-merchant cap
          </span>
        }
      >
        <div className="grid gap-6 lg:grid-cols-[1.1fr_1fr]">
          <div>
            <p className="prose-sans text-[1rem] leading-relaxed text-ink">
              The attacker converged on buy credentials, reset the password, then request refunds,
              and it repeated that last action until the forty-action episode budget ran out. That
              chain appears in{' '}
              <span className="text-holdout">none of the seven trained verticals</span>.
            </p>
            <p className="prose-sans mt-4 text-[1rem] leading-relaxed text-ink">
              It is also a reward hack, and that is the result worth reading twice. Nothing bounded
              what a single merchant would absorb inside one episode, so repeating one action at one
              merchant paid better than any real strategy. The policy found that missing control{' '}
              <span className="text-holdout">within one run</span>. A per-merchant value ceiling now
              exists because of it, with a test file asserting it holds.
            </p>
            <p className="prose-sans mt-4 text-[1rem] leading-relaxed text-ink">
              The half that survives the fix is the switch to reset password. It reaches a spendable
              state without minting a device, so{' '}
              <span className="num text-def">device_age_days</span>, the feature that dominates the
              detector by gain, never fires. Under the current per-merchant cap the same profile
              converges on buy credentials, reset password, then authorise, which is the chain the
              solution document reports at Table 7.
            </p>
            <p className="prose-sans mt-4 text-[1rem] leading-relaxed text-ink">
              What the defender scores on the two held-out verticals is not reported.
            </p>

            <div className="mt-4">
              <WithheldNote />
            </div>
          </div>

          <div>
            <Label>
              how the strategy mutated, sampled at each refit, {int(meta.population)} holder run
            </Label>
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
                      <span className="num w-10 shrink-0 text-[0.75rem] text-ink-3">
                        u{s.update}
                      </span>
                      <span className="min-w-0 flex-1 truncate text-[0.75rem] text-ink-2">
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
            <p className="prose-sans mt-3 text-[0.875rem] text-ink-2">
              IVR provisioning at update 11, phish and drain at 23 and 35, then the refund loop
              from 47 onward. It never left.
            </p>
          </div>
        </div>
      </Panel>

      <Panel
        name="attacker strategy stream"
        live
        aside={
          <span className="text-[0.75rem] uppercase tracking-[0.09em] text-ink-3">
            sampled at each refit &middot; {int(meta.population)} holder run
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
                <span className="num w-16 shrink-0 text-[0.8125rem] text-ink-3">
                  u{String(s.update).padStart(3, '0')}
                </span>
                <span className="min-w-0 flex-1 break-words text-[0.8125rem] text-ink">
                  {tapeText(s.runs)}
                  {s.truncated && <span className="text-ink-3"> &hellip;</span>}
                </span>
                <Chip tone={isRefund ? 'holdout' : isPhish ? 'atk' : 'def'}>
                  {isRefund ? 'refund' : isPhish ? 'phish' : 'ivr'}
                </Chip>
                <span className="num w-10 shrink-0 text-right text-[0.8125rem] text-ink-2">
                  {s.count}x
                </span>
              </li>
            )
          })}
        </ul>
        <div className="border-t border-rule px-4 py-3 text-[0.8125rem] text-ink-2">
          Chains are truncated by the run log. The untruncated final strategy is below.
        </div>
      </Panel>

      {final && (
        <Panel
          name="the final trained policy, at full length"
          tone="atk"
          aside={<Badge tone="atk">{final.count} occurrences</Badge>}
        >
          <Tape runs={final.runs} />
          <Note
            label="why the policy collapsed onto one action"
          >
            <p className="prose-sans mt-5 max-w-3xl pt-4 text-[0.9375rem] leading-relaxed text-ink-2">
            Two setup actions and then one action repeated, which exactly saturates the forty-action
            episode budget set in <span className="text-ink">configs/simulation.yaml</span>. Nobody
            scripted this chain, and nobody bounded it either: at the time of this run no ceiling
            capped what one merchant would absorb within an episode, so the cheapest repeatable path
            was also the most profitable one. That is what the policy found. The cap exists now, and{' '}
            <span className="text-ink">tests/test_anti_reward_hacking.py</span> holds it in place.
          </p>
          </Note>
        </Panel>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel name="what victim selection learned" tone="def">
          <PosteriorChart groups={coadapt.selection.groups} />
          <Note
            label="what victim selection actually learned"
          >
            <p className="prose-sans text-[0.9375rem] text-ink-2">
            A contextual bandit over {int(coadapt.selection.observations)} candidates, actively
            selecting. It learned to prefer cards older than three years, which carry higher limits
            and longer histories, so an authorisation on one is both worth more and less anomalous.
            Nothing told it which cards were worth attacking.
          </p>
            <p className="prose-sans mt-3 text-[0.9375rem] text-ink-2">
            The two middle bands sit at exactly zero because the selector never drew them, not
            because it weighed them and found them equal. The issuer BIN tier is the selector&rsquo;s
            other feature and is not shown: its coefficients change sign between runs, so the card
            age preference is the only part of this posterior stable enough to report.
          </p>
          </Note>
        </Panel>

        <Panel name={`how the loop was bootstrapped, ${int(meta.population)} holder run`} tone="pass">
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
                <dd className="num mt-0.5 text-[0.9375rem] text-ink">{v}</dd>
              </div>
            ))}
          </dl>
          <Note
            label="why stale labels are the real constraint"
          >
            <p className="prose-sans text-[0.9375rem] text-ink-2">
            The defender always trains on stale labels: a 4320 minute latency is three days, which
            is what a real chargeback cycle looks like. The critic loss is large because the reward
            scale is large, not because the critic failed.
          </p>
            <p className="prose-sans mt-3 text-[0.875rem] text-ink-3">
            Held out of training entirely:{' '}
            {[...HELD_OUT].map((id) => LABEL.get(id) ?? id).join(' and ')}.
          </p>
          </Note>
        </Panel>
      </div>
    </div>
  )
}
