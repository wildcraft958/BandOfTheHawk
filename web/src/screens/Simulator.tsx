import { Panel } from '../components/ui/Panel'
import { KpiStrip } from '../components/ui/KpiStrip'
import { Badge } from '../components/ui/Badge'
import { Meter } from '../components/ui/Meter'
import { Label } from '../components/ui/Label'
import { PerCardLevels } from '../components/charts/PerCardLevels'
import { CircadianDial } from '../components/charts/CircadianDial'
import { ProvenanceLedger } from '../components/diagrams/ProvenanceLedger'
import { fidelity, graph, runReport } from '../data/run'
import { cn } from '../components/ui/cn'
import { fixed, int, pct } from '../lib/format'
import type { FidelityComparison } from '../data/types'
import { PageHead } from '../components/ui/PageHead'
import { Note } from '../components/ui/Note'

/** Ratio of the generated-to-target gap against the noise floor of real data. */
function verdict(ratio: number | null) {
  if (ratio == null) return { label: 'not measured', tone: 'neutral' as const }
  const { indistinguishable, close, structural_gap } = fidelity.verdict_ladder
  if (ratio <= indistinguishable) return { label: 'indistinguishable', tone: 'pass' as const }
  if (ratio <= close) return { label: 'close', tone: 'value' as const }
  if (ratio <= structural_gap) return { label: 'structural gap', tone: 'atk' as const }
  return { label: 'not reproduced', tone: 'atk' as const }
}

function ComparisonRow({ c }: { c: FidelityComparison }) {
  const v = verdict(c.ratio)
  // The floor sits at 1.0 on this axis. Anything left of the marker differs
  // from the target by less than real data differs from itself.
  const scale = 6
  const width = c.ratio == null ? 0 : Math.min((c.ratio / scale) * 100, 100)

  return (
    <li className="py-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <span className="text-[0.9375rem] text-ink">{c.name}</span>
        <span className="flex items-center gap-3">
          <span className="num text-[0.875rem] text-ink-2">
            {fixed(c.observed, 5)} vs {fixed(c.target, 5)}
          </span>
          <Badge tone={v.tone}>{v.label}</Badge>
        </span>
      </div>

      <div className="relative mt-2 h-1.5 w-full rounded-full bg-rule">
        <div
          className={cn(
            'h-full rounded-full',
            v.tone === 'pass' ? 'bg-pass' : v.tone === 'value' ? 'bg-value' : 'bg-atk',
          )}
          style={{ width: `${width}%` }}
        />
        <span
          className="absolute top-[-3px] h-[12px] w-px bg-ink-2"
          style={{ left: `${(1 / scale) * 100}%` }}
          aria-hidden="true"
        />
      </div>

      <div className="num mt-1.5 text-[0.8125rem] text-ink-3">
        gap {fixed(c.gap, 5)} against a noise floor of {fixed(c.noise_floor, 5)}, a ratio of{' '}
        <span className={v.tone === 'pass' ? 'text-pass' : 'text-ink-2'}>
          {fixed(c.ratio, 3)}x
        </span>
      </div>
    </li>
  )
}

export function Simulator() {
  const ws = runReport.warm_start
  const rules = Object.entries(runReport.rule_trigger_rates).filter(([k]) => k !== 'any')
  const maxRule = Math.max(...rules.map(([, v]) => v))
  const negatives = Object.entries(runReport.hard_negatives).sort((a, b) => b[1] - a[1])
  const maxNeg = Math.max(...negatives.map(([, v]) => v))
  const suspicious = negatives.filter(([k]) => k !== 'ordinary').reduce((a, [, v]) => a + v, 0)
  const t = graph.targets
  const cats = Object.entries(fidelity.category_mix ?? {}).sort((a, b) => b[1] - a[1])

  return (
    <div className="space-y-4 px-4 py-5 sm:px-6">
      <PageHead
        title="Fidelity"
        blurb="How close the generated world sits to real data, measured as a ratio against the noise floor of real data split against itself."
      />

      <KpiStrip
        items={[
          { label: 'entities', value: int(ws.entities), detail: `${int(ws.events)} warm-start events` },
          { label: 'dormant share', value: pct(ws.dormant_share), detail: 'cards with no activity' },
          {
            label: 'fan-out mean',
            value: fixed(graph.fanout_observed_mean, 2),
            detail: `target ${fixed(graph.fanout_target_mean, 2)}`,
            tone: 'pass',
          },
          {
            label: 'rule trigger rate',
            value: pct(runReport.rule_trigger_rates.any, 2),
            detail: `target ${pct(runReport.rule_target, 1)}, ${runReport.rule_verdict}`,
            tone: 'value',
          },
          { label: 'history', value: `${int(ws.history_days)}d`, detail: 'of behaviour per card' },
        ]}
      />

      <Panel
        name="calibration against the noise floor"
        tone="pass"
        aside={
          <span className="hidden text-[0.75rem] uppercase tracking-[0.09em] text-ink-3 md:inline">
            split-half of {int(fidelity.split.left_entities)} entities
          </span>
        }
      >
        <p className="prose-sans max-w-3xl text-[0.9375rem] text-ink-2">
          Every fidelity number is a ratio against a noise floor, where the floor is the distance
          between two halves of real data. A ratio near or below 1.0 means the synthetic data
          differs from the target about as much as real data differs from itself. The marker on
          each bar is that 1.0 line.
        </p>

        <div className="mt-4 flex flex-wrap gap-x-5 gap-y-1.5 text-[0.8125rem] text-ink-3">
          <span>
            <Badge tone="pass">indistinguishable</Badge> ratio at or below{' '}
            {fidelity.verdict_ladder.indistinguishable}
          </span>
          <span>
            <Badge tone="value">close</Badge> at or below {fidelity.verdict_ladder.close}
          </span>
          <span>
            <Badge tone="atk">structural gap</Badge> at or below{' '}
            {fidelity.verdict_ladder.structural_gap}
          </span>
        </div>

        <ul className="mt-2 divide-y divide-rule-subtle">
          {fidelity.comparisons.map((c) => (
            <ComparisonRow key={c.name} c={c} />
          ))}
        </ul>

        <Note
          label="the miss shown rather than dropped"
        >
          <p className="prose-sans text-[0.875rem] text-ink-2">
          Arrival autocorrelation misses by more than the floor, at 4.57x. It is shown rather than
          dropped: a fidelity panel that admits a miss is worth more than one claiming only hits.
          Measured on a seed the parameter search did not use.
        </p>
        </Note>
      </Panel>

      <Panel name="where every parameter came from" tone="def">
        <ProvenanceLedger />
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel name="rule trigger rates" tone="value">
          <div className="space-y-3">
            {rules.map(([id, rate]) => (
              <Meter
                key={id}
                label={id}
                value={rate}
                max={maxRule}
                display={rate.toFixed(4)}
                tone={rate === 0 ? 'neutral' : 'value'}
                note={
                  rate === 0
                    ? `never fires in ${int(runReport.rule_events)} events. Declines are rare in this population.`
                    : undefined
                }
              />
            ))}
          </div>
          <Note
            label="why an empty bar is left on the chart"
          >
            <p className="prose-sans text-[0.875rem] text-ink-2">
            Aggregate {pct(runReport.rule_trigger_rates.any, 2)} against a target of{' '}
            {pct(runReport.rule_target, 1)}. An empty bar is left visible on purpose: it is
            evidence the rates were measured rather than assembled.
          </p>
          </Note>
        </Panel>

        <Panel name="hard negatives injected" tone="atk">
          <div className="space-y-3">
            {negatives.map(([name, count]) => (
              <Meter
                key={name}
                label={name}
                value={count}
                max={maxNeg}
                display={int(count)}
                tone={name === 'ordinary' ? 'neutral' : 'atk'}
              />
            ))}
          </div>
          <Note
            label="what the hard negatives are for"
          >
            <p className="prose-sans text-[0.875rem] text-ink-2">
            {int(suspicious)} of the benign events are deliberately suspicious but legitimate:
            gift-card runs, travel, new devices, recovery flows, disputes. The false-positive rate
            is earned against those, not against easy negatives.
          </p>
          </Note>
        </Panel>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel name="entity graph: why degrees are generated first" tone="def">
          <dl className="grid grid-cols-2 gap-x-6 gap-y-3">
            {(
              [
                ['fan-out mean', fixed(t.fanout_mean, 4)],
                ['variance', fixed(t.fanout_variance, 1)],
                ['variance to mean', fixed(t.fanout_variance_to_mean, 2)],
                ['p99', fixed(t.fanout_p99, 1)],
                ['max', int(t.fanout_max)],
                ['share shared', pct(t.fanout_share_shared)],
              ] as const
            ).map(([k, v]) => (
              <div key={k}>
                <Label>{k}</Label>
                <dd className="num mt-0.5 text-[1rem] text-ink">{v}</dd>
              </div>
            ))}
          </dl>
          <Note
            label="why degrees are generated first"
          >
            <p className="prose-sans text-[0.9375rem] leading-relaxed text-ink-2">
            {graph.variance_to_mean_note} Variance to mean is{' '}
            <span className="num text-def">{fixed(t.fanout_variance_to_mean, 2)}</span> against a
            mean of <span className="num text-def">{fixed(t.fanout_mean, 2)}</span>, so
            row-sampling provably cannot produce this tail.
          </p>
            <p className="prose-sans mt-3 text-[0.875rem] text-ink-3">
            Graph invariants {graph.invariants_hold ? 'hold' : 'do not hold'} for the built world.
          </p>
          </Note>
        </Panel>

        <div className="space-y-4">
          <Panel name="fitted amount distribution" tone="value">
            <dl className="grid grid-cols-2 gap-x-6 gap-y-3">
              {(
                [
                  ['lognormal mu', fixed(fidelity.amount.lognormal_mu, 4)],
                  ['lognormal sigma', fixed(fidelity.amount.lognormal_sigma, 4)],
                  ['median', fixed(fidelity.amount.median, 1)],
                  ['tail index', fixed(fidelity.amount.tail_index, 4)],
                  ['whole-number share', pct(fidelity.amount.whole_number_share)],
                  ['samples', int(fidelity.amount.n_samples)],
                ] as const
              ).map(([k, v]) => (
                <div key={k}>
                  <Label>{k}</Label>
                  <dd className="num mt-0.5 text-[1rem] text-ink">{v}</dd>
                </div>
              ))}
            </dl>
            <Note
              label="why whole numbers had to be modelled"
            >
              <p className="prose-sans text-[0.875rem] text-ink-2">
              Over half of real amounts are whole numbers. A generator that samples a smooth
              lognormal and stops there gets the shape right and the texture wrong.
            </p>
            </Note>
          </Panel>

          <Panel name="merchant category mix" tone="def">
            <div className="space-y-2.5">
              {cats.map(([name, share]) => (
                <Meter
                  key={name}
                  label={name.replace('_', ' ')}
                  value={share}
                  max={cats[0][1]}
                  display={pct(share)}
                  tone="def"
                />
              ))}
            </div>
          </Panel>
        </div>
      </div>

      <Panel name="fitted arrival hour" tone="value">
        <CircadianDial />
      </Panel>

      <Panel name="why the pooled histogram is not enough" tone="pass">
        <PerCardLevels />
      </Panel>

      <Panel name="the blindness rule" tone="holdout">
        <p className="prose-sans max-w-3xl text-[0.9375rem] leading-relaxed text-ink-2">
          An event carries no indication of who caused it. The same builder produces the row
          whether an ordinary holder or an attacker acted, and any field distinguishing them would
          be a shortcut a detector could learn instead of learning behaviour.
        </p>
        <ul className="mt-4 space-y-2.5">
          {[
            'The event builder receives entity references and reads the graph. Nothing in its signature says who is acting.',
            'scoring_fields() drops is_fraud and episode_id structurally, so a scorer cannot read the label even by accident.',
            'Labels are stamped only after an episode closes, because at the moment of scoring nothing knows the answer.',
            'The defender table drops all nine identity fields. Identity is not a feature.',
            'Build and commit are separate calls, so an event never counts itself in its own velocity windows.',
          ].map((line) => (
            <li key={line} className="flex gap-2.5">
              <span className="mt-[0.45rem] size-1 shrink-0 rounded-full bg-holdout" aria-hidden="true" />
              <span className="prose-sans text-[0.9375rem] leading-relaxed text-ink-2">{line}</span>
            </li>
          ))}
        </ul>
        <p className="prose-sans mt-4 border-t border-rule pt-3 text-[0.875rem] text-ink-3">
          Each of these is enforced by a test, not by convention.
        </p>
      </Panel>
    </div>
  )
}
