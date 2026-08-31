import { Link } from 'react-router-dom'
import { ArrowRight, GitBranch, ShieldCheck, Swords } from 'lucide-react'
import { KpiStrip } from '../components/ui/KpiStrip'
import { Panel } from '../components/ui/Panel'
import { ClosedLoop } from '../components/diagrams/ClosedLoop'
import { Seismograph } from '../components/charts/Seismograph'
import { coadapt, detectors, meta, points, runReport } from '../data/run'
import { VERTICALS } from '../data/taxonomy'
import { duration, fixed, int, pct } from '../lib/format'

const PILLARS = [
  {
    icon: Swords,
    name: 'Identify',
    body: `${VERTICALS.length} GenAI-enabled fraud verticals mapped across the account lifecycle, over one shared 20-action space. Seven of the twenty actions require a generated artifact.`,
  },
  {
    icon: GitBranch,
    name: 'Generate',
    body: 'A calibrated synthetic bank, fitted against real aggregate statistics and checked against the noise floor of real data split against itself.',
  },
  {
    icon: ShieldCheck,
    name: 'Defend',
    body: 'Five experts routed by event type under a learned combiner, refitting live while a reinforcement-learning attacker adapts underneath it.',
  },
]

export function Landing() {
  const full = detectors.configs.find((c) => c.id === 'gbdt_full')
  const textExpert = detectors.experts.find((e) => e.name === 'text')

  return (
    <div className="px-4 py-10 sm:px-6 sm:py-14">
      <p className="text-[0.6875rem] uppercase tracking-[0.14em] text-ink-3">
        Mastercard Innovation Challenge 2026 &middot; AI Defense Lab for Payment Security
      </p>

      <h1
        className="mt-6 font-display text-[clamp(2.5rem,8vw,6rem)] font-extrabold uppercase leading-[0.9] tracking-[-0.02em]"
        style={{ fontStretch: '80%' }}
      >
        Build the attack,
        <br />
        then build the defense.
      </h1>

      <p className="prose-sans mt-7 max-w-3xl text-[1rem] leading-relaxed text-ink-2">
        GAUNTLET invents GenAI payment fraud, simulates it against a synthetic bank, and trains a
        detector that catches it &mdash; as one closed loop where attacker and defender adapt against
        each other. The attacker is a reinforcement-learning agent that discovers strategies on its
        own. When the defender improves, the attacker finds new gaps; when the attacker escalates,
        the defender refits.
      </p>

      <div className="mt-8 flex flex-wrap items-center gap-3">
        <Link
          to="/dashboard"
          className="inline-flex items-center gap-2 rounded-panel border border-atk bg-atk/12 px-4 py-2.5 text-[0.75rem] uppercase tracking-[0.1em] text-atk no-underline transition-colors duration-150 hover:bg-atk/20"
        >
          Open the console <ArrowRight className="size-3.5" aria-hidden="true" />
        </Link>
        <Link
          to="/demo"
          className="inline-flex items-center gap-2 rounded-panel border border-rule px-4 py-2.5 text-[0.75rem] uppercase tracking-[0.1em] text-ink-2 no-underline transition-colors duration-150 hover:border-ink-3 hover:text-ink"
        >
          Explore the attacks
        </Link>
      </div>

      <div className="mt-12">
        <KpiStrip
          items={[
            {
              label: 'attack verticals',
              value: VERTICALS.length,
              detail: `${VERTICALS.filter((v) => v.simulated).length} simulated, 2 held out`,
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
              detail: 'on verticals never trained on',
              tone: 'holdout',
            },
            {
              label: 'rl updates',
              value: points.length,
              detail: `${coadapt.refit_updates.length} defender refits`,
              tone: 'value',
            },
            {
              label: 'one full run',
              value: duration(meta.total_seconds),
              detail: `${int(meta.population)} cardholders, profile ${meta.profile}`,
            },
          ]}
        />
      </div>

      <div className="mt-4">
        <Panel
          name="the arms race, from one real run"
          live
          bodyClassName="px-4 pb-3 pt-5"
          aside={
            <span className="hidden text-[0.625rem] uppercase tracking-[0.09em] text-ink-3 sm:inline">
              {points.length} updates &middot; symlog
            </span>
          }
        >
          <Seismograph points={points} refits={coadapt.refit_updates} />
          <p className="prose-sans mt-4 max-w-3xl text-[0.8125rem] text-ink-2">
            The attacker climbs to {int(coadapt.reads.extracted_max)} value extracted per episode.
            Each blue rule is a defender refit. After the first, extraction falls to{' '}
            <span className="text-def">exactly zero for {coadapt.reads.zeros} updates</span> &mdash;
            then the attacker finds a different channel and climbs again. That sawtooth is the
            closed loop.
          </p>
        </Panel>
      </div>

      <div className="mt-10 grid gap-4 lg:grid-cols-3">
        {PILLARS.map((p) => (
          <Panel key={p.name} name={p.name} tone="def">
            <p.icon className="size-4 text-ink-3" aria-hidden="true" />
            <p className="prose-sans mt-3 text-[0.8125rem] leading-relaxed text-ink-2">{p.body}</p>
          </Panel>
        ))}
      </div>

      <div className="mt-4">
        <Panel name="what closes the loop" tone="value">
          <ClosedLoop stages={meta.stages} total={meta.total_seconds ?? 1} />
        </Panel>
      </div>

      <div className="mt-4">
        <Panel name="the result worth reading twice" tone="holdout">
          <p className="prose-sans max-w-3xl text-[0.9375rem] leading-relaxed text-ink">
            The attacker independently converged on{' '}
            <span className="text-atk">buy credentials, reset the password, request refunds</span>{' '}
            &mdash; a refund-abuse loop appearing in none of the seven trained verticals. Refund
            abuse was one of the two verticals{' '}
            <span className="text-holdout">deliberately held out</span>. It rediscovered the
            held-out attack on its own, through the defender&rsquo;s thinnest expert: text, at{' '}
            <span className="text-def">
              {textExpert ? pct(textExpert.normalized_weight) : '4.7%'}
            </span>{' '}
            of combiner weight. The defender caught it anyway, at{' '}
            <span className="text-pass">{fixed(coadapt.zero_shot[0]?.recall, 3)} recall</span>,
            having never seen it.
          </p>
          <p className="prose-sans mt-4 text-[0.75rem] text-ink-3">
            {int(runReport.episodes)} episodes, {int(runReport.fraud_auths)} fraud authorisations
            against {int(runReport.benign_auths)} benign, at a{' '}
            {pct(runReport.fraud_auth_share, 2)} base rate.
          </p>
        </Panel>
      </div>
    </div>
  )
}
