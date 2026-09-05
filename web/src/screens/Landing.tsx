import { Link } from 'react-router-dom'
import { ArrowRight, GitBranch, ShieldCheck, Swords } from 'lucide-react'
import { KpiStrip } from '../components/ui/KpiStrip'
import { Panel } from '../components/ui/Panel'
import { detectors, meta } from '../data/run'
import { ABLATION, COADAPT_SETUP, COEVOLUTION, FRICTION } from '../data/paper'
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

  return (
    <div className="px-4 py-10 sm:px-6 sm:py-14">
      <p className="text-[0.8125rem] uppercase tracking-[0.14em] text-ink-3">
        <span className="text-ink-2">Overview</span> &middot; Mastercard Innovation Challenge 2026
        &middot; AI Defense Lab for Payment Security
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
        detector that catches it, as one closed loop where attacker and defender adapt against
        each other. The attacker is a reinforcement-learning agent that discovers strategies on its
        own. When the defender improves, the attacker finds new gaps; when the attacker escalates,
        the defender refits.
      </p>

      <div className="mt-8 flex flex-wrap items-center gap-3">
        <Link
          to="/attack-surface"
          className="inline-flex items-center gap-2 rounded-panel border border-atk bg-atk/12 px-4 py-2.5 text-[0.875rem] uppercase tracking-[0.1em] text-atk no-underline transition-colors duration-150 hover:bg-atk/20"
        >
          Run an attack <ArrowRight className="size-3.5" aria-hidden="true" />
        </Link>
        {/* Blue, because the panel it lands on is the defender's and the page
            leads with it. The two things a judge can actually operate come
            first; the results page follows them. */}
        <Link
          to="/co-evolution"
          className="inline-flex items-center gap-2 rounded-panel border border-def bg-def/12 px-4 py-2.5 text-[0.875rem] uppercase tracking-[0.1em] text-def no-underline transition-colors duration-150 hover:bg-def/20"
        >
          Run the loop yourself <ArrowRight className="size-3.5" aria-hidden="true" />
        </Link>
        <Link
          to="/detection"
          className="inline-flex items-center gap-2 rounded-panel border border-rule px-4 py-2.5 text-[0.875rem] uppercase tracking-[0.1em] text-ink-2 no-underline transition-colors duration-150 hover:border-ink-3 hover:text-ink"
        >
          See the detection results
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
              label: 'stealth uplift',
              value: `+${int(ABLATION.meanDifference)}`,
              detail: `95% interval [+${ABLATION.ci[0]}, +${ABLATION.ci[1]}]`,
              tone: 'holdout',
            },
            {
              label: 'co-evolution',
              value: `${COEVOLUTION.runsWithFullPattern} of ${COEVOLUTION.ofRuns}`,
              detail: `${COADAPT_SETUP.seeds} paired seeds, refit every ${COADAPT_SETUP.refitEvery}`,
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

      <div className="mt-10 grid gap-4 lg:grid-cols-3">
        {PILLARS.map((p) => (
          <Panel key={p.name} name={p.name} tone="def">
            <p.icon className="size-4 text-ink-3" aria-hidden="true" />
            <p className="prose-sans mt-3 text-[0.9375rem] leading-relaxed text-ink-2">{p.body}</p>
          </Panel>
        ))}
      </div>

      <div className="mt-4">
        <Panel name="the result worth reading twice" tone="holdout">
          <p className="prose-sans max-w-3xl text-[1rem] leading-relaxed text-ink">
            The loop closes. Across {COADAPT_SETUP.seeds} paired seeds, extraction rises above its
            opening level, a defender refit suppresses it, and the attacker recovers, in{' '}
            <span className="text-value">
              {COEVOLUTION.runsWithFullPattern} of {COEVOLUTION.ofRuns} runs
            </span>
            . Peak extraction beats the opening block by {COEVOLUTION.peakOverOpeningMin}x to{' '}
            {COEVOLUTION.peakOverOpeningMax}x in every one of the eight.
          </p>
          <p className="prose-sans mt-3 max-w-3xl text-[1rem] leading-relaxed text-ink">
            Take the attacker&rsquo;s stealth away and it loses{' '}
            <span className="text-holdout">{int(ABLATION.meanDifference)}</span> in mean post-refit
            extraction, with a 95% interval of [+{ABLATION.ci[0]}, +{ABLATION.ci[1]}] that excludes
            zero. One of the four seeds runs the other way, and that is on the page too.
          </p>
          <p className="prose-sans mt-3 max-w-3xl text-[0.9375rem] leading-relaxed text-ink-2">
            The defender&rsquo;s side of the ledger: it refuses{' '}
            <span className="text-def">{pct(FRICTION.full, 2)}</span> of genuine authorisations, so
            what it gains is bounded by the friction it imposes, not by recall alone.
          </p>
          <Link
            to="/co-evolution"
            className="mt-4 inline-flex items-center gap-2 text-[0.875rem] uppercase tracking-[0.1em] text-atk no-underline hover:text-accent-hover"
          >
            See how it got there <ArrowRight className="size-3.5" aria-hidden="true" />
          </Link>
        </Panel>
      </div>
    </div>
  )
}
