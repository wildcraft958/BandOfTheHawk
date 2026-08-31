import { useEffect, useRef, useState } from 'react'
import { ChevronDown, Pause, Play, RotateCcw } from 'lucide-react'
import { Panel } from '../components/ui/Panel'
import { KpiStrip } from '../components/ui/KpiStrip'
import { Badge } from '../components/ui/Badge'
import { Chip } from '../components/ui/Chip'
import { Label } from '../components/ui/Label'
import { cn } from '../components/ui/cn'
import { useReducedMotion } from '../lib/useReducedMotion'
import { detectors, fidelity, runReport } from '../data/run'
import {
  BAND_MITIGATION,
  BAND_TONE,
  sampleAuthorization,
  type Authorization,
} from '../stream/sample'
import { fixed, int, pct } from '../lib/format'

const SEED = 20260830
const MAX_ROWS = 40

const FEASIBILITY = [
  {
    name: 'what deploys today',
    tone: 'pass' as const,
    items: [
      'The five experts and the combiner are ordinary scikit-learn and XGBoost artifacts behind a single scoring call.',
      'The risk bands map a score onto an action a payment system already has: step up, hold, decline, block.',
      'Mitigation is graph surgery. Blocking unbinds a device, which removes an edge rather than flagging a row.',
    ],
  },
  {
    name: 'operating constraints',
    tone: 'value' as const,
    items: [
      'Labels arrive late. The run used a 4320 minute latency, three days, which is what a chargeback cycle looks like.',
      'Retention is asymmetric: a confirmed fraud is worth keeping far longer than a confirmed benign.',
      'The defender refits on a cadence, not continuously. Every refit is a deployment.',
    ],
  },
  {
    name: 'scalability',
    tone: 'def' as const,
    items: [
      'Scoring is one small ensemble per event. The 63.7 minute figure is training on a GPU, not inference.',
      'The simulation path imports no ML libraries at all, enforced by an AST-level import firewall at test time.',
      'The co-adaptation loop is an offline research process. Only the fitted defender goes near an auth path.',
    ],
  },
  {
    name: 'what this does not do',
    tone: 'atk' as const,
    items: [
      'Merchant collusion and bust-out are identified but not simulated: both need a settlement and clawback layer.',
      'Zero-shot recall of 1.000 is measured on a small positive count at a 0.5 threshold.',
      'Prevalence is stated rather than deployed, and the live defender is narrower than the static one.',
      'Capability tiers are ordinal. They are ranked, never treated as quantities.',
    ],
  },
]

export function Live() {
  const reduced = useReducedMotion()
  const [rows, setRows] = useState<Authorization[]>([])
  const [running, setRunning] = useState(!reduced)
  const [open, setOpen] = useState<number | null>(null)
  const [scoreMicros, setScoreMicros] = useState<number | null>(null)
  const next = useRef(0)

  useEffect(() => {
    if (!running) return
    const timer = window.setInterval(() => {
      const t0 = performance.now()
      const auth = sampleAuthorization(SEED, next.current++)
      const elapsed = (performance.now() - t0) * 1000
      setScoreMicros(elapsed)
      setRows((prev) => [auth, ...prev].slice(0, MAX_ROWS))
    }, 700)
    return () => window.clearInterval(timer)
  }, [running])

  const reset = () => {
    next.current = 0
    setRows([])
    setOpen(null)
  }

  const banded = rows.filter((r) => r.band !== 'approve').length
  const blocked = rows.filter((r) => r.band === 'block').length
  const bands = detectors.fitted_bands

  return (
    <div className="space-y-4 px-4 py-5 sm:px-6">
      <KpiStrip
        items={[
          { label: 'events scored', value: rows.length, detail: `rolling window of ${MAX_ROWS}` },
          {
            label: 'actioned',
            value: banded,
            detail: 'anything above the step-up band',
            tone: 'value',
          },
          { label: 'blocked', value: blocked, detail: 'device unbound', tone: 'atk' },
          {
            label: 'base rate',
            value: pct(runReport.fraud_auth_share, 2),
            detail: 'real card fraud sits under 1%',
            tone: 'pass',
          },
          {
            label: 'scoring time',
            value: scoreMicros == null ? 'n/a' : `${scoreMicros.toFixed(0)}us`,
            detail: 'in-page, per event',
            tone: 'def',
          },
        ]}
      />

      <Panel
        name="authorisation stream"
        live={running}
        aside={
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => setRunning((r) => !r)}
              className="inline-flex items-center gap-1.5 rounded-panel border border-rule px-2.5 py-1 text-[0.625rem] uppercase tracking-[0.09em] text-ink-2 transition-colors duration-150 hover:border-ink-3 hover:text-ink"
            >
              {running ? (
                <>
                  <Pause className="size-3" aria-hidden="true" /> pause
                </>
              ) : (
                <>
                  <Play className="size-3" aria-hidden="true" /> play
                </>
              )}
            </button>
            <button
              type="button"
              onClick={reset}
              className="inline-flex items-center gap-1.5 rounded-panel border border-rule px-2.5 py-1 text-[0.625rem] uppercase tracking-[0.09em] text-ink-2 transition-colors duration-150 hover:border-ink-3 hover:text-ink"
            >
              <RotateCcw className="size-3" aria-hidden="true" /> reset
            </button>
          </div>
        }
        bodyClassName="px-0 py-0"
      >
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 border-b border-rule-subtle px-4 py-2.5 text-[0.625rem] text-ink-3">
          <span>bands</span>
          {bands &&
            (
              [
                ['approve', 0],
                ['step_up', bands.step_up],
                ['hold', bands.hold],
                ['decline', bands.decline],
                ['block', bands.block],
              ] as const
            ).map(([name, threshold]) => (
              <span key={name} className="flex items-center gap-1.5">
                <Chip tone={BAND_TONE[name as keyof typeof BAND_TONE]}>{name}</Chip>
                <span className="num">{threshold.toFixed(2)}</span>
              </span>
            ))}
        </div>

        {rows.length === 0 ? (
          <p className="px-4 py-10 text-center text-[0.75rem] text-ink-3">
            Stream idle. Press play to score authorisations.
          </p>
        ) : (
          <ul className="divide-y divide-rule-subtle">
            {rows.map((r) => {
              const isOpen = open === r.id
              return (
                <li key={r.id} className={reduced ? undefined : 'rise'}>
                  <button
                    type="button"
                    onClick={() => setOpen(isOpen ? null : r.id)}
                    aria-expanded={isOpen}
                    className="flex w-full items-center gap-3 px-4 py-2 text-left hover:bg-surface-hover"
                  >
                    <ChevronDown
                      className={cn(
                        'size-3 shrink-0 text-ink-3 transition-transform duration-150',
                        isOpen && 'rotate-180',
                      )}
                      aria-hidden="true"
                    />
                    <span className="num w-16 shrink-0 text-[0.6875rem] text-ink-3">{r.ts}</span>
                    <span className="num w-20 shrink-0 text-[0.6875rem] text-ink-2">{r.cardId}</span>
                    <span className="num w-20 shrink-0 text-right text-[0.6875rem] text-ink">
                      {r.amount.toFixed(2)}
                    </span>
                    <span className="hidden w-28 shrink-0 text-[0.6875rem] text-ink-3 sm:inline">
                      {r.category.replace('_', ' ')}
                    </span>
                    <span className="hidden shrink-0 sm:inline">
                      <Chip tone={r.entryMode === 'card_not_present' ? 'atk' : undefined}>
                        {r.entryMode}
                      </Chip>
                    </span>
                    <span className="flex-1" />
                    <span className="num w-12 shrink-0 text-right text-[0.6875rem] text-ink-2">
                      {r.score.toFixed(2)}
                    </span>
                    <span className="w-20 shrink-0 text-right">
                      <Badge tone={BAND_TONE[r.band]}>{r.band}</Badge>
                    </span>
                  </button>

                  {isOpen && (
                    <div className="border-t border-rule-subtle bg-surface px-4 py-4 sm:px-12">
                      <div className="flex flex-wrap items-center gap-3">
                        <Label>why it scored {r.score.toFixed(2)}</Label>
                        <Badge tone={BAND_TONE[r.band]}>{BAND_MITIGATION[r.band]}</Badge>
                      </div>
                      <ul className="mt-3 space-y-1.5">
                        {r.contributions
                          .slice()
                          .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
                          .map((c) => (
                            <li key={c.feature} className="flex items-center gap-3">
                              <span className="w-52 shrink-0 truncate text-[0.6875rem] text-ink-2">
                                {c.feature}
                              </span>
                              <span className="num w-32 shrink-0 text-[0.6875rem] text-ink">
                                {c.value}
                              </span>
                              <span className="relative h-2.5 flex-1">
                                <span
                                  className="absolute inset-y-0 left-1/2 w-px bg-rule"
                                  aria-hidden="true"
                                />
                                <span
                                  className={cn(
                                    'absolute inset-y-0 rounded-[1px]',
                                    c.weight > 0 ? 'bg-atk' : 'bg-pass',
                                  )}
                                  style={
                                    c.weight > 0
                                      ? { left: '50%', width: `${Math.abs(c.weight) * 120}%` }
                                      : { right: '50%', width: `${Math.abs(c.weight) * 120}%` }
                                  }
                                />
                              </span>
                              <span
                                className={cn(
                                  'num w-14 shrink-0 text-right text-[0.6875rem]',
                                  c.weight > 0 ? 'text-atk' : 'text-pass',
                                )}
                              >
                                {c.weight > 0 ? `+${c.weight.toFixed(2)}` : c.weight.toFixed(2)}
                              </span>
                            </li>
                          ))}
                      </ul>
                    </div>
                  )}
                </li>
              )
            })}
          </ul>
        )}

        <div className="border-t border-rule px-4 py-4">
          <div className="flex flex-wrap items-center gap-3">
            <Badge tone="value">illustrative</Badge>
            <span className="text-[0.6875rem] text-ink-3">seed {SEED}, deterministic</span>
          </div>
          <p className="prose-sans mt-3 max-w-3xl text-[0.8125rem] leading-relaxed text-ink-2">
            <span className="text-ink">What is real here:</span> the amount distribution
            (lognormal mu {fixed(fidelity.amount.lognormal_mu, 4)}, sigma{' '}
            {fixed(fidelity.amount.lognormal_sigma, 4)}, with{' '}
            {pct(fidelity.amount.whole_number_share)} whole numbers), the two-component circadian
            mixture, the merchant category mix, and the four band thresholds, all fitted from real
            aggregate data. The feature names are the ones the real detector leads on by gain.
          </p>
          <p className="prose-sans mt-2 max-w-3xl text-[0.8125rem] leading-relaxed text-ink-2">
            <span className="text-ink">What is not:</span> the individual events are sampled, and
            the score is a transparent linear stand-in whose weights are shown above, not the
            trained XGBoost ensemble. What this panel demonstrates is the mitigation ladder and
            the shape of a scored authorisation, which is the part that has to fit inside an
            authorisation path.
          </p>
        </div>
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        {FEASIBILITY.map((group) => (
          <Panel key={group.name} name={group.name} tone={group.tone}>
            <ul className="space-y-2.5">
              {group.items.map((item) => (
                <li key={item} className="flex gap-2.5">
                  <span
                    className={cn(
                      'mt-[0.45rem] size-1 shrink-0 rounded-full',
                      group.tone === 'pass' && 'bg-pass',
                      group.tone === 'value' && 'bg-value',
                      group.tone === 'def' && 'bg-def',
                      group.tone === 'atk' && 'bg-atk',
                    )}
                    aria-hidden="true"
                  />
                  <span className="prose-sans text-[0.8125rem] leading-relaxed text-ink-2">
                    {item}
                  </span>
                </li>
              ))}
            </ul>
          </Panel>
        ))}
      </div>

      <Panel name="commercial viability" tone="pass">
        <p className="prose-sans max-w-3xl text-[0.875rem] leading-relaxed text-ink-2">
          The number a payments organisation actually buys is reviewer cost. At an alert budget of{' '}
          {detectors.alert_budget} events, precision is{' '}
          <span className="num text-pass">
            {fixed(detectors.operating_point?.precision, 4)}
          </span>
          , so a reviewer&rsquo;s queue is very nearly all true fraud. Recall at that budget is{' '}
          <span className="num text-ink">{fixed(detectors.operating_point?.recall, 4)}</span> and
          F1 is <span className="num text-ink">{fixed(detectors.operating_point?.f1, 4)}</span>,
          derived rather than measured because the Python does not compute F1.
        </p>
        <p className="prose-sans mt-3 max-w-3xl text-[0.8125rem] text-ink-3">
          The simulator is for stress-testing a detector against attacks that do not exist yet. It
          does not replace one. That distinction is the honest version of the pitch, and it is what
          makes the {int(runReport.episodes)} episode run worth running again next quarter.
        </p>
      </Panel>
    </div>
  )
}
