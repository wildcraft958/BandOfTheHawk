import { duration } from '../../lib/format'
import type { StageTiming } from '../../data/types'

/**
 * Where the run's time actually went.
 *
 * This replaced a node graph of the same six stages. The graph carried one fact
 * the boxes could not, that the stages form a cycle, and the closed loop above
 * already makes that point properly. What is left here is proportion, and a
 * proportion reads better as a bar than as a diagram.
 */
const ROLE: Record<string, { label: string; tone: string; note: string }> = {
  demo: { label: 'setup', tone: 'bg-ink-3', note: 'build the synthetic bank' },
  text: { label: 'setup', tone: 'bg-ink-3', note: 'generate and embed dispute text' },
  fraud: { label: 'generate', tone: 'bg-atk', note: 'inject scripted fraud episodes' },
  baseline: { label: 'detect', tone: 'bg-def', note: 'fit the flat detector' },
  mixture: { label: 'detect', tone: 'bg-def', note: 'fit five routed experts' },
  coadapt: { label: 'co-adapt', tone: 'bg-value', note: 'attacker and defender co-adapt' },
}

const TEXT: Record<string, string> = {
  'bg-ink-3': 'text-ink-3',
  'bg-atk': 'text-atk',
  'bg-def': 'text-def',
  'bg-value': 'text-value',
}

export function StageTimeline({ stages, total }: { stages: StageTiming[]; total: number }) {
  const known = stages.filter((s) => ROLE[s.stage])

  return (
    <div>
      <div className="flex h-7 w-full overflow-hidden rounded-panel border border-rule">
        {known.map((s) => {
          const role = ROLE[s.stage]
          const share = total > 0 ? s.seconds / total : 0
          return (
            <div
              key={s.stage}
              className={`${role.tone} h-full`}
              style={{ width: `${Math.max(share * 100, 0.4)}%` }}
              title={`${s.stage}: ${duration(s.seconds)}, ${(share * 100).toFixed(1)}%`}
            />
          )
        })}
      </div>

      <ul className="mt-4 divide-y divide-rule-subtle">
        {known.map((s) => {
          const role = ROLE[s.stage]
          const share = total > 0 ? s.seconds / total : 0
          return (
            <li key={s.stage} className="flex items-baseline gap-3 py-2">
              <span className={`size-1.5 shrink-0 rounded-full ${role.tone}`} aria-hidden="true" />
              <span className="w-20 shrink-0 text-[0.9375rem] text-ink">{s.stage}</span>
              <span
                className={`w-16 shrink-0 text-[0.6875rem] uppercase tracking-[0.1em] ${TEXT[role.tone]}`}
              >
                {role.label}
              </span>
              <span className="min-w-0 flex-1 truncate text-[0.8125rem] text-ink-3">
                {role.note}
              </span>
              <span className="num shrink-0 text-[0.875rem] text-ink-2">{duration(s.seconds)}</span>
              <span className="num w-12 shrink-0 text-right text-[0.875rem] text-ink-3">
                {(share * 100).toFixed(1)}%
              </span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
