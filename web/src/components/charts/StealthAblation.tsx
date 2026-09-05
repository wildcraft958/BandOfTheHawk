import { ABLATION } from '../../data/paper'
import { Badge } from '../ui/Badge'
import { int } from '../../lib/format'

/**
 * The stealth ablation, from Table 6.
 *
 * The two arms share a world per seed and differ in one respect: the ablated
 * arm has its posture head pinned to loud and its credential dump cut to a
 * single card, which is the attacker as it stood before those capabilities
 * existed. That makes this a paired comparison rather than two separate runs.
 *
 * The seed that goes the other way is drawn the same size as the three that
 * agree. An ablation with one contrary seed reported as if it had four is not an
 * ablation.
 */
export function StealthAblation() {
  const seeds = ABLATION.seeds
  const span = Math.max(...seeds.flatMap((s) => [s.full, s.ablated]))

  return (
    <div>
      <p className="prose-sans max-w-3xl text-[0.9375rem] leading-relaxed text-ink-2">
        Each pair shares a world. The only difference is that the ablated attacker has its posture
        head pinned to loud and its credential dump cut to one card, which is the attacker as it was
        before those capabilities existed.
      </p>

      <div className="mt-4 space-y-3">
        {seeds.map((s) => {
          const against = s.difference < 0
          return (
            <div key={s.seed}>
              <div className="flex items-baseline justify-between gap-3">
                <span className="num text-[0.875rem] text-ink-2">seed {s.seed}</span>
                <span className={`num text-[0.875rem] ${against ? 'text-def' : 'text-atk'}`}>
                  {s.difference > 0 ? '+' : ''}
                  {int(s.difference)}
                </span>
              </div>
              <div className="mt-1 space-y-1">
                <div className="flex items-center gap-2">
                  <span className="w-14 shrink-0 text-[0.6875rem] uppercase tracking-[0.09em] text-atk">
                    full
                  </span>
                  <div className="h-2 flex-1 rounded-full bg-rule">
                    <div
                      className="h-full rounded-full bg-atk"
                      style={{ width: `${(s.full / span) * 100}%` }}
                    />
                  </div>
                  <span className="num w-14 shrink-0 text-right text-[0.8125rem] text-ink-2">
                    {int(s.full)}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-14 shrink-0 text-[0.6875rem] uppercase tracking-[0.09em] text-ink-3">
                    ablated
                  </span>
                  <div className="h-2 flex-1 rounded-full bg-rule">
                    <div
                      className="h-full rounded-full bg-ink-3"
                      style={{ width: `${(s.ablated / span) * 100}%` }}
                    />
                  </div>
                  <span className="num w-14 shrink-0 text-right text-[0.8125rem] text-ink-2">
                    {int(s.ablated)}
                  </span>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      <div className="mt-5 flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-rule pt-4">
        <span className="text-[0.9375rem] text-ink">
          mean paired difference{' '}
          <span className="num text-atk">+{int(ABLATION.meanDifference)}</span>
        </span>
        <span className="num text-[0.875rem] text-ink-2">
          95% interval [+{ABLATION.ci[0]}, +{ABLATION.ci[1]}]
        </span>
        <span className="text-[0.8125rem] text-ink-3">
          {int(ABLATION.resamples)} bootstrap resamples
        </span>
        <Badge tone="value">excludes zero</Badge>
      </div>

      <p className="prose-sans mt-3 text-[0.875rem] leading-relaxed text-ink-2">
        The interval excludes zero, so the stealth capability is worth having at this scale. The
        caveats matter as much as the estimate: four seeds is a small sample, seed 2 runs the other
        way, and an interval this wide would not detect an effect much smaller than the one measured.
      </p>
      <p className="prose-sans mt-2 text-[0.875rem] leading-relaxed text-ink-3">
        Supported: the posture head and the multi-card dump improve the attacker inside the loop. Not
        supported: that they cause the co-evolution, which appears in the ablated arm too.
      </p>
    </div>
  )
}
