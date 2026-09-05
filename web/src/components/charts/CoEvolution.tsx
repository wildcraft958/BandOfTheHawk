import { BLOCKS, COADAPT_SETUP, COEVOLUTION, type Block } from '../../data/paper'
import { Badge } from '../ui/Badge'
import { int } from '../../lib/format'

/**
 * Co-evolution across the paired seeds, from Table 5.
 *
 * Drawn as one panel per run, not as an overlay with arm means, because the arm
 * mean is actively misleading here: averaging the full arm gives
 * 2496 to 4391 to 4801 to 5423, a monotonic rise with no suppression in it,
 * directly contradicting the claim the chart exists to support. Different seeds
 * get suppressed at different refits, so the mean smooths the sawtooth away.
 *
 * Per run, the shape is legible and the count is checkable: the paper's rule is
 * applied here rather than asserted, so "five of eight" can be counted off the
 * badges instead of taken on trust.
 */
const COLS = ['pre', 'r1', 'r2', 'r3'] as const
const W = 168
const H = 76
const PAD = 9

const ARM_INK = { Full: 'var(--color-atk)', Ablated: 'var(--color-ink-2)' }

/** The paper's own rule for the full pattern. */
export function showsPattern(b: Block): boolean {
  const post = [b.r1, b.r2, b.r3]
  const rose = post.some((v) => v > b.pre)
  const dropped = post.some((v, i) => v < (i === 0 ? b.pre : post[i - 1]))
  return rose && dropped
}

function Spark({ b, maxY }: { b: Block; maxY: number }) {
  const vals = COLS.map((c) => b[c])
  const x = (i: number) => PAD + (i / (COLS.length - 1)) * (W - PAD * 2)
  const y = (v: number) => H - PAD - (v / maxY) * (H - PAD * 2)
  const d = vals.map((v, i) => `${i ? 'L' : 'M'}${x(i)} ${y(v)}`).join(' ')
  const ink = ARM_INK[b.arm]
  const ok = showsPattern(b)
  const peakAt = vals.indexOf(Math.max(...vals))

  return (
    <div className="rounded-panel border border-rule bg-surface-card px-3 py-2.5">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[0.8125rem]">
          <span className={b.arm === 'Full' ? 'text-atk' : 'text-ink-2'}>{b.arm}</span>
          <span className="num ml-1.5 text-ink-3">seed {b.seed}</span>
        </span>
        {ok ? (
          <Badge tone="value">pattern</Badge>
        ) : (
          <span className="text-[0.6875rem] uppercase tracking-[0.09em] text-ink-3">no</span>
        )}
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="mt-1.5 w-full" aria-hidden="true">
        {/* Refits sit between blocks. */}
        {[0, 1, 2].map((i) => (
          <line
            key={i}
            x1={(x(i) + x(i + 1)) / 2}
            y1={PAD - 3}
            x2={(x(i) + x(i + 1)) / 2}
            y2={H - PAD + 3}
            stroke="var(--color-def)"
            strokeOpacity="0.3"
            strokeWidth="1"
            strokeDasharray="3 3"
          />
        ))}
        <line
          x1={PAD}
          y1={y(b.pre)}
          x2={W - PAD}
          y2={y(b.pre)}
          stroke="var(--color-ink-3)"
          strokeOpacity="0.45"
          strokeWidth="1"
          strokeDasharray="2 3"
        />
        <path d={d} fill="none" stroke={ink} strokeWidth="1.8" strokeLinecap="round" />
        {vals.map((v, i) => (
          <circle
            key={i}
            cx={x(i)}
            cy={y(v)}
            r={i === peakAt ? 2.8 : 1.8}
            fill={i === peakAt ? ink : 'var(--color-surface)'}
            stroke={ink}
            strokeWidth="1.2"
          />
        ))}
      </svg>

      <div className="num mt-1 flex items-baseline justify-between text-[0.75rem] text-ink-3">
        <span>opening {int(b.pre)}</span>
        <span className="text-ink-2">peak {int(b.peak)}</span>
      </div>
    </div>
  )
}

export function CoEvolution({ compact = false }: { compact?: boolean } = {}) {
  const maxY = Math.max(...BLOCKS.flatMap((b) => [b.pre, b.r1, b.r2, b.r3])) * 1.08
  const withPattern = BLOCKS.filter(showsPattern).length

  return (
    <div>
      {!compact && (
        <p className="prose-sans max-w-3xl text-[0.9375rem] leading-relaxed text-ink-2">
          {COADAPT_SETUP.holders} holders, {COADAPT_SETUP.updates} updates, a refit every{' '}
          {COADAPT_SETUP.refitEvery}, across {COADAPT_SETUP.seeds} paired seeds, so{' '}
          {COADAPT_SETUP.runs} runs. Each panel is one run: mean value extracted per episode in the
          opening block and after each defender refit, on a shared scale.
        </p>
      )}

      <div className={`grid gap-2.5 sm:grid-cols-2 lg:grid-cols-4 ${compact ? '' : 'mt-4'}`}>
        {BLOCKS.map((b) => (
          <Spark key={`${b.arm}-${b.seed}`} b={b} maxY={maxY} />
        ))}
      </div>

      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-[0.75rem] text-ink-3">
        <span>
          <span className="text-atk">red</span> full attacker,{' '}
          <span className="text-ink-2">grey</span> stealth ablated
        </span>
        <span>
          <span className="text-def">dashed vertical</span> a defender refit
        </span>
        <span>dotted horizontal is that run&rsquo;s opening level</span>
        <span>
          <Badge tone="value">pattern</Badge> rose above opening, and a refit knocked it back
        </span>
      </div>

      {!compact && (
        <>
          {/* The hero caption already carries these figures, so this only
              appears where the chart stands on its own. */}
          <p className="prose-sans mt-3 text-[0.875rem] leading-relaxed text-ink-2">
            {withPattern} of {COEVOLUTION.ofRuns} panels carry the badge, and peak extraction beats
            the opening level by {COEVOLUTION.peakOverOpeningMin}x to{' '}
            {COEVOLUTION.peakOverOpeningMax}x in all eight. The badge is computed from the table by
            the paper&rsquo;s own rule, so it can be counted rather than taken on trust.
          </p>
          <ul className="mt-3 space-y-1.5 border-t border-rule pt-3">
            {COEVOLUTION.exceptions.map((e) => (
              <li key={e.which} className="prose-sans text-[0.875rem] leading-relaxed text-ink-3">
                <span className="text-ink-2">{e.which}:</span> {e.what}
              </li>
            ))}
          </ul>
          <p className="prose-sans mt-3 text-[0.875rem] leading-relaxed text-ink-3">
            Shown per run rather than as an arm average on purpose. Averaging the full arm gives a
            monotonic rise with no suppression in it, because different seeds are suppressed at
            different refits, and that average would contradict the very pattern this figure is
            about. Both arms carry the pattern, so co-evolution is a property of the loop rather
            than of any one attacker capability.
          </p>
        </>
      )}
    </div>
  )
}
