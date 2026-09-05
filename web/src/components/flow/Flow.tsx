import type { ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { cn } from '../ui/cn'

/**
 * The flow chart engine for the architecture diagram.
 *
 * Every part takes a `delay` and nothing schedules itself; the caller reads one
 * timetable from clock.ts. `dart` is a pass counter: changing it remounts the
 * animated elements so the whole graph replays together.
 */

const TONE = {
  neutral: 'border-rule',
  atk: 'border-atk/55',
  def: 'border-def/55',
  value: 'border-value/55',
  holdout: 'border-holdout/55',
} as const

const BADGE = {
  neutral: 'border-rule bg-surface-raised text-ink',
  atk: 'border-atk/35 bg-atk/10 text-atk',
  def: 'border-def/35 bg-def/10 text-def',
  value: 'border-value/35 bg-value/10 text-value',
  holdout: 'border-holdout/35 bg-holdout/10 text-holdout',
} as const

export type FlowTone = keyof typeof TONE

const DART = {
  neutral: 'var(--color-ink-3)',
  atk: 'var(--color-atk)',
  def: 'var(--color-def)',
  value: 'var(--color-value)',
  holdout: 'var(--color-holdout)',
} as const

/**
 * The vertical run between stacked stages, drawn as one stretched SVG so the
 * geometry survives any column width. The paths sit in a two-column grid, so a
 * fork has to land on 25% and 75% to meet the middle of each node.
 */
/**
 * The vertical run between stacked stages, drawn as one stretched SVG so the
 * geometry survives any column width. The paths sit in a two-column grid, so an
 * arm has to land on 25% or 75% to meet the middle of a node.
 *
 * Each arm is a separate path, drawn in the direction the flow actually travels.
 * A single compound path cannot do this: one stroke-dashoffset animation sweeps
 * in the path's own drawing order, so a fork's two arms would both animate the
 * same way round and its horizontal segment would always run left to right,
 * whichever way the flow goes. A fork diverges outward from the centre; a join
 * converges inward to it.
 *
 * Darts are normalised with pathLength, so the short centre stem and the longer
 * arms sweep at the same apparent speed.
 */
export function Branch({
  kind = 'straight',
  dart,
  delay = 0,
  tone = 'neutral',
}: {
  kind?: 'straight' | 'fork' | 'join'
  dart: number
  delay?: number
  tone?: FlowTone
}) {
  // Drawn in flow order. Fork: down the stem, then outward and down each arm.
  // Join: down and inward along each arm, then down the stem.
  const segments: Array<{ d: string; at: number }> =
    kind === 'straight'
      ? [{ d: 'M50 2 V26', at: 0 }]
      : kind === 'fork'
        ? [
            { d: 'M50 2 V14', at: 0 },
            { d: 'M50 14 H25 V26', at: 260 },
            { d: 'M50 14 H75 V26', at: 260 },
          ]
        : [
            { d: 'M25 2 V14 H50', at: 0 },
            { d: 'M75 2 V14 H50', at: 0 },
            { d: 'M50 14 V26', at: 260 },
          ]

  return (
    <svg
      viewBox="0 0 100 28"
      preserveAspectRatio="none"
      aria-hidden="true"
      data-branch={kind}
      className="h-7 w-full overflow-visible text-rule"
    >
      {segments.map((seg) => (
        <path
          key={`bg-${seg.d}`}
          d={seg.d}
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          vectorEffect="non-scaling-stroke"
          strokeDasharray="5 4"
          className="motion-safe:animate-[branch-drift_2.4s_linear_infinite]"
        />
      ))}
      {segments.map((seg) => (
        <path
          key={`${dart}-${seg.d}`}
          d={seg.d}
          fill="none"
          stroke={DART[tone]}
          strokeWidth="2"
          vectorEffect="non-scaling-stroke"
          pathLength={1}
          strokeDasharray="0.2 0.8"
          style={{ animationDelay: `${delay + seg.at}ms` }}
          className="opacity-0 motion-safe:animate-[branch-dart_0.7s_cubic-bezier(.4,0,.2,1)_both]"
        />
      ))}
    </svg>
  )
}

export function Node({
  kind,
  name,
  body,
  metric,
  tone = 'neutral',
  chip,
  note,
  icon: Icon,
  dart,
  delay = 0,
  onClick,
  expanded,
  children,
}: {
  kind: string
  name: string
  body: ReactNode
  metric?: string
  tone?: FlowTone
  chip?: ReactNode
  /** The stage's mark, carried in a tone-matched badge beside its label. */
  icon?: LucideIcon
  /** A one-line claim the node carries, shown under a rule. */
  note?: string
  dart: number
  delay?: number
  onClick?: () => void
  expanded?: boolean
  children?: ReactNode
}) {
  const Wrapper = onClick ? 'button' : 'div'

  return (
    <div
      key={dart}
      style={{ animationDelay: `${delay}ms` }}
      className={cn(
        'relative min-w-0 rounded-panel border bg-surface-card shadow-sm',
        TONE[tone],
        expanded && 'ring-1 ring-inset',
        expanded && tone === 'atk' && 'ring-atk/40',
        expanded && tone === 'def' && 'ring-def/40',
        'motion-safe:animate-[node-wake_1.1s_ease-out_both]',
      )}
    >
      <Wrapper
        {...(onClick ? { type: 'button' as const, onClick, 'aria-expanded': expanded } : {})}
        className={cn(
          'block w-full px-3 py-2.5 text-left',
          onClick && 'cursor-pointer transition-colors duration-150 hover:bg-surface-hover',
        )}
      >
        <div className="flex items-start justify-between gap-2">
          <div className="flex min-w-0 items-center gap-2">
            {Icon && (
              <span
                className={cn(
                  'grid size-6 shrink-0 place-items-center rounded-chip border',
                  BADGE[tone],
                )}
              >
                <Icon className="size-3.5" aria-hidden="true" />
              </span>
            )}
            <div className="min-w-0">
              <p className="text-[0.6875rem] font-semibold uppercase tracking-[0.11em] text-ink-3">
                {kind}
              </p>
              <p className="text-[0.9375rem] font-semibold leading-tight text-ink">{name}</p>
            </div>
          </div>
          {metric && (
            <span
              className={cn(
                'num shrink-0 rounded-chip border px-1.5 py-[0.15rem] text-[0.75rem] font-semibold',
                BADGE[tone],
              )}
            >
              {metric}
            </span>
          )}
        </div>
        <div className="mt-2 text-[0.8125rem] leading-relaxed text-ink-2">{body}</div>
        {chip && <div className="mt-2">{chip}</div>}
        {note && (
          <p className="mt-2 border-t border-rule pt-2 text-[0.6875rem] font-semibold uppercase tracking-[0.1em] text-ink-3">
            {note}
          </p>
        )}
        {onClick && (
          <p className="mt-2.5 flex items-center gap-1 text-[0.6875rem] uppercase tracking-[0.1em] text-ink-3">
            <ChevronDown
              className={cn('size-3 transition-transform duration-200', expanded && 'rotate-180')}
              aria-hidden="true"
            />
            {expanded ? 'hide mechanics' : 'mechanics'}
          </p>
        )}
      </Wrapper>
      {expanded && children && <div className="border-t border-rule px-3 py-3">{children}</div>}
    </div>
  )
}
