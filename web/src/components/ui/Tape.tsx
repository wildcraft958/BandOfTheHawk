import type { ActionRun } from '../../data/types'
import { cn } from './cn'

/**
 * An action chain rendered at full length rather than summarised.
 *
 * The trained attacker's top sequence is two setup actions followed by
 * request_refund thirty-eight times, which exactly saturates the forty-action
 * episode budget. Collapsing that to "x38" hides the thing worth seeing, so
 * every repeat gets a cell.
 */
export function Tape({ runs, expandUpTo = 40 }: { runs: ActionRun[]; expandUpTo?: number }) {
  const cells: Array<{ action: string; index: number; repeat: boolean }> = []
  for (const run of runs) {
    const n = Math.min(run.times, expandUpTo)
    for (let i = 0; i < n; i++) {
      cells.push({ action: run.action, index: i, repeat: run.times > 1 })
    }
  }

  return (
    <div className="overflow-x-auto">
      <div className="flex min-w-max items-stretch gap-[2px]">
        {cells.map((c, i) => (
          <div
            key={i}
            title={c.action}
            className={cn(
              'flex w-[1.15rem] items-end justify-center border-b-2 py-1.5 text-[0.6875rem]',
              c.repeat
                ? 'border-b-atk bg-atk/10 text-atk'
                : 'border-b-ink-3 bg-surface-hover text-ink-2',
            )}
          >
            {c.repeat ? c.index + 1 : ''}
          </div>
        ))}
      </div>
      <div className="mt-2 flex min-w-max gap-4 text-[0.8125rem]">
        {runs.map((r, i) => (
          <span key={i} className="text-ink-2">
            {r.action}
            {r.times > 1 && <span className="text-atk"> x{r.times}</span>}
          </span>
        ))}
      </div>
    </div>
  )
}
