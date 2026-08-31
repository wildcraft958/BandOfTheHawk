import { Label } from '../ui/Label'
import { cn } from '../ui/cn'
import type { PosteriorLevel } from '../../data/types'

/**
 * Victim selection, learned by a contextual bandit. Coefficients are
 * differences from a reference level, so they are signed and read against a
 * centre line.
 */
export function PosteriorChart({
  groups,
}: {
  groups: Array<{ name: string; levels: PosteriorLevel[] }>
}) {
  const max = Math.max(
    ...groups.flatMap((g) => g.levels.map((l) => Math.abs(l.coef))),
    1,
  )

  return (
    <div className="space-y-6">
      {groups.map((g) => (
        <div key={g.name}>
          <Label>{g.name}</Label>
          <ul className="mt-2.5 space-y-2">
            {g.levels.map((l) => {
              const frac = Math.abs(l.coef) / max
              const positive = l.coef > 0
              return (
                <li key={l.label} className="flex items-center gap-3">
                  <span className="w-24 shrink-0 truncate text-[0.6875rem] text-ink-2">
                    {l.label.replace(/^(bin tier|age) /, '')}
                  </span>
                  <span className="relative h-3 flex-1">
                    <span
                      className="absolute inset-y-0 left-1/2 w-px bg-rule"
                      aria-hidden="true"
                    />
                    {!l.reference && l.coef !== 0 && (
                      <span
                        className={cn(
                          'absolute inset-y-0 rounded-[1px]',
                          positive ? 'bg-atk' : 'bg-def',
                        )}
                        style={
                          positive
                            ? { left: '50%', width: `${(frac * 50).toFixed(2)}%` }
                            : { right: '50%', width: `${(frac * 50).toFixed(2)}%` }
                        }
                      />
                    )}
                  </span>
                  <span
                    className={cn(
                      'num w-20 shrink-0 text-right text-[0.6875rem]',
                      l.reference ? 'text-ink-3' : positive ? 'text-atk' : 'text-def',
                    )}
                  >
                    {l.reference ? 'reference' : l.coef > 0 ? `+${l.coef}` : l.coef}
                  </span>
                </li>
              )
            })}
          </ul>
        </div>
      ))}
    </div>
  )
}
