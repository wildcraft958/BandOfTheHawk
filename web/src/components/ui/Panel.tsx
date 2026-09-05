import type { ReactNode } from 'react'
import { cn } from './cn'

/**
 * The console's container. Header is a coloured dash plus a mono uppercase
 * name, so a judge can scan panel titles down the page.
 */
export function Panel({
  name,
  live,
  aside,
  children,
  className,
  bodyClassName,
  tone = 'atk',
}: {
  name: ReactNode
  live?: boolean
  aside?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
  tone?: 'atk' | 'def' | 'value' | 'pass' | 'holdout'
}) {
  const dash = {
    atk: 'bg-atk',
    def: 'bg-def',
    value: 'bg-value',
    pass: 'bg-pass',
    holdout: 'bg-holdout',
  }[tone]

  return (
    <section className={cn('rounded-card border border-rule bg-surface-card', className)}>
      <header className="flex items-center justify-between gap-4 border-b border-rule-subtle px-4 py-3">
        <h3 className="flex min-w-0 items-center gap-2.5">
          <span className={cn('h-[2px] w-5 shrink-0', dash)} aria-hidden="true" />
          <span className="truncate text-[0.8125rem] font-semibold uppercase tracking-[0.11em] text-ink">
            {name}
          </span>
          {live && (
            <span
              className="size-1.5 shrink-0 rounded-full bg-atk"
              aria-label="live"
              title="live"
            />
          )}
        </h3>
        {aside}
      </header>
      <div className={bodyClassName ?? 'px-4 py-4'}>{children}</div>
    </section>
  )
}
