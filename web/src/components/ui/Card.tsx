import type { ReactNode } from 'react'
import { cn } from './cn'

interface CardProps {
  /** Small mono identifier above the title, as a check or record id. */
  label?: ReactNode
  title?: ReactNode
  subtitle?: ReactNode
  /** Top-right slot, usually a Badge. */
  aside?: ReactNode
  children?: ReactNode
  className?: string
  padded?: boolean
}

export function Card({
  label,
  title,
  subtitle,
  aside,
  children,
  className,
  padded = true,
}: CardProps) {
  const hasHeader = label || title || subtitle || aside
  return (
    <section
      className={cn(
        'rounded-card border border-rule bg-surface-card transition-colors duration-150',
        className,
      )}
    >
      {hasHeader && (
        <header
          className={cn(
            'flex items-start justify-between gap-4 border-b border-rule-subtle',
            padded ? 'px-5 py-4' : 'px-4 py-3',
          )}
        >
          <div className="min-w-0">
            {label && (
              <div className="font-mono text-[0.625rem] uppercase tracking-[0.1em] text-ink-3">
                {label}
              </div>
            )}
            {title && (
              <h3 className="mt-0.5 text-[1.0625rem] font-semibold leading-snug text-ink">
                {title}
              </h3>
            )}
            {subtitle && <p className="mt-1 text-[0.8125rem] text-ink-2">{subtitle}</p>}
          </div>
          {aside}
        </header>
      )}
      {children && <div className={padded ? 'px-5 py-4' : 'px-4 py-3'}>{children}</div>}
    </section>
  )
}
