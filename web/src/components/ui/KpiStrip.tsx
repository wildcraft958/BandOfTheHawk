import type { ReactNode } from 'react'
import { cn } from './cn'

const TONE = {
  ink: 'text-ink',
  atk: 'text-atk',
  def: 'text-def',
  value: 'text-value',
  pass: 'text-pass',
  holdout: 'text-holdout',
} as const

export interface Kpi {
  label: string
  value: ReactNode
  detail?: string
  tone?: keyof typeof TONE
}

/** One bordered row of divided cells, as an ops console reports its headline numbers. */
export function KpiStrip({ items }: { items: Kpi[] }) {
  return (
    <div className="grid grid-cols-2 divide-rule rounded-card border border-rule bg-surface-card sm:grid-cols-3 sm:divide-x lg:grid-cols-5">
      {items.map((k) => (
        <div key={k.label} className="border-b border-rule px-4 py-4 last:border-b-0 sm:border-b-0">
          <div className="text-[0.625rem] font-semibold uppercase tracking-[0.1em] text-ink-3">
            {k.label}
          </div>
          <div
            className={cn(
              'num mt-2 text-[1.75rem] font-bold leading-none tracking-tight',
              TONE[k.tone ?? 'ink'],
            )}
          >
            {k.value}
          </div>
          {k.detail && <div className="mt-2 text-[0.6875rem] text-ink-2">{k.detail}</div>}
        </div>
      ))}
    </div>
  )
}
