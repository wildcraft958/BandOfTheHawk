import type { ReactNode } from 'react'
import { cn } from './cn'

export type BadgeTone = 'atk' | 'def' | 'value' | 'pass' | 'holdout' | 'neutral'

const TONE: Record<BadgeTone, string> = {
  atk: 'bg-atk/12 text-atk',
  def: 'bg-def/12 text-def',
  value: 'bg-value/12 text-value',
  pass: 'bg-pass/12 text-pass',
  holdout: 'bg-holdout/14 text-holdout',
  neutral: 'bg-surface-hover text-ink-2',
}

export function Badge({
  tone = 'neutral',
  children,
  className,
}: {
  tone?: BadgeTone
  children: ReactNode
  className?: string
}) {
  return (
    <span
      className={cn(
        'inline-flex shrink-0 items-center rounded-full px-2 py-[0.15rem] font-mono text-[0.625rem] font-semibold uppercase tracking-[0.08em]',
        TONE[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}
