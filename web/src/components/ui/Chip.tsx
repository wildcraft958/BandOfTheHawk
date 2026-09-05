import type { ReactNode } from 'react'
import { cn } from './cn'

export function Chip({
  children,
  tone,
  title,
}: {
  children: ReactNode
  tone?: 'atk' | 'def' | 'value' | 'holdout' | 'pass'
  title?: string
}) {
  return (
    <span
      title={title}
      className={cn(
        'inline-flex items-center rounded-chip border px-1.5 py-[0.15rem] font-mono text-[0.8125rem]',
        tone === 'atk' && 'border-atk/40 text-atk',
        tone === 'def' && 'border-def/40 text-def',
        tone === 'value' && 'border-value/40 text-value',
        tone === 'holdout' && 'border-holdout/45 text-holdout',
        tone === 'pass' && 'border-pass/40 text-pass',
        !tone && 'border-rule bg-surface text-ink-2',
      )}
    >
      {children}
    </span>
  )
}

export function ChipGroup({ children }: { children: ReactNode }) {
  return <div className="mt-1.5 flex flex-wrap gap-1.5">{children}</div>
}
