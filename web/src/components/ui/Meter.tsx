import { cn } from './cn'

const TONE = {
  def: 'bg-def',
  atk: 'bg-atk',
  value: 'bg-value',
  pass: 'bg-pass',
  holdout: 'bg-holdout',
  neutral: 'bg-ink-3',
} as const

export function Meter({
  label,
  value,
  max,
  display,
  tone = 'def',
  note,
}: {
  label: string
  value: number
  max: number
  display?: string
  tone?: keyof typeof TONE
  note?: string
}) {
  const pct = max > 0 ? Math.max((value / max) * 100, 0) : 0
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <span className="truncate font-mono text-[0.75rem] text-ink">{label}</span>
        <span className="num shrink-0 text-[0.8125rem] text-ink-2">{display ?? value}</span>
      </div>
      <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-rule">
        <div className={cn('h-full rounded-full', TONE[tone])} style={{ width: `${pct}%` }} />
      </div>
      {note && <div className="mt-1 text-[0.6875rem] text-ink-3">{note}</div>}
    </div>
  )
}
