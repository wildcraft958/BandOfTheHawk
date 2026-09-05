import { Badge } from '../ui/Badge'
import { WITHHELD } from '../../data/paper'

/**
 * The measurement this project decided not to report, and why.
 *
 * An earlier version of this site led with zero-shot recall of 1.000. The
 * solution document withholds it, because the held-out design was contaminated:
 * the held-out action stayed legal in the attacker's action space, so the
 * defender trained on the traffic it was then asked to generalise to. Saying so
 * is worth more than the number would have been.
 */
export function WithheldNote({ compact = false }: { compact?: boolean }) {
  return (
    <div className="rounded-panel border border-holdout/45 bg-holdout/5 px-4 py-3.5">
      <div className="flex flex-wrap items-center gap-2.5">
        <Badge tone="holdout">withheld</Badge>
        <span className="text-[0.9375rem] font-semibold text-ink">{WITHHELD.what}</span>
        <span className="text-[0.75rem] uppercase tracking-[0.09em] text-ink-3">
          {WITHHELD.cite}
        </span>
      </div>
      <p className="prose-sans mt-2.5 max-w-3xl text-[0.9375rem] leading-relaxed text-ink-2">
        {WITHHELD.why} {WITHHELD.consequence}
      </p>
      {!compact && (
        <p className="prose-sans mt-2 max-w-3xl text-[0.875rem] leading-relaxed text-ink-3">
          {WITHHELD.fix} It is reported here as a limitation rather than left off the page, because a
          held-out claim that cannot be trusted is worth less than an honest gap.
        </p>
      )}
    </div>
  )
}
