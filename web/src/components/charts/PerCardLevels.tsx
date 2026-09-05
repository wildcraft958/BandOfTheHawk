import { useMemo } from 'react'
import { mulberry32, gaussian } from '../../lib/rng'
import { Label } from '../ui/Label'
import { fidelity } from '../../data/run'
import { Note } from '../ui/Note'

/**
 * Why matching the pooled distribution is not enough.
 *
 * Both panels below hold exactly the same numbers. The only difference is which
 * card each amount is assigned to, so the pooled histograms are identical by
 * construction rather than by coincidence. On the left every card draws from one
 * shared curve; on the right each card has its own level.
 *
 * This matters because of what the detector actually reads. Its per-entity
 * features ask "is this unusual for *this* card": amount_vs_median,
 * is_first_txn_this_merchant, within_usual_hours. A generator that is right on
 * average and wrong per card is wrong exactly where the detector looks, and a
 * pooled histogram cannot show that.
 *
 * The variance decomposition is measured. The samples are drawn from it to make
 * the mechanism visible.
 */

const BINS = 11
const CARDS = 3
const PER_CARD = 320

function histogram(values: number[], lo: number, hi: number): number[] {
  const counts = new Array(BINS).fill(0)
  for (const v of values) {
    const i = Math.min(BINS - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * BINS)))
    counts[i] += 1
  }
  return counts
}

function sd(values: number[]): number {
  if (values.length < 2) return 0
  const mean = values.reduce((a, v) => a + v, 0) / values.length
  const variance = values.reduce((a, v) => a + (v - mean) ** 2, 0) / values.length
  return Math.sqrt(variance)
}

function Bars({ counts, max, tone }: { counts: number[]; max: number; tone: string }) {
  return (
    <div className="flex h-9 items-end gap-[2px]">
      {counts.map((c, i) => (
        <div
          key={i}
          className={tone}
          style={{ height: `${max > 0 ? Math.max((c / max) * 100, 1.5) : 1.5}%`, flex: 1 }}
        />
      ))}
    </div>
  )
}

function Panel({
  title,
  note,
  cards,
  pooled,
  lo,
  hi,
  betweenSd,
  tone,
}: {
  title: string
  note: string
  cards: number[][]
  pooled: number[]
  lo: number
  hi: number
  betweenSd: number
  tone: 'atk' | 'pass'
}) {
  const bar = tone === 'pass' ? 'bg-pass/70' : 'bg-atk/70'
  const text = tone === 'pass' ? 'text-pass' : 'text-atk'
  const cardHists = cards.map((c) => histogram(c, lo, hi))
  const pooledHist = histogram(pooled, lo, hi)
  const cardMax = Math.max(...cardHists.flat(), 1)
  const pooledMax = Math.max(...pooledHist, 1)

  return (
    <div className="rounded-panel border border-rule bg-surface px-4 py-3">
      <Label>{title}</Label>
      <p className="prose-sans mt-1 text-[0.8125rem] leading-snug text-ink-3">{note}</p>

      <div className="mt-3 space-y-2">
        {cardHists.map((counts, i) => (
          <div key={i}>
            <div className="mb-0.5 text-[0.6875rem] uppercase tracking-[0.1em] text-ink-3">
              card {String.fromCharCode(65 + i)}
            </div>
            <Bars counts={counts} max={cardMax} tone={bar} />
          </div>
        ))}
      </div>

      <div className="mt-3 border-t border-rule pt-2">
        <div className="mb-0.5 text-[0.6875rem] uppercase tracking-[0.1em] text-ink">pooled</div>
        <Bars counts={pooledHist} max={pooledMax} tone="bg-ink-3" />
      </div>

      <p className="num mt-3 border-t border-rule pt-2 text-[0.8125rem] text-ink-2">
        spread of these {cards.length} card levels <span className={text}>{betweenSd.toFixed(4)}</span>
      </p>
    </div>
  )
}

export function PerCardLevels() {
  const het = fidelity.amount_heterogeneity

  const model = useMemo(() => {
    if (!het) return null
    const rand = mulberry32(7)
    const grand = het.grand_mean
    const between = het.between_sd
    const within = het.within_sd

    // Right-hand world: each card gets its own level, then its own events around
    // that level.
    //
    // The three levels are rescaled so their spread is exactly the measured
    // between-card figure. Three draws from a distribution with sd 0.5663 can
    // easily land at 0.88, and a panel reporting 0.88 beside a measured 0.5663
    // reads as a contradiction rather than as sampling noise. This is an
    // illustration of a measured quantity, so it should show that quantity.
    const raw = Array.from({ length: CARDS }, () => gaussian(rand))
    const rawMean = raw.reduce((a, v) => a + v, 0) / CARDS
    const centred = raw.map((v) => v - rawMean)
    const rawSd = Math.sqrt(centred.reduce((a, v) => a + v * v, 0) / CARDS) || 1
    const levels = centred.map((v) => grand + (v / rawSd) * between)

    const levelled: number[][] = levels.map((level) => {
      const values: number[] = []
      for (let k = 0; k < PER_CARD; k++) values.push(level + gaussian(rand) * within)
      return values
    })

    // Left-hand world: the exact same numbers, shuffled and dealt round-robin,
    // so every card is a sample from the one shared curve. The pooled histogram
    // cannot tell the two apart because it is literally the same multiset.
    const pooled = levelled.flat()
    const shuffled = pooled.slice()
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1))
      ;[shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]]
    }
    const shared: number[][] = Array.from({ length: CARDS }, () => [])
    shuffled.forEach((v, i) => shared[i % CARDS].push(v))

    return {
      shared,
      levelled,
      pooled,
      lo: Math.min(...pooled),
      hi: Math.max(...pooled),
      // Every card in the shared world draws from the one grand mean, so the
      // level spread is exactly zero by construction.
      sharedBetween: 0,
      levelledBetween: sd(levels),
    }
  }, [het])

  if (!model || !het) return null

  return (
    <div>
      <p className="prose-sans max-w-3xl text-[0.9375rem] leading-relaxed text-ink-2">
        Both panels hold the same numbers. Only the card each amount belongs to differs, so the
        pooled histograms are identical by construction, not by luck. A generator judged on its
        pooled distribution alone passes both.
      </p>

      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <Panel
          title="one shared curve"
          note="Every card samples from the same distribution. Correct on average."
          cards={model.shared}
          pooled={model.pooled}
          lo={model.lo}
          hi={model.hi}
          betweenSd={model.sharedBetween}
          tone="atk"
        />
        <Panel
          title="per-card levels"
          note="Each card has its own level, drawn with the measured between-card spread."
          cards={model.levelled}
          pooled={model.pooled}
          lo={model.lo}
          hi={model.hi}
          betweenSd={model.levelledBetween}
          tone="pass"
        />
      </div>

      <div className="mt-4 rounded-panel border border-rule bg-surface px-4 py-3">
        <Label>measured on real data</Label>
        <dl className="num mt-2 grid grid-cols-2 gap-x-6 gap-y-1.5 text-[0.875rem] sm:grid-cols-4">
          {(
            [
              ['between-card sd', het.between_sd.toFixed(4)],
              ['within-card sd', het.within_sd.toFixed(4)],
              ['total sd', het.total_sd.toFixed(4)],
              ['variance between cards', `${(het.between_share * 100).toFixed(1)}%`],
            ] as const
          ).map(([k, v]) => (
            <div key={k}>
              <dt className="text-[0.6875rem] uppercase tracking-[0.1em] text-ink-3">{k}</dt>
              <dd className="mt-0.5 text-ink">{v}</dd>
            </div>
          ))}
        </dl>
        <Note
          label="why this is the gap that matters"
        >
          <p className="prose-sans text-[0.9375rem] leading-relaxed text-ink-2">
          <span className="text-pass">{(het.between_share * 100).toFixed(1)}%</span> of amount
          variance sits between cards rather than within them, across{' '}
          {het.n_entities.toLocaleString('en-US')} entities and{' '}
          {het.n_events.toLocaleString('en-US')} events. A generator that samples every card from
          one shared curve puts that figure at zero by construction while reproducing the pooled
          histogram exactly.
        </p>
          <p className="prose-sans mt-2 text-[0.9375rem] leading-relaxed text-ink-2">
          That is the gap that matters, because the detector&rsquo;s per-entity features ask
          whether an event is unusual{' '}
          <span className="text-ink">for this card</span>: amount_vs_median,
          is_first_txn_this_merchant, within_usual_hours. Being right on average and wrong per card
          is being wrong exactly where the detector reads.
        </p>
        </Note>
      </div>
    </div>
  )
}
