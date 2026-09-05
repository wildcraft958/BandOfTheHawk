import { useMemo, useState } from 'react'
import { ChevronRight, Play, RotateCcw, ShieldCheck, Swords } from 'lucide-react'
import { Badge } from '../../components/ui/Badge'
import { Chip } from '../../components/ui/Chip'
import { Label } from '../../components/ui/Label'
import { cn } from '../../components/ui/cn'
import { ACTIONS, STAGE_BLURB, STAGE_ORDER, type Stage, type Vertical } from '../../data/taxonomy'
import { BANDS, TIERS, TIER_NOTE } from '../../data/paper'
import { MITIGATION, runEpisode, type Decision } from './episode'
import { int } from '../../lib/format'
import { Note } from '../../components/ui/Note'

/**
 * Pick an attack vector, run one episode, see where it gets stopped.
 *
 * The mechanism is the real one: the stage machine, the actions each stage
 * admits, the three transitions that advance it, the five decisions the
 * defender's boundaries separate, and the per-episode jitter that moves those
 * boundaries. The score is a stand-in and says so.
 *
 * It reports no success rate, because the paper reports none. What it shows is
 * where an attack has to get to before it is worth anything, and how many
 * chances the defender has to stop it on the way.
 */
const GENAI = new Set(ACTIONS.filter((a) => a.genaiTool).map((a) => a.id))
const ACTION_BLURB = new Map(ACTIONS.map((a) => [a.id, a.description]))

const DECISION_TONE: Record<Decision, 'pass' | 'value' | 'atk' | 'def'> = {
  approve: 'pass',
  'step up': 'def',
  hold: 'value',
  decline: 'atk',
  block: 'atk',
}

function StageRail({ current, entry }: { current: Stage; entry: Stage }) {
  const reached = STAGE_ORDER.indexOf(current)
  const from = STAGE_ORDER.indexOf(entry)
  return (
    <ol className="flex flex-wrap items-center gap-1.5">
      {STAGE_ORDER.filter((s) => s !== 'terminal').map((s, i) => {
        // A vertical that enters at BOUND was never in NONE or ACQUIRED, so
        // those are skipped rather than reached.
        const skipped = i < from
        const done = i >= from && i <= reached
        const start = i === from
        return (
          <li key={s} className="flex items-center gap-1.5">
            <span
              title={skipped ? `skipped: this vector enters at ${entry}` : STAGE_BLURB[s]}
              className={cn(
                'rounded-chip border px-2 py-[0.15rem] font-mono text-[0.75rem] uppercase tracking-[0.08em]',
                done && 'border-atk/50 bg-atk/10 text-atk',
                skipped && 'border-rule/60 text-ink-3 line-through opacity-55',
                !done && !skipped && 'border-rule text-ink-3',
                start && 'ring-1 ring-inset ring-atk/40',
              )}
            >
              {s}
            </span>
            {i < 3 && <ChevronRight className="size-3 text-ink-3" aria-hidden="true" />}
          </li>
        )
      })}
    </ol>
  )
}

export function Replay({ vertical }: { vertical: Vertical }) {
  const [tier, setTier] = useState(2)
  const [seed, setSeed] = useState(7)

  const episode = useMemo(
    () => runEpisode({ seed, entryStage: vertical.entryStage, tier }),
    [seed, vertical.entryStage, tier],
  )

  const won = !episode.stopped && episode.finalStage === 'monetized'
  const chances = episode.steps.filter((s) => s.scored).length

  return (
    <div>
      <div className="grid gap-5 lg:grid-cols-[minmax(0,17rem)_1fr]">
        <div>
          <div>
            <Label>capability tier</Label>
            <div
              className="mt-2 flex overflow-hidden rounded-panel border border-rule"
              role="group"
              aria-label="Capability tier"
            >
              {TIERS.map((t) => (
                <button
                  key={t.tier}
                  type="button"
                  onClick={() => setTier(t.tier)}
                  aria-pressed={tier === t.tier}
                  title={t.length}
                  className={cn(
                    'num flex-1 px-2 py-1.5 text-[0.875rem] transition-colors duration-150',
                    tier === t.tier ? 'bg-holdout/15 text-holdout' : 'text-ink-3 hover:text-ink-2',
                  )}
                >
                  {t.tier}
                </button>
              ))}
            </div>
            <p className="prose-sans mt-2 text-[0.8125rem] leading-relaxed text-ink-3">
              {TIERS[tier].length}. {TIER_NOTE.claim}, and nothing more precise: the ladder is
              ordinal, so no per-tier success rate is claimed here.
            </p>
          </div>

          <div className="mt-5 flex items-center gap-2">
            <button
              type="button"
              onClick={() => setSeed((s) => s + 1)}
              className="inline-flex flex-1 items-center justify-center gap-2 rounded-panel border border-atk/55 bg-atk/10 px-3 py-2 text-[0.8125rem] uppercase tracking-[0.09em] text-atk transition-colors duration-150 hover:bg-atk/15"
            >
              <Play className="size-3" aria-hidden="true" /> run again
            </button>
            <button
              type="button"
              onClick={() => setSeed(7)}
              title="Back to the first episode"
              aria-label="Reset the seed"
              className="grid size-8 shrink-0 place-items-center rounded-panel border border-rule text-ink-2 transition-colors duration-150 hover:border-ink-3 hover:text-ink"
            >
              <RotateCcw className="size-3" aria-hidden="true" />
            </button>
          </div>
          <p className="num mt-2 text-[0.75rem] text-ink-3">
            seed {seed} &middot; same seed always replays identically
          </p>
        </div>

        <div className="min-w-0">
          <div
            className={cn(
              'rounded-panel border px-4 py-3.5',
              won ? 'border-atk/55 bg-atk/8' : 'border-pass/50 bg-pass/5',
            )}
          >
            <div className="flex flex-wrap items-center gap-2.5">
              {won ? (
                <>
                  <Swords className="size-4 text-atk" aria-hidden="true" />
                  <span className="text-[1rem] font-semibold text-ink">
                    Reached monetisation
                  </span>
                  <Badge tone="atk">attacker</Badge>
                </>
              ) : (
                <>
                  <ShieldCheck className="size-4 text-pass" aria-hidden="true" />
                  <span className="text-[1rem] font-semibold text-ink">
                    {episode.stoppedBy === 'block' ? 'Blocked' : 'Declined'} on action{' '}
                    {episode.steps.length}
                  </span>
                  <Badge tone="pass">defender</Badge>
                </>
              )}
            </div>
            <p className="prose-sans mt-2 text-[0.9375rem] leading-relaxed text-ink-2">
              {won ? (
                <>
                  {episode.steps.length} actions, {chances} of them scored, and none crossed the
                  decline boundary. Value extracted{' '}
                  <span className="num text-value">{int(episode.extracted)}</span> against an action
                  cost of <span className="num">{episode.totalCost}</span>.
                </>
              ) : (
                <>
                  It died in the{' '}
                  <span className="text-atk">
                    {episode.steps[episode.steps.length - 1]?.stage ?? vertical.entryStage}
                  </span>{' '}
                  stage, which means {MITIGATION[episode.stoppedBy ?? 'block']}. The defender had{' '}
                  {chances} scored events to work with, and this one crossed the{' '}
                  {episode.stoppedBy} boundary of{' '}
                  <span className="num">
                    {(episode.stoppedBy === 'block'
                      ? episode.bands.block
                      : episode.bands.decline
                    ).toFixed(3)}
                  </span>
                  .
                </>
              )}
            </p>
          </div>

          <div className="mt-4">
            <Label>where it got to</Label>
            <div className="mt-2">
              <StageRail current={episode.finalStage} entry={vertical.entryStage} />
            </div>
            <p className="prose-sans mt-2 text-[0.8125rem] text-ink-3">
              {vertical.label} enters at <span className="text-atk">{vertical.entryStage}</span>.
              Nothing is worth anything before monetised, which is why the defender has room to act.
            </p>
          </div>

          <div className="mt-5">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <Label>the tape</Label>
              <span className="num text-[0.75rem] text-ink-3">
                boundaries this episode: {episode.bands.stepUp.toFixed(3)} /{' '}
                {episode.bands.hold.toFixed(3)} / {episode.bands.decline.toFixed(3)} /{' '}
                {episode.bands.block.toFixed(3)}
              </span>
            </div>
            <ol className="mt-2 divide-y divide-rule-subtle">
              {episode.steps.map((s) => (
                <li key={s.n} className="flex flex-wrap items-baseline gap-x-3 gap-y-1 py-2">
                  <span className="num w-4 shrink-0 text-[0.75rem] text-ink-3">{s.n}</span>
                  <Chip tone={GENAI.has(s.action) ? 'atk' : undefined}>{s.action}</Chip>
                  <span className="min-w-0 flex-1 truncate text-[0.8125rem] text-ink-3">
                    {ACTION_BLURB.get(s.action)}
                  </span>
                  {s.advancedTo && (
                    <span className="font-mono text-[0.6875rem] uppercase tracking-[0.08em] text-value">
                      advances to {s.advancedTo}
                    </span>
                  )}
                  {s.scored ? (
                    <span className="flex shrink-0 items-center gap-2">
                      <span className="num text-[0.8125rem] text-ink-2">
                        {s.score?.toFixed(3)}
                      </span>
                      <Badge tone={DECISION_TONE[s.decision ?? 'approve']}>{s.decision}</Badge>
                    </span>
                  ) : (
                    <span className="shrink-0 font-mono text-[0.6875rem] uppercase tracking-[0.08em] text-ink-3">
                      emits no event
                    </span>
                  )}
                </li>
              ))}
            </ol>
          </div>
        </div>
      </div>

      <Note
        label="what is real in this episode, and what is not"
        lede={
          <>
            The score is a transparent stand-in built from the action, the events already emitted
            and the capability tier, not the trained ensemble. No success rate is claimed from it.
          </>
        }
      >
        <p className="prose-sans max-w-3xl text-[0.875rem] leading-relaxed text-ink-2">
          <span className="text-ink">Real:</span> the five stages and the actions each one admits,
          the three transitions that advance a stage, the per-action costs, the five decisions the
          defender&rsquo;s four boundaries separate, and the jitter that moves those boundaries once
          per episode so a policy cannot find a threshold and sit under it. Actions in red require a
          generated artifact, which is where a generative model changes the attacker&rsquo;s cost.
          Silent actions emit no event, so the defender never sees them.
        </p>
        <p className="prose-sans mt-2 max-w-3xl text-[0.875rem] leading-relaxed text-ink-2">
          <span className="text-ink">Not real:</span> the boundaries shown are one refit&rsquo;s
          grid-search output ({BANDS.stepUp} / {BANDS.hold} / {BANDS.decline} / {BANDS.block}) with
          this episode&rsquo;s jitter applied. Outcomes here demonstrate the ladder and the gating.
        </p>
      </Note>
    </div>
  )
}
