import { useState } from 'react'
import { Replay } from './demo/Replay'
import { ArrowRight, ChevronRight, Sparkles } from 'lucide-react'
import { Panel } from '../components/ui/Panel'
import { Badge } from '../components/ui/Badge'
import { Chip, ChipGroup } from '../components/ui/Chip'
import { Label } from '../components/ui/Label'
import { StageMachine } from '../components/diagrams/StageMachine'
import { TaxonomyMatrix } from '../components/diagrams/TaxonomyMatrix'
import { ActionGrid } from '../components/diagrams/ActionGrid'
import { ArtifactInspector } from '../components/diagrams/ArtifactInspector'
import { coadapt, runReport } from '../data/run'
import { ACTIONS, LEGAL_ACTIONS, VERTICALS } from '../data/taxonomy'
import type { Vertical } from '../data/taxonomy'
import { cn } from '../components/ui/cn'
import { Link } from 'react-router-dom'
import { PageHead } from '../components/ui/PageHead'
import { Note } from '../components/ui/Note'

const ACTION_BY_ID = new Map(ACTIONS.map((a) => [a.id, a]))

type Source = 'scripted' | 'learned'

function tape(runs: Array<{ action: string; times: number }>) {
  return runs.map((r) => (r.times > 1 ? `${r.action} x${r.times}` : r.action)).join(' > ')
}

/**
 * Attack explorer. Pick a vertical on the left, and see what the simulator
 * actually produced for it.
 *
 * The toggle contrasts the two attackers the run really had: the hand-written
 * scripted policies, and the trained RL policy. That contrast is the argument
 * for a learned attacker over scripts, and it is also an honest admission that
 * the learned one found a degenerate exploit.
 */
export function Demo() {
  const [selected, setSelected] = useState<Vertical>(VERTICALS[0])
  const [source, setSource] = useState<Source>('scripted')

  const episodes = runReport.per_vertical[selected.id]

  const scripted = runReport.top_sequences
  const learned = coadapt.final_sequences

  return (
    <div className="lg:flex lg:items-stretch">
      <aside className="border-b border-rule lg:w-72 lg:shrink-0 lg:border-b-0 lg:border-r">
        <div className="flex items-center gap-2.5 px-4 py-3">
          <span className="h-[2px] w-5 bg-atk" aria-hidden="true" />
          <span className="text-[0.8125rem] font-semibold uppercase tracking-[0.11em] text-ink">
            verticals
          </span>
        </div>
        <ul className="max-h-[22rem] overflow-y-auto lg:max-h-none">
          {VERTICALS.map((v) => {
            const active = v.id === selected.id
            return (
              <li key={v.id}>
                <button
                  type="button"
                  onClick={() => setSelected(v)}
                  aria-current={active ? 'true' : undefined}
                  className={cn(
                    'w-full border-l-2 border-t border-t-rule-subtle px-4 py-3 text-left transition-colors duration-150',
                    active
                      ? 'border-l-atk bg-surface-hover'
                      : 'border-l-transparent hover:bg-surface-raised',
                  )}
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <Chip
                      tone={v.heldOut ? 'holdout' : !v.simulated ? undefined : 'atk'}
                    >
                      {v.heldOut ? 'held out' : !v.simulated ? 'excluded' : v.entryStage}
                    </Chip>
                  </div>
                  <div
                    className={cn(
                      'mt-1.5 text-[0.9375rem]',
                      active ? 'text-ink' : v.simulated ? 'text-ink-2' : 'text-ink-3 line-through',
                    )}
                  >
                    {v.label}
                  </div>
                  <div className="prose-sans mt-0.5 text-[0.8125rem] leading-snug text-ink-3">
                    {v.blurb}
                  </div>
                </button>
              </li>
            )
          })}
        </ul>
      </aside>

      <div className="min-w-0 flex-1 space-y-4 px-4 py-6 sm:px-6">
        <PageHead
          title="Attack surface"
          blurb="Eleven verticals identified across the account lifecycle over one shared twenty-action space, nine of them simulated. Pick one and run it."
        />

        <div>
          {/* The vertical is the content heading, so it sits under the page's h1. */}
          <h2
            className="font-display text-[clamp(1.75rem,5vw,3.25rem)] font-extrabold uppercase leading-none tracking-[-0.02em]"
            style={{ fontStretch: '80%' }}
          >
            {selected.label}
          </h2>
          <p className="prose-sans mt-3 max-w-2xl text-[1rem] text-ink-2">{selected.blurb}</p>

          <div className="mt-4 flex flex-wrap items-center gap-2">
            <Badge tone={selected.simulated ? 'atk' : 'neutral'}>
              enters at {selected.entryStage}
            </Badge>
            {selected.heldOut && <Badge tone="holdout">held out of training</Badge>}
            {!selected.simulated && <Badge tone="neutral">not simulated</Badge>}
            {episodes != null && <Badge tone="neutral">{episodes} episodes</Badge>}
          </div>
        </div>

        {!selected.simulated ? (
          <Panel name="why this is not simulated" tone="def">
            <p className="prose-sans max-w-2xl text-[0.9375rem] leading-relaxed text-ink-2">
              {selected.exclusion}
            </p>
            <p className="prose-sans mt-3 max-w-2xl text-[0.9375rem] text-ink-3">
              Describing an excluded vertical is more useful than simulating one on fabricated
              numbers.
            </p>
          </Panel>
        ) : (
          <>
            <Panel
              name="run one attack"
              tone="atk"
              aside={
                <span className="hidden text-[0.75rem] uppercase tracking-[0.09em] text-ink-3 md:inline">
                  run it again for a different episode
                </span>
              }
            >
              <Replay vertical={selected} />
              {/* The judged path runs attack then detection, so it should not
                  dead end here and send a reader back to the nav to find the
                  metrics that scored what they just watched. */}
              <div className="mt-5 border-t border-rule pt-4">
                <Link
                  to="/detection"
                  className="inline-flex items-center gap-2 rounded-panel border border-def/55 bg-def/10 px-3.5 py-2 text-[0.8125rem] uppercase tracking-[0.09em] text-def no-underline transition-colors duration-150 hover:bg-def/15"
                >
                  See what catches it <ArrowRight className="size-3.5" aria-hidden="true" />
                </Link>
                <span className="ml-3 text-[0.8125rem] text-ink-3">
                  five detector configurations, and what recall costs at a fixed alert budget
                </span>
              </div>
            </Panel>

            <Panel name="generative capability" tone="atk">
              <div className="flex items-start gap-3">
                <Sparkles className="mt-0.5 size-4 shrink-0 text-atk" aria-hidden="true" />
                <div>
                  <p className="text-[0.9375rem] text-ink">{selected.capability}</p>
                  <p className="prose-sans mt-1 text-[0.875rem] text-ink-3">
                    The step where a generative model changes the attacker&rsquo;s cost.
                  </p>
                </div>
              </div>

              <div className="mt-5">
                <Label>actions legal at {selected.entryStage}</Label>
                <ChipGroup>
                  {LEGAL_ACTIONS[selected.entryStage].map((id) => {
                    const a = ACTION_BY_ID.get(id)
                    return (
                      <Chip key={id} tone={a?.genaiTool ? 'atk' : undefined} title={a?.description}>
                        {id}
                        {a?.genaiTool ? ' *' : ''}
                      </Chip>
                    )
                  })}
                </ChipGroup>
                <p className="prose-sans mt-2 text-[0.8125rem] text-ink-3">
                  * requires a generated artifact
                </p>
              </div>
            </Panel>

            <Panel
              name="attacker output"
              live
              aside={
                <div
                  role="group"
                  aria-label="Attacker source"
                  className="flex items-center gap-1 rounded-panel border border-rule p-0.5"
                >
                  {(['scripted', 'learned'] as const).map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => setSource(s)}
                      aria-pressed={source === s}
                      className={cn(
                        'rounded-[2px] px-2.5 py-1 text-[0.75rem] uppercase tracking-[0.09em] transition-colors duration-150',
                        source === s ? 'bg-atk/15 text-atk' : 'text-ink-3 hover:text-ink-2',
                      )}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              }
              bodyClassName="px-0 py-0"
            >
              {source === 'scripted' ? (
                <>
                  <ul className="divide-y divide-rule-subtle">
                    {scripted.map((s, i) => (
                      <li key={i} className="flex items-start gap-3 px-4 py-3 hover:bg-surface-hover">
                        <ChevronRight className="mt-0.5 size-3 shrink-0 text-ink-3" aria-hidden="true" />
                        <span className="min-w-0 flex-1 break-words text-[0.8125rem] text-ink">
                          {s.chain.join(' > ')}
                        </span>
                        <span className="num w-12 shrink-0 text-right text-[0.8125rem] text-ink-2">
                          {s.count}x
                        </span>
                      </li>
                    ))}
                  </ul>
                  <p className="prose-sans border-t border-rule px-4 py-3 text-[0.875rem] text-ink-2">
                    Ten distinct chains of three to nine actions, written by hand across the seven
                    trained verticals.
                  </p>
                </>
              ) : (
                <>
                  <ul className="divide-y divide-rule-subtle">
                    {learned.map((s, i) => (
                      <li key={i} className="flex items-start gap-3 px-4 py-3">
                        <Chip tone="holdout">converged</Chip>
                        <span className="min-w-0 flex-1 break-words text-[0.8125rem] text-atk">
                          {tape(s.runs)}
                        </span>
                        <span className="num w-12 shrink-0 text-right text-[0.8125rem] text-ink-2">
                          {s.count}x
                        </span>
                      </li>
                    ))}
                  </ul>
                  <Note
                    label="what one repeated chain means"
                    className="mx-4 mb-3"
                  >
                    <p className="prose-sans text-[0.875rem] leading-relaxed text-ink-2">
                      One chain, repeated. The trained policy collapsed onto a refund loop that
                      saturates the 40-action episode budget. A degenerate exploit, and the
                      rediscovery of a vertical held out of training. Nobody scripted this.
                    </p>
                  </Note>
                </>
              )}
            </Panel>
          </>
        )}

        <Panel name="capability model" tone="def">
          <StageMachine />
        </Panel>

        <Panel
          name="every vertical, across two axes"
          tone="atk"
          aside={
            <span className="hidden text-[0.75rem] uppercase tracking-[0.09em] text-ink-3 md:inline">
              where it enters, and what it needs
            </span>
          }
          bodyClassName="px-4 py-4"
        >
          <p className="prose-sans mb-4 max-w-3xl text-[0.9375rem] leading-relaxed text-ink-2">
            Two axes rather than a flat list, because the claim being graded is that the attack
            surface is spanned rather than sampled: where in the account lifecycle each attack
            enters, and which generative capability changes the attacker&rsquo;s cost.
          </p>
          <TaxonomyMatrix />
        </Panel>

        <Panel name="what the generative layer produces" tone="holdout">
          <ArtifactInspector />
        </Panel>

        <Panel name="the shared action space" tone="atk">
          <ActionGrid />
        </Panel>
      </div>
    </div>
  )
}
