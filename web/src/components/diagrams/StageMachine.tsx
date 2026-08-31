import { useState } from 'react'
import { ChevronRight } from 'lucide-react'
import { ACTIONS, ADVANCES, LEGAL_ACTIONS, STAGE_BLURB, STAGE_ORDER } from '../../data/taxonomy'
import type { Stage } from '../../data/taxonomy'

const COST = new Map(ACTIONS.map((a) => [a.id, a.cost]))
const TOOL = new Map(ACTIONS.map((a) => [a.id, a.genaiTool]))

/**
 * The five stages and what is legal in each. Selecting a stage shows its legal
 * actions and marks the ones that advance it, which is the whole capability
 * model: an actor cannot spend a card nobody has bound.
 */
export function StageMachine() {
  const [stage, setStage] = useState<Stage>('none')

  const legal = LEGAL_ACTIONS[stage]
  const advance = ADVANCES.find((a) => a.from === stage)
  const advancing = new Set(advance?.via ?? [])

  return (
    <div>
      <div className="flex flex-wrap items-center gap-1.5">
        {STAGE_ORDER.map((s, i) => (
          <div key={s} className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => setStage(s)}
              aria-pressed={stage === s}
              className={`rounded-[3px] border px-3 py-1.5 font-mono text-[0.6875rem] uppercase tracking-[0.1em] transition-colors duration-150 ${
                stage === s
                  ? 'border-ink bg-ground-hover text-ink'
                  : 'border-rule-strong text-ink-3 hover:border-ink-3 hover:text-ink-2'
              }`}
            >
              {s}
            </button>
            {i < STAGE_ORDER.length - 1 && (
              <ChevronRight className="size-3.5 text-rule-strong" aria-hidden="true" />
            )}
          </div>
        ))}
      </div>

      <p className="mt-4 text-[0.875rem] text-ink-2">
        <span className="font-mono text-[0.6875rem] uppercase tracking-[0.1em] text-ink">
          {stage}
        </span>{' '}
        &middot; {STAGE_BLURB[stage]}
        {advance && (
          <>
            {' '}Advances to{' '}
            <span className="font-mono text-[0.6875rem] uppercase text-value">{advance.to}</span> on
            a successful move marked below.
          </>
        )}
      </p>

      {legal.length === 0 ? (
        <p className="mt-5 font-mono text-[0.75rem] text-ink-3">
          No legal actions. The episode is over.
        </p>
      ) : (
        <ul className="mt-5 flex flex-wrap gap-2">
          {legal.map((id) => {
            const advances = advancing.has(id)
            const tool = TOOL.get(id)
            return (
              <li
                key={id}
                className={`rounded-[3px] border px-2.5 py-1.5 ${
                  advances ? 'border-value/50 bg-value/5' : 'border-rule'
                }`}
              >
                <span className="font-mono text-[0.75rem] text-ink">{id}</span>
                <span className="num ml-2 font-mono text-[0.625rem] text-ink-3">
                  cost {COST.get(id)?.toFixed(1)}
                </span>
                {tool && (
                  <span className="ml-2 font-mono text-[0.625rem] text-atk" title={`requires ${tool}`}>
                    genai
                  </span>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
