import { ACTIONS, PHASE_LABEL } from '../../data/taxonomy'
import type { Phase } from '../../data/taxonomy'

const PHASES: Phase[] = ['acquire', 'bind', 'spend', 'extract']

/**
 * All twenty actions, grouped by the four phases of an episode. The seven that
 * require a generated artifact are marked, because that mapping is the concrete
 * answer to where the GenAI actually enters the fraud rather than a claim that
 * it does.
 */
export function ActionGrid() {
  const genaiCount = ACTIONS.filter((a) => a.genaiTool).length

  return (
    <div>
      <p className="mb-6 max-w-3xl text-[0.9375rem] text-ink-2">
        One shared action space across every vertical, so a per-vertical recall figure reflects a
        difference in behaviour rather than in machinery.{' '}
        <span className="text-atk">{genaiCount} of {ACTIONS.length} actions</span> require a
        generated artifact: a cloned voice, a deepfake selfie, a written pretext, a dispute
        narrative. Those are the points where a generative model changes the attacker&rsquo;s cost.
      </p>

      <div className="grid gap-x-8 gap-y-8 sm:grid-cols-2 lg:grid-cols-4">
        {PHASES.map((phase) => (
          <div key={phase}>
            <h3 className="border-b border-rule-strong pb-2 font-mono text-[0.75rem] uppercase tracking-[0.12em] text-ink-3">
              {PHASE_LABEL[phase]}
            </h3>
            <ul className="mt-3 space-y-3">
              {ACTIONS.filter((a) => a.phase === phase).map((a) => (
                <li key={a.id}>
                  <div className="flex items-baseline justify-between gap-2">
                    <span
                      className={`font-mono text-[0.875rem] ${a.genaiTool ? 'text-atk' : 'text-ink'}`}
                    >
                      {a.id}
                    </span>
                    <span className="num shrink-0 font-mono text-[0.75rem] text-ink-3">
                      {a.cost.toFixed(1)}
                    </span>
                  </div>
                  <div className="text-[0.875rem] leading-snug text-ink-3">{a.description}</div>
                  {a.genaiTool && (
                    <div className="mt-0.5 font-mono text-[0.75rem] text-atk/80">
                      needs {a.genaiTool}
                    </div>
                  )}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  )
}
