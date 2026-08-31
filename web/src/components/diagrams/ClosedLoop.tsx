import { useState } from 'react'
import { useReducedMotion } from '../../lib/useReducedMotion'
import { duration } from '../../lib/format'
import type { StageTiming } from '../../data/types'

/**
 * The pipeline drawn as what it actually is: two setup stages feeding a
 * four-stage cycle.
 *
 *   demo -> text -> [ fraud -> baseline -> mixture -> coadapt ] and back to fraud
 *
 * That cycle is the closed loop the competition asks to see demonstrated. The
 * attacks generated in `fraud` train the detector in `baseline` and `mixture`;
 * `coadapt` runs attacker and defender against each other and feeds the gaps it
 * finds back into new attacks. Drawing it as a ring rather than a bar is the
 * honest shape.
 *
 * Node area is proportional to real runtime, so co-adaptation dominates exactly
 * as much as it did in the run (3321s of 3819s).
 */

const CX = 400
const CY = 150
const R = 104

const SETUP = ['demo', 'text'] as const
const CYCLE = ['fraud', 'baseline', 'mixture', 'coadapt'] as const

// Left, top, right, bottom. Travel is clockwise from fraud.
const CYCLE_POS: Record<string, { x: number; y: number }> = {
  fraud: { x: CX - R, y: CY },
  baseline: { x: CX, y: CY - R },
  mixture: { x: CX + R, y: CY },
  coadapt: { x: CX, y: CY + R },
}

const SETUP_POS: Record<string, { x: number; y: number }> = {
  demo: { x: 52, y: CY },
  text: { x: 158, y: CY },
}

const BLURB: Record<string, string> = {
  demo: 'build the synthetic bank',
  text: 'generate and embed dispute text',
  fraud: 'inject scripted fraud episodes',
  baseline: 'fit the flat detector',
  mixture: 'fit five routed experts',
  coadapt: 'attacker and defender co-adapt',
}

/** Clockwise arc between two points on the cycle. */
function arc(from: string, to: string): string {
  const a = CYCLE_POS[from]
  const b = CYCLE_POS[to]
  return `M ${a.x} ${a.y} A ${R} ${R} 0 0 1 ${b.x} ${b.y}`
}

function radius(seconds: number, max: number): number {
  // Area, not diameter, carries the proportion, so the small stages stay legible.
  return 9 + 21 * Math.sqrt(seconds / max)
}

export function ClosedLoop({ stages, total }: { stages: StageTiming[]; total: number }) {
  const reduced = useReducedMotion()
  const [active, setActive] = useState<string | null>(null)

  const byName = new Map(stages.map((s) => [s.stage, s]))
  const max = Math.max(...stages.map((s) => s.seconds))

  const edges = [
    { id: 'e-fraud-baseline', d: arc('fraud', 'baseline') },
    { id: 'e-baseline-mixture', d: arc('baseline', 'mixture') },
    { id: 'e-mixture-coadapt', d: arc('mixture', 'coadapt') },
    { id: 'e-coadapt-fraud', d: arc('coadapt', 'fraud'), feedback: true },
  ]

  const shown = active ? byName.get(active) : null

  return (
    <div>
      <svg
        viewBox="0 0 580 306"
        className="block w-full max-w-2xl"
        role="img"
        aria-labelledby="loop-title"
      >
        <title id="loop-title">
          The pipeline as a cycle: demo and text build the synthetic world, then fraud, baseline,
          mixture and coadapt form a closed loop in which generated attacks train the detector and
          the detector&rsquo;s gaps feed new attacks. Node size is proportional to runtime;
          co-adaptation took 3321 of the run&rsquo;s 3819 seconds.
        </title>

        <defs>
          <marker id="arrow" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5" markerHeight="5" orient="auto">
            <path d="M 0 0 L 8 4 L 0 8 z" fill="var(--color-rule-strong)" />
          </marker>
          {edges.map((e) => (
            <path key={e.id} id={e.id} d={e.d} />
          ))}
        </defs>

        {/* setup chain */}
        <line
          x1={SETUP_POS.demo.x}
          y1={CY}
          x2={SETUP_POS.text.x}
          y2={CY}
          stroke="var(--color-rule-strong)"
          strokeWidth={1}
          markerEnd="url(#arrow)"
        />
        <line
          x1={SETUP_POS.text.x}
          y1={CY}
          x2={CYCLE_POS.fraud.x - 26}
          y2={CY}
          stroke="var(--color-rule-strong)"
          strokeWidth={1}
          markerEnd="url(#arrow)"
        />

        {/* the cycle */}
        {edges.map((e) => (
          <path
            key={e.id}
            d={e.d}
            fill="none"
            stroke={e.feedback ? 'var(--color-def)' : 'var(--color-rule-strong)'}
            strokeWidth={e.feedback ? 1.5 : 1}
            strokeDasharray={e.feedback ? '4 4' : undefined}
            opacity={e.feedback ? 0.85 : 1}
          />
        ))}

        {/* Packets circulating the loop. This is the one thing on the page that
            animates continuously, because the loop running is the claim. */}
        {!reduced &&
          edges.map((e, i) => (
            <circle key={`p-${e.id}`} r={3} fill={e.feedback ? 'var(--color-def)' : 'var(--color-value)'}>
              <animateMotion dur="2.6s" begin={`${i * 0.65}s`} repeatCount="indefinite" rotate="auto">
                <mpath href={`#${e.id}`} />
              </animateMotion>
            </circle>
          ))}

        <text
          x={CX}
          y={CY - 6}
          textAnchor="middle"
          className="fill-ink-3 font-mono text-[9px] uppercase"
          style={{ letterSpacing: '0.14em' }}
        >
          the closed loop
        </text>
        <text
          x={CX}
          y={CY + 9}
          textAnchor="middle"
          className="fill-def font-mono text-[9px]"
        >
          12 defender refits
        </text>

        {/* The feedback direction is the thing being graded, so it is named
            rather than left for the reader to infer from an arrowhead. */}
        <text
          textAnchor="middle"
          className="fill-def font-mono text-[8.5px]"
          style={{ letterSpacing: '0.06em' }}
        >
          <textPath href="#e-coadapt-fraud" startOffset="50%">
            gaps feed new attacks
          </textPath>
        </text>

        {/* nodes */}
        {[...SETUP, ...CYCLE].map((name) => {
          const stage = byName.get(name)
          if (!stage) return null
          const pos = SETUP_POS[name] ?? CYCLE_POS[name]
          const rr = radius(stage.seconds, max)
          const isCoadapt = name === 'coadapt'
          const isActive = active === name
          return (
            <g
              key={name}
              tabIndex={0}
              role="button"
              aria-label={`${name}: ${duration(stage.seconds)}, ${BLURB[name]}`}
              onMouseEnter={() => setActive(name)}
              onMouseLeave={() => setActive(null)}
              onFocus={() => setActive(name)}
              onBlur={() => setActive(null)}
              className="cursor-default outline-none"
            >
              <circle
                cx={pos.x}
                cy={pos.y}
                r={rr}
                fill={isCoadapt ? 'var(--color-value)' : 'var(--color-ground-raised)'}
                fillOpacity={isCoadapt ? 0.16 : 1}
                stroke={
                  isActive
                    ? 'var(--color-ink)'
                    : isCoadapt
                      ? 'var(--color-value)'
                      : 'var(--color-rule-strong)'
                }
                strokeWidth={isActive ? 2 : 1.25}
              />
              <text
                x={pos.x}
                y={pos.y + rr + 15}
                textAnchor="middle"
                className="fill-ink font-mono text-[10px] uppercase"
                style={{ letterSpacing: '0.1em' }}
              >
                {name}
              </text>
              <text
                x={pos.x}
                y={pos.y + rr + 28}
                textAnchor="middle"
                className="fill-ink-3 font-mono text-[9px]"
              >
                {duration(stage.seconds)}
              </text>
            </g>
          )
        })}
      </svg>

      <p className="mt-3 min-h-[1.5rem] text-[0.8125rem] text-ink-2">
        {shown ? (
          <>
            <span className="font-mono text-[0.6875rem] uppercase tracking-[0.1em] text-ink">
              {shown.stage}
            </span>{' '}
            &middot; {BLURB[shown.stage]} &middot;{' '}
            <span className="num">{duration(shown.seconds)}</span>, or{' '}
            <span className="num">{((shown.seconds / total) * 100).toFixed(1)}%</span> of the run
          </>
        ) : (
          <span className="text-ink-3">
            Hover or tab a stage for its real runtime. Node area is proportional to time spent.
          </span>
        )}
      </p>
    </div>
  )
}
