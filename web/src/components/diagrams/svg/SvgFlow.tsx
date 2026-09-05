import { useReducedMotion } from '../../../lib/useReducedMotion'
/**
 * The diagram engine: boxes and wires as plain SVG, with a dart travelling each
 * wire so the direction of flow is visible rather than inferred.
 *
 * Deliberately not a graph library. These diagrams have fixed, hand-placed
 * geometry, so a layout engine has nothing to contribute, and hand-authored SVG
 * cannot fail to draw a connector.
 *
 * Every wire declares pathLength 1. Without it a short wire and a long one get
 * the same dash length in user units, so the short one appears to move faster.
 * Normalising means one duration reads as one speed everywhere.
 */

export type Tone = 'atk' | 'def' | 'pass' | 'value' | 'holdout'

const INK: Record<Tone, string> = {
  atk: 'var(--color-atk)',
  def: 'var(--color-def)',
  pass: 'var(--color-pass)',
  value: 'var(--color-value)',
  holdout: 'var(--color-holdout)',
}

export interface SvgBox {
  id: string
  x: number
  y: number
  w?: number
  h?: number
  eyebrow?: string
  label: string
  sub?: string
  tone: Tone
}

export interface SvgWire {
  d: string
  tone: Tone
  label?: string
  lx?: number
  ly?: number
  /** Dashed, for something learned travelling backwards. */
  feedback?: boolean
  /**
   * When this step happens, in seconds from the start of the diagram's cycle.
   * Steps that genuinely happen together share a value; steps that depend on
   * each other are ordered by it. This is what turns movement into flow.
   */
  at: number
  /** Seconds this traversal takes. Slower reads as a heavier step. */
  travel?: number
}

export interface SvgGroup {
  x: number
  y: number
  w: number
  h: number
  title: string
  note?: string
  tone: Tone
}

function Group({ g }: { g: SvgGroup }) {
  return (
    <g>
      <rect
        x={g.x}
        y={g.y}
        width={g.w}
        height={g.h}
        rx={5}
        fill="var(--color-surface-raised)"
        fillOpacity="0.35"
        stroke="var(--color-rule)"
        strokeWidth="1"
        strokeDasharray="4 4"
      />
      {/* Title and note in one text run, so the gap comes from a tspan offset
          rather than from guessing the title's rendered width. */}
      <text
        x={g.x + 12}
        y={g.y + 18}
        fontSize="8.5"
        letterSpacing="1.1"
        fontFamily="var(--font-mono)"
      >
        <tspan fill={INK[g.tone]}>{g.title.toUpperCase()}</tspan>
        {g.note && (
          <tspan dx="12" fill="var(--color-ink-3)" fontSize="8">
            {g.note.toUpperCase()}
          </tspan>
        )}
      </text>
    </g>
  )
}

function Box({ b }: { b: SvgBox }) {
  const w = b.w ?? 150
  const h = b.h ?? (b.sub ? 52 : 38)
  const hasEyebrow = !!b.eyebrow
  return (
    <g>
      <rect
        x={b.x}
        y={b.y}
        width={w}
        height={h}
        rx={4}
        fill="var(--color-surface-card)"
        stroke={INK[b.tone]}
        strokeOpacity="0.6"
        strokeWidth="1"
      />
      {hasEyebrow && (
        <text
          x={b.x + 9}
          y={b.y + 14}
          fill={INK[b.tone]}
          fontSize="7"
          letterSpacing="0.85"
          fontFamily="var(--font-mono)"
        >
          {b.eyebrow!.toUpperCase()}
        </text>
      )}
      <text
        x={b.x + 9}
        y={b.y + (hasEyebrow ? 29 : 23)}
        fill="var(--color-ink)"
        fontSize="11.5"
        fontWeight="600"
        fontFamily="var(--font-mono)"
      >
        {b.label}
      </text>
      {b.sub && (
        <text
          x={b.x + 9}
          y={b.y + (hasEyebrow ? 43 : 37)}
          fill="var(--color-ink-3)"
          fontSize="8"
          fontFamily="var(--font-mono)"
        >
          {b.sub}
        </text>
      )}
    </g>
  )
}

function Wire({
  w,
  idPrefix,
  cycle,
  animate,
}: {
  w: SvgWire
  idPrefix: string
  cycle: number
  animate: boolean
}) {
  // A wire crossing in more than a second is one whose cost is time: a latency,
  // a refit. Those get the slow keyframe so the eye reads the weight.
  const slow = (w.travel ?? 0) > 1

  return (
    <g>
      <path
        d={w.d}
        fill="none"
        stroke={INK[w.tone]}
        strokeWidth="1.25"
        strokeOpacity={w.feedback ? 0.5 : 0.65}
        strokeDasharray={w.feedback ? '5 4' : undefined}
        markerEnd={`url(#${idPrefix}-arrow-${w.tone})`}
      />
      <path
        d={w.d}
        fill="none"
        stroke={INK[w.tone]}
        strokeWidth="2.4"
        strokeLinecap="round"
        pathLength={1}
        strokeDasharray="0.1 0.9"
        opacity={0}
        className={animate ? (slow ? 'wire-step-slow' : 'wire-step') : undefined}
        style={
          animate
            ? { animationDuration: `${cycle}s`, animationDelay: `${w.at}s` }
            : undefined
        }
      />
      {w.label && (
        <text x={w.lx} y={w.ly} fill={INK[w.tone]} fontSize="8" fontFamily="var(--font-mono)">
          {w.label}
        </text>
      )}
    </g>
  )
}

const TONES: Tone[] = ['atk', 'def', 'pass', 'value', 'holdout']

export function SvgFlow({
  id,
  viewBox,
  groups,
  boxes,
  wires,
  ariaLabel,
  cycle,
}: {
  /** Namespaces the marker ids, so two diagrams on one page cannot collide. */
  id: string
  viewBox: string
  groups?: SvgGroup[]
  boxes: SvgBox[]
  wires: SvgWire[]
  ariaLabel: string
  /** Seconds for one full pass of the diagram. Every wire shares it. */
  cycle: number
}) {
  const reduced = useReducedMotion()
  return (
    <svg viewBox={viewBox} className="w-full" role="img" aria-label={ariaLabel}>
      <defs>
        {TONES.map((t) => (
          <marker
            key={t}
            id={`${id}-arrow-${t}`}
            viewBox="0 0 8 8"
            refX="7"
            refY="4"
            markerWidth="5.5"
            markerHeight="5.5"
            orient="auto-start-reverse"
          >
            <path d="M0 0 L8 4 L0 8 z" fill={INK[t]} />
          </marker>
        ))}
      </defs>

      {groups?.map((g) => <Group key={g.title} g={g} />)}
      {wires.map((w) => (
        <Wire
          key={w.d + (w.label ?? '')}
          w={w}
          idPrefix={id}
          cycle={cycle}
          animate={!reduced}
        />
      ))}
      {boxes.map((b) => <Box key={b.id} b={b} />)}
    </svg>
  )
}

export function Legend({ children }: { children: React.ReactNode }) {
  return (
    <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-[0.75rem] text-ink-3">{children}</div>
  )
}
