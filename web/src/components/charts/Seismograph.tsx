import { scaleLinear, scaleSymlog } from '@visx/scale'
import { LinePath } from '@visx/shape'
import { useReducedMotion } from '../../lib/useReducedMotion'
import type { CoadaptPoint } from '../../data/types'

const W = 1000
const H = 300

/**
 * The hero object: value extracted per episode across all 150 co-adaptation
 * updates, with the twelve defender refits as vertical scars.
 *
 * No axes, no ticks, no labels. The instrumented version of the same series
 * lives on The Loop; here it is a signature mark, and the one thing it has to
 * make legible is the shape. The attacker climbs to 23,391, the refit at update
 * 11 takes it to exactly 0.0 for sixteen updates, and it claws back through a
 * different channel.
 *
 * Scale is symlog rather than linear. The series spans 0 to 28,781.9 and
 * contains sixteen true zeros: on a linear axis the entire collapse, the
 * dramatic heart of the run, flattens into an unreadable line on the baseline,
 * and a log axis cannot represent the zeros at all.
 */
export function Seismograph({
  points,
  refits,
}: {
  points: CoadaptPoint[]
  refits: number[]
}) {
  const reduced = useReducedMotion()

  const x = scaleLinear({ domain: [0, points.length - 1], range: [0, W] })
  const y = scaleSymlog({
    domain: [0, Math.max(...points.map((p) => p.extracted))],
    range: [H, 8],
    constant: 100,
  })

  const zeroY = y(0)

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="block h-[clamp(170px,24vh,260px)] w-full"
      role="img"
      aria-labelledby="seismograph-title"
      preserveAspectRatio="none"
    >
      <title id="seismograph-title">
        Value extracted per episode across 150 co-adaptation updates. The attacker climbs to
        23,391 by update 11, the defender refits, and extraction falls to exactly zero for
        sixteen consecutive updates before recovering through a different attack channel and
        converging.
      </title>

      {/* The twelve refits. Each is the moment the defender retrained. */}
      {refits.map((update) => (
        <line
          key={update}
          x1={x(update)}
          x2={x(update)}
          y1={0}
          y2={H}
          stroke="var(--color-def)"
          strokeWidth={1}
          opacity={0.28}
        />
      ))}

      {/* The contested boundary. Where the curve sits on this rule, the
          defender has shut the attacker out completely. */}
      <line
        x1={0}
        x2={W}
        y1={zeroY}
        y2={zeroY}
        stroke="var(--color-def)"
        strokeWidth={1.5}
        opacity={0.9}
      />

      <LinePath
        data={points}
        x={(p) => x(p.update)}
        y={(p) => y(p.extracted)}
        stroke="var(--color-value)"
        strokeWidth={1.75}
        vectorEffect="non-scaling-stroke"
        className={reduced ? undefined : 'seismo-trace'}
        pathLength={1}
      />
    </svg>
  )
}
