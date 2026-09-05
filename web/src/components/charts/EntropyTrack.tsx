import { Group } from '@visx/group'
import { scaleLinear } from '@visx/scale'
import { LinePath } from '@visx/shape'
import type { CoadaptPoint } from '../../data/types'

const W = 1000
const H = 110
const M = { top: 8, right: 16, bottom: 8, left: 58 }
const IW = W - M.left - M.right
const IH = H - M.top - M.bottom

/**
 * Policy entropy, which does not fall monotonically.
 *
 * It holds near 3.5 through the first refit, jumps above 4.1 for the twenty
 * updates that follow, then decays to 2.844. The refit did not only crush the
 * attacker's income: it destroyed a converged policy and threw it back into
 * exploration. The recovery converges tighter than the start.
 */
export function EntropyTrack({ points, refits }: { points: CoadaptPoint[]; refits: number[] }) {
  const lo = Math.min(...points.map((p) => p.entropy))
  const hi = Math.max(...points.map((p) => p.entropy))
  const x = scaleLinear({ domain: [0, points.length - 1], range: [0, IW] })
  const y = scaleLinear({ domain: [lo - 0.1, hi + 0.1], range: [IH, 0] })

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="block w-full" role="img" aria-labelledby="ent-title">
      <title id="ent-title">
        Policy entropy across 150 updates. It sits near 3.5, spikes above 4.2 after the first
        defender refit as the attacker is forced back into exploration, then decays to 2.844,
        tighter than where it began.
      </title>
      <Group left={M.left} top={M.top}>
        {refits.map((u) => (
          <line
            key={u}
            x1={x(u)}
            x2={x(u)}
            y1={0}
            y2={IH}
            stroke="var(--color-def)"
            strokeWidth={1}
            opacity={0.25}
          />
        ))}
        <LinePath
          data={points}
          x={(p) => x(p.update)}
          y={(p) => y(p.entropy)}
          stroke="var(--color-def)"
          strokeWidth={1.25}
          vectorEffect="non-scaling-stroke"
        />
        <text x={-6} y={y(hi) + 3} textAnchor="end" className="fill-ink-3" fontSize={9} fontFamily="var(--font-mono)">
          {hi.toFixed(2)}
        </text>
        <text x={-6} y={y(lo) + 3} textAnchor="end" className="fill-ink-3" fontSize={9} fontFamily="var(--font-mono)">
          {lo.toFixed(2)}
        </text>
      </Group>
    </svg>
  )
}
