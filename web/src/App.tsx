import { scaleLinear, scaleSymlog } from '@visx/scale'
import { LinePath } from '@visx/shape'
import { Group } from '@visx/group'
import { ShieldAlert } from 'lucide-react'

// Spike only: a real slice of the co-adaptation series from run.log, covering
// the first refit. Extraction climbs, the defender refits at update 11, and the
// attacker's take goes to exactly zero. The full 150 rows arrive via the
// extractor script.
const SLICE: Array<[number, number]> = [
  [0, 4245.8],
  [1, 6723.8],
  [2, 10342.8],
  [3, 16474.7],
  [4, 14586.8],
  [5, 9100.7],
  [6, 11442.8],
  [7, 10299.0],
  [8, 21461.6],
  [9, 20221.9],
  [10, 15723.2],
  [11, 23391.5],
  [12, 0.0],
  [13, 0.0],
  [14, 0.0],
  [15, 0.0],
]

const W = 560
const H = 220

export default function App() {
  const x = scaleLinear({ domain: [0, 15], range: [0, W] })
  const y = scaleSymlog({ domain: [0, 28781.9], range: [H, 0], constant: 100 })

  return (
    <main id="main" className="mx-auto max-w-3xl px-6 py-16">
      <p className="font-mono text-[0.6875rem] uppercase tracking-[0.14em] text-ink-3">
        Build spike &middot; verifying the single-file pipeline
      </p>

      <h1 className="mt-4 font-display text-6xl font-extrabold tracking-[-0.03em]"
        style={{ fontStretch: '75%' }}>GAUNTLET</h1>

      <div className="mt-8 flex items-baseline gap-3">
        <ShieldAlert className="size-5 text-def" aria-hidden="true" />
        <span className="num font-display text-4xl font-bold text-def">0.9879</span>
        <span className="text-ink-2">flat GBDT PR-AUC</span>
      </div>

      <figure className="mt-10 rounded-panel border border-rule bg-ground-raised p-5">
        <svg width="100%" viewBox={`0 0 ${W} ${H + 20}`} role="img" aria-label="Value extracted per episode over the first sixteen updates, collapsing to zero after the defender refit at update eleven.">
          <Group top={6}>
            <line x1={0} x2={W} y1={y(0)} y2={y(0)} stroke="var(--color-def)" strokeWidth={1.5} />
            <line
              x1={x(11)}
              x2={x(11)}
              y1={0}
              y2={H}
              stroke="var(--color-def)"
              strokeWidth={1}
              strokeDasharray="2 3"
            />
            <LinePath
              data={SLICE}
              x={(d) => x(d[0])}
              y={(d) => y(d[1])}
              stroke="var(--color-value)"
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
            />
          </Group>
        </svg>
        <figcaption className="mt-3 font-mono text-[0.6875rem] text-ink-3">
          symlog scale &middot; the dashed rule is the defender refit at update 11
        </figcaption>
      </figure>
    </main>
  )
}
