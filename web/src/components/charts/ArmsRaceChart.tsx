import { useState } from 'react'
import { AxisBottom, AxisLeft } from '@visx/axis'
import { Group } from '@visx/group'
import { scaleLinear, scaleSymlog } from '@visx/scale'
import { LinePath } from '@visx/shape'
import type { CoadaptPoint } from '../../data/types'
import { int } from '../../lib/format'
import { cn } from '../ui/cn'

const W = 1000
const H = 340
const M = { top: 12, right: 16, bottom: 34, left: 58 }
const IW = W - M.left - M.right
const IH = H - M.top - M.bottom

// Linear near zero, logarithmic beyond. The series spans 0 to 28,781.9 and holds
// sixteen true zeros, so a linear axis flattens the collapse into nothing and a
// log axis cannot represent the zeros at all.
const TICKS = [0, 25, 100, 1000, 10000]

export function ArmsRaceChart({
  points,
  refits,
  selected,
  onSelect,
}: {
  points: CoadaptPoint[]
  refits: number[]
  selected: number | null
  onSelect: (update: number | null) => void
}) {
  const [scaleKind, setScaleKind] = useState<'symlog' | 'linear'>('symlog')
  const [hover, setHover] = useState<CoadaptPoint | null>(null)

  const maxY = Math.max(...points.map((p) => p.extracted))
  const x = scaleLinear({ domain: [0, points.length - 1], range: [0, IW] })
  const y =
    scaleKind === 'symlog'
      ? scaleSymlog({ domain: [0, maxY], range: [IH, 0], constant: 100 })
      : scaleLinear({ domain: [0, maxY], range: [IH, 0] })

  const readout = hover ?? (selected != null ? points[selected] : null)

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div
          role="group"
          aria-label="Vertical scale"
          className="flex items-center gap-1 rounded-panel border border-rule p-0.5"
        >
          {(['symlog', 'linear'] as const).map((k) => (
            <button
              key={k}
              type="button"
              onClick={() => setScaleKind(k)}
              aria-pressed={scaleKind === k}
              className={cn(
                'rounded-[2px] px-2.5 py-1 text-[0.625rem] uppercase tracking-[0.09em] transition-colors duration-150',
                scaleKind === k ? 'bg-value/15 text-value' : 'text-ink-3 hover:text-ink-2',
              )}
            >
              {k}
            </button>
          ))}
        </div>

        <div className="num min-h-[1.25rem] text-[0.6875rem] text-ink-2">
          {readout ? (
            <>
              update <span className="text-ink">{readout.update}</span>
              {'  '}extracted <span className="text-value">{int(readout.extracted)}</span>
              {'  '}return <span className="text-ink">{readout.policyReturn.toFixed(2)}</span>
              {'  '}entropy <span className="text-def">{readout.entropy.toFixed(3)}</span>
              {readout.refit && <span className="ml-2 text-atk">refit</span>}
            </>
          ) : (
            <span className="text-ink-3">Hover the chart, or pick a refit marker.</span>
          )}
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="block w-full" role="img" aria-labelledby="ar-title">
        <title id="ar-title">
          Value extracted per episode over 150 co-adaptation updates, with twelve defender refits
          marked. Extraction climbs to 23,391 by update 11, collapses to exactly zero for sixteen
          updates after the first refit, then recovers through a different attack channel.
        </title>

        <Group left={M.left} top={M.top}>
          {scaleKind === 'symlog' &&
            TICKS.map((t) => (
              <line
                key={t}
                x1={0}
                x2={IW}
                y1={y(t)}
                y2={y(t)}
                stroke="var(--color-rule)"
                strokeWidth={1}
              />
            ))}

          {/* Refit markers. Each is focusable and selectable, because the
              strategy the attacker was running before each one is the story. */}
          {refits.map((update) => {
            const isSel = selected === update
            return (
              <g
                key={update}
                tabIndex={0}
                role="button"
                aria-label={`Defender refit at update ${update}. Show the attacker strategy sampled before it.`}
                aria-pressed={isSel}
                onClick={() => onSelect(isSel ? null : update)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    onSelect(isSel ? null : update)
                  }
                }}
                className="cursor-pointer outline-none"
              >
                {/* Wider transparent hit area than the visible rule. */}
                <rect x={x(update) - 6} y={0} width={12} height={IH} fill="transparent" />
                <line
                  x1={x(update)}
                  x2={x(update)}
                  y1={0}
                  y2={IH}
                  stroke={isSel ? 'var(--color-atk)' : 'var(--color-def)'}
                  strokeWidth={isSel ? 1.75 : 1}
                  opacity={isSel ? 1 : 0.4}
                />
                <circle
                  cx={x(update)}
                  cy={-4}
                  r={isSel ? 3.5 : 2.5}
                  fill={isSel ? 'var(--color-atk)' : 'var(--color-def)'}
                />
              </g>
            )
          })}

          <line
            x1={0}
            x2={IW}
            y1={y(0)}
            y2={y(0)}
            stroke="var(--color-def)"
            strokeWidth={1.5}
          />

          <LinePath
            data={points}
            x={(p) => x(p.update)}
            y={(p) => y(p.extracted)}
            stroke="var(--color-value)"
            strokeWidth={1.5}
            vectorEffect="non-scaling-stroke"
          />

          {readout && (
            <circle
              cx={x(readout.update)}
              cy={y(readout.extracted)}
              r={3}
              fill="var(--color-value)"
              stroke="var(--color-surface)"
              strokeWidth={1.5}
            />
          )}

          <AxisLeft
            scale={y}
            tickValues={scaleKind === 'symlog' ? TICKS : undefined}
            numTicks={5}
            stroke="var(--color-rule)"
            tickStroke="var(--color-rule)"
            tickFormat={(v) => int(Number(v))}
            tickLabelProps={() => ({
              fill: 'var(--color-ink-3)',
              fontSize: 9,
              fontFamily: 'var(--font-mono)',
              textAnchor: 'end',
              dx: -4,
              dy: 3,
            })}
          />
          <AxisBottom
            top={IH}
            scale={x}
            numTicks={10}
            stroke="var(--color-rule)"
            tickStroke="var(--color-rule)"
            tickLabelProps={() => ({
              fill: 'var(--color-ink-3)',
              fontSize: 9,
              fontFamily: 'var(--font-mono)',
              textAnchor: 'middle',
              dy: 2,
            })}
            label="update"
            labelProps={{
              fill: 'var(--color-ink-3)',
              fontSize: 9,
              fontFamily: 'var(--font-mono)',
              textAnchor: 'middle',
            }}
          />

          {/* One overlay for hover. Inverse-maps x to an index, so it is O(1)
              rather than a listener per point. */}
          <rect
            x={0}
            y={0}
            width={IW}
            height={IH}
            fill="transparent"
            onPointerMove={(e) => {
              const svg = e.currentTarget.ownerSVGElement
              if (!svg) return
              const rect = svg.getBoundingClientRect()
              const px = ((e.clientX - rect.left) / rect.width) * W - M.left
              const i = Math.round(x.invert(Math.max(0, Math.min(px, IW))))
              setHover(points[Math.max(0, Math.min(i, points.length - 1))] ?? null)
            }}
            onPointerLeave={() => setHover(null)}
          />
        </Group>
      </svg>
    </div>
  )
}
