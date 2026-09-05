import { useEffect, useMemo, useState } from 'react'
import { Pause, Play, RotateCcw, StepForward } from 'lucide-react'
import { Badge } from '../ui/Badge'
import { Label } from '../ui/Label'
import { cn } from '../ui/cn'
import { ablation, refitUpdates, runFor, type AblationRun } from '../../data/ablation'
import { useReducedMotion } from '../../lib/useReducedMotion'
import { int, pct } from '../../lib/format'
import { Note } from '../ui/Note'

/**
 * The loop, closing, one update at a time.
 *
 * Every other view of the co-adaptation shows what the loop produced. This one
 * shows it happening: extraction climbs, a defender refit fires, extraction is
 * knocked back, and the attacker works its way up again.
 *
 * The series is derived from a run file tracked in the repository, so the shape
 * on screen can be checked against artifacts/ablation/. It is not the run the
 * solution document reports, and the panel says so rather than letting a reader
 * assume the headline figures came from here.
 */
const W = 760
const H = 260
const PAD = { top: 16, right: 14, bottom: 30, left: 54 }
const TICK_MS = 420

type Phase = 'climbing' | 'refit' | 'after'

function phaseAt(refits: number[], update: number): Phase {
  if (refits.includes(update)) return 'refit'
  if (refits.includes(update - 1)) return 'after'
  return 'climbing'
}

/**
 * What the block on screen is, said without promising an outcome.
 *
 * A refit does not always knock extraction down on the next block, and the
 * paper reports the same thing: the pattern holds in five of eight runs. So the
 * step after a refit states the two numbers and lets them speak.
 */
function phaseCopy(
  phase: Phase,
  run: AblationRun,
  update: number,
): { label: string; tone: 'atk' | 'def' | 'value'; what: string } {
  if (phase === 'refit') {
    return {
      label: 'defender retrains',
      tone: 'def',
      what: 'this is the last block the current defender sees. At the end of it the defender retrains on the fraud labels that have cleared the latency window, so the opponent changes underneath the attacker',
    }
  }
  if (phase === 'after') {
    const before = run.extraction[update - 2]
    const now = run.extraction[update - 1]
    const direction = now < before ? 'the refit cost the attacker' : 'the refit did not hold it back'
    return {
      label: 'first block against the new defender',
      tone: 'value',
      what: `extraction went from ${int(before)} to ${int(now)}, so ${direction}`,
    }
  }
  return {
    label: 'attacker adapting',
    tone: 'atk',
    what: 'no refit at either end of this block, so the attacker is playing the same defender it played last time',
  }
}

export function LoopStepper() {
  const reduced = useReducedMotion()
  const [seed, setSeed] = useState(0)
  const [arm, setArm] = useState<'full' | 'ablated'>('full')
  const [update, setUpdate] = useState(1)
  const [playing, setPlaying] = useState(false)

  const run = runFor(arm, seed)
  const total = ablation.updates

  useEffect(() => {
    if (!playing || reduced) return undefined
    const timer = window.setInterval(() => {
      setUpdate((u) => {
        if (u >= total) {
          setPlaying(false)
          return u
        }
        return u + 1
      })
    }, TICK_MS)
    return () => window.clearInterval(timer)
  }, [playing, reduced, total])

  // Changing the run restarts the walk, so the chart never shows a playhead
  // beyond a series it does not belong to.
  useEffect(() => {
    setUpdate(1)
    setPlaying(false)
  }, [arm, seed])

  const geometry = useMemo(() => {
    if (!run) return null
    const maxY = Math.max(...ablation.runs.flatMap((r) => r.extraction)) * 1.06
    const x = (u: number) => PAD.left + ((u - 1) / (total - 1)) * (W - PAD.left - PAD.right)
    const y = (v: number) => H - PAD.bottom - (v / maxY) * (H - PAD.top - PAD.bottom)
    const drawn = run.extraction.slice(0, update)
    return {
      maxY,
      x,
      y,
      path: drawn.map((v, i) => `${i ? 'L' : 'M'}${x(i + 1)} ${y(v)}`).join(' '),
      ghost: run.extraction.map((v, i) => `${i ? 'L' : 'B'}${x(i + 1)} ${y(v)}`).join(' ').replace('B', 'M'),
    }
  }, [run, update, total])

  if (!run || !geometry) return null

  const value = run.extraction[update - 1]
  const opening = run.extraction[0]
  const refits = refitUpdates(run)
  const phase = phaseAt(refits, update)
  const copy = phaseCopy(phase, run, update)
  const passedRefits = refits.filter((r) => r <= update)

  return (
    <div>
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex flex-wrap items-end gap-4">
          <div>
            <Label>seed</Label>
            <div className="mt-1.5 flex overflow-hidden rounded-panel border border-rule">
              {[0, 1, 2, 3].map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => setSeed(s)}
                  aria-pressed={seed === s}
                  className={cn(
                    'num px-2.5 py-1 text-[0.875rem] transition-colors duration-150',
                    seed === s ? 'bg-atk/15 text-atk' : 'text-ink-3 hover:text-ink-2',
                  )}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
          <div>
            <Label>attacker</Label>
            <div className="mt-1.5 flex overflow-hidden rounded-panel border border-rule">
              {(
                [
                  ['full', 'full'],
                  ['ablated', 'stealth ablated'],
                ] as const
              ).map(([k, label]) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => setArm(k)}
                  aria-pressed={arm === k}
                  className={cn(
                    'px-2.5 py-1 text-[0.8125rem] uppercase tracking-[0.08em] transition-colors duration-150',
                    arm === k ? 'bg-atk/15 text-atk' : 'text-ink-3 hover:text-ink-2',
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => {
              if (update >= total) setUpdate(1)
              setPlaying((p) => !p)
            }}
            aria-label={playing ? 'Pause the walk' : 'Play the walk'}
            className="inline-flex items-center gap-2 rounded-panel border border-atk/55 bg-atk/10 px-3 py-1.5 text-[0.8125rem] uppercase tracking-[0.09em] text-atk transition-colors duration-150 hover:bg-atk/15"
          >
            {playing ? (
              <>
                <Pause className="size-3" aria-hidden="true" /> pause
              </>
            ) : (
              <>
                <Play className="size-3" aria-hidden="true" /> {update >= total ? 'replay' : 'play'}
              </>
            )}
          </button>
          <button
            type="button"
            onClick={() => {
              setPlaying(false)
              setUpdate((u) => Math.min(u + 1, total))
            }}
            aria-label="Step one update"
            className="grid size-8 place-items-center rounded-panel border border-rule text-ink-2 transition-colors duration-150 hover:border-ink-3 hover:text-ink"
          >
            <StepForward className="size-3" aria-hidden="true" />
          </button>
          <button
            type="button"
            onClick={() => {
              setPlaying(false)
              setUpdate(1)
            }}
            aria-label="Back to the first update"
            className="grid size-8 place-items-center rounded-panel border border-rule text-ink-2 transition-colors duration-150 hover:border-ink-3 hover:text-ink"
          >
            <RotateCcw className="size-3" aria-hidden="true" />
          </button>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2">
        <Badge tone={copy.tone}>{copy.label}</Badge>
        <span className="num text-[0.9375rem] text-ink">
          update {update} of {total}
        </span>
        <span className="num text-[0.9375rem] text-value">{int(value)} extracted</span>
        <span className="num text-[0.875rem] text-ink-3">
          {(value / opening).toFixed(2)}x its opening block
        </span>
        <span className="num text-[0.875rem] text-ink-3">
          {passedRefits.length} of {refits.length} refits fired
        </span>
      </div>
      <p className="prose-sans mt-2 max-w-3xl text-[0.875rem] leading-relaxed text-ink-2">
        {copy.what}.
      </p>

      <div className="mt-3 overflow-x-auto">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full min-w-[30rem]"
          role="img"
          aria-label={`Value extracted per episode across ${total} updates for the ${arm} attacker on seed ${seed}, currently at update ${update} with ${int(value)} extracted. The defender refits at the end of updates ${refits.join(', ')}.`}
        >
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <g key={f}>
              <line
                x1={PAD.left}
                y1={geometry.y(geometry.maxY * f)}
                x2={W - PAD.right}
                y2={geometry.y(geometry.maxY * f)}
                stroke="var(--color-rule)"
                strokeWidth="1"
              />
              <text
                x={PAD.left - 8}
                y={geometry.y(geometry.maxY * f) + 3}
                textAnchor="end"
                fill="var(--color-ink-3)"
                fontSize="8.5"
                fontFamily="var(--font-mono)"
              >
                {Math.round((geometry.maxY * f) / 1000)}k
              </text>
            </g>
          ))}

          {refits.map((r) => {
            const fired = r <= update
            // The refit lands at the end of update r, so the marker sits between
            // that block and the next one rather than on top of the block that
            // was measured before it happened.
            const rx = r < total ? (geometry.x(r) + geometry.x(r + 1)) / 2 : geometry.x(r)
            return (
              <g key={r}>
                <line
                  x1={rx}
                  y1={PAD.top}
                  x2={rx}
                  y2={H - PAD.bottom}
                  stroke="var(--color-def)"
                  strokeOpacity={fired ? 0.75 : 0.22}
                  strokeWidth={fired ? 1.4 : 1}
                  strokeDasharray="4 4"
                />
                <text
                  x={rx}
                  y={PAD.top - 5}
                  textAnchor="middle"
                  fill="var(--color-def)"
                  fillOpacity={fired ? 1 : 0.35}
                  fontSize="7.5"
                  fontFamily="var(--font-mono)"
                >
                  refit
                </text>
              </g>
            )
          })}

          {/* The whole series faintly, so the walk reads as progress through a
              known shape rather than a line appearing from nowhere. */}
          <path
            d={geometry.ghost}
            fill="none"
            stroke="var(--color-ink-3)"
            strokeOpacity="0.28"
            strokeWidth="1"
          />
          <path
            d={geometry.path}
            fill="none"
            stroke="var(--color-atk)"
            strokeWidth="2"
            strokeLinecap="round"
          />
          <circle
            cx={geometry.x(update)}
            cy={geometry.y(value)}
            r="3.4"
            fill="var(--color-surface)"
            stroke="var(--color-atk)"
            strokeWidth="1.8"
          />

          <text
            x={PAD.left}
            y={H - 8}
            fill="var(--color-ink-3)"
            fontSize="8.5"
            fontFamily="var(--font-mono)"
          >
            update 1
          </text>
          <text
            x={W - PAD.right}
            y={H - 8}
            textAnchor="end"
            fill="var(--color-ink-3)"
            fontSize="8.5"
            fontFamily="var(--font-mono)"
          >
            {total}
          </text>
        </svg>
      </div>

      {run.friction.length > 0 && (
        <p className="prose-sans mt-2 text-[0.875rem] leading-relaxed text-ink-3">
          Genuine authorisations refused at each refit in this run:{' '}
          {run.friction.map((f) => pct(f, 2)).join(', ')}. The defender&rsquo;s gains cost
          something, and this is what.
        </p>
      )}

      <Note
        label="where this series comes from"
        lede={
          <>
            This is a different run from the one the solution document reports, and its figures are
            never mixed with the document&rsquo;s.
          </>
        }
      >
        <p className="prose-sans max-w-3xl text-[0.875rem] leading-relaxed text-ink-2">
          Derived, not transcribed: the series is read from{' '}
          <span className="num text-ink">{run.file}</span>, which is tracked in the repository, so
          the shape above can be checked against the file rather than taken on trust. The direction
          agrees with the document and the magnitude does not, so every headline figure on this page
          stays the document&rsquo;s.
        </p>
      </Note>
    </div>
  )
}
