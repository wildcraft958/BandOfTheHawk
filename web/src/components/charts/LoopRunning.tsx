import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Pause, Play, RotateCcw, StepForward } from 'lucide-react'
import { Label } from '../ui/Label'
import { cn } from '../ui/cn'
import {
  DEFAULT_CONFIG,
  initSim,
  stepSim,
  type Frame,
  type SimConfig,
  type SimState,
} from '../../sim/model'
import { TACTICS } from '../../sim/tactics'
import { useReducedMotion } from '../../lib/useReducedMotion'
import { int, pct } from '../../lib/format'
import { Note } from '../ui/Note'

/**
 * The loop, running.
 *
 * Everything else on this page is evidence that co-adaptation happened. This is
 * the loop actually happening: both sides are learning in the browser while the
 * panel is open. The attacker reallocates across seven tactic chains by
 * exponential weights, the defender is five logistic experts and a logistic
 * combiner refitted from scratch on the labels that have cleared its latency
 * window, and neither side is following a script.
 *
 * The two controls are the two that decide whether a loop exists at all. Too
 * little review capacity and both sides settle into a low quality equilibrium
 * that never moves. Too much and the defender covers every channel at once, the
 * attacker loses its gradient, and learning stops on both sides. Neither of those
 * is a demonstration, and a judge can put the panel into either one.
 */
const UPDATES = 24
const TICK_MS = 460

const W = 760
const H = 190
const PAD = { top: 14, right: 14, bottom: 26, left: 52 }

/**
 * Seven shades of the attacker hue, because every tactic is the attacker and
 * borrowing the defender's blue or the money gold for one would break the
 * page's colour language. Seven steps of one hue are hard to tell apart, so the
 * bar carries a hairline between segments and names whichever one it is mostly
 * made of.
 */
const TACTIC_FILL = TACTICS.map((_, i) => 1 - i * 0.115)

/**
 * The refit lands at the end of the update it is recorded on, so the marker sits
 * just past that block. Clamped, because the last refit of a run falls on the
 * final update and would otherwise be drawn off the plot.
 */
const markerX = (x: number) => Math.min(x + 6, W - PAD.right)

const CADENCES = [4, 6, 8, 12] as const
const CAPACITIES = [
  { rate: 0.0015, label: 'tight' },
  { rate: 0.0025, label: 'as run' },
  { rate: 0.006, label: 'loose' },
  { rate: 0.015, label: 'wide' },
] as const

function useSim(config: SimConfig) {
  const [state, setState] = useState<SimState>(() => initSim())
  const [playing, setPlaying] = useState(false)
  const reduced = useReducedMotion()

  // The config object is rebuilt every render, so the ticker reads it through a
  // ref. Putting it in the effect's dependencies restarts the interval on every
  // render and the sim never advances.
  const cfg = useRef(config)
  cfg.current = config

  const advance = useCallback(() => {
    setState((s) => (s.t >= UPDATES ? s : stepSim(s, cfg.current)))
  }, [])

  const reset = useCallback(() => {
    setPlaying(false)
    setState(initSim())
  }, [])

  useEffect(() => {
    if (!playing) return undefined
    // Reduced motion gates the ticker itself, not just the CSS. Step stays
    // available, so the panel is still fully usable.
    if (reduced) {
      setPlaying(false)
      return undefined
    }
    const timer = window.setInterval(() => {
      setState((s) => {
        if (s.t >= UPDATES) {
          setPlaying(false)
          return s
        }
        return stepSim(s, cfg.current)
      })
    }, TICK_MS)
    return () => window.clearInterval(timer)
  }, [playing, reduced])

  return { state, playing, setPlaying, advance, reset, reduced }
}

function Bars({
  rows,
  tone,
}: {
  rows: Array<{ key: string; label: string; value: number; hint?: string }>
  tone: 'def' | 'atk'
}) {
  return (
    <div className="space-y-1.5">
      {rows.map((r) => (
        <div key={r.key} className="flex items-center gap-2">
          <span className="w-[7.75rem] shrink-0 text-[0.8125rem] text-ink-2">{r.label}</span>
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-active">
            <div
              className={cn(
                'h-full rounded-full transition-[width] duration-300 ease-out',
                tone === 'def' ? 'bg-def' : 'bg-atk',
              )}
              style={{ width: `${Math.max(0, Math.min(1, r.value)) * 100}%` }}
            />
          </div>
          <span className="num w-10 shrink-0 text-right text-[0.8125rem] text-ink-3">
            {r.hint ?? r.value.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  )
}

export function LoopRunning() {
  const [cadence, setCadence] = useState<number>(6)
  const [capacity, setCapacity] = useState<number>(0.0025)

  const config = useMemo<SimConfig>(
    () => ({ ...DEFAULT_CONFIG, refitEvery: cadence, alertRate: capacity }),
    [cadence, capacity],
  )

  const { state, playing, setPlaying, advance, reset, reduced } = useSim(config)
  const frames = state.frames
  const last: Frame | undefined = frames[frames.length - 1]

  // Changing a control restarts the run, because half a run under one setting
  // and half under another is not a reading of either.
  useEffect(() => {
    reset()
  }, [cadence, capacity, reset])

  const geometry = useMemo(() => {
    const maxY = Math.max(4000, ...frames.map((f) => f.extracted)) * 1.12
    const x = (t: number) => PAD.left + (t / Math.max(UPDATES - 1, 1)) * (W - PAD.left - PAD.right)
    const y = (v: number) => H - PAD.bottom - (v / maxY) * (H - PAD.top - PAD.bottom)
    return {
      maxY,
      x,
      y,
      path: frames.map((f, i) => `${i ? 'L' : 'M'}${x(f.t)} ${y(f.extracted)}`).join(' '),
      refits: frames.filter((f) => f.refit).map((f) => f.t),
    }
  }, [frames])

  const started = frames.length > 0
  const done = state.t >= UPDATES
  const dominant = useMemo(() => {
    let best = -1
    let at = 0
    state.weights.forEach((w, i) => {
      if (w > best) {
        best = w
        at = i
      }
    })
    return started ? { label: TACTICS[at].label, share: best } : null
  }, [state.weights, started])

  return (
    <div>
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex flex-wrap items-end gap-4">
          <div>
            <Label>refit every</Label>
            <div className="mt-1.5 flex overflow-hidden rounded-panel border border-rule">
              {CADENCES.map((c) => (
                <button
                  key={c}
                  type="button"
                  onClick={() => setCadence(c)}
                  aria-pressed={cadence === c}
                  className={cn(
                    'num px-2.5 py-1 text-[0.875rem] transition-colors duration-150',
                    cadence === c ? 'bg-def/15 text-def' : 'text-ink-3 hover:text-ink-2',
                  )}
                >
                  {c}
                </button>
              ))}
            </div>
          </div>
          <div>
            <Label>review capacity</Label>
            <div className="mt-1.5 flex overflow-hidden rounded-panel border border-rule">
              {CAPACITIES.map((c) => (
                <button
                  key={c.rate}
                  type="button"
                  onClick={() => setCapacity(c.rate)}
                  aria-pressed={capacity === c.rate}
                  className={cn(
                    'px-2.5 py-1 text-[0.8125rem] uppercase tracking-[0.08em] transition-colors duration-150',
                    capacity === c.rate ? 'bg-def/15 text-def' : 'text-ink-3 hover:text-ink-2',
                  )}
                >
                  {c.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => {
              if (done) reset()
              setPlaying((p) => !p)
            }}
            aria-label={playing ? 'Pause the run' : 'Run the loop'}
            className="inline-flex items-center gap-2 rounded-panel border border-atk/55 bg-atk/10 px-3 py-1.5 text-[0.8125rem] uppercase tracking-[0.09em] text-atk transition-colors duration-150 hover:bg-atk/15"
          >
            {playing ? (
              <>
                <Pause className="size-3" aria-hidden="true" /> pause
              </>
            ) : (
              <>
                <Play className="size-3" aria-hidden="true" /> {done ? 'again' : 'run'}
              </>
            )}
          </button>
          <button
            type="button"
            onClick={() => {
              setPlaying(false)
              advance()
            }}
            aria-label="Advance one update"
            className="grid size-8 place-items-center rounded-panel border border-rule text-ink-2 transition-colors duration-150 hover:border-ink-3 hover:text-ink"
          >
            <StepForward className="size-3" aria-hidden="true" />
          </button>
          <button
            type="button"
            onClick={reset}
            aria-label="Back to an untrained defender"
            className="grid size-8 place-items-center rounded-panel border border-rule text-ink-2 transition-colors duration-150 hover:border-ink-3 hover:text-ink"
          >
            <RotateCcw className="size-3" aria-hidden="true" />
          </button>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2">
        <span className="num text-[0.9375rem] text-ink">
          update {state.t} of {UPDATES}
        </span>
        {last ? (
          <>
            <span className="num text-[0.9375rem] text-value">
              {int(last.extracted)} extracted
            </span>
            <span className="num text-[0.875rem] text-ink-3">
              {last.reviewed} reviewed of {int(last.fraudEvents)} fraud events
            </span>
            <span className="num text-[0.875rem] text-def">
              {pct(last.precisionAtBudget, 0)} of reviews were fraud
            </span>
            <span className="num text-[0.875rem] text-ink-3">
              {pct(last.baseRate, 2)} of traffic is fraud
            </span>
            <span className="text-[0.8125rem] uppercase tracking-[0.08em] text-atk">
              {last.topTactic}
            </span>
          </>
        ) : (
          <span className="text-[0.875rem] text-ink-3">
            the defender starts untrained. Press run.
          </span>
        )}
      </div>

      <div className="mt-3 overflow-x-auto">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full min-w-[30rem]"
          role="img"
          aria-label={
            started
              ? `Value extracted per update in the live loop, at update ${state.t} of ${UPDATES}, currently ${int(last?.extracted ?? 0)} model units.`
              : 'The live loop has not been started yet.'
          }
        >
          {[0, 0.5, 1].map((f) => (
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

          {geometry.refits.map((t) => (
            <g key={t}>
              <line
                x1={markerX(geometry.x(t))}
                y1={PAD.top}
                x2={markerX(geometry.x(t))}
                y2={H - PAD.bottom}
                stroke="var(--color-def)"
                strokeOpacity="0.7"
                strokeWidth="1.4"
                strokeDasharray="4 4"
              />
              <text
                x={markerX(geometry.x(t))}
                y={PAD.top - 4}
                textAnchor="middle"
                fill="var(--color-def)"
                fontSize="7.5"
                fontFamily="var(--font-mono)"
              >
                refit
              </text>
            </g>
          ))}

          <path
            d={geometry.path}
            fill="none"
            stroke="var(--color-atk)"
            strokeWidth="2"
            strokeLinecap="round"
          />
          {last ? (
            <circle
              cx={geometry.x(last.t)}
              cy={geometry.y(last.extracted)}
              r="3.4"
              fill="var(--color-surface)"
              stroke="var(--color-atk)"
              strokeWidth="1.8"
            />
          ) : null}

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
            {UPDATES}
          </text>
        </svg>
      </div>

      <div className="mt-4 grid gap-5 lg:grid-cols-2">
        <div>
          <Label>where the attacker is spending</Label>
          <div className="mt-2 flex h-5 overflow-hidden rounded-panel border border-rule">
            {TACTICS.map((t, i) => (
              <div
                key={t.id}
                title={`${t.label} ${pct(state.weights[i] ?? 0, 0)}`}
                className={cn(
                  'h-full transition-[width] duration-300 ease-out',
                  i > 0 && 'border-l border-surface',
                )}
                style={{
                  width: `${(state.weights[i] ?? 0) * 100}%`,
                  backgroundColor: 'var(--color-atk)',
                  opacity: TACTIC_FILL[i],
                }}
              />
            ))}
          </div>
          <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1">
            {TACTICS.map((t, i) => {
              const leading = dominant?.label === t.label
              return (
                <span
                  key={t.id}
                  className={cn(
                    'flex items-center gap-1.5 text-[0.8125rem]',
                    leading ? 'text-atk' : 'text-ink-3',
                  )}
                >
                  <span
                    className="size-2 shrink-0 rounded-[1px]"
                    style={{ backgroundColor: 'var(--color-atk)', opacity: TACTIC_FILL[i] }}
                    aria-hidden="true"
                  />
                  {t.label}
                  <span className={cn('num', leading ? 'font-semibold text-atk' : 'text-ink-2')}>
                    {pct(state.weights[i] ?? 0, 0)}
                  </span>
                </span>
              )
            })}
          </div>
          <p className="prose-sans mt-2.5 text-[0.875rem] leading-relaxed text-ink-2">
            Seven chains over the real twenty action space, each carrying its real summed action
            cost. The attacker reallocates toward whatever is still paying, so this bar moving is
            the attacker adapting.
          </p>
        </div>

        <div>
          <Label>what the defender is stopping</Label>
          <div className="mt-2">
            <Bars
              tone="def"
              rows={TACTICS.map((t, i) => ({
                key: t.id,
                label: t.label,
                value: last?.blockedShare[i] ?? 0,
                hint: started ? pct(last?.blockedShare[i] ?? 0, 0) : 'n/a',
              }))}
            />
          </div>
          <p className="prose-sans mt-2.5 text-[0.875rem] leading-relaxed text-ink-2">
            Share of each tactic&rsquo;s takings the defender stopped this update. Read it against
            the bar on the left: the attacker&rsquo;s weight sits wherever these are lowest, and a
            channel that reaches the top of this list is one the attacker abandons within a few
            updates. That inverse relationship is the loop, and it is not scripted anywhere.
          </p>
        </div>
      </div>

      <Note
        label="what is real here, and what is tuned"
        lede={
          <>
            Logistic experts and exponential weights running in this browser, not the trained PPO
            policy, and not a rerun of the experiment the rest of this page reports.
          </>
        }
      >
        <p className="prose-sans max-w-3xl text-[0.875rem] leading-relaxed text-ink-2">
          Five logistic experts and a logistic combiner, fitted here by gradient descent on the
          events the attacker is generating, against exponential weights over the tactic chains. The
          same family as the deployed combiner, which is a logistic regression too, at roughly a
          thousandth of the scale. The run&rsquo;s own attacker is PPO with a 512 unit hidden layer
          over eighty episodes per update. Action costs, refit cadence, label latency, asymmetric
          label retention and the review budget framing come from the system. Event yields and the
          feature means are tuned. Values are model units, never currency.
        </p>
        <p className="prose-sans mt-2 max-w-3xl text-[0.875rem] leading-relaxed text-ink-2">
          One number here is deliberately not the document&rsquo;s: this runs at roughly the 0.5%
          fraud prevalence a card portfolio actually carries, where the reported co-adaptation was
          measured at 2% because reaching 0.5% would have needed around six hundred thousand more
          benign rows per refit window. A seven feature model in a browser can afford that; the full
          pipeline could not, at the run counts the comparison needed. So this is the deployment
          regime, not a rerun of the experiment, and the two sets of figures are not comparable.
          {reduced ? ' Motion is reduced, so the ticker is off. Step still works.' : ''}
        </p>
      </Note>
    </div>
  )
}
