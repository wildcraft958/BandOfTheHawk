import { useMemo } from 'react'
import { Label } from '../ui/Label'
import { fidelity } from '../../data/run'
import { fixed, int, pct } from '../../lib/format'
import { Note } from '../ui/Note'

/**
 * The fitted arrival-hour distribution.
 *
 * Timing is not uniform and it is not a single peak, so the pipeline fits a
 * two-component von Mises mixture on the circle rather than a histogram on a
 * line: midnight and 23:59 are neighbours, and a linear fit cannot know that.
 *
 * Every curve below is the density evaluated from the fitted parameters, and the
 * modes are found by scanning it at one-minute resolution rather than being read
 * off the component means. That distinction matters here, because the mixture's
 * taller mode is not the mean of its heavier component.
 */

const CX = 100
const CY = 100
const R_ZERO = 34
const R_MAX = 82
const DRAW_STEPS = 288
const SCAN_STEPS = 1440

/**
 * Modified Bessel function of the first kind, order zero. Each component's
 * normalising constant depends on its own concentration, so an unnormalised
 * mixture would render the wrong relative peak heights. The series converges in
 * a few terms at the fitted concentrations, both under 2.3.
 */
function besselI0(x: number): number {
  let sum = 1
  let term = 1
  for (let k = 1; k <= 32; k += 1) {
    term *= (x * x) / (4 * k * k)
    sum += term
    if (term < 1e-15) break
  }
  return sum
}

/** Hour 0 at the top, running clockwise, the way a clock face reads. */
function hoursToAngle(hour: number): number {
  return (hour / 24) * 2 * Math.PI - Math.PI / 2
}

function point(hour: number, radius: number): [number, number] {
  const a = hoursToAngle(hour)
  return [CX + radius * Math.cos(a), CY + radius * Math.sin(a)]
}

function clockTime(hour: number): string {
  let h = Math.floor(hour)
  let m = Math.round((hour - h) * 60)
  if (m === 60) {
    m = 0
    h = (h + 1) % 24
  }
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`
}

const TONES = ['var(--color-value)', 'var(--color-def)']

export function CircadianDial() {
  const c = fidelity.circadian
  const weights = (c?.weights as number[]) ?? []
  const means = (c?.means as number[]) ?? []
  const kappas = (c?.concentrations as number[]) ?? []
  const nSamples = (c?.n_samples as number) ?? null
  const resultant = (c?.resultant_length as number) ?? null

  const fit = useMemo(() => {
    if (weights.length === 0) return null

    const component = (i: number, hour: number) =>
      (weights[i] * Math.exp(kappas[i] * Math.cos(hoursToAngle(hour) - hoursToAngle(means[i])))) /
      (2 * Math.PI * besselI0(kappas[i]))
    const mixture = (hour: number) =>
      weights.reduce((acc, _, i) => acc + component(i, hour), 0)

    // Scan at one-minute resolution so the reported times do not depend on the
    // drawing resolution.
    const scan = Array.from({ length: SCAN_STEPS }, (_, i) => (i / SCAN_STEPS) * 24)
    const scanned = scan.map(mixture)
    const lo = Math.min(...scanned)
    const hi = Math.max(...scanned)

    const modes: number[] = []
    let trough = scan[0]
    let saddle: number | null = null
    for (let i = 0; i < SCAN_STEPS; i += 1) {
      const prev = scanned[(i - 1 + SCAN_STEPS) % SCAN_STEPS]
      const here = scanned[i]
      const next = scanned[(i + 1) % SCAN_STEPS]
      if (here > prev && here >= next) modes.push(scan[i])
      if (here < prev && here <= next) {
        if (here === lo) trough = scan[i]
        else saddle = scan[i]
      }
    }
    modes.sort((a, b) => mixture(b) - mixture(a))

    const span = hi - lo || 1
    const radius = (v: number) => R_ZERO + ((v - lo) / span) * (R_MAX - R_ZERO)
    const ring = (value: (hour: number) => number) =>
      Array.from({ length: DRAW_STEPS }, (_, i) => {
        const h = (i / DRAW_STEPS) * 24
        const [x, y] = point(h, radius(value(h)))
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`
      }).join(' ') + ' Z'

    return {
      mixturePath: ring(mixture),
      componentPaths: weights.map((_, i) => ring((h) => component(i, h))),
      modes,
      trough,
      saddle,
      radius,
      mixture,
    }
  }, [weights, means, kappas])

  if (!fit) {
    return <p className="prose-sans text-[0.9375rem] text-ink-3">No circadian fit in this run.</p>
  }

  const heavier = weights.indexOf(Math.max(...weights))
  const tallestMode = fit.modes[0]

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,16rem)_1fr] lg:items-start">
      <svg
        viewBox="-10 -10 220 220"
        className="w-full max-w-[16rem]"
        role="img"
        aria-label={`Fitted arrival-hour density on a 24 hour dial. Modes at ${fit.modes
          .map(clockTime)
          .join(' and ')}, trough at ${clockTime(fit.trough)}.`}
      >
        <circle cx={CX} cy={CY} r={R_MAX} fill="none" stroke="var(--color-rule)" strokeWidth="1" />
        <circle
          cx={CX}
          cy={CY}
          r={R_ZERO}
          fill="none"
          stroke="var(--color-rule)"
          strokeWidth="1"
          strokeDasharray="2 3"
        />

        {Array.from({ length: 24 }, (_, h) => h).map((h) => {
          const major = h % 6 === 0
          const [x1, y1] = point(h, R_MAX)
          const [x2, y2] = point(h, R_MAX + (major ? 5 : 2.5))
          return (
            <line
              key={h}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke={major ? 'var(--color-ink-3)' : 'var(--color-rule)'}
              strokeWidth="1"
            />
          )
        })}

        {[0, 6, 12, 18].map((h) => {
          const [lx, ly] = point(h, R_MAX + 14)
          return (
            <text
              key={h}
              x={lx}
              y={ly}
              textAnchor="middle"
              dominantBaseline="middle"
              fill="var(--color-ink-3)"
              fontSize="8"
              fontFamily="var(--font-mono)"
            >
              {String(h).padStart(2, '0')}
            </text>
          )
        })}

        {/* The mixture, filled. Then each component on top as a thin ring, so
            "two-component" is visible rather than asserted. */}
        <path
          d={fit.mixturePath}
          fill="var(--color-value)"
          fillOpacity="0.13"
          stroke="var(--color-value)"
          strokeWidth="1.6"
        />
        {fit.componentPaths.map((d, i) => (
          <path
            key={i}
            d={d}
            fill="none"
            stroke={TONES[i % TONES.length]}
            strokeWidth="0.9"
            strokeOpacity="0.75"
            strokeDasharray="2.5 2"
          />
        ))}

        {/* The two modes, read off the one-minute scan rather than the means. */}
        {fit.modes.map((h) => {
          const [x, y] = point(h, fit.radius(fit.mixture(h)))
          return (
            <g key={h}>
              <circle cx={x} cy={y} r="2.6" fill="var(--color-surface)" />
              <circle
                cx={x}
                cy={y}
                r="2.6"
                fill="none"
                stroke="var(--color-value)"
                strokeWidth="1.4"
              />
            </g>
          )
        })}
      </svg>

      <div>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-3">
          {means.map((mean, i) => (
            <div key={mean}>
              <Label>component {i + 1}</Label>
              <dd className="num mt-0.5 text-[1rem]" style={{ color: TONES[i] }}>
                {clockTime(mean)}
              </dd>
              <dd className="num mt-0.5 text-[0.8125rem] text-ink-3">
                weight {pct(weights[i])}, concentration {fixed(kappas[i], 4)}
              </dd>
            </div>
          ))}
          <div>
            <Label>modes</Label>
            <dd className="num mt-0.5 text-[1rem] text-ink">
              {fit.modes.map(clockTime).join(', ')}
            </dd>
          </div>
          <div>
            <Label>trough</Label>
            <dd className="num mt-0.5 text-[1rem] text-ink">{clockTime(fit.trough)}</dd>
          </div>
          <div>
            <Label>resultant length</Label>
            <dd className="num mt-0.5 text-[1rem] text-ink">{fixed(resultant, 4)}</dd>
          </div>
          <div>
            <Label>samples</Label>
            <dd className="num mt-0.5 text-[1rem] text-ink">{int(nSamples)}</dd>
          </div>
        </dl>

        <Note
          label="why the taller mode is not the heavier one"
        >
          <p className="prose-sans text-[0.875rem] leading-relaxed text-ink-2">
          Genuinely bimodal, and not in the obvious way. The heavier component carries{' '}
          {pct(weights[heavier])} of arrivals at {clockTime(means[heavier])}, yet the taller mode of
          the mixture sits at {clockTime(tallestMode)}, because the lighter component is the more
          concentrated one ({fixed(Math.max(...kappas), 4)} against{' '}
          {fixed(Math.min(...kappas), 4)}). Weight and height are not the same thing on a circle.
          {fit.saddle != null && ` The two modes are separated by a saddle at ${clockTime(fit.saddle)}.`}
        </p>
          <p className="prose-sans mt-2 text-[0.875rem] leading-relaxed text-ink-2">
          Resultant length {fixed(resultant, 4)} places the population between the extremes, where 0
          would be arrivals spread evenly around the clock and 1 would be every arrival at a single
          instant. The quiet hour is {clockTime(fit.trough)}.
        </p>
          <p className="prose-sans mt-2 text-[0.875rem] leading-relaxed text-ink-3">
          The hour noise floor is {fidelity.all_floors.hour_jsd.toExponential(2)} JSD, so two halves
          of the real data agree on arrival hour to within that. Unlike the four metrics on the
          calibration panel, this run recorded no observed-against-target pair for arrival hour, so
          the fit is shown without a verdict.
        </p>
        </Note>
      </div>
    </div>
  )
}
