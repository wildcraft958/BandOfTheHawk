import { SvgFlow, Legend, type SvgBox, type SvgGroup, type SvgWire } from './SvgFlow'
import { detectors } from '../../../data/run'
import { BANDS } from '../../../data/paper'
import { pct } from '../../../lib/format'

/**
 * The defender, from one event to one action.
 *
 * Schema decides which experts an event reaches, the combiner folds their scores
 * into one number, the fitted cost curve turns that number into a band, and the
 * band picks a mitigation that edits the entity graph. Below, the refit path:
 * the same event only reaches training once the label latency has elapsed, which
 * is why the defender always trains on stale labels.
 *
 * Expert weights are the run's own normalised combiner weights, so the thin
 * channel is visible rather than asserted.
 */
const MITIGATIONS = ['Freeze card', 'Unbind device', 'Detach payee', 'Blocklist device']

const GROUPS: SvgGroup[] = [
  { x: 8, y: 8, w: 1154, h: 384, title: 'one event, one action', tone: 'def' },
  { x: 60, y: 418, w: 1050, h: 156, title: 'refit path', note: 'always stale', tone: 'value' },
]

export function DefenderDiagram() {
  const experts = detectors.experts
  const b = BANDS
  const text = experts.find((e) => e.name === 'text')

  const expertBoxes: SvgBox[] = experts.map((e, i) => ({
    id: `x-${e.name}`,
    x: 226,
    y: 44 + i * 66,
    w: 138,
    eyebrow: `E${i + 1}`,
    label: e.name,
    sub: `weight ${pct(e.normalized_weight)}`,
    tone: e.name === 'text' ? 'atk' : 'pass',
  }))

  const mitigationBoxes: SvgBox[] = MITIGATIONS.map((m, i) => ({
    id: `m-${i}`,
    x: 828,
    y: 60 + i * 62,
    w: 152,
    h: 38,
    label: m,
    tone: 'pass',
  }))

  const boxes: SvgBox[] = [
    { id: 'event', x: 26, y: 112, w: 146, eyebrow: 'in', label: 'Event', sub: 'no field says who acted', tone: 'value' },
    { id: 'schema', x: 26, y: 218, w: 146, label: 'Schema lookup', sub: 'not a learned gate', tone: 'value' },
    ...expertBoxes,
    { id: 'combiner', x: 430, y: 168, w: 140, eyebrow: 'fold', label: 'Combiner', sub: 'logistic regression', tone: 'atk' },
    { id: 'bands', x: 620, y: 168, w: 158, eyebrow: 'cost curve', label: 'Risk bands', sub: `${b.stepUp} / ${b.hold} / ${b.decline} / ${b.block}, one search`, tone: 'atk' },
    ...mitigationBoxes,
    { id: 'graph', x: 1016, y: 168, w: 136, eyebrow: 'out', label: 'Graph edit', sub: 'an edge disappears', tone: 'value' },

    { id: 'latency', x: 88, y: 470, w: 146, eyebrow: 'wait', label: 'Label latency', sub: '4320 minutes', tone: 'value' },
    { id: 'retention', x: 300, y: 470, w: 156, label: 'Retention buffer', sub: 'fraud kept 3x longer', tone: 'pass' },
    { id: 'subsample', x: 520, y: 470, w: 176, label: 'Prevalence subsample', sub: 'holds the base rate', tone: 'pass' },
    { id: 'retrain', x: 760, y: 470, w: 186, eyebrow: 'refit', label: 'Experts and combiner', sub: 'trains on stale labels', tone: 'atk' },
  ]

  // 13 second clock. One event walks to a decision, then the same event takes
  // the long way round and comes back as new weights.
  const wires: SvgWire[] = [
    { d: 'M99 164 V218', tone: 'value', at: 0 },
    ...experts.map((e, i) => ({
      d: `M172 244 H198 V${70 + i * 66} H226`,
      tone: (e.name === 'text' ? 'atk' : 'pass') as SvgWire['tone'],
      at: 0.9 + i * 0.1,
    })),
    ...experts.map((e, i) => ({
      d: `M364 ${70 + i * 66} H400 V194 H430`,
      tone: (e.name === 'text' ? 'atk' : 'pass') as SvgWire['tone'],
      at: 2.1 + i * 0.1,
    })),
    { d: 'M570 194 H620', tone: 'atk', at: 3.3 },
    // One band fires per event, but which one is not knowable here, so the four
    // are shown as the options the score chooses between.
    ...MITIGATIONS.map((_, i) => ({
      d: `M778 194 H802 V${79 + i * 62} H828`,
      tone: 'pass' as const,
      at: 4.2 + i * 0.12,
    })),
    ...MITIGATIONS.map((_, i) => ({
      d: `M980 ${79 + i * 62} H998 V194 H1016`,
      tone: 'value' as const,
      at: 5.3 + i * 0.12,
    })),

    // The refit path. Slow on both the wait and the return, because those are
    // the two steps whose cost is time.
    { d: 'M99 164 V400 H88 V470', tone: 'value', label: 'every event', lx: 104, ly: 396, feedback: true, at: 6.7, travel: 1.4 },
    { d: 'M234 496 H300', tone: 'value', at: 8.3 },
    { d: 'M456 496 H520', tone: 'pass', at: 9.2 },
    { d: 'M696 496 H760', tone: 'pass', at: 10.1 },
    { d: 'M853 470 V410 H500 V220', tone: 'atk', label: 'new weights', lx: 512, ly: 404, feedback: true, at: 11.1, travel: 1.4 },
  ]

  return (
    <div>
      <SvgFlow
        id="def"
        viewBox="0 0 1170 590"
        groups={GROUPS}
        boxes={boxes}
        wires={wires}
        cycle={13}
        ariaLabel="The defender. An event is routed by schema lookup to the five experts its type admits. The combiner folds their scores into one number, the fitted cost curve turns that number into one of four bands, and the band selects a mitigation that edits the entity graph. Below, the refit path: every event reaches training only after the 4320 minute label latency, through a retention buffer that keeps fraud three times longer and a prevalence subsample that holds the base rate, before new weights return to the combiner."
      />
      <Legend>
        <span>
          <span className="text-atk">Text at {pct(text?.normalized_weight ?? 0)}</span> is the
          thinnest channel, and the one the attacker converged on
        </span>
        <span><span className="text-ink-2">Dashed</span> the refit path, which always trains on stale labels</span>
      </Legend>
    </div>
  )
}
