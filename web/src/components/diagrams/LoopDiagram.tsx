import { SvgFlow, Legend, type SvgBox, type SvgWire } from './svg/SvgFlow'
import { coadapt, detectors, runReport } from '../../data/run'
import { int, pct } from '../../lib/format'
import { PREVALENCE_CAVEAT } from '../../data/paper'

/**
 * The closed loop.
 *
 * Calibrated benign data warm-starts the world. Attack episodes and ordinary
 * traffic land in one event stream, and the defender in force scores it. Three
 * paths run backwards, and those are what make this a loop rather than a
 * pipeline: mitigation edits the world, label latency delays what reaches
 * training, and every refit forces the attacker to adapt.
 *
 * Numbers on the boxes come from the run, so the figure is not merely
 * conceptual.
 */
export function LoopDiagram() {
  const ws = runReport.warm_start
  const flat = detectors.configs.find((c) => c.id === 'gbdt_full')

  const boxes: SvgBox[] = [
    { id: 'benign', x: 40, y: 24, w: 176, eyebrow: 'calibrated', label: 'Benign data', sub: `${int(ws.events)} events over ${int(ws.history_days)}d`, tone: 'value' },
    { id: 'world', x: 362, y: 24, w: 176, eyebrow: 'world', label: 'Simulation world', sub: `${int(ws.entities)} entities`, tone: 'value' },
    { id: 'episode', x: 40, y: 196, w: 176, eyebrow: 'red team', label: 'Attack episode', sub: `${int(runReport.episodes)} run, ${int(runReport.reached_monetized)} monetised`, tone: 'atk' },
    { id: 'stream', x: 362, y: 196, w: 176, eyebrow: 'shared builder', label: 'Event stream', sub: `${pct(PREVALENCE_CAVEAT.measuredAt, 0)} fraud, blind to who acted`, tone: 'holdout' },
    { id: 'force', x: 684, y: 196, w: 176, eyebrow: 'blue team', label: 'Defender in force', sub: `PR-AUC ${flat?.metrics.pr_auc.toFixed(4)}`, tone: 'def' },
    { id: 'buffer', x: 684, y: 330, w: 176, eyebrow: 'process', label: 'Retention buffer', sub: 'fraud kept 3x longer', tone: 'holdout' },
    { id: 'refit', x: 362, y: 380, w: 176, eyebrow: 'blue team', label: 'Defender refit', sub: `${coadapt.refit_updates.length} refits over ${coadapt.rows.length} updates`, tone: 'def' },
  ]

  // One pass of the loop on a 9 second clock. Steps that genuinely happen at the
  // same moment share a time; everything else is ordered by what it depends on.
  // The gap before the cycle restarts is deliberate: a beat of stillness is what
  // makes the restart read as "again" rather than as continuous churn.
  const wires: SvgWire[] = [
    { d: 'M216 55 H362', tone: 'value', label: 'warm start', lx: 232, ly: 48, at: 0 },
    { d: 'M450 76 V196', tone: 'value', at: 1.0 },
    { d: 'M216 227 H362', tone: 'atk', at: 1.0 },
    { d: 'M538 227 H684', tone: 'holdout', at: 2.1 },
    { d: 'M772 248 V330', tone: 'def', at: 3.2 },
    // Mitigation fires at the moment of the decision, not after it.
    { d: 'M860 217 H890 V55 H538', tone: 'def', label: 'mitigation', lx: 700, ly: 48, feedback: true, at: 3.2, travel: 1.3 },
    // The delay is the point of this one, so it crosses slowly.
    { d: 'M450 248 V300 H772 V330', tone: 'holdout', label: 'label latency 4320 min', lx: 480, ly: 294, feedback: true, at: 4.4, travel: 1.5 },
    { d: 'M684 361 H450 V380', tone: 'def', label: 'prevalence subsample', lx: 470, ly: 354, at: 6.0 },
    { d: 'M362 406 H128 V248', tone: 'atk', label: 'attacker adapts', lx: 150, ly: 300, feedback: true, at: 7.0, travel: 1.2 },
  ]

  return (
    <div>
      <SvgFlow
        id="loop"
        viewBox="0 0 940 460"
        boxes={boxes}
        wires={wires}
        cycle={9}
        ariaLabel="The closed loop. Calibrated benign data warm-starts the simulation world. Attack episodes and ordinary traffic share one event stream, which the defender in force scores. Three feedback paths close the loop: mitigation edits the world, label latency delays what reaches the retention buffer, and every defender refit forces the attacker to adapt."
      />
      <Legend>
        <span><span className="text-ink-2">Solid</span> the forward path</span>
        <span><span className="text-ink-2">Dashed</span> the three feedback paths that make it a loop</span>
        <span>
          <span className="text-atk">red</span> attacker,{' '}
          <span className="text-def">blue</span> defender,{' '}
          <span className="text-value">gold</span> the calibrated world,{' '}
          <span className="text-holdout">violet</span> the shared path that is blind to who acted
        </span>
      </Legend>
    </div>
  )
}
