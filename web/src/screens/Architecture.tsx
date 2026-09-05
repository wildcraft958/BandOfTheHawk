import { useEffect, useState } from 'react'
import { Panel } from '../components/ui/Panel'
import { Chip, ChipGroup } from '../components/ui/Chip'
import { Label } from '../components/ui/Label'
import { Meter } from '../components/ui/Meter'
import { Branch, Node } from '../components/flow/Flow'
import { cn } from '../components/ui/cn'
import { StageTimeline } from '../components/diagrams/StageTimeline'
import { AttackerDiagram } from '../components/diagrams/svg/AttackerDiagram'
import { DefenderDiagram } from '../components/diagrams/svg/DefenderDiagram'
import { LoopDiagram } from '../components/diagrams/LoopDiagram'
import { WithheldNote } from '../components/diagrams/WithheldNote'
import { AT, RUN, feed } from '../components/flow/clock'
import {
  Database,
  Pause,
  Play,
  Gauge,
  GitMerge,
  Landmark,
  Network,
  RefreshCw,
  Swords,
  Table2,
  UsersRound,
} from 'lucide-react'
import { useReducedMotion } from '../lib/useReducedMotion'
import { coadapt, detectors, graph, meta, runReport } from '../data/run'
import { ACTIONS, POSTURES, STAGE_ORDER, VERTICALS } from '../data/taxonomy'
import { duration, fixed, int, pct } from '../lib/format'
import { BANDS } from '../data/paper'
import { PageHead } from '../components/ui/PageHead'

type Open = 'attack' | 'benign' | 'builder' | 'log' | 'flat' | 'mixture' | 'bands' | 'refit' | null

const GENAI = ACTIONS.filter((a) => a.genaiTool)

/**
 * How one closed-loop run is produced.
 *
 * The shape is the argument. Attack traffic and ordinary traffic fork apart and
 * then converge on a single event builder, because the blindness rule means the
 * same code path produces both and no field distinguishes them. If the two sides
 * had separate builders, a detector could learn which builder wrote a row
 * instead of learning behaviour, and the geometry would show that.
 *
 * The loop closes at the bottom: a refit changes what the attacker faces, so the
 * next pass through the graph is not the same pass.
 */
export function Architecture() {
  const reduced = useReducedMotion()
  const [dart, setDart] = useState(0)
  const [open, setOpen] = useState<Open>(null)
  const [playing, setPlaying] = useState(true)

  useEffect(() => {
    if (reduced || !playing) return undefined
    const timer = window.setInterval(() => setDart((d) => d + 1), RUN)
    return () => window.clearInterval(timer)
  }, [reduced, playing])

  // Resuming replays at once. Waiting out the rest of a 7.6 second interval
  // makes the button feel broken.
  const togglePlaying = () => {
    setPlaying((p) => {
      if (!p) setDart((d) => d + 1)
      return !p
    })
  }

  const toggle = (k: Exclude<Open, null>) => () => setOpen((o) => (o === k ? null : k))
  const flat = detectors.configs.find((c) => c.id === 'gbdt_full')
  const learned = detectors.configs.find((c) => c.id === 'experts_learned')
  const maxWeight = Math.max(...detectors.experts.map((e) => e.weight))
  const bands = BANDS

  /**
   * The mechanics for the four paired stages live here rather than inside
   * their cards. Two stages side by side have to stay the same size, and a
   * card that grows when opened would stretch its partner into an empty box.
   * Rendered full width below the pair, dense content also gets the room it
   * wants instead of half a column.
   */
  const mixtureDetail = (
    <>
      <Label>learned combiner weights</Label>
      <div className="mt-2 space-y-2.5">
        {detectors.experts.map((e) => (
          <Meter
            key={e.name}
            label={e.name}
            value={e.weight}
            max={maxWeight}
            display={pct(e.normalized_weight)}
            tone={e.name === 'text' ? 'atk' : 'def'}
            note={
              e.name === 'text'
                ? 'the thinnest channel, and the one the attacker converged on'
                : undefined
            }
          />
        ))}
      </div>
      <p className="prose-sans mt-3 border-t border-rule pt-2 text-[0.8125rem] text-ink-3">
        The mixture loses to the flat tree on this run, {fixed(learned?.metrics.pr_auc, 4)} against{' '}
        {fixed(flat?.metrics.pr_auc, 4)}. Structural decomposition costs accuracy at this scale and
        buys per-event-type attribution and independent retraining.
      </p>
    </>
  )
  const flatDetail = (
    <>
      <Label>top features by gain</Label>
      <div className="mt-2 space-y-2">
        {detectors.feature_gains.slice(0, 5).map((f) => (
          <Meter
            key={f.name}
            label={f.name}
            value={f.gain}
            max={detectors.feature_gains[0].gain}
            display={f.gain.toFixed(1)}
            tone={f.per_entity ? 'holdout' : 'def'}
          />
        ))}
      </div>
      <p className="prose-sans mt-3 text-[0.8125rem] text-ink-3">
        Per-entity features added {fixed(detectors.per_entity_lift, 4)} PR-AUC. The run&rsquo;s own
        verdict: they add little here.
      </p>
    </>
  )
  const benignDetail = (
    <>
      <Label>hard negatives injected</Label>
      <div className="mt-2 space-y-2">
        {Object.entries(runReport.hard_negatives)
          .sort((a, b) => b[1] - a[1])
          .map(([name, count]) => (
            <Meter
              key={name}
              label={name}
              value={count}
              max={Math.max(...Object.values(runReport.hard_negatives))}
              display={int(count)}
              tone={name === 'ordinary' ? 'neutral' : 'def'}
            />
          ))}
      </div>
      <p className="prose-sans mt-3 text-[0.8125rem] text-ink-3">
        Gift-card runs, travel, new devices, recovery flows and disputes are all legitimate and all
        look suspicious. The false-positive rate is earned against these, not against easy
        negatives.
      </p>
    </>
  )
  const attackDetail = (
    <>
      <Label>stage machine</Label>
      <ChipGroup>
        {STAGE_ORDER.map((st) => (
          <Chip key={st} tone="atk">
            {st}
          </Chip>
        ))}
      </ChipGroup>
      <p className="prose-sans mt-2 text-[0.8125rem] text-ink-3">
        Gating is structural. The mask is applied to the policy&rsquo;s logits before the softmax,
        so no probability is ever placed on an impossible action.
      </p>

      <div className="mt-4">
        <Label>policy heads</Label>
        <ul className="mt-1.5 space-y-1 text-[0.8125rem] text-ink-2">
          <li>action, a 20-way masked softmax</li>
          <li>amount, 1.0 to 5000.0</li>
          <li>delay, 0 to 4320 minutes</li>
          <li>posture, 4-way: {POSTURES.map((p) => p.id).join(', ')}</li>
        </ul>
      </div>

      <div className="mt-4">
        <Label>where the GenAI enters</Label>
        <ChipGroup>
          {GENAI.map((a) => (
            <Chip key={a.id} tone="atk" title={a.genaiTool ?? undefined}>
              {a.id}
            </Chip>
          ))}
        </ChipGroup>
        <p className="prose-sans mt-2 text-[0.8125rem] text-ink-3">
          {GENAI.length} of {ACTIONS.length} actions require a generated artifact: a cloned voice, a
          deepfake selfie, a written pretext, a dispute narrative.
        </p>
      </div>

      <div className="mt-4">
        <Label>victim selection</Label>
        <p className="prose-sans mt-1 text-[0.8125rem] text-ink-2">
          A contextual bandit over {int(coadapt.selection.observations)} candidates. It learned to
          prefer cards older than three years. Nothing told it which cards were worth attacking.
        </p>
      </div>
    </>
  )

  return (
    <div className="space-y-4 px-4 py-5 sm:px-6">
      <PageHead
        title="Architecture"
        blurb="One pass of the closed loop, then each side in detail: how the attacker learns what to try, and how the defender turns one event into one action."
      />

      <Panel
        name="architecture: one pass of the closed loop"
        live={!reduced}
        tone="value"
        aside={
          <span className="flex items-center gap-3">
            <span className="hidden text-[0.75rem] uppercase tracking-[0.09em] text-ink-3 md:inline">
              select any stage for its mechanics
            </span>
            {!reduced && (
              <button
                type="button"
                onClick={togglePlaying}
                aria-label={playing ? 'Pause the flow animation' : 'Play the flow animation'}
                aria-pressed={!playing}
                title={playing ? 'Pause the flow animation' : 'Play the flow animation'}
                className="grid size-6 shrink-0 place-items-center rounded-chip border border-rule text-ink-2 transition-colors duration-150 hover:border-ink-3 hover:text-ink"
              >
                {playing ? (
                  <Pause className="size-3" aria-hidden="true" />
                ) : (
                  <Play className="size-3" aria-hidden="true" />
                )}
              </button>
            )}
          </span>
        }
      >
        <p className="prose-sans mb-6 max-w-3xl text-[0.9375rem] leading-relaxed text-ink-2">
          Two sides fork apart and converge on a single event builder. That convergence is the
          blindness rule: the same code path produces an attacker&rsquo;s row and an ordinary
          holder&rsquo;s row, and no field distinguishes them. Separate builders would let a
          detector learn which builder wrote a row instead of learning behaviour.
        </p>

        <div className="mx-auto max-w-3xl">
          <Node
            kind="World"
            icon={Landmark}
            name="Calibrated synthetic bank"
            body={`${int(runReport.warm_start.entities)} entities and ${int(runReport.warm_start.events)} warm-start events over ${int(runReport.warm_start.history_days)} days, fitted against real aggregate statistics.`}
            metric={`fan-out ${fixed(graph.fanout_observed_mean, 2)} vs ${fixed(graph.fanout_target_mean, 2)}`}
            dart={dart}
            delay={AT.world}
            note="no cardholder data at any point"
          />

          <Branch kind="fork" dart={dart} delay={feed(AT.sides)} tone="value" />

          <div className="grid grid-cols-2 gap-3">
            <Node
              kind="Red team"
              icon={Swords}
              name="Attacker"
              body={`${VERTICALS.filter((v) => v.simulated).length} verticals over one 20-action space. Scripted policies plus a reinforcement-learning policy that discovers its own.`}
              metric={`${int(runReport.fraud_auths)} fraud auths`}
              tone="atk"
              dart={dart}
              delay={AT.sides}
              onClick={toggle('attack')}
              expanded={open === 'attack'}
            />

            <Node
              kind="Blue world"
              icon={UsersRound}
              name="Ordinary traffic"
              body={`${int(runReport.benign_auths)} benign authorisations, including deliberately suspicious but legitimate behaviour.`}
              metric={`${pct(runReport.fraud_auth_share, 1)} fraud share`}
              tone="def"
              dart={dart}
              delay={AT.sides}
              onClick={toggle('benign')}
              expanded={open === 'benign'}
            />
          </div>

          {(open === 'attack' || open === 'benign') && (
            <div
              className={cn(
                'mt-3 rounded-panel border bg-surface-card px-4 py-4',
                open === 'attack' ? 'border-atk/55' : 'border-def/55',
              )}
            >
              <p className="mb-3 text-[0.6875rem] font-semibold uppercase tracking-[0.11em] text-ink-3">
                {open === 'attack' ? 'Red team' : 'Blue world'} mechanics
              </p>
              {open === 'attack' ? attackDetail : benignDetail}
            </div>
          )}

          <Branch kind="join" dart={dart} delay={feed(AT.builder)} tone="value" />

          <Node
            kind="Shared"
            icon={GitMerge}
            name="Event builder"
            body="One builder for both sides. It receives entity references and reads the graph, and nothing in its signature says who is acting."
            tone="holdout"
            dart={dart}
            delay={AT.builder}
            note="the blindness rule"
            onClick={toggle('builder')}
            expanded={open === 'builder'}
          >
            <ul className="space-y-2">
              {[
                'No event field indicates who caused it. Any such field would be a shortcut a detector could learn instead of learning behaviour.',
                'scoring_fields() drops is_fraud and episode_id structurally, so a scorer cannot read the label even by accident.',
                'Build and commit are separate calls, so an event never counts itself in its own velocity windows.',
                'The defender table drops all nine identity fields. Identity is not a feature.',
              ].map((line) => (
                <li key={line} className="flex gap-2">
                  <span
                    className="mt-[0.4rem] size-1 shrink-0 rounded-full bg-holdout"
                    aria-hidden="true"
                  />
                  <span className="prose-sans text-[0.8125rem] leading-relaxed text-ink-2">
                    {line}
                  </span>
                </li>
              ))}
            </ul>
            <p className="prose-sans mt-3 border-t border-rule pt-2 text-[0.75rem] text-ink-3">
              Each of these is enforced by a test, not by convention.
            </p>
          </Node>

          <Branch dart={dart} delay={feed(AT.log)} tone="holdout" />

          <Node
            kind="Store"
            icon={Database}
            name="Event log"
            body="Labels are stamped only after an episode closes, because at the moment of scoring nothing knows the answer."
            metric={`${int(detectors.train_rows)} train / ${int(detectors.test_rows)} test`}
            dart={dart}
            delay={AT.log}
            onClick={toggle('log')}
            expanded={open === 'log'}
          >
            <Label>feature table</Label>
            <ul className="mt-1.5 space-y-1 text-[0.8125rem] text-ink-2">
              <li>14 event types across authorisation, binding, text and network</li>
              <li>18 compound window features over 1h, 24h and 7d</li>
              <li>8 nullable fields, each with a companion missing flag</li>
              <li>256 embedding columns when the text pool is embedded</li>
            </ul>
            <p className="prose-sans mt-3 text-[0.8125rem] text-ink-3">
              Label latency in the live loop is 4320 minutes, three days, which is what a chargeback
              cycle looks like. The defender always trains on stale labels.
            </p>
          </Node>

          <Branch kind="fork" dart={dart} delay={feed(AT.detectors)} tone="def" />

          <div className="grid grid-cols-2 gap-3">
            <Node
              kind="Path A"
              icon={Table2}
              name="Flat detector"
              body="One gradient-boosted tree over the whole table. The benchmark the mixture has to beat."
              metric={`PR-AUC ${fixed(flat?.metrics.pr_auc, 4)}`}
              tone="def"
              dart={dart}
              delay={AT.detectors}
              onClick={toggle('flat')}
              expanded={open === 'flat'}
            />

            <Node
              kind="Path B"
              icon={Network}
              name="Five experts, routed by schema"
              body="Each expert reads only the event types it applies to. Routing is a schema fact, not a learned gate."
              metric={`PR-AUC ${fixed(learned?.metrics.pr_auc, 4)}`}
              tone="def"
              dart={dart}
              delay={AT.detectors}
              onClick={toggle('mixture')}
              expanded={open === 'mixture'}
            />
          </div>

          {(open === 'flat' || open === 'mixture') && (
            <div
              className={cn(
                'mt-3 rounded-panel border bg-surface-card px-4 py-4',
                open === 'flat' ? 'border-def/55' : 'border-def/55',
              )}
            >
              <p className="mb-3 text-[0.6875rem] font-semibold uppercase tracking-[0.11em] text-ink-3">
                {open === 'flat' ? 'Path A' : 'Path B'} mechanics
              </p>
              {open === 'flat' ? flatDetail : mixtureDetail}
            </div>
          )}

          <Branch kind="join" dart={dart} delay={feed(AT.bands)} tone="def" />

          <Node
            kind="Decision"
            icon={Gauge}
            name="Risk bands to mitigation"
            body="A score becomes an action a payment system already has. Mitigation is graph surgery: a block unbinds a device, removing an edge."
            metric="relative cost"
            tone="value"
            dart={dart}
            delay={AT.bands}
            onClick={toggle('bands')}
            expanded={open === 'bands'}
          >
            {bands && (
              <ul className="space-y-2">
                {(
                  [
                    ['step_up', bands.stepUp, 'challenge the cardholder', 'def'],
                    ['hold', bands.hold, 'freeze the card 24h', 'value'],
                    ['decline', bands.decline, 'freeze 72h', 'value'],
                    ['block', bands.block, 'unbind device, add to blocklist', 'atk'],
                  ] as const
                ).map(([name, threshold, mitigation, tone]) => (
                  <li key={name} className="flex items-baseline gap-3">
                    <span className="num w-10 shrink-0 text-[0.875rem] text-ink">
                      {threshold.toFixed(2)}
                    </span>
                    <Chip tone={tone}>{name}</Chip>
                    <span className="text-[0.8125rem] text-ink-2">{mitigation}</span>
                  </li>
                ))}
              </ul>
            )}
            <p className="prose-sans mt-3 border-t border-rule pt-2 text-[0.8125rem] leading-relaxed text-ink-3">
              One refit&rsquo;s grid-search output, not fixed thresholds. The four boundaries are
              re-searched against the cost curve at every refit, and jitter moves them slightly each
              episode so a policy cannot find a threshold and sit under it. The axis is relative
              cost, never a currency figure: amounts are unit where event value is not to hand.
            </p>
          </Node>

          <Branch dart={dart} delay={feed(AT.refit)} tone="atk" />

          <Node
            kind="Loop"
            icon={RefreshCw}
            name="Defender refit"
            body={`The defender retrains on the labels that have cleared latency, and what the attacker faces changes. In the ${int(meta.population)} holder run replayed here, ${coadapt.refit_updates.length} refits over ${coadapt.rows.length} updates.`}
            metric={`${coadapt.reads.zeros} updates at zero`}
            tone="atk"
            dart={dart}
            delay={AT.refit}
            note="the next pass is not the same pass"
            onClick={toggle('refit')}
            expanded={open === 'refit'}
          >
            <p className="prose-sans text-[0.8125rem] leading-relaxed text-ink-2">
              This edge is what makes the graph a loop rather than a pipeline. After the first refit
              the attacker&rsquo;s take fell to exactly zero for {coadapt.reads.zeros} updates and
              its policy entropy rose from {fixed(coadapt.reads.entropy_start, 3)} to{' '}
              {fixed(coadapt.reads.entropy_peak, 3)}: the refit destroyed a converged policy and
              forced re-exploration. What it converged on next was a vertical held out of training
              entirely.
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              <WithheldNote compact />
            </div>
          </Node>
        </div>
      </Panel>

      <Panel name="the closed loop, end to end" tone="value">
        <p className="prose-sans mb-4 max-w-3xl text-[0.9375rem] leading-relaxed text-ink-2">
          Calibrated benign data warm-starts the world. Attack episodes and ordinary traffic
          share one event stream, and the defender in force scores it. Three arrows run backwards,
          and they are what make this a loop rather than a pipeline: mitigation edits the world,
          label latency delays what reaches training, and every refit forces the attacker to
          adapt.
        </p>
        <LoopDiagram />
      </Panel>

      <Panel
        name="pipeline stages and where the time went"
        tone="value"
        aside={
          <span className="hidden text-[0.75rem] uppercase tracking-[0.09em] text-ink-3 md:inline">
            {duration(meta.total_seconds)} total
          </span>
        }
      >
        <p className="prose-sans mb-4 max-w-3xl text-[0.9375rem] leading-relaxed text-ink-2">
          The same system seen as six stages rather than as a data flow. Two setup stages feed a
          four-stage cycle, and the bar in each node is its share of the run. Co-adaptation is
          almost all of it, which is why this page replays one run rather than launching another.
        </p>
        <StageTimeline stages={meta.stages} total={meta.total_seconds ?? 1} />
      </Panel>

      <div className="border-t border-rule pt-5">
        <p className="text-[0.8125rem] uppercase tracking-[0.12em] text-ink-3">
          the two sides in detail
        </p>
        <p className="prose-sans mt-2 max-w-3xl text-[0.9375rem] leading-relaxed text-ink-2">
          The loop above shows how the pieces connect. These two show what happens inside each
          side: how the attacker learns what to try, and how the defender turns one event into one
          action.
        </p>
      </div>

      <Panel
        name="the learned attacker"
        tone="atk"
        aside={
          <span className="text-[0.75rem] uppercase tracking-[0.09em] text-ink-3">
            sizes from the {int(meta.population)} holder run
          </span>
        }
      >
        <AttackerDiagram />
      </Panel>

      <Panel name="the defender, one event to one action" tone="def">
        <DefenderDiagram />
      </Panel>
    </div>
  )
}
