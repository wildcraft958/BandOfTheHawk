import { SvgFlow, Legend, type SvgBox, type SvgGroup, type SvgWire } from './SvgFlow'
import { coadapt } from '../../../data/run'
import { int } from '../../../lib/format'

/**
 * The learned attacker, in three phases.
 *
 * Offline, scripted policies are cloned into an actor and a critic is
 * warm-started against them, so PPO never begins from noise. Online, the policy
 * runs against the world and every defender refit throws it back into
 * exploration. Below, one decision: a bandit picks the victim, then four heads
 * choose what to do to them.
 */
const GROUPS: SvgGroup[] = [
  { x: 8, y: 8, w: 566, h: 258, title: 'warm start', note: 'offline', tone: 'value' },
  { x: 596, y: 8, w: 566, h: 258, title: 'co-adaptation', note: 'online', tone: 'atk' },
  { x: 8, y: 292, w: 1154, h: 300, title: 'one decision', tone: 'def' },
]

export function AttackerDiagram() {
  const boxes: SvgBox[] = [
    { id: 'scripted', x: 26, y: 56, w: 152, eyebrow: 'seed', label: 'Scripted policies', sub: '9 verticals, hand written', tone: 'pass' },
    { id: 'actor0', x: 26, y: 176, w: 152, label: 'Initialised actor', tone: 'pass' },
    { id: 'bc', x: 218, y: 112, w: 158, eyebrow: 'offline fit', label: 'Behaviour cloning', sub: `loss ${coadapt.warm_start.bc_final_loss ?? 'not recorded'}`, tone: 'value' },
    { id: 'critic0', x: 410, y: 40, w: 148, label: 'Initialised critic', tone: 'pass' },
    { id: 'criticws', x: 410, y: 110, w: 148, label: 'Critic warm start', sub: `loss ${coadapt.warm_start.critic_final_loss ?? 'not recorded'}`, tone: 'value' },
    { id: 'imitated', x: 410, y: 200, w: 148, label: 'Imitated actor', tone: 'pass' },

    { id: 'ppo', x: 620, y: 56, w: 176, eyebrow: 'policy', label: 'Warm started PPO', sub: '512 hidden, 80 eps per update', tone: 'def' },
    { id: 'env', x: 860, y: 56, w: 158, eyebrow: 'world', label: 'Environment', sub: 'holders, devices, payees', tone: 'value' },
    { id: 'refit', x: 740, y: 190, w: 152, eyebrow: 'blue team', label: 'Defender refit', sub: 'every 12 updates', tone: 'pass' },

    { id: 'bandit', x: 26, y: 400, w: 158, eyebrow: 'who', label: 'Contextual bandit', sub: `${int(coadapt.selection.observations)} candidates`, tone: 'value' },
    { id: 'policy', x: 246, y: 400, w: 152, eyebrow: 'what', label: 'PPO policy', sub: 'masked before softmax', tone: 'def' },
    { id: 'h1', x: 466, y: 316, w: 150, label: 'Action head', sub: '20 way, masked', tone: 'pass' },
    { id: 'h2', x: 466, y: 378, w: 150, label: 'Amount head', sub: '1.0 to 5000.0', tone: 'pass' },
    { id: 'h3', x: 466, y: 440, w: 150, label: 'Delay head', sub: '0 to 4320 min', tone: 'pass' },
    { id: 'h4', x: 466, y: 502, w: 150, label: 'Posture head', sub: '4 way stealth', tone: 'pass' },
    { id: 'exec', x: 700, y: 400, w: 152, eyebrow: 'act', label: 'Attack executes', sub: 'against the live world', tone: 'value' },
    { id: 'outcome', x: 916, y: 400, w: 180, eyebrow: 'result', label: 'Scored by defender', sub: 'value extracted, or nothing', tone: 'atk' },
  ]

  // 14 second clock, read in the order the system actually runs: the offline
  // warm start first, then the online loop, then a single decision.
  const wires: SvgWire[] = [
    // Offline. Cloning consumes both the scripted policies and the blank actor.
    { d: 'M178 82 H218 V112', tone: 'pass', at: 0 },
    { d: 'M178 195 H218 V164', tone: 'pass', at: 0 },
    { d: 'M376 138 H410 V200', tone: 'value', at: 0.95 },
    { d: 'M484 78 V110', tone: 'pass', at: 1.9 },
    { d: 'M484 200 V162', tone: 'pass', at: 1.9 },
    { d: 'M558 136 H620 V82', tone: 'value', label: 'warm start', lx: 566, ly: 100, at: 2.85 },

    // Online. The reward returns, then a refit throws the policy back out.
    { d: 'M796 82 H860', tone: 'def', at: 4.0 },
    { d: 'M939 108 V150 H700 V108', tone: 'value', label: 'reward', lx: 716, ly: 144, feedback: true, at: 5.0, travel: 1.2 },
    { d: 'M740 216 H660 V108', tone: 'pass', label: 'must adapt', lx: 620, ly: 232, feedback: true, at: 6.5, travel: 1.2 },

    // One decision. Victim first, then all four heads together, then the act.
    { d: 'M184 426 H246', tone: 'value', label: 'victim', lx: 192, ly: 418, at: 8.2 },
    { d: 'M398 426 H432 V342 H466', tone: 'def', at: 9.1 },
    { d: 'M398 426 H432 V404 H466', tone: 'def', at: 9.2 },
    { d: 'M398 426 H432 V466 H466', tone: 'def', at: 9.3 },
    { d: 'M398 426 H432 V528 H466', tone: 'def', at: 9.4 },
    { d: 'M616 342 H652 V426 H700', tone: 'pass', at: 10.3 },
    { d: 'M616 404 H652 V426 H700', tone: 'pass', at: 10.4 },
    { d: 'M616 466 H652 V426 H700', tone: 'pass', at: 10.5 },
    { d: 'M616 528 H652 V426 H700', tone: 'pass', at: 10.6 },
    { d: 'M852 426 H916', tone: 'atk', at: 11.6 },
  ]

  return (
    <div>
      <SvgFlow
        id="atk"
        viewBox="0 0 1170 600"
        groups={GROUPS}
        boxes={boxes}
        wires={wires}
        cycle={14}
        ariaLabel="The learned attacker. Offline, scripted policies are cloned into an actor and a critic is warm-started against them, so PPO never begins from noise. Online, the warm-started policy runs against the environment, and both the reward and every defender refit travel back into it. Below, one decision: a contextual bandit picks the victim, then four heads choose the action, the amount, the delay and the stealth posture before the attack executes and the defender scores it."
      />
      <Legend>
        <span><span className="text-ink-2">Offline</span> runs once, before the loop starts</span>
        <span><span className="text-ink-2">Dashed</span> the reward, and the refit that forces adaptation</span>
        <span>Four heads, so one decision sets what, how much, when and how quietly</span>
      </Legend>
    </div>
  )
}
