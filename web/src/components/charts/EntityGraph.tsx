import { useMemo, useState } from 'react'
import { Badge } from '../ui/Badge'
import { Label } from '../ui/Label'
import { cn } from '../ui/cn'
import { mulberry32 } from '../../lib/rng'
import { graph } from '../../data/run'
import { pct } from '../../lib/format'

/**
 * Mitigation as graph surgery, and the reason a blocklist has to know what it is
 * blocking.
 *
 * A block does not flag a row: it unbinds a device, which deletes edges. Watching
 * those edges disappear is the clearest statement of what the defender actually
 * does to the world.
 *
 * The second argument is the one that decides whether this survives production.
 * Heavy device sharing is mostly fingerprint collision, many unrelated people on
 * the same OS, browser and screen, rather than real sharing. Blocking a device
 * removes one binding; blocking everything behind a shared fingerprint removes
 * every card in the bucket, most of them belonging to people who did nothing.
 *
 * The graph is generated from the fitted fan-out shape rather than exported from
 * the run, which the panel states. The degree statistics beside it are measured.
 */

const W = 720
const H = 340
const DEVICE_X = 130
const CARD_X = 580

interface Device {
  id: string
  bucket: number
  y: number
  fraud: boolean
}
interface Card {
  id: string
  y: number
}

export function EntityGraph() {
  const [blocked, setBlocked] = useState<Set<string>>(new Set())
  const [mode, setMode] = useState<'device' | 'bucket'>('device')

  const { devices, cards, edges } = useMemo(() => {
    const rand = mulberry32(11)
    const N_DEVICES = 9
    const N_CARDS = 16
    // Three fingerprint buckets. One is deliberately crowded, which is what a
    // real OS-browser-screen collision looks like.
    const bucketOf = [0, 0, 0, 0, 1, 1, 2, 2, 2]

    const devices: Device[] = Array.from({ length: N_DEVICES }, (_, i) => ({
      id: `dev_${i}`,
      bucket: bucketOf[i],
      y: 34 + i * ((H - 68) / (N_DEVICES - 1)),
      fraud: i === 3,
    }))
    const cards: Card[] = Array.from({ length: N_CARDS }, (_, i) => ({
      id: `card_${i}`,
      y: 22 + i * ((H - 44) / (N_CARDS - 1)),
    }))

    // Most devices bind one or two cards; one binds many, which is the heavy
    // tail the fitted fan-out has and row-sampling cannot produce.
    const edges: Array<{ device: string; card: string }> = []
    devices.forEach((d, i) => {
      const degree = i === 3 ? 6 : i === 0 ? 3 : 1 + Math.floor(rand() * 2)
      const taken = new Set<number>()
      for (let k = 0; k < degree; k++) {
        let c = Math.floor(rand() * N_CARDS)
        let guard = 0
        while (taken.has(c) && guard++ < 20) c = Math.floor(rand() * N_CARDS)
        taken.add(c)
        edges.push({ device: d.id, card: cards[c].id })
      }
    })
    return { devices, cards, edges }
  }, [])

  const isBlocked = (deviceId: string) => {
    if (blocked.has(deviceId)) return true
    if (mode === 'bucket') {
      const d = devices.find((x) => x.id === deviceId)
      return devices.some((x) => blocked.has(x.id) && d && x.bucket === d.bucket)
    }
    return false
  }

  const liveEdges = edges.filter((e) => !isBlocked(e.device))
  const cutEdges = edges.filter((e) => isBlocked(e.device))
  const cardsCut = new Set(cutEdges.map((e) => e.card))
  const cardsStillReachable = new Set(liveEdges.map((e) => e.card))
  const collateral = [...cardsCut].filter((c) => !cardsStillReachable.has(c))
  const fraudDevice = devices.find((d) => d.fraud)
  const t = graph.targets

  const toggle = (id: string) =>
    setBlocked((prev) => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })

  return (
    <div>
      <p className="prose-sans max-w-3xl text-[0.9375rem] leading-relaxed text-ink-2">
        A block is not a flag on a row. It unbinds a device, which deletes edges from the graph.
        Select a device to block it and watch its bindings go.
      </p>

      <div className="mt-4 flex flex-wrap items-center gap-4">
        <div>
          <Label>what a block removes</Label>
          <div className="mt-1.5 flex gap-1">
            {(
              [
                ['device', 'this device only'],
                ['bucket', 'everything sharing its fingerprint'],
              ] as const
            ).map(([m, label]) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                aria-pressed={mode === m}
                className={cn(
                  'rounded-chip px-2.5 py-1 text-[0.8125rem] transition-colors duration-150',
                  mode === m
                    ? 'bg-atk/15 text-atk'
                    : 'text-ink-3 hover:bg-surface-hover hover:text-ink-2',
                )}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
        {blocked.size > 0 && (
          <button
            type="button"
            onClick={() => setBlocked(new Set())}
            className="rounded-panel border border-rule px-2.5 py-1 text-[0.75rem] uppercase tracking-[0.09em] text-ink-2 hover:border-ink-3 hover:text-ink"
          >
            restore all
          </button>
        )}
      </div>

      <div className="mt-4 overflow-x-auto">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="block h-[21rem] w-full min-w-[38rem]"
          role="img"
          aria-label="A device to card binding graph. Blocking a device deletes its bindings; blocking everything behind a shared fingerprint deletes far more, most of it belonging to unrelated cardholders."
        >
          <text x={DEVICE_X} y={14} textAnchor="middle" className="fill-ink-3" fontSize={9}>
            DEVICES
          </text>
          <text x={CARD_X} y={14} textAnchor="middle" className="fill-ink-3" fontSize={9}>
            CARDS
          </text>

          {edges.map((e, i) => {
            const d = devices.find((x) => x.id === e.device)!
            const c = cards.find((x) => x.id === e.card)!
            const cut = isBlocked(e.device)
            const mid = (DEVICE_X + CARD_X) / 2
            return (
              <path
                key={i}
                d={`M ${DEVICE_X + 26} ${d.y} C ${mid} ${d.y}, ${mid} ${c.y}, ${CARD_X - 12} ${c.y}`}
                fill="none"
                stroke={cut ? 'var(--color-atk)' : d.fraud ? 'var(--color-value)' : '#2f2f36'}
                strokeWidth={cut ? 1 : 1.25}
                strokeDasharray={cut ? '3 3' : undefined}
                opacity={cut ? 0.22 : 1}
                style={{ transition: 'opacity 320ms ease, stroke 320ms ease' }}
              />
            )
          })}

          {devices.map((d) => {
            const cut = isBlocked(d.id)
            const directly = blocked.has(d.id)
            return (
              <g
                key={d.id}
                role="button"
                tabIndex={0}
                aria-pressed={directly}
                aria-label={`${d.id}, fingerprint bucket ${d.bucket}${d.fraud ? ', the attacker device' : ''}`}
                onClick={() => toggle(d.id)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    toggle(d.id)
                  }
                }}
                className="cursor-pointer outline-none"
              >
                <rect
                  x={DEVICE_X - 26}
                  y={d.y - 9}
                  width={52}
                  height={18}
                  rx={2}
                  fill={directly ? 'rgb(229 72 77 / 0.18)' : 'var(--color-surface-card)'}
                  stroke={
                    directly
                      ? 'var(--color-atk)'
                      : cut
                        ? 'rgb(229 72 77 / 0.5)'
                        : d.fraud
                          ? 'var(--color-value)'
                          : 'var(--color-rule)'
                  }
                  strokeWidth={directly ? 1.5 : 1}
                  strokeDasharray={cut && !directly ? '3 2' : undefined}
                />
                <text
                  x={DEVICE_X}
                  y={d.y + 3}
                  textAnchor="middle"
                  fontSize={7.5}
                  className={cut ? 'fill-ink-3' : 'fill-ink-2'}
                >
                  {d.id}
                </text>
                <text x={DEVICE_X - 34} y={d.y + 3} textAnchor="end" fontSize={7} className="fill-ink-3">
                  fp{d.bucket}
                </text>
              </g>
            )
          })}

          {cards.map((c) => {
            const orphaned = !cardsStillReachable.has(c.id) && cardsCut.has(c.id)
            return (
              <g key={c.id}>
                <circle
                  cx={CARD_X}
                  cy={c.y}
                  r={4}
                  fill={orphaned ? 'rgb(229 72 77 / 0.25)' : 'var(--color-surface-card)'}
                  stroke={orphaned ? 'var(--color-atk)' : 'var(--color-rule)'}
                  strokeWidth={1}
                  style={{ transition: 'stroke 320ms ease, fill 320ms ease' }}
                />
              </g>
            )
          })}
        </svg>
      </div>

      <div className="mt-3 grid gap-3 sm:grid-cols-3">
        <div className="rounded-panel border border-rule bg-surface px-3 py-2">
          <Label>bindings removed</Label>
          <p className="num mt-1 text-[1.1rem] text-atk">
            {cutEdges.length}
            <span className="text-[0.875rem] text-ink-3"> / {edges.length}</span>
          </p>
        </div>
        <div className="rounded-panel border border-rule bg-surface px-3 py-2">
          <Label>cards left unreachable</Label>
          <p className="num mt-1 text-[1.1rem] text-atk">{collateral.length}</p>
        </div>
        <div className="rounded-panel border border-rule bg-surface px-3 py-2">
          <Label>mode</Label>
          <p className="mt-1 text-[0.9375rem] text-ink">
            {mode === 'device' ? 'one device' : 'whole fingerprint bucket'}
          </p>
        </div>
      </div>

      <div className="mt-4 rounded-panel border border-rule bg-surface px-4 py-3">
        <div className="flex flex-wrap items-center gap-3">
          <Badge tone="value">the attacker device</Badge>
          <span className="text-[0.875rem] text-ink-2">
            {fraudDevice?.id} carries the gold bindings. Blocking it is the intended outcome.
          </span>
        </div>
        <p className="prose-sans mt-3 max-w-3xl text-[0.9375rem] leading-relaxed text-ink-2">
          Switch the mode above. Blocking one device removes one binding. Blocking everything
          behind a shared fingerprint removes every card in that bucket, and{' '}
          <span className="text-ink">
            heavy device sharing is mostly fingerprint collision rather than real sharing
          </span>{' '}
          : many unrelated people on the same operating system, browser and screen. Conflating the
          two is how a blocklist takes out customers who did nothing.
        </p>
        <p className="prose-sans mt-2 text-[0.875rem] text-ink-3">
          Measured on the real graph: fan-out mean {t.fanout_mean.toFixed(2)}, variance to mean{' '}
          {t.fanout_variance_to_mean.toFixed(2)}, p99 {t.fanout_p99.toFixed(0)}, max{' '}
          {t.fanout_max.toFixed(0)}, and {pct(t.fanout_share_shared)} of nodes shared. The graph
          above is drawn from that shape rather than exported from the run, so treat the picture as
          the mechanism and the numbers as the measurement.
        </p>
      </div>
    </div>
  )
}
