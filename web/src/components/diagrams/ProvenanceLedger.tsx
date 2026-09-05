import { Chip, ChipGroup } from '../ui/Chip'
import { Label } from '../ui/Label'
import { fidelity } from '../../data/run'
import { int } from '../../lib/format'
import { Note } from '../ui/Note'

/**
 * The parameter ledger.
 *
 * Every parameter in the simulator carries one of four provenance tags. Two of
 * them are recorded by name in the calibration artifact and are listed here
 * exactly as it wrote them, so this panel cannot drift from what the pipeline
 * actually did. The other two are design-time choices the artifact does not
 * enumerate, and no count is asserted for them.
 */
export function ProvenanceLedger() {
  const p = fidelity.provenance

  return (
    <div>
      <p className="prose-sans max-w-3xl text-[0.9375rem] leading-relaxed text-ink-2">
        A number whose origin is visible is worth more than a number that is merely large. These are
        the parameter groups the calibration recorded, under its own names.
      </p>

      <div className="mt-4 grid gap-5 md:grid-cols-2">
        <div>
          <div className="flex items-baseline gap-2">
            <Label>fitted</Label>
            <span className="num text-[0.9375rem] text-pass">{p.fitted.length}</span>
            <span className="text-[0.8125rem] text-ink-3">measured from real data</span>
          </div>
          <ChipGroup>
            {p.fitted.map((name) => (
              <Chip key={name} tone="pass">
                {name}
              </Chip>
            ))}
          </ChipGroup>
        </div>

        <div>
          <div className="flex items-baseline gap-2">
            <Label>swept</Label>
            <span className="num text-[0.9375rem] text-value">{p.swept.length}</span>
            <span className="text-[0.8125rem] text-ink-3">unmeasurable, reported across a range</span>
          </div>
          <ChipGroup>
            {p.swept.map((name) => (
              <Chip key={name} tone="value">
                {name}
              </Chip>
            ))}
          </ChipGroup>
        </div>
      </div>

      <dl className="mt-5 grid grid-cols-2 gap-x-6 gap-y-3 border-t border-rule pt-4 sm:grid-cols-4">
        <div>
          <Label>noise floors</Label>
          <dd className="num mt-0.5 text-[1rem] text-ink">{int(p.n_noise_floors)}</dd>
        </div>
        <div>
          <Label>calibration targets</Label>
          <dd className="num mt-0.5 text-[1rem] text-ink">{int(p.n_targets)}</dd>
        </div>
        <div>
          <Label>split entities</Label>
          <dd className="num mt-0.5 text-[1rem] text-ink">
            {int(fidelity.split.left_entities)}
          </dd>
        </div>
        <div>
          <Label>fitted on</Label>
          <dd className="num mt-0.5 text-[1rem] text-ink">
            {fidelity.created_utc?.slice(0, 10) ?? 'not recorded'}
          </dd>
        </div>
      </dl>

      <Note
        label="the two tags not shown here"
      >
        <p className="prose-sans text-[0.875rem] leading-relaxed text-ink-2">
        Two more tags exist in the system. <span className="text-ink">Cited</span> parameters come
        from published figures and <span className="text-ink">free</span> ones are design choices,
        both declared in the configuration rather than produced by calibration. The artifact keeps no
        ledger of those, so no count is claimed for them here.
      </p>
        <p className="prose-sans mt-2 text-[0.875rem] leading-relaxed text-ink-3">
        A swept parameter is the honest case: nothing in the available data pins down how many
        devices a household shares or how far a cardholder travels from home, so those are varied
        across a range instead of being asserted as a measurement.
      </p>
      </Note>
    </div>
  )
}
