# What Was Built, and Where It Diverged

> The other documents in this directory are the plan. This one is the record of
> what the simulation actually does and why it differs, with the measurement
> behind each decision. Where this document and the others conflict, this one is
> current.
>
> Divergence is not failure here. Every entry below exists because something was
> measured that the plan could not have known, and in four cases the measurement
> contradicted an assumption the plan was resting on.

---

## 1. The per-entity axis — a layer the fidelity protocol did not have

`sink12_fidelity_protocol.md` requires conditionals, and means conditional on
**category**: `amount | mcc_cluster`, `hour | mcc_cluster`. That is a different
axis from the one that matters most.

Commit `a711d64` found transaction amount drawn from a single shared curve, so
any two cards differed only by sampling noise: real cards' mean log amount
spreads at 0.605, the generator produced 0.38, and all of it was noise. **No
marginal metric can see this**, and neither can a category conditional. Pooling
every transaction ignores which card produced it, so a generator can match the
pooled shape — and every per-category slice of it — while distributing it across
cards entirely wrongly. A detector reading a value against a card's own history
reads precisely what pooling gets wrong.

**L1 therefore gains a per-entity layer**, implemented in
`fraudsim/calibration/entity_stats.py`:

| Quantity | Statistic | Correction it needs |
|---|---|---|
| amount | between/within variance decomposition | subtract `within²/k` sampling term |
| hour | circular resultant, within and between entity | Fisher: `sqrt((n·R²−1)/(n−1))` |
| category, merchant | Simpson concentration vs a shuffled null | unbiased `Σnᵢ(nᵢ−1)/n(n−1)` |

Every one needs a small-sample correction for the same reason: an entity seen
*k* times looks concentrated purely for having been seen *k* times. One event has
a circular resultant of exactly 1 and a Simpson index of exactly 1, whatever the
process behind it. Uncorrected, the hour estimate reads 0.565 against a true
0.492 and drifts to 0.496 as the cutoff rises to twenty events — so a generator
tuned at one cutoff is wrong at every other. Corrected, it holds near 0.48.

Comparisons are **matched on event count** (`matched_by_event_count`). Pooling
sparse and dense entities reports a difference in census as a difference in
behaviour.

---

## 2. Hour: fitted per holder, and a parameter that had to be abandoned

**Plan** (`FINAL_DESIGN.md` B.2b): fit a von Mises to *the holder's* history,
use it as both the SINK 1 generator and the detector feature — "fit once, use
twice."

**What was there:** the mixture was fitted and used *only* as a feature. No event
was ever placed by it; the warm start had no time-of-day term at all. And the
feature tested every holder against one *population* interval, which is the same
test for everyone and so says nothing about anybody.

**Measured on IEEE-CIS benign** (569,877 rows / 40,866 entities, min_events=10):

| | before | after | real |
|---|---|---|---|
| marginal R | 0.149 | 0.421 | 0.434 |
| within-entity R | 0.226 | 0.452 | 0.492 |
| between-entity R | 0.163 | 0.735 | 0.792 |

The between-entity term is the one no marginal metric could have reported.

**A third parameter was planned and abandoned.** The theory was that holders
differ in how tightly they keep their hours and the tighter ones transact more,
carrying extra weight in the marginal. Measurement says the opposite: busier
entities keep **looser** hours (corrected R falls 0.492 → 0.452 across activity
bands, correlating −0.08 with log activity), and spreading the tightness moves
the marginal the *wrong way* monotonically, because the resultant is concave in
the concentration. What remains is that real holders are not von Mises about one
hour — they have a midday and an evening peak — so a single component
reproducing their resultant under-delivers the marginal by 6%. That is corrected
by a gain on the concentration, recorded as `marginal_gain` rather than folded in
silently, **because it compensates for a shape mismatch rather than estimating
anything.**

A two-component per-holder mixture would remove the need for the gain. Not
attempted; the gain is honest and the cost is one parameter.

---

## 3. Merchant and category: swept, not fitted — and why Sparkov cannot anchor them

**Plan:** category clusters and `cnp_fraction` from Sparkov (Part C.2), merchant
popularity swept.

**What was there:** merchant selection was a uniform draw over the entire roster
for every card, and the Zipf popularity vector the builder computed was read by
nothing. 99.7% of transactions were a card's first at that merchant. Category was
worse — the archetype affinities, `build_profiles`, and `sample_category` were
all built, stored, and never called, so a transaction's category was whatever the
uniformly chosen merchant happened to be.

**The Sparkov gate, measured:**

| | Sparkov | reading |
|---|---|---|
| merchants visited per card | **572 of 693** (83%) | a population with no habits |
| top-1 merchant share | 0.67% | no favourite |
| per-card Simpson vs marginal | 1.05× | significant (+39σ) but tiny |
| per-card category share SD | 1–4 pts on 8–10 pt shares | near a shared curve |

The 1.05× excess is **geographic** — Sparkov places merchants near each customer,
which shrinks the reachable set — not loyalty. And IEEE-CIS carries no merchant
entity at all.

**So both are SWEPT**, joining geo and `amount|category` in the demotions already
recorded for the same underlying cause. Sparkov's figure is a **lower bound from
a generator**, recorded as such, not an estimate.

Sweep ranges reach down to the behaviour they replace (`merchant_loyalty` low =
0.0), so a claim holds across a range containing the defect as well as the
correction. **No noise floor is recorded for merchant loyalty** — none is
measurable, and a Sparkov-derived floor would launder a generator's artifact into
a measurement.

| | before | after |
|---|---|---|
| category concentration | 1.07× null (z 6) | 1.26× (z 20) |
| merchant concentration | 1.25× null (z 2) | 4.18× (z 58) |
| first at merchant | 0.997 | 0.861 |

**Draw order is inverted**: category first, merchant within category. Layering
loyalty onto merchant alone leaves category an uncontrolled by-product of which
merchants land in the preferred set.

---

## 4. Divergences inherited from earlier work

| Plan | Built | Why |
|---|---|---|
| Hawkes inter-arrival, sweep η | **Rejected**; gamma renewal under an AR(1) drifting rate | Failed its time-rescaling gate. The fitted decay meant the excitation term was absorbing the spread of per-entity rates rather than describing bursts. Recorded as a rejection in the artifact, with a session model rejected alongside it |
| Sparkov supplies geo | **Swept** | Its geography is an annulus: p10 distance 35 km, no local spend at all |
| Sparkov supplies `amount \| category` | **Swept** | Its per-category amounts are inverted — travel below grocery |
| Device fan-out (SINK 4) | Split into `bucket` (fingerprint) vs physical `Device` | The plan conflates them. A fingerprint composite over-merges unrelated devices; the two need separate entities, and only the bucket carries the measured heavy tail |
| Amount = lognormal + Pareto | That, **plus** a between/within decomposition | The plan has no concept of per-card levels |
| `ARCHETYPE_AMOUNT_SHIFT` | **Deleted** | Superseded by `ARCHETYPE_LEVEL_TILT`; an absolute shift double-counted, taking the spread from 0.379 to 0.740 against a target of 0.605 |

---

## 5. Known defects, recorded rather than absorbed

**~~Warm-start autocorrelation has the wrong sign.~~ Withdrawn — this was a
measurement error, not a generator defect.**

The claim was that generated inter-event autocorrelation ran at −0.096 against
a real +0.040, outside its noise floor, and that L3-P1 therefore failed.

That comparison was invalid. Lag-1 autocorrelation carries a small-sample bias
of about **−1/(n−1)**: with five gaps per entity it is computed against a mean
estimated from those same five gaps, which forces it negative. The generated
population averages ~7 events per card; the real target was measured at
`min_events=10` and dominated by far longer histories. The number being
compared was history length, not behaviour.

Real data shows the same effect, which settles it:

| history | real | generated |
|---|---|---|
| 5–9 events | **−0.131** | −0.165 |
| 10–19 | +0.007 | −0.043 |
| 20–49 | +0.056 | +0.023 |

Real cardholders with 5–9 events measure −0.13. The same real population at 50+
events measures +0.07. One process, opposite signs. And the same arrival model
that produces −0.180 at 7 events per card produces **+0.046** at 40 — identical
parameters, only the history length differs.

Matched on history length, the generator is mildly **under-clustered**, not
inverted. Burstiness matches at 0.84–1.17 across bands. **L3-P1 passes.**

This is the mistake `matched_by_event_count` exists to prevent, made in the one
place it was not applied. Per-band targets are now recorded in the artifact
(`autocorrelation_events_5_9` and siblings) so the pooled figure cannot be used
as a target for a population that does not match it, and two tests pin the
bias.

The residual under-clustering is real but small, and is the honest remaining
item: generated traffic sits below its band target in every band.

**Travel runs ~3 points above its configured category share.** Correct
behaviour, not a defect: the travel injector fires on under 1% of slots but emits
several authorisations each, all travel. People on a trip spend on travel. The
docstring warns against "fixing" it by constraining the pool, which would remove
the hard negative rather than the discrepancy.

**`ARCHETYPE_GEO_SCALE` is still dead.** `_local_distance` ignores archetype
entirely, so a traveller ranges no further from home than a homebody. A separate
defect, left for its own change.

**Merchant concentration was underpowered before loyalty existed.** With 2000
merchants and ~7 events per card, the Simpson index was estimated from accidental
collisions and its null spread ±0.21. The report says so rather than printing a
number that looks like evidence. It is meaningful now (null ±0.06).

---

## 6. Calibration status

`python -m fraudsim.calibration.pipeline` produces `artifacts/fitted_params.json`:
7 fitted models / 54 parameters, 9 swept, 13 noise floors, 3 targets.

Until this was first run, **all six tests covering `resolve()` skipped** — the
entire fitted/swept config merge, the provenance collision guard, and the
`allow_override` escape hatch had no executing coverage.

The naive-rule trip rate is **0.0647 against a target of 0.065**, no rule
carrying more than half. Three inconsistent acceptance bands for that one
quantity (target 0.065, a hardcoded ±0.02 in the renderer, and [0.02, 0.15] in
the test) are now one configured tolerance read by both — previously a
regression landing at 0.14 would print "off target" and pass the suite.

## 7. Verification

```
python -m pytest -q                                    # 336 passing
python -m fraudsim.calibration.pipeline                # regenerate the artifact
python -m fraudsim.analysis.cli entity-stats --judge Dataset
python -m fraudsim.engine.cli demo                     # trip rate + graph invariants
```
