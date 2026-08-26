# Closed-Loop GenAI Payment Fraud Red/Blue System
## Design Specification

> The single design document for this system. Two references sit alongside it:
> **`data_research.md`** (the dataset audit behind every dataset claim here) and
> **`literature_addendum.md`** (published sources and their caveats).
>
> Parts A–G specify the simulation, which is built and measured. Parts H–J
> specify the defender, attacker and generative layer, which are defined as
> interfaces with null implementations and are the remaining work.
>
> Every number stated as measured is reproducible by the commands in Part L.

---

# PART 0 — THE GOVERNING PRINCIPLE

**A benign world that is too uniform hands the detector free signal, and every
downstream result becomes an artifact.**

This is the constraint the whole design serves. If synthetic cardholders are
interchangeable — same spending curve, same hours, same merchants — then a
classifier separating fraud from benign is separating *generated* from
*designed*, and the reported AUC measures the generator rather than the
detection problem.

The specific failure mode is subtle enough to state precisely, because it drives
several decisions later in this document:

**A marginal metric cannot see a per-entity error.** Pooling every transaction
and comparing the distribution discards which card produced each one. A
generator can match the pooled shape exactly — and every per-category slice of
it — while distributing that shape across cards entirely wrongly. Amounts drawn
from one shared curve produce a perfect marginal and a per-card spread of 0.38
against a real 0.566: every card identical up to sampling noise.

This matters because **the features a detector actually uses are per-entity
comparisons.** `amount_vs_median`, `is_first_txn_this_merchant`,
`within_usual_hours` all ask "is this unusual *for this card*". A generator that
is right marginally and wrong per-entity is wrong in exactly the place those
features read.

Four behavioural features therefore carry per-entity structure by design:

| Feature | Mechanism | Provenance |
|---|---|---|
| amount | per-card level, drawn once and kept | FITTED (IEEE-CIS) |
| hour | per-holder preferred hour and concentration | FITTED (IEEE-CIS) |
| category | per-card Dirichlet mix around its archetype | SWEPT |
| merchant | per-card preferred set, per category | SWEPT |

**Validation follows the same axis.** Fidelity is measured per-entity as well as
marginally, because a marginal-only protocol would certify the defective
generator described above. See Part G.

---

# PART A — PARAMETER PROVENANCE

Roughly 400 numbers parameterise this system. They are not one artifact and must
never be presented as one. Four provenance classes, and every configuration
field carries its class:

| Class | Meaning | If wrong |
|---|---|---|
| **FITTED** | estimated from real data by a fitting script against a held-out entity-level split | fidelity dies; discriminator AUC rises |
| **SWEPT** | genuinely unmeasurable; claimed across a range, never at a point | a claim rests on an invented number |
| **CITED** | published operating point | a reviewer asks where it came from |
| **FREE** | design choice, tuned until the simulation behaves | the sim behaves oddly and you notice |

Provenance is **enforced, not asserted**: config resolution raises if YAML tries
to overwrite a fitted value without an explicit override, and the resolved
config can report the origin of any field.

Current artifact: **8 fitted models / 63 parameters, 9 swept, 16 noise floors,
19 measured targets.**

**The swept set, with the reason each cannot be fitted:**

| Parameter | Value | Range | Why unmeasurable |
|---|---|---|---|
| `recovery_chain_probability` | 1e-5 | 1e-7 – 1e-4 | no bank, regulator or paper publishes P(reset ∧ call ∧ rebind within 1h \| legit) |
| `merchant_loyalty` | 0.55 | **0.0** – 0.9 | judge dataset has no merchant entity; taxonomy source has no habits |
| `merchant_preferred_set_mean` | 12 | 3 – 60 | unmeasured in both sources |
| `category_concentration` | 8 | 1 – 100 | taxonomy figure is a lower bound from a generator |
| `merchant_popularity_exponent` | 1.2 | 0.6 – 2.2 | taxonomy merchant traffic is nearly flat |
| `geo_home_radius_km` | 12 | 3 – 40 | taxonomy geography is an annulus with no local spend |
| `amount_by_category_spread` | 0.35 | 0.1 – 0.8 | taxonomy per-category amounts are inverted |
| `device_household_mean` / `max` | 2.1 / 8 | 1.5–4 / 4–15 | no source separates a physical device from a fingerprint |

**Sweep ranges are chosen to contain the null.** `merchant_loyalty` reaching 0.0
means the range includes uniform merchant selection, so a claim can be stated as
holding across a range that contains the degenerate case as well as the intended
one.

---

# PART B — DATASET ROLES

Full audit in `data_research.md`. The roles below are its conclusions.

| Dataset | Role | Never used for |
|---|---|---|
| **IEEE-CIS** | **The judge.** Noise floors, amount shape and tail, per-card amount heterogeneity, inter-arrival timing, per-entity hour structure, fingerprint fan-out | structure or semantics — it is anonymised |
| **Sparkov** | **Taxonomy only.** 14 categories → 8 clusters, card-not-present split, demographics | ⛔ timing, geography, per-category amounts, merchant loyalty |
| **CFPB** | Text negatives and template-similarity reference. 573k narratives, **≤2022 only** | transaction fidelity |
| **PaySim** | Transfer and layering topology for mule verticals | benign behaviour — too simplistic |
| ~~Amazon Fraud~~ | ❌ **rejected** | everything |

**Roles are split across sources, not fit/holdout within one source.** This tests
transfer across datasets rather than within one dataset's artifacts.

**Why Sparkov is taxonomy-only.** It is itself a generator, and its artifacts are
measurable:

| Measured | Value | Consequence |
|---|---|---|
| merchants visited per card | **572 of 693** (83% of roster) | no cardholder has habits |
| top-1 merchant share | 0.67% | no favourite merchant |
| per-card concentration vs marginal | 1.05× | significant (+39σ) but negligible, and *geographic* — merchants are placed near each customer |
| transactions per card per day | ~2.4, near-uniform spacing | vs. a real US average near 0.7 |
| geography | annulus, p10 = 35 km, no local spend | unusable |
| per-category amounts | inverted — travel below grocery | unusable |

Merchant and category concentration are therefore **SWEPT**, with Sparkov's
figure recorded as a lower bound from a generator rather than an estimate.

**Why Amazon Fraud is rejected:** var/mean 0.59, fraud rate rising 3%→95%
monotonically with fan-out, one row per user. That is synthetic ring-stamping,
not organic sharing, and using it as a graph anchor would stamp "sharing = fraud"
into the benign population.

**CFPB date cut:** 2023+ is 63% of the corpus and carries LLM-authorship
contamination risk. Training a "human text" class on partly-machine text would
attack the validity of the text-detection result, so it is held out as a
comparison probe instead.

**Calibration scope is declared US.** Sparkov geography and IEEE-CIS are US, and
the validation anchors are US. The fraud-*type* mix is a labelled proxy from
European and UK sources, since no US equivalent breakdown exists — stated rather
than silently mixed.

---

# PART C — THE PARAMETER SINKS

Parameters are organised by **where they are consumed**, not by which dataset
they came from. Thirteen sinks. The first five are the simulation and are built;
the remainder belong to components specified in Parts H–J.

## SINK 1 — Behavioural generators ✅ built

The largest consumer. Per-archetype behaviour across six archetypes
(`commuter, homebody, online_heavy, traveller, senior, business`).

| Quantity | Family | Provenance |
|---|---|---|
| amount level per card | lognormal body + truncated Pareto tail | FITTED |
| amount between/within split | variance decomposition | FITTED |
| inter-arrival | gamma renewal under an AR(1) drifting rate | FITTED |
| hour of day | two-level von Mises (holder within population) | FITTED |
| category mix per card | Dirichlet around archetype weights | SWEPT |
| merchant choice | preferred set drawn by popularity, per category | SWEPT |
| activity tier | four tiers with rate multipliers | FITTED shape |
| geo radius | lognormal km | SWEPT |
| entry mode mix | categorical | FREE |

**Amount** is a per-card level plus a spliced tail. The tail is **bounded** —
unbounded at the fitted index it reaches far past anything real, which widens
every distance while each summary statistic still looks correct. **51.6% of
amounts land on a whole currency unit**, because amounts are prices; a continuous
draw has none of that and the cents digit alone would separate it from real data.

Fitted: **between-card SD 0.566, within-card 0.706 — a 39%/61% variance split.**
A pooled fit collapses the first into the second and reaches the right total by
accident.

Archetype effects are **tilts that redistribute the fitted spread**, not shifts
added on top. The spread was measured across real cardholders who are already a
mixture of habits, so an absolute shift double-counts: layering one on took the
generated spread from 0.379 to 0.740 against a target near 0.6.

**Inter-arrival.** Two richer models were fitted and rejected against their
gates. A self-exciting (Hawkes) kernel failed time-rescaling — its fitted decay
meant the excitation term was absorbing the spread of per-entity rates rather
than describing bursts. A session model produced negative lag-1 autocorrelation
against a positive target. Decomposition explains both: raw consecutive gaps
correlate at about +0.06, but after dividing each by a local rolling median the
correlation vanishes. **Neighbouring gaps resemble each other because they were
drawn under a similar rate, not because one event triggered the next.** Both
rejected models describe short-range clustering, which is why neither could reach
it. The rejections are recorded in the artifact with their statistics.

**Hour of day** is two-level because one level cannot express the population.
Each holder draws a preferred hour about a population mode and concentrates
around it:

| Parameter | Fitted |
|---|---|
| population mode | 20.50 |
| κ between holders | 3.685 |
| κ within holder | 1.175 |
| `marginal_gain` | 1.066 |

For a von Mises within a von Mises, resultants multiply — so two parameters
would suffice for a homogeneous population. Real holders are not homogeneous and
are not von Mises about a single hour: they have a midday and an evening peak, so
a single component reproducing their per-holder concentration under-delivers the
population marginal by 6%. `marginal_gain` corrects that and is **recorded as a
separate quantity rather than folded into the concentration, because it
compensates for a shape mismatch rather than estimating anything.** A
two-component per-holder mixture would remove the need for it.

Activity is deliberately long-tailed: the judge dataset has a **median of two
transactions per entity and 39.5% with exactly one**. A uniformly active
population gives every history-dependent feature more to work with than it would
ever have in practice.

**Archetype category tilts are renormalised** so the archetype-weighted
population mix returns the configured mix exactly, while archetypes still differ
from one another by the same ratios. Tilting and normalising each archetype alone
drifts the population mix — travel worst, since the two archetypes favouring it
favour it strongly.

## SINK 2 — Hard-negative injectors ✅ built

Ordinary behaviour that looks suspicious. **Without these a false-positive rate
has nothing to measure**: every rule keyed on travel, bursts, unseen devices, or
recovery sequences stays silent forever against traffic that never does any of
those things, and the resulting rate says only that ordinary spending is
ordinary.

Eight kinds: `ordinary`, `large_purchase`, `gift_card`, `travel`, `session`,
`new_device`, `dispute`, `recovery`.

Each produces a **plan**, not a transaction, because most are not about the
amount. Travel changes *where*. A session changes *how many and how close
together*. A recovery is not a transaction at all.

**Calibration target: a small share of ordinary traffic trips the naive rule
engine.** This is an objective to tune toward, not an input. Currently
**0.0647 against a target of 0.065**, with no single rule carrying more than
half — measured with the R1–R8 set in Part E, which is what makes the figure
verifiable rather than asserted.

**Provenance is uneven and stated as such.** Only the dispute rate comes from a
published figure. Device replacement is a proxy from upgrade cycles; travel and
session rates are tuned toward the target.

**The one that matters most** is the recovery chain: P(password reset ∧ support
call ∧ device rebind within 1h | legitimate). The binding detector keys on
exactly that sequence, so if the benign rate is effectively zero the detector
gets a free perfect signal and the chained-takeover result is an artifact of the
generator. **It is genuinely unmeasured** — swept across three orders of
magnitude, and no claim may rest on a point estimate of it.

**Travel deliberately runs above its configured category share** by about three
points: the injector fires on under 1% of slots but emits several
authorisations each, all travel. People on a trip spend on travel. This is
behaviour, not drift, and must not be "corrected" by constraining the pool — that
would remove the hard negative rather than the discrepancy.

## SINK 3 — Population structure ✅ built

The static entities behaviour runs on, as distinct from the behaviour itself.

```
Cardholder   home_geo, household_id, archetype, activity_tier, age_band,
             tenure_days
Card         holder_id, issued_ts, credit_line, bin, status, median_amount*
Device       bucket_id, first_seen_ts, household_id, os_code, browser_code,
             app_version, ip_asn
Bucket       fingerprint composite shared across unrelated devices
Account      holder_id, balance, opened_ts, kyc_level
Merchant     category, avg_ticket, chargeback_rate, risk_tier,
             is_high_liquidity, is_card_not_present, popularity_rank
Payee        account_id, added_ts, is_mule
```

**`median_amount` is derived from realised transactions, never sampled.**
Sampling it independently of the transactions it summarises creates an internal
inconsistency a detector can exploit as a leak.

**Device and fingerprint bucket are separate entities.** A fingerprint composite
over-merges unrelated hardware sharing a configuration. The heavy-tailed sharing
measured in the judge dataset belongs to the *bucket*; a *physical device* stays
at household scale. Collapsing them would either make ordinary device sharing
look like fraud or make the measured tail vanish.

**Device ages are seeded from a distribution, not zero.** Every device at age 0
makes `device_age_days` useless through the burn-in.

## SINK 4 — Graph topology and warm start ✅ built

A graph detector needs a **benign** graph with realistic structure to contrast
against. If benign fan-out is trivially 1:1 while attacker fan-out is 40:1, the
graph result is a threshold check with a model wrapped around it.

Anchored to the judge dataset's benign rows:

| Property | Measured |
|---|---|
| var/mean | **112.5** |
| mean fan-out | 8.83 |
| maximum | 829 |
| share touching >1 card | **51.4%** |
| fraud rate across fan-out bands | **flat, 3.5–7.5%** |

**The flat fraud rate is the load-bearing part: device sharing is normal
behaviour, not a fraud signal.**

Independent per-row attribute assignment caps dispersion at 1.0 whatever
distribution it draws from, so a measured 112.5 rules it out by construction.
Fan-out is therefore generated **degree-first** — degrees drawn, then cards
matched onto them — rather than by sampling an attribute per row.

*Caveat for any writeup:* identity coverage is 20.1% of transactions and the key
is a fingerprint composite, so treat the numbers as an order-of-magnitude anchor
for the *shape* — heavy tail, majority sharing — and sweep the knee.

**Warm start.** A cold graph at t=0 leaves device age, tenure and every prior
count degenerate through the burn-in, silently corrupting the earliest events a
detector trains on. History is generated **by running the real step method**, not
written into state: derived fields are genuinely realised, rolling windows arrive
populated, and a bug here surfaces here. Warm-start rows are flagged so training
can exclude them, since they are feature-poorer by construction — the history
they would read is what they are creating.

Plans are expanded into timed events and **globally sorted before execution**.
Executing plan-by-plan pushes the monotonic clock past later slots, dragging
subsequent events off their intended time of day.

## SINK 5 — Control thresholds ✅ built, CITED

Operating points on the channel controls, cited rather than fitted:
`base_decline_rate` 0.03, `voice_similarity_threshold` 0.85,
`liveness_threshold` 0.90, `document_forensic_threshold` 0.75,
`step_up_challenge_rate` 0.12, `step_up_abandon_rate` 0.15,
`payee_cooling_off_hours` 24, `manual_review_cost` $8.

Vendor-sourced figures are labelled as such and swept rather than cited as
measured — manual review $2–25/case, challenge abandonment 5–30%.

## SINK 6 — Generative capability tiers ⏳ Part J

**The circularity trap, and the most important thing to get right in the
generative layer.** If capability tiers are invented *and* the thresholds they
are compared against are invented, then "attack success rises with generative
capability" is arithmetic on two of our own assumptions wearing the clothes of an
empirical finding.

Each tool returns multiple scores (~17 across all tools), and they must be
**correlated within a tier, not sampled independently** — independent sampling
hands the defender a free decorrelation signal.

Mitigations, specified in Part J.

## SINK 7 — Text scoring functions ⏳ Part J

Real generated text must be scored by something defined. CFPB supplies the
real-narrative reference corpus for template similarity, and the human-text class
for the text expert.

## SINK 8 — Attacker resource model ⏳ Part I

Credential quality tiers and what a quality level means operationally, credential
prices, attacker capital, per-action cost, cash-out haircut, and credential decay
as stolen cards get reported.

## SINK 9 — Reward weights ⏳ Part I

Pure FREE. Detection penalty, burn penalty, terminal bonus, per-action costs,
episode caps, diversity bonus. **Tuned until the simulation behaves, and never
presented as empirical.**

## SINK 10 — Risk bands ⏳ Part H

Band boundaries are free parameters, but should be **grid-searched against a
business cost curve** (fraud loss + friction + review cost) rather than left at
round numbers. That converts a FREE parameter into a CITED one and yields a
stronger claim.

## SINK 11 — Prevalence and mix ⏳ Part H

Base fraud rate **0.5%**, matching published card-fraud rates. Never 50/50.

Vertical mix directly determines per-vertical recall and training balance. Card
testing is enormously more common in reality than voice-clone provisioning; a
uniform simulation would let rare verticals dominate aggregate PR-AUC.

## SINK 12 — Fidelity validation ✅ built

Part G in full.

## SINK 13 — Zero-shot holdout ⏳ Part I

Two verticals held out entirely. Merchant collusion / bust-out is **excluded from
simulation** — see Part F.

---

# PART D — THE WORLD

## D.1 Edges

```
holder  —owns→        card
card    —provisioned→ device    { bind_ts, bind_method }
account —added→       payee     { add_ts, cooling_off_until }
card    —transacts→   merchant  { first_ts, count, total }
device  —used_by→     account
```

**Fraud creates an edge that should not exist. Mitigation deletes one or raises
its cost.** That symmetry is the core of the model and does not change.

## D.2 Category taxonomy

Sparkov's 14 categories cluster to 8: `grocery, fuel_transit, dining, retail,
online, entertainment, health, travel`. The `_net`/`_pos` suffix distinction
gives card-not-present directly and is kept as its own flag rather than lost in
the merge.

## D.3 Draw order: category, then merchant

A card draws its category first and its merchant **within** that category — the
order the behaviour actually has. Someone decides to buy petrol, then goes to
their usual station.

Preferred merchants are drawn **without replacement weighted by popularity**,
which reproduces population-level concentration and per-card loyalty from one
mechanism rather than two that could disagree. The roster is sized **per card,
not per category**: sized per category, a card accumulates a hundred regulars
against a median of five transactions and never revisits any of them, so the
habit exists on paper only.

---

# PART E — EVENTS AND ACTIONS

## E.1 The blindness rule

The event builder receives entity references and reads the graph. **Nothing in
its signature says who is acting.** The same builder produces the row whether an
ordinary holder or an attacker acted; any field distinguishing them would be a
shortcut a detector learns instead of learning behaviour.

Labels are stamped **after an episode closes**, never as events are written,
because at the moment of scoring nothing knows the answer. The scoring view
excludes the label structurally rather than by convention.

**Reading and writing are two calls, never one.** `build` reads state as it
stands; `commit` folds the event into state afterwards. Collapsing them makes
every count include the event describing it — an off-by-one that leaves every
number looking reasonable while every feature is wrong.

## E.2 Event schema

Fourteen event types. `AUTH_ATTEMPT` carries identity and context, device
posture, velocity aggregates, per-entity history comparisons, time, and tenure —
plus **compound window aggregates** keyed on card × {category, entry mode, risk
tier} × {1h, 24h, 168h}, which are the highest-lift feature class in the
published ablations because they add a second matching criterion to aggregates
that otherwise key on card alone.

Fields requiring history are **optional**. A card with no past has no median, and
reporting zero would be a claim rather than an absence.

`BindingEvent` covers device binds, payee additions and credential resets, with
the `preceded_by` chain fields that account-takeover detection keys on. A reset,
a call and a binding are unremarkable alone; the sequence carries the signal.

## E.3 Action space — 20 actions

Four stages, gated:

```
NONE       phish_holder, buy_creds, make_synth_id, harvest_voice, harvest_face
ACQUIRED   call_ivr_provision, submit_kyc, add_device_selfserve, sim_swap,
           reset_password, add_payee, open_ticket, escalate_limit
BOUND      attempt_auth, complete_3ds, transfer_p2p, request_refund
MONETIZED  file_dispute, cash_out, launder_chain
```

The stage gate produces the legal-action mask. Every action has a resolver that
does what it claims or fails.

**The money model is deliberately shallow.** Cash-out is instantaneous: no
settlement delay, no clawback, no merchant reserve. This is a stated limitation
rather than a hidden one, and it is why merchant collusion is excluded (Part F).

## E.4 The naive rule baseline

Eight velocity rules, used both as a baseline and as the instrument that makes
the hard-negative calibration verifiable.

| Rule | Reads | Threshold |
|---|---|---|
| R1 | `auths_last_1h` | > 3 |
| R2 | `distinct_merchants_24h` | > 5 |
| R3 | `amount_sum_24h` | > $1,000 |
| R4 | `account_age_days` ∧ `auths_last_24h` | < 7d ∧ > 3 |
| R5 | `card_n_devices` | > 4 |
| R6 | `amount_vs_median` | > 6.0 |
| R7 | `declines_last_1h` | > 2 |
| R8 | `distinct_ips_24h` | > 3 |

The published benchmark could run only six of these; the schema here supports all
eight.

**R5 reads devices-per-card, not cards-per-device.** The latter is the shared
fingerprint fan-out, which is heavy-tailed among ordinary holders **by design**,
so a rule keyed on it fires on half of all legitimate traffic and measures the
generator rather than the behaviour.

---

# PART F — VERTICALS

Nine simulated, one documented.

| Vertical | Status |
|---|---|
| Voice-clone provisioning | simulated |
| Deepfake synthetic onboarding | simulated |
| Agentic phishing → account takeover | simulated |
| AI-drafted friendly fraud | simulated, real text |
| Card testing at scale | simulated |
| Support-channel social engineering | simulated, real text |
| SIM-swap → OTP interception | simulated — **zero-shot holdout** |
| Mule-network layering | simulated |
| Refund abuse via generated evidence | simulated, real text — **zero-shot holdout** |
| Merchant collusion / bust-out | ❌ **documented only** |

**Merchant collusion is excluded deliberately.** It requires a settlement and
clawback layer the money model does not implement, and no available dataset
carries merchant-onboarding or settlement data. **Describing a mechanism we
cannot calibrate would be worse than declining to simulate it**, and the
exclusion is stated in the writeup rather than hidden.

The zero-shot pair tests generalisation to an unseen attack *and* — through the
refund vertical — generalisation across the text modality.

---

# PART G — FIDELITY VALIDATION

## G.1 Everything is a ratio against a measured floor

```
degradation = distance(real, generated) / distance(real_A, real_B)
```

`real_A/B` is a random **entity-level** 50/50 split of real data — the
irreducible divergence between two samples of the same thing. A ratio of 1.0 is
indistinguishable from a real split. **No hand-picked thresholds survive this
construction**, which is the point.

**Split by entity, never by row.** A row split leaks the same card into both
halves, collapsing the floor and inflating every ratio.

**Wasserstein-1 is primary for continuous quantities, not Kolmogorov-Smirnov.**
KS keys on the single point of maximum CDF separation and is largely blind to the
tail — which is exactly where the amount distribution matters and where the
legitimate large purchase lives.

**Hours use a circular Wasserstein.** The linear form calls 23:30 and 00:30
twenty-three hours apart and returns it without complaint.

## G.2 Three layers

**L1 — statistical.** Marginals and conditionals: W₁ on continuous, JSD on
categorical, correlation-matrix error. Reported conditionally, not only
marginally.

**L1-E — per-entity.** The axis Part 0 describes, and the one a marginal-only
protocol cannot supply. Per-entity amount spread, circular hour concentration
within and between entities, categorical concentration against a shuffled null.

**Every per-entity statistic requires a small-sample correction**, because an
entity seen *k* times looks concentrated purely for having been seen *k* times —
one event has a circular resultant of exactly 1 and a concentration index of
exactly 1 whatever produced it:

| Statistic | Bias | Correction |
|---|---|---|
| between-entity variance | `within²/k` | subtract the sampling term |
| circular resultant | `≈1/k` | `√((n·R²−1)/(n−1))` |
| categorical concentration | `≈(1−S)/n` | `Σnᵢ(nᵢ−1)/n(n−1)` |
| lag-1 autocorrelation | `≈−1/(n−1)` | compare within event-count bands |

Uncorrected estimates **drift with the cutoff**: the hour resultant reads 0.565
at a five-event cutoff and 0.496 at twenty, where the corrected value holds near
0.48 throughout. A generator tuned against an uncorrected estimate is wrong at
every cutoff but the one it was tuned at.

**Comparisons are matched on history length.** Real and generated populations
have different event-count distributions, and every statistic here varies with
event count, so an unmatched comparison reports a difference in census as a
difference in behaviour.

This is not a refinement. Lag-1 autocorrelation measured without matching reads
**−0.13 for real cardholders with 5–9 events and +0.07 for the same real
population at 50+** — same process, opposite sign, nothing but history length
between them. Per-band targets are recorded in the artifact for exactly this
reason.

**L2 — downstream utility.** Train-on-synthetic/test-on-real against
train-on-real as the floor, scored by PR-AUC.

**L3 — behavioural.** Inter-event autocorrelation and burstiness, burst structure
and lifetime, graph motifs, and velocity-rule trigger rates.

**L2 passing licenses nothing about L3.** A generator can reach strong downstream
utility while being simultaneously worst on graph structure.

**The benign class is what gets validated.** The attacker is a policy, not a
generator, so what needs validating is that the benign world is real.

## G.3 Two failure modes provable in advance

**Independent attribute assignment forces thin-tailed fan-out** — dispersion at
most 1.0 against a measured 112.5. **Independent timestamp sampling forces
non-positive autocorrelation** against a positive real target.

The design escapes both **by construction**, being an agent generator rather than
a row sampler. These propositions are why inter-arrival is modelled as its own
per-entity process and why fan-out is generated degree-first.

## G.4 Current state

| Statistic | Generated | Real | Ratio |
|---|---|---|---|
| hour marginal R | 0.416 | 0.434 | 0.96 |
| hour within-entity R | 0.459 | 0.492 | 0.93 |
| **hour between-entity R** | **0.715** | **0.792** | **0.90** |
| category concentration | 1.24× null (z 18) | — | swept |
| merchant concentration | 3.77× null (z 49) | — | swept |
| naive rule trip rate | 0.0647 | target 0.065 | — |
| fan-out mean | 8.06 | 8.14 | 0.99 |

The between-entity term is the one no marginal metric could report: a population
where everyone shops in the evening and one where each holder keeps tight but
unrelated hours share a marginal exactly.

**The reporting harness names its own blind spots.** Where a statistic has no
power — too many possible values against too little history per entity — it says
so rather than printing a number that looks like evidence.

## G.5 Honest remainders

- Generated traffic sits **below its band target for clustering in every band**:
  the world is somewhat less bursty than the real one.
- Archetype geographic range is not yet differentiated — a traveller ranges no
  further from home than a homebody.
- **Fidelity claims cover the stated field intersection only.** An AUC on seven
  fields is not an AUC on twenty-five.

**The honest-reporting rule:** L3 failures get reported, not suppressed. A
submission naming a structural limitation and showing it was measured is stronger
than one reporting a single flattering aggregate — and such failures invalidate
specific downstream claims, so they must be known before those results are
written up.

---

# PART H — DEFENDER

> Specified, not built. Exists as a scoring interface with a null implementation.

## H.1 Order of work

**A single gradient-boosted tree on the flat table first.** This is the fallback
and the comparison point, and it produces a real number early. **Then** a mixture
of experts, reporting the delta. If the mixture loses, an honest ablation is
still a result.

## H.2 Five experts

Identity, binding, transaction, text, network.

The mixture is justified by **structurally different feature spaces**, not by
"many attack types": a KYC submission has no amount, an authorisation has no
liveness score, a dispute carries a text embedding appearing nowhere else.

The binding expert should use a **calibrated likelihood ratio** over feature
vectors, a published design calibrated on 31.3M real logins, for exactly the job
it does here.

## H.3 Metrics

**PR-AUC primary**, plus precision at a fixed alert budget and recall at 0.1% and
1% FPR, per-vertical recall, and zero-shot recall. For a detection task where
investigators review only the top alerts, plain AUC and accuracy are the wrong
targets. **Simulate at a 0.5% base rate. Never 50/50.**

## H.4 Retraining

Keep **all** past fraud; keep only recent genuine transactions. Adopt the
asymmetry explicitly rather than approximating it by upweighting.

## H.5 Mitigation

Risk bands drive graph write-back — the mitigation half of the edge symmetry in
Part D.1. Band boundaries grid-searched against the business cost curve.

## H.6 The open question

**Whether the per-entity features from Part 0 carry signal is not yet known.**
`is_first_txn_this_merchant` went from firing on 0.2% of events to 15%. Whether
that helps a classifier separate fraud is exactly what the baseline exists to
answer, and it should be answered before the attacker is built on top.

---

# PART I — ATTACKER

> Specified, not built. Exists as a policy interface with a null implementation.
> The action layer it drives is built and complete.

## I.1 Scripted first

**One scripted policy per vertical, no reinforcement learning.** These are
sequences of actions that already exist and resolve. Scripted attackers alone
give a complete, submittable closed loop; RL is upside.

## I.2 If RL

Masked discrete head over the legal-action mask, continuous heads for amount and
delay. Trained against a **frozen** defender. **Diversity bonus included** —
attack diversity is a scored property and should be optimised directly rather
than hoped for.

## I.3 Anti-reward-hacking is mandatory

Especially this: **log the top-10 action sequences each round and read them.**
Absurd sequences mean a simulator bug, not a clever attacker. Every economic
exploit found this way is a hole a real attacker would also find, so they are
worth fixing rather than patching around.

## I.4 The observation boundary

A policy sees stage, a legal-action mask, and a feature mapping — **never the
graph.** An attacker that could read the graph would be solving a different
problem from the one a real attacker faces, and any resulting success rate would
describe that different problem.

---

# PART J — GENERATIVE LAYER

> Specified, not built. Exists as an artifact interface with a null
> implementation.

## J.1 Two modes

**Mode A — real generation** for the text verticals. Real model output, scored.
The scores are *measured*, so the capability-tier circularity is much weaker
here. **Prefer Mode A wherever possible.**

**Mode B — parametric** for voice and video. Sample the detector-facing score;
generate no deepfake media. Defensible as modelling the detector-facing signal
rather than the media.

## J.2 Breaking the circularity

Capability tiers must be anchored to **published benchmark ladders, used as
ordinal only** — cross-edition detector error rates are not cardinally
comparable, and the benchmark organisers say so.

Three requirements on the headline claim:

1. **Sweep the threshold.** Report "success rises with capability across every
   threshold in the range", not a point estimate.
2. **Measure end-to-end episode success under a learned defender**, not a
   single-gate pass rate. That makes the result include the defender's adaptation
   instead of restating one minus a cumulative distribution function.
3. **Correlate the scores.** Published ensemble work reports pairwise correlation
   around 0.9 between complementary detector streams on the same artifact.
   Sample from a copula and sweep the correlation.

**The threshold that changes the design:** if any headline result flips between
low and high correlation, correlation is load-bearing and must be measured
empirically rather than assumed.

---

# PART K — WHAT MAY AND MAY NOT BE CLAIMED

**Supported:**
- Benign fidelity as a degradation ratio against a measured real-data noise
  floor, per layer, with the field list stated
- Per-entity structure in amount, hour, category and merchant, each with its
  small-sample correction, each compared within matched history bands
- Graph fan-out reproducing a measured heavy tail that independent per-row
  assignment provably cannot produce
- Robustness across the full swept range of unmeasurable benign base rates
- Closed-loop co-evolution, once Parts H–I are built
- Zero-shot recall on the two held-out verticals
- Mixture-versus-single ablation, honest either way

**Not supported:**
- Any absolute merchant-loyalty or category-concentration figure — both swept
- Any specific benign account-takeover chain false-positive rate — unmeasured,
  swept across three orders of magnitude
- Any absolute per-tier attack success rate — tiers are ordinal
- Merchant collusion detection — not simulated
- Any specific dollar ROI — review cost is vendor-sourced
- Fidelity beyond the stated field intersection

---

# PART L — REPRODUCING THE NUMBERS

```bash
python -m pytest -q                                    # 338 passing, no skips
python -m fraudsim.calibration.pipeline                # fit and write the artifact
python -m fraudsim.analysis.cli entity-stats --judge Dataset
python -m fraudsim.analysis.cli compare                # fan-out vs measurement
python -m fraudsim.engine.cli demo                     # warm start + trip rate
python -m fraudsim.rules.cli describe                  # the eight rules
python -m fraudsim.config.cli                          # config + provenance table
```

Datasets are not committed. `Dataset/` requires `ieee-fraud-detection/`,
`sparkov/`, `paysim/` and `complaints/` as parquet; install the calibration
extra for the fitting tier.

**Build order, for reference.** Noise floors and behavioural targets are measured
*before* any generator code, because the two propositions in G.3 predict the
failure modes analytically: measuring the real targets first costs hours, while
discovering the failure after the population exists costs a generator rewrite and
invalidates every number produced in between. The fidelity gate is a gate, not a
checkpoint.
