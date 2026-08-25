# SINK 12 — Fidelity Validation Protocol

> Supersedes the fidelity section of `design.md:683-693`.
>
> The original plan was a single discriminator AUC against hand-picked cutoffs
> (0.50 excellent / 0.65 acceptable / 0.90 fix it). Those cutoffs are arbitrary,
> and a single aggregate number hides the failure mode that matters most:
> a generator can look near-real on marginals while being structurally wrong
> about entity and timing structure.
>
> This replaces it with three layers, each measured against an empirical noise
> floor rather than a chosen threshold.

---

## Why the old plan is insufficient

Two independent reasons, both established rather than speculative.

**1. The aggregate hides the structural failure.** One recent benchmark
(arXiv 2604.13125, single-author preprint — treat the specific figures as
"one benchmark found", not settled literature) reports a generator scoring
**0.798 TSTR AUROC** — nearly indistinguishable from real on downstream
utility — while being simultaneously **worst of all tested generators on
graph-motif fidelity**. Marginal realism and structural realism are not
correlated. Measuring one and declaring the other is the central trap.

**2. Our own datasets demonstrate it.** Sparkov is documented as "too
learnable": ensemble models score near-perfectly on it and transfer poorly to
real EU data. A discriminator run against Sparkov alone would reward us for
reproducing Sparkov's artifacts, not reality.

**Consequence for the metric:** SINK 1 (marginals) and SINK 4 (graph topology)
must be validated **separately**. Passing L1 says nothing about L3.

---

## The noise floor — the core idea

Every threshold in the old plan was a guess. Replace all of them with a
measured baseline:

```
                    metric_m( D_real , D_syn )
DR(G, m)  =  ─────────────────────────────────────
              metric_m( D_real,A , D_real,B )
```

where `D_real,A` / `D_real,B` is a **random 50/50 split of the real training
data**. The denominator is the noise floor: the irreducible divergence between
two halves of the same real dataset. `DR = 1.0` is indistinguishable from a
real split; `DR = 5.0` is five times more divergent than real sampling
variability.

Three properties make this the right anchor:

1. **No ground-truth labels needed** — any dataset produces its own baseline.
2. **Comparable across metric scales** — a `DR` of 30× on autocorrelation means
   the same thing as 30× on a Wasserstein distance, even though the raw values
   differ by orders of magnitude. This is what lets P1–P4 be averaged at all.
3. **50/50 is the most stringent split.** Sampling variability scales as
   `1/√n`, so a smaller half (e.g. 30%) has *higher* variance, a *larger*
   denominator, and therefore *lower* reported degradation. The equal split
   minimizes noise-floor variance and maximizes the ratio. **Reported values
   are upper bounds across split choices** — a useful sentence for the writeup.

Interpretation bands (ours, not the source's — state them as our reading):

| DR | Reading |
|---|---|
| **≈ 1** | indistinguishable from a real-vs-real split |
| **1 – 3** | close; name which sub-metric carries the gap |
| **3 – 10** | a real structural difference exists |
| **> 10** | the property is not being reproduced |

For reference, the source paper found all four tested row-independent
generators failing at **> 20×** on behavioral fidelity — treat those specific
figures as *one preprint's benchmark*, not settled literature.

> **Implementation note — entity-level splitting.** Split by **entity**, not by
> row. A random row split leaks the same card into both halves and collapses
> the floor toward zero, inflating every degradation ratio. Split on `card1`
> (IEEE-CIS) or `cc_num` (Sparkov), then assign that entity's full history to
> one side. This matters most for P1–P3, which are all defined per-entity.

### Composite score

```
BF(G) = K⁻¹ Σ_k DR(G, m_k)
```

Equal-weighted mean over the `K` evaluated sub-metrics. **Always report the
disaggregated sub-metrics alongside the composite** — the composite exists for
tracking across rounds, not for hiding a single catastrophic failure inside an
average.

---

## Layer 1 — Statistical fidelity (marginals and conditionals)

**Validates:** SINK 1, SINK 3.
**Question:** do single-variable and low-order conditional distributions match?

Following SDMetrics convention, as the source framework does:

| Metric | Applied to |
|---|---|
| **Wasserstein-1** | continuous: amount, inter-arrival, distance-from-home |
| **Jensen–Shannon divergence** | categorical: MCC cluster, hour-of-day, day-of-week, entry-mode |
| Mean absolute pairwise correlation difference | the full correlation matrix |
| Kolmogorov–Smirnov | optional secondary on continuous columns |

**Use W₁ as the primary continuous metric, not KS.** KS is driven by the point
of maximum CDF separation and is largely insensitive to tail mass — which is
exactly what the log-normal + Pareto finding says matters, and what the "legit
large purchase" hard negative depends on. Keeping W₁ primary also makes L1
commensurable with the L3 metrics, which are all W₁-based.

**Report conditionally, not just marginally.** A generator can match the
aggregate amount distribution while getting every per-MCC amount wrong. Report
KS on `amount | mcc_cluster` for each cluster, and on `hour | mcc_cluster`.
The design already requires these conditionals; validation has to follow them.

**Anchors** (from `data_research.md`, all MEASURED):
- mean credit transaction **$83**, debit **$70** (Fed Diary of Consumer Payment
  Choice, Oct 2024)
- ~**264 credit purchases/cardholder/year** (Fed NY, via Capital One Shopping —
  INDUSTRY ESTIMATE)
- amount distribution = **log-normal body + Pareto tail**, not pure log-normal
  (Ghosh et al., arXiv 0912.5420 — proxy population, see caveat)

Fitting a pure log-normal makes legitimate large purchases artificially rare,
which directly corrupts the "legit large purchase" hard negative in SINK 2.
Fit the mixture; sweep the tail index α ∈ [1.05, 3].

**Pass condition:** degradation ≤ 3 on amount, inter-arrival, and MCC; every
conditional slice reported, not only the pooled figure.

---

## Layer 2 — Downstream utility (TSTR / TRTS)

**Validates:** that the synthetic data is usable for the task it exists for.
**Question:** does a detector trained on synthetic transfer to real?

- **TSTR** — Train on Synthetic, Test on Real
- **TRTS** — Train on Real, Test on Synthetic
- **TRTR** — Train on Real, Test on Real ← *this is the noise floor*

Use the C6 baseline GBDT (`design.md:762`), identical hyperparameters across
all three runs.

```
utility_degradation = PR-AUC(TRTR) / PR-AUC(TSTR)
```

**Report PR-AUC, not ROC-AUC.** At a ~0.5% base rate ROC-AUC is misleading —
this is already the design's own position (`design.md:770`) and it applies to
fidelity measurement exactly as it does to detector evaluation.

**A high TSTR score is necessary but NOT sufficient.** This is precisely the
0.798-while-worst-on-motifs result above. Layer 2 passing does not license
skipping Layer 3.

---

## Layer 3 — Behavioral fidelity ⚠️ the one that matters

**Validates:** SINK 4 (graph topology), SINK 1's timing process, SINK 2 (burst
hard negatives).
**Question:** does cross-row and cross-entity structure match?

Four patterns, P1–P4, following the source framework. **Every sub-metric is
reported as a degradation ratio**, which is what makes a Wasserstein distance
in seconds comparable to a dimensionless autocorrelation gap.

### Notation

For entity `u` (card fingerprint or account), let `T_u = ⟨(t_1,x_1)…(t_n,x_n)⟩`
be its chronologically ordered transaction sequence. `F` = fraud entities,
`N` = non-fraud. `W₁(P,Q)` is the Wasserstein-1 distance between empirical
distributions.

**Our adaptation:** the source paper reports the fraud class (`F`) because it
benchmarks fraud-data generators. **We care primarily about the benign class
`N`** — our attacker is an RL policy, not a generator, so what needs validating
is that the *benign population* has realistic structure. Compute both; treat
`N` as the pass/fail criterion and `F` as a secondary check on scripted
attackers.

---

### P1 — Inter-event time distribution

Inter-event times for entity `u` with `n_u ≥ 2`:

```
Δ_u = ⟨δ_i⟩,  δ_i = t_{i+1} − t_i ≥ 0
IETD_c(D) = ⋃_{u∈C} Δ_u
```

Within-entity lag-1 autocorrelation — the burst-regularity fingerprint:

```
ρ_u = Corr( ⟨δ_i⟩_{i=1..n-2} , ⟨δ_{i+1}⟩_{i=1..n-2} )
```

**Metrics:**

```
B₁ᶜ  = W₁( IETD_c(D_real) , IETD_c(D_syn) )          distributional shift
B₁ᴬᶜ = | E_{u∈C}[ρ_u] − E_{u∈C̃}[ρ_u] |               autocorrelation collapse
```

`B₁ᴬᶜ` is the direct test of Proposition 2. Independently sampled timestamps
give `E[ρ] ≤ 0`; real sequences are positively autocorrelated. A large gap here
means the generator has no burst structure at all.

**Why it matters for us:** `auths_last_60s`, `auths_last_1h`,
`distinct_mcc_last_1h`, `time_since_last_auth_s` (`design.md:506-509`) are E3's
primary signal. If benign traffic has no burst structure, **every velocity
burst is fraud by construction** and E3's result is a generator artifact.
This also determines whether SINK 2's "legit velocity burst" hard negative is
real or decorative.

No published Hawkes characterization of consumer card sequences exists — fit it
from IEEE-CIS and sweep branching ratio η ∈ [0, 0.6].

---

### P2 — Burst structure and active lifetime

For gap threshold `δ ∈ {1 min, 5 min, 30 min}`, a **burst** is a maximal
contiguous subsequence of `T_u` with consecutive gaps ≤ δ. `B_u(δ)` is the
burst set, `L(b) = |b|` the burst length. **Active lifetime**
`AL_u = t_n − t_1` (0 for singletons).

**Metrics:**

```
B₂,ᴬᴸ    = W₁( {AL_u}_{u∈C} , {AL_ũ}_{u∈C̃} )
B₂,ᴮᴸ(δ) = W₁( {L(b)}_{b∈∪B_u(δ)} , {L(b̃)}_{b̃∈∪B_ũ(δ)} )
B̄₂,ᴮᴸ    = ⅓ Σ_δ B₂,ᴮᴸ(δ)                    averaged over the three thresholds
```

**Our reading:** the source motivates this by fraud's *short* active lifetime
under time pressure. For us the benign side carries the load — legitimate
accounts should show long lifetimes with scattered low-density activity, and a
realistic shopping-session burst length. This is the metric that says whether
SINK 2's session model is right.

---

### P3 — Shared-infrastructure graph motifs

Bipartite entity–attribute graph `G = (U, A, E)`: `U` entities, `A` shared
attributes (device IDs, IPs, payees), `(u,a) ∈ E` if entity `u` used attribute
`a`. Fan-out `FO(a) = |{u : (u,a) ∈ E}|`. The **entity projection** `G_U`
connects two entities sharing ≥ 1 attribute.

**Metrics:**

```
B₃,ᶠᴼ = W₁( {FO(a)}_{a∈A_real} , {FO(a)}_{a∈A_syn} )
B₃,ᶜᶜ = | CC(G_U^real) − CC(G_U^syn) |              global clustering coefficient
B₃,△  = | log( (|△(G_U^real)|+1) / (|△(G_U^syn)|+1) ) |    triangle count, log ratio
```

Note the metric shapes: **fan-out is a W₁ over the full degree distribution**
(not a variance-to-mean summary), clustering is an absolute difference in
`CC ∈ [0,1]`, and triangles use a **log ratio** because counts span orders of
magnitude. The `+1` prevents a divide-by-zero when the synthetic graph has no
triangles at all — which is the expected failure, not an edge case.

**Proposition 1** is what this tests: row-independent assignment of shared
attributes samples from a marginal, collapsing fan-out toward 1 for every
attribute node. The source's Figure 1 shows it directly — a real fraud ring
where four users share `d0` becomes eight users with eight distinct devices.
Variance-to-mean ≤ 1 remains a useful *diagnostic signature* of this collapse,
but `B₃,ᶠᴼ` is the metric.

**Why this is the highest-stakes sub-metric for us:** if benign fan-out is
trivially 1:1 while attacker fan-out is 40:1, **E5 scores ~1.0 AUC and the
result is meaningless.** That is the graph-level equivalent of the code-path
leak the design already warns about (`design.md:654`).

**Reference data problem.** No public dataset carries real benign
device–card–payee topology. Anchors in descending quality:

1. ~~**Amazon Fraud (Kaggle)**~~ — ❌ **REJECTED. Measured 2026-08-25 on
   `Dataset/fraud_amazon/Fraud_Data.csv` (151,112 rows).** It has `device_id`
   and `ip_address`, but is unusable as a *benign* anchor:

   - **Fraud rate rises monotonically with fan-out, 3% → 95%.** IP fan-out ≥ 2
     is 91% fraud. Sharing here is a synthetically stamped fraud signal, not an
     organic distribution — there is no ambiguous middle.
   - **The degree distribution is non-monotonic**: node counts run
     131,781 → 5,327 → 90 → **4** → 13 → 29 → 50 → … climbing to a hard cap at
     exactly **20**. Real degree distributions decay. This is two separately
     generated populations — benign users each assigned a unique device, plus
     fraud rings stamped in with fan-out drawn ~uniformly over 5–20.
   - **var/mean = 0.59 (device), 0.55 (IP)** — both < 1, the exact
     Proposition 1 signature. The benign half was generated row-independently.
   - **One row per user** (151,112 rows = 151,112 users), so no transaction
     sequences exist. Useless for P1/P2 as well — which explains the source
     paper's **K=1** on this dataset.

   Calibrating benign fan-out against this would be *worse than having no
   anchor*: we would tune to a generator that provably cannot produce realistic
   sharing, then measure our own generator against it and pass.

2. **IEEE-CIS `card1` entity sizes** — median 4, max 14,932 txns/entity. Weak:
   no device-sharing or payee edges.
3. **Household derivation** — Census/Pew household size and device-per-household
   → benign device→card fan-out. Derived, not measured, but defensible.
   **Now the primary path.**
4. **Sweep** the fan-out knee regardless.

> **General lesson.** Run the var/mean and monotonicity check on *any* candidate
> graph anchor before trusting it. Two lines of pandas, and it caught a dataset
> that a published preprint used for this exact metric.

---

### P4 — Velocity-rule trigger rates

A rule `r = (f, θ, w, op)` triggers at `(u,t)` if `f(T_u, t, w) op θ`.
Class-conditioned trigger rate `τᶜ(r,D) = |C|⁻¹ Σ_{u∈C} 1_r(u)`.

```
B₄ = |R|⁻¹ Σ_{r∈R} | τᶜ(r, D_real) − τᶜ(r, D_syn) |
```

**The canonical rule set R** (Bahnsen 2016 / Dal Pozzolo 2014):

| Rule | Feature | Op | Threshold | Window |
|---|---|---|---|---|
| R1 | Transaction count per card | > | 3 | 1 hour |
| R2 | Distinct merchants per card | > | 5 | 24 hours |
| R3 | Sum of amounts per card | > | $1,000 | 24 hours |
| R4 | Transaction count per card | > | 1 | Account age < 7 days |
| R5 | Distinct payment methods | > | 2 | 7 days |
| R6 | Max / median amount ratio | > | 3.0 | 30-day history |
| R7 | Failed transactions per card | > | 2 | 1 hour |
| R8 | Distinct IPs per card | > | 3 | 24 hours |

**Diff against our schema — we can compute all eight.** R7 and R8 were excluded
from the source's benchmark only because their 48-column training subset lacked
the columns; our `AUTH_ATTEMPT` schema has `declines_last_1h` (R7) and `ip_asn`
(R8), so we have full coverage. That is a small advantage worth stating.

Mapping to `design.md:499-511`: R1 → `auths_last_1h`; R2 →
`distinct_merchants_last_24h`; R3 → needs a 24h amount sum (**not currently in
the schema — add it**); R4 → `device_age_days` + `time_since_account_open_d`;
R5 → `entry_mode` / `device_n_cards_seen`; R6 → `amount_vs_card_median`;
R7 → `declines_last_1h`; R8 → `ip_asn`.

**R3 is the one genuine gap.** Add a rolling 24h amount sum to the event
builder.

**Why P4 is worth doing:** it is the only sub-metric that connects fidelity to
*operational miscalibration*. If benign trigger rates are wrong, the SINK 2
target of "5-8% of benign traffic trips a naive rule engine"
(`design.md:679`) is unverifiable — P4 against R is precisely how you measure
that number rather than asserting it.

---

## Which dataset plays which role

**Do not use one dataset for both fitting and judging.** Split by role, not by
random rows:

| Dataset | Role |
|---|---|
| **Sparkov** | *Structural source* — MCC semantics, geo, customer profiles. Real category names. Fit SINK 1's structure here. **Never the fidelity judge** (documented as "too learnable"). |
| **IEEE-CIS** | *Realism judge* — amount and velocity texture, entity sizes. Real Vesta transactions. Primary discriminator holdout and the L1/L2/L3 noise floor. |
| **PaySim** | *Transfer topology only* — mule/layering structure for E5. No device or payee sharing. Benign behavior is too simplistic to be ground truth. |
| **Amazon Fraud** | *Candidate fan-out anchor* — pending the verification above. |
| **CFPB** | *Text corpus* — E4 negatives + template-similarity reference. Not part of transaction fidelity. |

Splitting roles this way is stronger than a fit/holdout split within a single
dataset, because it tests transfer across sources rather than within one
source's artifacts.

---

## The field-coverage caveat — state it explicitly

Our events carry ~25 fields. IEEE-CIS is largely anonymized (V1–V339);
Sparkov has ~10 usable columns. **The discriminator can only run on the
intersection** — realistically amount, hour, MCC, geo distance, inter-arrival,
and entity-derived velocity.

> **An AUC of 0.55 on 5 fields is not the same claim as 0.55 on 25 fields.**

Report the field list alongside every fidelity number. A reader cannot
interpret the figure without it.

---

## Reporting template

```
FIDELITY REPORT — round N          class: BENIGN (N)     judge: IEEE-CIS

L1 statistical                       raw     floor      DR
  amount                    W1      0.031   0.019      1.6
  amount | mcc=5411         W1      0.044   0.022      2.0
  ...  one row per conditional slice
  inter-arrival             W1      0.088   0.021      4.2   <-- flagged
  mcc mix                   JSD     0.012   0.008      1.5
  hour-of-day               JSD     0.019   0.011      1.7
  pairwise corr matrix      MAE     0.041   0.026      1.6

L2 downstream
  TRTR PR-AUC 0.412   (noise floor)
  TSTR PR-AUC 0.371   DR 1.11
  TRTS PR-AUC 0.388

L3 behavioral                        raw     floor      DR
  P1  B_1^N   IET W1               412.7    18.9      21.8   <-- FAIL
      B_1^AC  lag-1 AC gap          0.29     0.03       9.7   <-- FAIL
  P2  B_2,AL  active lifetime W1    88.2     9.4        9.4   <-- FAIL
      B_2,BL  burst length W1        3.1     0.4        7.8   <-- FAIL
  P3  B_3,FO  fan-out W1             6.44    0.21      30.7   <-- FAIL
      B_3,CC  clustering diff        0.18    0.02       9.0   <-- FAIL
      B_3,tri triangle log-ratio     4.11    0.09      45.7   <-- FAIL
  P4  B_4     mean trigger-rate gap  0.14    0.02       7.0   <-- FAIL

  BF(G) composite = mean(DR) over K=8 = 17.6

FIELDS COVERED: amount, hour, mcc_cluster, geo_dist, inter_arrival,
                auths_last_1h, distinct_merchants_24h   (7 of 25)

HONEST SUMMARY: L1 and L2 pass (DR < 3). L3 fails across all four patterns.
P3 fan-out at 30.7x and triangles at 45.7x are the signature of Proposition 1:
device assignment is row-independent, so no attribute is shared by more than
one entity. P1's autocorrelation gap is the Proposition 2 signature: timestamps
are sampled independently within entity. Both were predicted analytically
before measurement.

INVALIDATED CLAIMS until fixed:
  - any E5 graph-level detection result  (P3)
  - any E3 velocity-based result         (P1, P2)
  - the "5-8% of benign trips naive rules" figure (P4)
```

The illustrative numbers above are the *expected shape of a first run* on a
naive generator, not measurements. Real values replace them.

**Report L3 failures rather than suppressing them.** A submission that names a
structural limitation and shows it was measured is stronger than one reporting
a single flattering aggregate — and L3 failures invalidate specific downstream
claims (E5 for 3a, E3 for 3b), so they have to be known before those results
are written up.

---

## Order of work

1. **Entity-level 50/50 split of IEEE-CIS → compute every noise floor first.**
   All eight L3 denominators plus the L1 denominators. Every later number is a
   ratio against these, so nothing else can be interpreted until this exists.
2. **Measure P1 and P2 on real data** — the target `E[ρ_u]`, burst-length and
   active-lifetime distributions the generator must hit. This is also the
   Hawkes fit.
3. **L1** on the fitted SINK 1 output.
4. **P1 + P2 on synthetic** — before building C5 in full.
5. **L2** once the C6 baseline exists (`design.md:762`).
6. **P3** once SINK 4's graph construction exists; **P4** once the event builder
   emits the R1–R8 features (note the R3 gap above).
7. Re-run all layers every round — fidelity drifts as the population is tuned.

**Steps 1, 2 and 4 come before C5 is written.** Proposition 2 guarantees the
failure analytically if timestamps are sampled independently within entity;
Proposition 1 does the same for row-independent attribute assignment. Both are
knowable before a single synthetic row exists. Measuring early costs hours;
discovering it after C5 is built costs a rewrite of the generator *and*
invalidates every E3/E5 number produced in the meantime.

---

## Provenance note

The P1–P4 taxonomy, the degradation ratio, the three-layer framing, and the
canonical rule set are taken from a **single-author, unreviewed arXiv preprint
(2604.13125, 2026)**. Handle its contents at two different confidence levels:

- **Propositions 1 and 2, and the metric definitions** — self-contained math.
  Verify them directly; they stand without the paper's authority, and the
  framework is worth adopting on its merits.
- **The specific benchmark figures** (>20× degradation across four generators;
  0.798 TSTR while worst on motifs) — cite as *"one recent benchmark reports…"*,
  never as an established result. One author, two datasets, specific
  hyperparameters, no peer review.

The rule set itself traces to **Bahnsen et al. (2016)** and **Dal Pozzolo et al.
(2014)**, which are the citable sources for R1–R8 — prefer those over the
preprint when referencing the rules.
