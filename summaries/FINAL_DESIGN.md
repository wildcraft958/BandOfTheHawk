# Closed-Loop GenAI Payment Fraud Red/Blue System
## Final Build Specification

> **See `RECONCILIATION.md` for what was actually built.** The simulation has
> diverged from this document in several places, each for a measured reason:
> the timing model was rejected on its goodness-of-fit gate, merchant and
> category concentration proved unfittable from the available sources, and the
> fidelity protocol gained a per-entity layer this document does not describe.
> Where the two conflict, that one is current.

> Supersedes `design.md` where they conflict. `design.md` remains the reference
> for the component contract, payload schemas, and the per-turn loop — none of
> that changes. This document records **what actually gets built**, after the
> data audit, literature review, and a viability pass that killed two things
> the original design assumed.
>
> **Priority stated by the user:** C1/C2 — the world model and simulator — must
> be excellent. Classification (C6) and the agent side (C3) are flexible and can
> be simplified without hurting the submission.

---

# PART 0 — WHAT CHANGED AND WHY

Five findings from the audit that alter the build. Everything else in
`design.md` stands.

| # | Finding | Effect |
|---|---|---|
| 1 | **V9 is not buildable.** `bust_out` needs a settlement/clawback layer that exists nowhere in the money model, and no dataset carries merchant-onboarding or settlement data. | **V9 cut from simulation.** Action space stays at 20. Zero-shot holdout becomes V7 + V10. |
| 2 | **Amazon Fraud rejected as a graph anchor.** var/mean 0.59, fraud rate 3%→95% monotonic with fan-out, one row per user. Synthetic ring-stamping, not organic sharing. | Removed from the dataset roster. |
| 3 | **IEEE-CIS *does* carry real benign device fan-out.** Benign-only var/mean **112.5**, monotonic decay, 51% of devices touch >1 card, max 829, and **fraud rate is flat across fan-out buckets** (3.5%–7.5%). | **SINK 4 upgrades from "derive + sweep" to MEASURED.** This was the weakest sink; it is now anchored. |
| 4 | **Sparkov cannot be the timing source.** 983 cards over 18 months at a mean of 1,319 txn/card — roughly 2.4 txn/card/day with near-uniform spacing. Real US average is ~0.7/day. Its inter-arrival structure is a generator artifact. | Sparkov is **structure only** (category, geo, demographics). All timing comes from IEEE-CIS. |
| 5 | **"Sparkov gives real MCC" was wrong.** It has **14 coarse categories**, not MCC codes. | The 8–12 cluster scheme is built from these 14 directly. No MCC mapping is possible or needed. |

---

# PART A — THE WORLD (C1) — build this well

## A.1 Nodes

Unchanged from `design.md:359-382` except as noted.

```
Cardholder  holder_id, home_geo(lat,long), city_pop, job, dob->age_band,
            tenure_days, archetype, voice_embedding, face_embedding

Card        card_id, holder_id, issue_date, credit_line, bin,
            median_txn_amount*, category_histogram*, status
            (* derived from realized behavior, never sampled independently)

Device      device_id, fingerprint_hash, first_seen_ts, os, browser,
            app_version, ip_asn, geo_estimate, is_emulator_flag,
            reputation_score

Account     account_id, holder_id, balance, open_date, kyc_level,
            kyc_passed_via

Merchant    merchant_id, category, avg_ticket, chargeback_rate,
            risk_tier, is_high_liquidity, popularity_rank

Payee       payee_id, account_id_target, first_added_ts, is_mule_flag
```

**`median_txn_amount` and `category_histogram` are derived, not sampled.**
Sampling them independently of realized transactions creates an internal
inconsistency that E3 can exploit as a leak. Compute them after the warm-start
window, then update online.

## A.2 Edges

```
holder  —owns→        card
card    —provisioned→ device   { bind_ts, bind_method, bind_trust, challenge_required }
account —added→       payee    { add_ts, add_method, cooling_off_until }
card    —transacts→   merchant { first_ts, count, total }
device  —used_by→     account  { count }
```

Fraud = create an edge you shouldn't be able to. Mitigation = delete it or
raise its cost. This symmetry is the core of the design and does not change.

## A.3 Warm start — do not skip

A cold graph at `t=0` makes `device_age_days`, `tenure_days`, and every
`prior_*` count degenerate through the burn-in, which silently corrupts E2/E3
training on the earliest events.

**Warm-start requirement, and it is cheaper than feared:** Wiefling et al.
(31.3M real logins) find a "stable setup" — novelty features stop firing — after
just **4–8 historical events per entity**. Generate ~10 events of history per
entity before `t=0`, plus:

- device age distribution seeded from a real spread, not all zero
- `tenure_days` seeded from a tenure distribution
- **activity sparsity**: Wiefling reports median 2 logins/user and 48.3% of
  users active less than monthly. A uniformly-active population makes every
  history feature unrealistically rich. Model the long tail of near-dormant
  entities explicitly.

## A.4 Graph topology — MEASURED (SINK 4)

Anchored to IEEE-CIS `train_identity` pseudo-device
(`DeviceInfo|id_30|id_31|id_33`) → distinct `card1`:

| Property | Benign-only value |
|---|---|
| var/mean | **112.5** (heavy-tailed) |
| mean fan-out | 8.83 |
| max | 829 |
| share of devices touching >1 card | **51.4%** |
| fraud rate across fan-out buckets | **flat, 3.5%–7.5%** |

The flat fraud rate is the important part: **device sharing is normal
behavior**, not a fraud signal. If our benign population is 1 device : 1 card,
E5 gets a free ~1.0 AUC and the graph result is meaningless.

**Caveats to state in the writeup:** identity coverage is only 20.1% of
transactions; the pseudo-device key is a fingerprint composite, not a true
device ID, so it over-merges common configurations. Treat the numbers as an
order-of-magnitude anchor for the *shape* (heavy tail, >50% sharing), and sweep
the exact knee.

---

# PART B — THE SIMULATOR (C2) — build this well

## B.1 Action space — 20 actions, FINAL

Unchanged from `design.md:474-493`. `onboard_merchant`, `self_auth`, and
`bust_out` are **not added**. See Part 0, finding 1.

The stage gate (`design.md:456-464`) is unchanged.

## B.2 Event schema — three additions to `AUTH_ATTEMPT`

`design.md:499-511` plus:

**(a) Compound-conditioned window aggregates** — Bahnsen 2016's `agg2`, the
single highest-lift feature class in their ablation (+201% → +252% savings).
Every existing aggregate in the schema keys on card alone; agg2's contribution
is a *second* matching criterion.

```
count and amount-sum, per card, within window w,
  conditioned additionally on: same category_cluster | same entry_mode | same geo_bucket
windows w ∈ {1h, 24h, 168h}          (Bahnsen used 1,3,6,12,18,24,72,168)
```

A 3-window × 3-criterion cross is 18 features and captures the effect.

**(b) von Mises time-periodicity feature** — Bahnsen's +13% on top of agg1+agg2.
Transaction time is **circular**: 23:00 and 01:00 are adjacent, not 22 hours
apart, and a raw `hour_of_day` bucket cannot represent that.

```
fit von Mises to the holder's transaction-time history over trailing tp >= 7 days
emit: is this transaction's time inside the alpha-confidence interval?
```

**The same fitted von Mises serves as the SINK 1 benign hour-of-day generator.**
Fit once, use twice — as the generator and as the detector feature.

**(c) Rolling 24h amount sum** — required by velocity rule R3 (below), currently
absent.

## B.3 Velocity rules — full 8-rule coverage

Canonical set from Bahnsen 2016 / Dal Pozzolo 2014. Used both as a naive-rule
baseline and as the SINK 12 P4 metric.

| Rule | Feature | Op | Threshold | Window | Our field |
|---|---|---|---|---|---|
| R1 | txn count per card | > | 3 | 1h | `auths_last_1h` |
| R2 | distinct merchants per card | > | 5 | 24h | `distinct_merchants_last_24h` |
| R3 | sum of amounts per card | > | $1,000 | 24h | **new (B.2c)** |
| R4 | txn count per card | > | 1 | account age <7d | `time_since_account_open_d` |
| R5 | distinct payment methods | > | 2 | 7d | `device_n_cards_seen` |
| R6 | max/median amount ratio | > | 3.0 | 30d | `amount_vs_card_median` |
| R7 | failed txns per card | > | 2 | 1h | `declines_last_1h` |
| R8 | distinct IPs per card | > | 3 | 24h | `ip_asn` |

The source benchmark could only run R1–R6 (their 48-column subset lacked the
rest). **We have all eight** — worth stating.

## B.4 Money model — deliberately shallow

`cash_out` is instantaneous. **No settlement delay, no clawback, no merchant
reserve.** This is why V9 was cut, and the limitation is stated rather than
worked around. Every other vertical's economics work without it.

---

# PART C — BENIGN POPULATION (C5)

## C.1 Archetypes — 6, fitted not typed

```
commuter, homebody, online_heavy, traveler, senior, business
```

Per archetype, **fitted from data, ~250 parameters total**:

| Parameter | Family | Source |
|---|---|---|
| `amount \| (archetype, category_cluster)` | **log-normal body + Pareto tail** | IEEE-CIS |
| `txn_per_day` | Negative Binomial (over-dispersed) | IEEE-CIS; sweep dispersion k ∈ [0.1,5] |
| `inter_arrival` | **Hawkes, exponential kernel** | IEEE-CIS; sweep η ∈ [0,0.6] |
| `category_histogram` | categorical over 8–12 clusters | Sparkov |
| `hour_of_day \| category` | von Mises mixture | IEEE-CIS + Sparkov |
| `day_of_week` | 7 weights | Sparkov |
| `geo` radius / trip model | log-normal km + trip process | Sparkov (has lat/long + merch_lat/long) |
| `cnp_fraction \| category` | Bernoulli | Sparkov (`*_net` vs `*_pos` categories) |
| `entry_mode` mix | categorical | assumption, sweep |

**Amount must be a log-normal + Pareto mixture, not pure log-normal.** Fitting
a pure log-normal makes legitimate large purchases artificially rare, which
directly breaks the "legit large purchase" hard negative. Sweep tail index
α ∈ [1.05, 3].

**Validation anchors:** mean credit txn **$83**, debit **$70** (Fed Diary of
Consumer Payment Choice, Oct 2024); ~**264** credit purchases/cardholder/year.

## C.2 Category clusters — built from Sparkov's 14

Sparkov has **14 categories, not MCC codes**. Cluster to ~8:

```
grocery      <- grocery_pos, grocery_net
fuel_transit <- gas_transport
dining       <- food_dining
retail       <- shopping_pos, home, kids_pets
online       <- shopping_net, misc_net
entertain    <- entertainment
health       <- health_fitness, personal_care
travel       <- travel
misc         <- misc_pos
```

`*_net` vs `*_pos` gives CNP directly — keep that split as a separate flag
rather than losing it in the merge.

## C.3 Hard negatives — target 5–8% tripping naive rules

Seven injectors (`design.md:669-679`). **Base rates are largely unavailable —
tune to hit the target and say so.** The target is a calibration objective, not
an input.

Measured with the R1–R8 rule set from B.3, which is what makes the 5–8% figure
verifiable rather than asserted.

**The one that matters most:** `P(password reset ∧ auth call ∧ device rebind
within 1h | legit)`. E2's `preceded_by_*` features key on exactly this chain.
**Genuinely unmeasured — no bank, regulator, or paper publishes it.** Sweep
**10⁻⁷ to 10⁻⁴ per user-day** and claim only *"separation holds across all
plausible benign co-occurrence rates."*

> ⚠️ The earlier claim that Wiefling et al. measured "new-device login ≈ 1–10%
> of logins" is **withdrawn** — that statistic is not in the paper; it reports
> threshold-dependent re-authentication rates. See `literature_addendum.md` §3.

---

# PART D — GENERATIVE LAYER (C4)

## D.1 Two modes, unchanged

- **Mode A (real generation)** — text verticals V4, V6, V10. Real LLM output,
  scored. The scores are *measured*, so the capability-tier circularity is much
  weaker here. **Prefer Mode A wherever possible.**
- **Mode B (parametric)** — voice and video. Sample the detector-facing score;
  generate no deepfake media. Defensible: *"we model the detector-facing signal,
  not the media."*

## D.2 Capability tiers — externally anchored, NOT invented

The original Beta(2,8)/Beta(5,5)/Beta(8,2) made the headline chart circular:
we picked both the tier distributions *and* the thresholds they are compared
against, so "success rises with capability" was arithmetic on our own
assumptions.

**Anchor to published benchmark ladders instead:**

| Modality | Tier 0 | Tier 1 | Tier 2 | Source |
|---|---|---|---|---|
| Voice (detector EER) | ~1–5% | ~10–16% | ~20–24% | ASVspoof 2015→2019→2021 DF→ASVspoof5 |
| Face (detector AUC) | 95–99% | 80–90% | 60–75% | deepfake AUC-by-generation; NIST FATE PAD (IR 8491) |
| Text (detector AUROC) | ~0.95+ | ~0.80 | 0.5–0.75 | RAID (Dugan et al., ACL 2024) |
| Document forensics | — | — | — | **no public benchmark; sweep AUC 0.6–0.99** |

**Ordinal only.** ASVspoof organizers explicitly warn cross-edition EERs are not
cardinally comparable (different databases and protocols). Use as ordered tiers,
never as absolute rates.

## D.3 Correlated score sampling — mandatory

Each tool returns multiple scores (~17 across all tools). **Sampling them
independently hands the defender a free decorrelation signal.**

Published deepfake-ensemble work reports pairwise correlation **ρ ≈ 0.90–0.92**
between complementary detector streams on the same artifact.

```
Gaussian copula
  within-modality  rho ~ 0.6-0.9
  cross-modality   rho ~ 0.1-0.3
  SWEEP rho
```

**Threshold that changes the design:** if any headline result flips between
ρ=0.3 and ρ=0.9, correlation is load-bearing and must be measured empirically
(run several open detectors on the same artifacts) rather than assumed.

## D.4 Reporting the ablation

Not a single point. **Sweep the threshold** and report
*"success rises with capability across every threshold in [0.75, 0.95]"*, and
measure **end-to-end episode success under a learned defender**, not the
single-gate pass rate. That makes the result include C6's adaptation instead of
being a restatement of `1 − BetaCDF(threshold)`.

---

# PART E — DEFENDER (C6) — flexible, keep simple

Per the stated priority, C6 is where to economize if time runs short.

## E.1 Order of work — unchanged and important

1. **Baseline single GBDT on the flat table.** Get a real number early. This is
   the fallback and the comparison point.
2. **Then MoE**, report the delta. If MoE loses, an honest ablation is still a
   result.

## E.2 Five experts

E1 Identity / E2 Binding / E3 Transaction / E4 Text / E5 Network, as
`design.md:737-745`. MoE is justified by **structurally different feature
spaces**, not by "many attack types" — `KYC_SUBMIT` has no amount,
`AUTH_ATTEMPT` has no liveness, `DISPUTE` has a 768-dim embedding appearing
nowhere else.

**E2 architecture suggestion** — Wiefling's calibrated likelihood ratio:

```
S_u(FV) = PROD_k [ p(FV^k) / p(FV^k | u, legit) ] * p(u|attack)/p(u|legit)
weights: IP 0.6 / ASN 0.3 / country 0.1
         UA-string ~0.54 / browser ~0.28 / OS ~0.17 / device-type 0.01
```

Calibrated on 31.3M real logins — a published design for exactly what E2 does.

## E.3 Metrics — PrecisionRank, not just AUC

Dal Pozzolo 2014's argument: for a *detection* task where investigators review
only the top-α alerts, AUC and accuracy are the wrong targets.

**Report:** PR-AUC (primary), **PrecisionRank at a fixed alert budget**,
recall @1% FPR and @0.1% FPR, per-vertical recall, zero-shot recall.
Simulate at ~**0.5% base rate**. Never 50/50.

## E.4 Retraining — the Forget strategy

Dal Pozzolo's winner was **Forget + EasyEnsemble + daily retrain**: keep **all**
past fraud, keep only the last `Kgen` days of genuine transactions.

This is the published form of what `design.md:813-815` does by instinct
(cumulative data + upweight `evaded_detection`). Adopt the asymmetry explicitly.

## E.5 Mitigation

Risk bands and graph mutations unchanged (`design.md:776-781`). Band
boundaries are free parameters — **grid-search them against the business cost
curve** (fraud loss + λ₁·friction + λ₂·review cost) rather than leaving them at
0.3/0.6/0.8/0.95. Sweep review cost **$2–25/case** and challenge abandonment
**5–30%**; both are vendor-sourced, not measured.

---

# PART F — ATTACKER (C3) — flexible

PPO, MLP 2×256, masked discrete head + continuous heads for amount/delay/
category/channel. Unchanged from `design.md:569-614`.

**All five anti-reward-hacking measures are mandatory** (`design.md:595-601`),
especially #5: log the top-10 action sequences each round and *read them*.
Absurd sequences mean a simulator bug, not a clever attacker.

Train against a **frozen** C6. Add the diversity bonus — attack diversity is a
scored criterion, so optimize it directly.

**If time is short:** scripted attackers (one per vertical) already give a
complete submittable system. RL is upside, per `design.md:950`.

---

# PART G — VERTICALS — 9 simulated, 1 documented

| # | Vertical | Status |
|---|---|---|
| V1 | Voice-clone provisioning | simulated |
| V2 | Deepfake synthetic onboarding | simulated |
| V3 | Agentic phishing → ATO | simulated |
| V4 | AI-drafted friendly fraud | simulated (**Mode A**) |
| V5 | Card testing at scale | simulated |
| V6 | Support-channel social engineering | simulated (**Mode A**) |
| V7 | SIM-swap → OTP interception | simulated — **zero-shot holdout** |
| V8 | Mule network layering | simulated |
| V9 | Merchant collusion / bust-out | ❌ **documented only** |
| V10 | Refund abuse via generated evidence | simulated (**Mode A**) — **zero-shot holdout** |

**V9 rationale, stated in the writeup:** requires a settlement/clawback model
the money layer does not implement, and no available dataset carries
merchant-onboarding or settlement data. Describing a mechanism we cannot
calibrate is more honest than simulating it with invented parameters.

**Zero-shot holdout is V7 + V10.** Both run on existing actions, both are
calibratable. V10 additionally tests text-vertical generalization, which V9
would not have.

---

# PART H — FIDELITY (SINK 12)

Full spec in `sink12_fidelity_protocol.md`. Summary:

**Everything is a degradation ratio against a measured noise floor:**

```
DR(G,m) = metric(D_real, D_syn) / metric(D_real_A, D_real_B)
```

`D_real_A/B` = random **50/50 entity-level** split of real data. DR = 1.0 is
indistinguishable from a real split. No hand-picked thresholds survive.

**Split by entity, never by row** — a row split leaks the same card into both
halves, collapsing the floor and inflating every ratio.

**Three layers, validated independently:**

- **L1 statistical** — W₁ on continuous (primary; KS is tail-blind), JSD on
  categorical, correlation-matrix MAE. Report **conditionals**, not just
  marginals.
- **L2 downstream** — TSTR / TRTS against TRTR as the floor. PR-AUC.
- **L3 behavioral** — P1 inter-event times + autocorrelation, P2 burst/lifetime,
  P3 graph motifs (fan-out W₁, clustering, triangle log-ratio), P4 velocity-rule
  trigger rates.

**L2 passing licenses nothing about L3.** One benchmark found a generator at
0.798 TSTR AUROC that was simultaneously *worst* on graph motifs.

**We validate the BENIGN class.** The source framework reports the fraud class
because it benchmarks fraud-data generators; our attacker is an RL policy, so
what needs validating is that the benign world is real.

**Two failure modes are provable in advance:**
- **Prop 1** — row-independent attribute assignment forces thin-tailed fan-out.
  Our IEEE-CIS anchor shows real var/mean = 112.5; independent sampling gives ≤1.
- **Prop 2** — independent timestamp sampling forces autocorrelation ≤ 0. Real
  sequences are positively autocorrelated.

Our archetype-generator design escapes both **by construction** — these are
PaySim-class agent generators, not CTGAN-class row samplers. The propositions
explain *why* the design was right to model inter-arrival as its own process.

---

# PART I — DATASET ROLES — final

| Dataset | Role | Never used for |
|---|---|---|
| **IEEE-CIS** | **The judge.** Noise floors, amount tail, inter-arrival/Hawkes fit, entity sequences, **device fan-out (SINK 4)** | structure/semantics (anonymized) |
| **Sparkov** | **Structure only.** 14 categories, geo (lat/long + merch_lat/long), demographics, CNP split | ⛔ **timing** (2.4 txn/card/day, near-uniform — artifact); ⛔ **fidelity judging** ("too learnable") |
| **CFPB** ✅ | E4 text negatives + template-similarity reference. **573,065 narratives, 109,905 fraud-issue.** Built. | transaction fidelity |
| **PaySim** | Transfer/layering topology for V8, E5 | benign behavior (too simplistic) |
| ~~Amazon Fraud~~ | ❌ **rejected** — synthetic ring-stamping | everything |

**Roles are split across sources, not fit/holdout within one source.** This
tests transfer across datasets rather than within one dataset's artifacts.

**CFPB date cut:** use **≤2022** for E4's human-text class. 2023+ is 63% of the
corpus (2025 alone is 161,649) and carries LLM-authorship contamination risk —
training a "human text" class on partly-machine text attacks the validity of the
text-detection result. Hold 2023+ out as a comparison probe.

**Calibration scope:** declare **US** — Sparkov geo and IEEE-CIS are US, and the
Fed anchors ($83/$70, 264 txn/yr, 17.6 bps) are US. The fraud-*type* mix is then
a labeled proxy from ECB/EBA and UK Finance, since no US equivalent to the UK
Finance breakdown exists. State this rather than silently mixing.

---

# PART J — PARAMETER PROVENANCE

Three artifacts, three provenance stories. Never blend them.

**1. FITTED (~250 params)** — SINK 1, 3, 4. Produced by a fitting script from
IEEE-CIS + Sparkov with a held-out split. Stored as a fitted-params artifact,
not a hand-written YAML.

**2. CITED (~50)** — SINK 5, 6, 10, 11. Every one carries source, population,
vintage, and confidence label. Vendor figures are labeled as such and swept, not
cited as measured.

**3. SWEPT** — everything genuinely unavailable. Claim robustness across the
range, never a point estimate:

| Parameter | Range |
|---|---|
| Benign ATO chain (reset ∧ call ∧ rebind within 1h) | 10⁻⁷ – 10⁻⁴ /user-day |
| Document-forensic detector AUC | 0.6 – 0.99 |
| Manual review cost | $2 – 25 /case |
| 3DS challenge / abandonment | 5–20% / 5–30% |
| AVS-CVV conditionals | CVV-match\|fraud 0.3–0.9 |
| NB dispersion k | 0.1 – 5 |
| Hawkes branching ratio η | 0 – 0.6 |
| Amount tail index α | 1.05 – 3 |
| Artifact score correlation ρ | 0.3 – 0.9 |
| Device fan-out knee | around the measured var/mean 112.5 |

**4. FREE** — reward weights, action costs, λ_detect, λ_burn, episode caps.
Tuned until the sim behaves. **Never presented as empirical.**

---

# PART K — BUILD ORDER

```
 0. Entity-level 50/50 split of IEEE-CIS; compute ALL noise floors     <- FIRST
 1. Measure P1/P2 on real data: E[rho_u], burst lengths, lifetimes
    Fit Hawkes (Ozaki MLE + Ogata O(n) recursion)
    Validate with time-rescaling K-S test against Exp(1)
 2. Fit SINK 1/3/4 params -> fitted-params artifact
 3. C1 graph + clock + event_log + WARM START
 4. C2 action API + stage gate + event builder (incl. B.2 additions)
 5. C5 benign population
 6. FIDELITY GATE: L1 + L3-P1/P2  <- must pass before proceeding
 7. C6 baseline single GBDT + the R1-R8 naive rule engine
 8. Scripted attackers, one per vertical (9 verticals) - NO RL yet
 9. C6 retrained on scripted attacks
10. FIDELITY GATE: L3-P3/P4 once the graph and rules are live
11. C3 RL replaces scripted attackers
12. C4 generative layer (Mode A for V4/V6/V10)
13. C6 MoE split + ablation vs baseline
14. C6 mitigation write-back to C1
15. C7 co-evolution rounds (5-10)
16. Web prototype
```

**Steps 0–9 give a complete, submittable system with zero RL.** Everything after
is upside.

**Steps 0 and 1 come before any generator code.** Propositions 1 and 2 predict
the failure modes analytically — measuring the real targets first costs hours;
discovering the failure after C5 exists costs a generator rewrite *and*
invalidates every E3/E5 number produced in between.

**Step 6 is a gate, not a checkpoint.** If L3-P1 fails, the inter-arrival model
is wrong and every velocity result downstream is an artifact. Fix before
proceeding.

---

# PART L — WHAT WE CLAIM, AND WHAT WE DON'T

**Entitled to claim:**
- Closed-loop co-evolution with a non-zero diagonal in the round matrix
- Attack success rises with generative capability, **across a swept threshold
  range**, measured end-to-end under a learned defender
- Benign fidelity by degradation ratio against a measured real-data noise floor,
  reported per-layer with the field list stated
- Zero-shot recall on V7 and V10, held out entirely
- MoE vs single-model ablation, honest either way
- Business-cost-optimized risk bands, not just AUC
- Detection robust across the full swept range of unmeasurable benign base rates

**NOT entitled to claim:**
- Any absolute per-tier attack success rate (tier distributions are ordinal)
- Any specific benign ATO-chain false-positive number (unmeasured)
- Any absolute fake-ID pass rate (no public benchmark)
- Any specific dollar ROI (review cost is vendor-sourced)
- Merchant-collusion detection (V9 is not simulated)
- Fidelity beyond the stated field intersection — *an AUC on 7 fields is not an
  AUC on 25*

**The honest-reporting rule:** L3 failures get reported, not suppressed. A
submission that names a structural limitation and shows it was measured is
stronger than one reporting a single flattering aggregate — and L3 failures
invalidate specific downstream claims (P3→E5, P1/P2→E3), so they must be known
before those results are written up.
