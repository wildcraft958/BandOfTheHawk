I've read the whole doc. Let me give you the sink-by-sink data map — organized by *where the number gets consumed*, not by dataset.

First, a correction to the other agent's framing: **"~30-40 numbers in one YAML" is wrong by roughly an order of magnitude.** Just C5's archetype table alone is 6 archetypes × ~7 parameters = 42 numbers, and that's one of about fourteen distinct sinks. The real count is 300-500 parameters. That matters because it changes your build plan — you can't hand-tune 400 numbers, so most of them have to be *fitted programmatically* from the datasets, not typed into a config file.

---

# The bifurcation that actually matters

Not "data / benchmarks / design choices." The real axis is **what breaks if the number is wrong:**

| Class | Wrong number causes | Must come from |
|---|---|---|
| **A. Distributional** | Your discriminator AUC hits 0.9 and the Fidelity metric dies | Fitted from real data |
| **B. Structural** | Graph features (E5, velocity) are meaningless | Fitted or derived from real data |
| **C. Operating points** | Nothing breaks; a reviewer asks "where'd 0.85 come from?" | Cited literature |
| **D. Free parameters** | Sim behaves stupidly, you notice and retune | Pure design, tuned in-loop |
| **E. Adversarially load-bearing** | Your headline claim is circular | ⚠️ see the trap section |

Class E is the one nobody warned you about. I'll get to it.

---

# SINK 1 — C5 archetype generators (`design.md:658-665`)

This is the single biggest data consumer. Six archetypes, each needs a full parameter block:

**Per archetype (×6):**
- `txn_per_day` — not a scalar. Needs a **rate + dispersion**: Negative Binomial or Poisson-Gamma, because "2-4/day" hides whether that's 3±0.5 or 3±3. Traveler is explicitly bursty, so it needs a Hawkes process or a burst-probability + burst-size pair instead.
- `amount | mcc` — **conditional, not marginal.** The doc writes `LogNormal(2.8, 0.5)` for commuter as if amount is archetype-level. It isn't. A commuter's grocery amount and their gas amount are different distributions. You need μ,σ **per (archetype, mcc_cluster)** — that's 6 × ~8 clusters × 2 params = ~96 numbers. This is why hand-tuning fails.
- `mcc_histogram` — categorical over ~8-12 clusters (not raw 300 MCCs; cluster them or E3's `distinct_mcc_last_1h` feature gets sparse).
- `hour_of_day` — 24-bin multinomial, or a 2-3 component von Mises mixture (better: it's cyclic and you get smooth interpolation). Must be **conditional on MCC** — nobody buys groceries at 3am but online purchases happen at 3am.
- `day_of_week` — 7 weights.
- `geo` — home radius (LogNormal on km), plus P(out-of-radius) and the out-of-radius distance tail. Traveler needs a separate trip model entirely: trip frequency, trip duration, trip distance.
- `cnp_fraction` — conditional on MCC again.
- `entry_mode` mix — chip/contactless/cnp/token. `AUTH_ATTEMPT` has this field (`design.md:503`) and it's a strong feature; if all your benign traffic is one mode, E3 learns entry_mode as a leak.
- `inter_arrival` — you need this *separately* from txn/day, because `time_since_last_auth_s` and `auths_last_60s` are direct features (`design.md:506-509`). A daily rate with uniform spread produces zero velocity bursts and E3 gets a free ride.

**Source:** Sparkov gives you archetype-ish customer profiles + real MCC names + geo (it's generated from real demographic distributions). IEEE-CIS gives amount shape and the velocity/timing texture but has anonymized categoricals — you can't get MCC semantics from it. Use Sparkov for structure, IEEE-CIS for shape.

**Realistic count: ~250 parameters.** Fit them with a script, store as a fitted-params artifact, not a hand-written YAML.

---

# SINK 2 — C5 hard-negative injectors (`design.md:669-679`)

This is a **separate sink** and the other agent didn't mention it at all. Each of the 7 hard negatives needs its own parameter set, and the target "5-8% of benign trips a naive rule engine" is a *calibration target you tune against*, not an input.

Per injector you need a **base rate** (how often does a real person do this?) and a **shape**:

| Injector | Needs |
|---|---|
| Legit new device | P(device replacement / holder / year) — ~0.5-0.7 from phone upgrade cycles; plus the post-bind txn burst pattern (people test a new phone) |
| Legit travel | trip rate/year by archetype, trip duration, destination distance dist, foreign MCC mix, amount inflation factor during travel |
| Legit large purchase | P(txn > 10× median) — this is just the tail of your amount dist, so it's **free if SINK 1 is fitted right**, and *wrong* if you fit a LogNormal to something that's actually heavy-tailed |
| Legit dispute | dispute rate per txn (~0.05-0.1% of txns; chargeback rates are published by card networks), + reason-code distribution |
| Legit account recovery | P(password reset/account/year), and critically **P(reset → IVR → rebind within 1h \| legit)** — this is the exact chain E2's `preceded_by_*` features key on (`design.md:547`). If this joint probability is ~0 in benign traffic, E2 gets a perfect free signal and your whole ATO detection result is fake |
| Legit gift-card buy | P(high-liquidity MCC purchase) by archetype |
| Legit velocity burst | P(shopping session), session length dist, intra-session inter-arrival |

**The one that matters most:** the legit-recovery chain probability. It's the difference between "we detect chained ATO" being a real result and being an artifact of your generator.

**Source:** mostly public industry stats + your own judgment, and honestly this is where you *tune to hit the 5-8% target* — declare that openly.

---

# SINK 3 — C1 population structure (`design.md:359-382`)

Distinct from SINK 1: SINK 1 is *behavior*, this is *the static entities behavior runs on*. Node-by-node:

**Cardholder:** age band dist, income band dist, `tenure_days` dist (matters — `time_since_account_open_d` is an E2 feature), archetype mixture weights (what % of the population is each archetype — this drives your whole aggregate distribution), home geo spatial distribution (clustered, not uniform — uniform geo makes `geo_distance_from_home_km` behave nothing like reality).

**Card:** cards per holder (~2.3 US avg, needs the full dist not the mean), `credit_line` dist **conditional on income band**, credit_line-to-spend ratio (drives `escalate_limit` realism), BIN tier mix, `median_txn_amount` — note this is *derived* from SINK 1, not independent; if you sample it independently you create an inconsistency E3 can exploit.

**Device:** devices per holder, `first_seen_ts` / device age distribution at simulation start (**critical**: if every device is age 0 at t=0, `device_age_days` is useless for the first N days of sim — you need a warm-start age distribution), OS/browser/app_version mix, ASN distribution, emulator base rate in benign population (should be small but non-zero — some legit users run emulators).

**Account:** balance dist, KYC level mix, `kyc_passed_via` mix.

**Merchant:** the merchant population's MCC distribution (different from *transaction* MCC distribution — power law, few merchants get most volume), `avg_ticket` per MCC, `chargeback_rate` per MCC (published by networks), `risk_tier`, high-liquidity flag rate. Plus the **merchant popularity distribution** (Zipf) — this drives `is_first_txn_this_merchant` and `distinct_merchants_last_24h`.

**Payee:** payees per account, payee add rate, `cooling_off` policy.

**Source:** Sparkov has merchants + customers + geo. Census/Fed data for income, cards-per-holder, tenure. Merchant popularity is a Zipf you pick the exponent for.

---

# SINK 4 — Graph topology / warm start ⚠️ NOT IN THE DOC AT ALL

E5 is a GNN detecting "device→multi-card fan-out, shared payees, mule rings" (`design.md:743-745`). **A GNN needs a benign graph with realistic structure to contrast against.** The doc never says where the *benign* graph structure comes from.

You need:
- Benign device→card fan-out distribution (families share tablets; a device touching 3-4 cards is normal, 40 is not — where's the knee?)
- Benign shared-payee structure (roommates paying the same landlord, families)
- Benign account→account transfer graph density and clustering coefficient
- The **warm-start state**: how much history exists at t=0. Cold-start at t=0 means every velocity feature, every `prior_*` count, every age is degenerate for the burn-in period.

If benign fan-out is trivially 1:1 and attacker fan-out is 40:1, **E5 gets ~1.0 AUC and it means nothing.** This is the E5 equivalent of the code-path leak the doc warns about at `design.md:654`.

**Source:** PaySim for transfer-graph topology (it's built for exactly this). Fan-out distributions are largely a judgment call — declare and sensitivity-test.

---

# SINK 5 — C2 control thresholds (`design.md:180-186`)

Class C — cite, don't fit:

- `voice_threshold` (0.85 in the trace) — from speaker-verification EER curves. Cite a specific system's DET curve and pick an operating point; don't just say "0.85."
- `liveness_threshold` — published FAR/FRR from PAD (presentation attack detection) literature / NIST FATE.
- `doc_forensic_threshold`
- `face_match_threshold`
- Base decline rate (~3%) — published issuer decline rates.
- Step-up **pass rate given attacker credential quality** — the trace at `design.md:310` has the attacker pass a step-up on retry, but the doc never specifies P(pass step-up | cred_quality, has_cvv). This is a needed mapping, not a scalar.
- 3DS challenge rate and pass rate.
- AVS/CVV result distributions conditional on legit vs. stolen credential.
- Payee cooling-off period — real bank policy, publicly documented.
- Knowledge-question pass rate in IVR given attacker info level.

---

# SINK 6 — C4 capability tiers (`design.md:634-638`) ⚠️ THIS IS CLASS E

The doc gives Beta(2,8) / Beta(5,5) / Beta(8,2) for `voice_sim`. Two problems the other agent's "just make them ordered" answer misses:

**Problem 1 — it's under-specified.** Every tool returns *multiple* scores, and the doc only parameterizes one:
- `clone_voice` → similarity, artifact_detect, dur_s
- `deepfake_selfie` → liveness, doc_forensic, blink_rate, head_movement_variance, frame_consistency
- `write_dispute` → coherence, template_sim, sentiment, length
- `write_ticket` → social_pressure, inconsistency_count
- `write_phish` → personalization, urgency_markers

That's ~17 score distributions × 3 tiers = **51 distributions**, not 3. And they must be **correlated within a tier** — a tier-2 deepfake with liveness 0.95 shouldn't independently roll doc_forensic 0.2. Independent sampling gives the defender a free decorrelation signal. You need a covariance structure, not 51 marginals.

**Problem 2 — the circularity.** Your headline result is the tier ablation chart (`design.md:640`). But the tier distributions are *chosen by you*, and `voice_threshold` is *also* chosen by you. Success rate at each tier is then just `1 - BetaCDF(threshold)` — an arithmetic consequence of two of your own assumptions dressed up as an empirical finding.

**This is the most important thing to fix in the whole design.** Mitigations, in order of strength:
1. Anchor tier means to *something real* — published ASV similarity scores for TTS-cloned vs. genuine speech at each generation of tooling. There is literature (ASVspoof challenge series across years) that gives you actual ordered numbers.
2. Report the ablation as **sensitivity across a threshold sweep**, not a single point. "Success rises with capability across every threshold in [0.75, 0.95]" is a defensible claim; "tier 2 succeeds 61% of the time" is not.
3. Show the ablation on the **end-to-end episode success rate under a learned defender**, not the single-gate pass rate. Then the result includes C6's adaptation and isn't pure arithmetic.

For Mode A (text verticals V4/V6/V10), the scores are *actually measured* from generated text, so tier = model quality and the circularity is much weaker. **That's an argument for pushing more verticals to Mode A.**

---

# SINK 7 — C4 Mode A scoring functions

Separate sink from SINK 6, and undefined in the doc. Mode A generates real text, then **something has to score it.** You need:
- `coherence_score` — a defined scorer (perplexity? an LLM judge? readability?)
- `template_similarity` — needs a **corpus of real dispute letters/tickets to compare against.** You don't have one. Options: CFPB Consumer Complaint Database (public, huge, real complaint narratives — this is your best bet), or synthesize a template bank and be explicit that it's synthetic.
- `entity_consistency_score` — a checker that cross-references claims in text against graph facts.
- `text_embedding[768]` — which encoder, frozen or fine-tuned.

**CFPB complaints are the highest-value dataset the other agent didn't mention.** Real consumer dispute narratives, public, and they give you both the E4 training negatives and the template-similarity reference corpus.

---

# SINK 8 — Attacker resource model

Class D mostly, but note the doc under-specifies:
- Credential quality tiers: what `quality=0.6` means operationally — P(has CVV), P(expiry valid), P(already-reported-stolen).
- Credential prices per tier (dark-market prices are actually published in threat-intel reports — you can cite these).
- Attacker capital, cost per action, cash-out haircut by channel.
- Credential decay — stolen cards get reported over time.

---

# SINK 9 — RL reward weights (`design.md:583-593`)

Pure Class D: λ_detect, λ_burn, terminal bonus/penalty, per-action costs, episode caps, per-merchant value cap, threshold jitter magnitude, diversity bonus weight. Tune until the top-10 sequence log (`design.md:601`) looks sane. **Never present as empirical.**

---

# SINK 10 — Risk band boundaries (`design.md:776-781`)

The 0.3/0.6/0.8/0.95 bands are Class D, **but** the doc's own optional-RL section (`design.md:791-797`) tells you what to do: grid-search them against the business cost curve. To do that you need λ₁ (friction cost per legit user stepped up) and λ₂ (manual review cost) in **the same units as fraud loss** — i.e. dollars. Published: manual review runs ~$2-5/case, and there's survey data on cart abandonment from step-up friction. That converts Class D into Class C and gives you a genuinely stronger claim.

---

# SINK 11 — Prevalence and mix

- Base fraud rate: 0.5% (`design.md:769`) — declared, matches published card-fraud bps.
- **Vertical mix**: what fraction of your fraud is V1 vs V5 vs V8? Undefined in the doc, and it directly determines per-vertical recall and your gate's training balance. Card testing (V5) is enormously more common in reality than voice-clone provisioning (V1); if you simulate them uniformly your aggregate PR-AUC is dominated by rare verticals.
- Attacker population size, episode concurrency.

---

# SINK 12 — Fidelity validation (`design.md:683-693`)

Meta-sink. The discriminator needs a **real held-out transaction set** in the same schema as your synthetic events. Two hard problems the doc glosses:

1. **Schema mismatch.** Your events have ~25 fields. IEEE-CIS has anonymized V1-V339. Sparkov has ~10 usable columns. You can only run the discriminator on the *intersection* — realistically amount, hour, MCC, geo distance, inter-arrival. **Say which fields the AUC covers.** An AUC of 0.55 on 5 fields is not the same claim as 0.55 on 25.
2. **You must hold out data from fitting.** If you fit archetype params on all of Sparkov and then discriminate against Sparkov, the AUC is measuring overfit, not fidelity. Split before you fit.

---

# SINK 13 — Zero-shot holdout (`design.md:847`)

V7 and V9 held out entirely. V9 (merchant collusion / bust-out) needs `onboard_merchant`, `self_auth`, `bust_out` — **three actions that appear nowhere in the 20-action table** (`design.md:474-493`). Either the action space needs extending or V9 can't be simulated. Flag this now; it's a design gap, not a data gap, but it kills your strongest planned result if unnoticed.

---

# The three biggest gaps, ranked

1. **SINK 6 circularity** — your headline GenAI-uplift chart is currently arithmetic on your own assumptions. Fix with ASVspoof anchoring + threshold sweep + end-to-end measurement.
2. **SINK 4 benign graph topology** — completely absent from the doc, and without it E5's result is a fan-out threshold check with a GNN wrapped around it.
3. **SINK 2 legit-recovery chain probability** — a single joint probability that determines whether your chained-ATO detection is real or generator-induced.

And the framing correction: this isn't a 30-40 number YAML. It's ~250 *fitted* parameters (Sinks 1, 3, 4) that need a fitting script and a held-out split, ~50 *cited* operating points (Sinks 5, 6, 10), and ~40 *tuned* free parameters (Sinks 8, 9, 11). Treat them as three different artifacts with three different provenance stories, because reviewers will ask about each differently.

**Dataset shortlist:** Sparkov (MCC semantics, geo, customer structure), IEEE-CIS (amount + velocity shape, held-out discriminator set), PaySim (transfer-graph topology for E5), **CFPB Consumer Complaints** (E4 real-text negatives + template-similarity corpus — the missing one), **ASVspoof** (tier anchoring for voice), NIST FATE/PAD (liveness operating points).
