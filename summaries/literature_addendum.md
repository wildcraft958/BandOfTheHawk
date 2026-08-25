# Literature Addendum — Corrections and Additions

> Reading notes on Bahnsen 2016, Dal Pozzolo 2014, Wiefling 2022, and Hawkes
> fitting mechanics. Amends `data_requirements.md`, `data_research.md`, and
> `sink12_fidelity_protocol.md`.
>
> **Contains one material correction** to a claim in `data_research.md`. See §3.

---

## 1. Bahnsen et al. 2016 — feature engineering

Free: `albahnsen.github.io/files/Feature Engineering Strategies for Credit Card Fraud Detection_published.pdf`

Three feature tiers with measured lift (their production system, so treat the
percentages as one system's results, not universal constants):

| Tier | What it is | Lift vs. raw |
|---|---|---|
| **Raw** | txn id, time, account, card, type, entry mode, amount, merchant code/group, country, country-of-residence, card type, gender, age, bank | baseline |
| **agg1** | per-card count + sum-of-amount over last `tp` hours (Whitrow et al.) | **+201%** avg savings |
| **agg2** | same window, but conditioned on a **second** matching criterion — same card AND same country, or AND same entry mode, etc. | **+252%** (only *with* agg1; alone it underperforms agg1) |
| **periodic** | von Mises fit to the user's transaction-time history | **+13%** on top → **+287%** total |

`tp` values in their production system: **1, 3, 6, 12, 18, 24, 72, 168 hours**,
crossed with grouping criteria **country, transaction type, entry mode, merchant
code, merchant group** → 280 aggregated features from one base pair.

### Two concrete gaps against `AUTH_ATTEMPT` (`design.md:499-511`)

**Gap A — no compound-conditioned window aggregates.** Our schema has
single-key aggregates only: `auths_last_60s`, `auths_last_1h`, `auths_last_24h`,
`distinct_mcc_last_1h`, `distinct_merchants_last_24h`. Every one keys on card
alone. **agg2's entire contribution is that compound conditioning catches what
agg1 misses** — and it carried the largest single lift in their ablation.

*Add:* per-card count and amount-sum within window, **conditioned on a second
attribute** — same-MCC-cluster, same-entry-mode, same-country/geo-bucket.
Even a small cross (3 windows × 3 criteria) captures the effect.

**Gap B — `hour_of_day` is a raw bucket, not a periodicity feature.** A raw
hour is not the same signal as *"is this transaction time inside this user's
fitted confidence interval."* The latter earned the +13%.

Transaction time is **circular** — 23:00 and 01:00 are adjacent, not 22 hours
apart. A raw hour bucket cannot represent that. Fit a von Mises (periodic
normal) to each holder's transaction-time history over the last `tp` ≥ 7 days
(shorter windows give degenerate fits) and emit a binary: is the new
transaction within the α-confidence interval? Closed-form periodic mean/std in
their Appendix A.

**This also lands in SINK 1.** `data_requirements.md` already specifies
hour-of-day as "a 2-3 component von Mises mixture (better: it's cyclic)."
Bahnsen confirms it — the *same* distribution serves both as the benign
generator and as the detector feature. Fit once, use twice.

---

## 2. Dal Pozzolo et al. 2014 — methodology, not features

Free: `dalpozz.github.io/static/pdf/FraudDetectionPaper_8.pdf`

Their dataset is one Belgian processor, 2012-13 — **the specific K/M/window
values are dataset-specific; cite the methods, not the numbers.**

**Evaluation metric argument.** AUC and accuracy are wrong for a *detection*
task where investigators can only review the top-α ranked alerts. They argue
for **Average Precision** and **PrecisionRank** (precision within the top-α
alerts specifically).

This sharpens `design.md:766-771`, which already says report PR-AUC over
ROC-AUC. Dal Pozzolo goes further: the operationally meaningful quantity is
precision *at the alert volume an analyst can actually review*. Our
`recall @ 1% FPR` is close in spirit — add **PrecisionRank at a fixed alert
budget** alongside it, and this becomes the citation for why.

**Incremental learning** — three strategies compared:

| Strategy | Description |
|---|---|
| Static | train once |
| Update | retrain every K days, ensemble last M models |
| **Forget** | keep **all** past fraud + only the last `Kgen` days of genuine transactions |

Their winner: **Forget + EasyEnsemble + daily retrain.**

**Direct relevance to C7.** `design.md:813-815` says retrain C6 on *cumulative*
data with `evaded_detection=True` upweighted, to prevent catastrophic
forgetting. Forget is the published version of exactly that intuition — keep
all rare positives, decay the abundant negatives. Cite it, and consider
adopting the asymmetry explicitly rather than relying on upweighting alone.

Their §4 `X^H_iλ` historical-window notation also generalizes Bahnsen's
aggregation to any function (mean/max/min/count), not just count/sum.

---

## 3. ⚠️ CORRECTION — Wiefling et al., ACM TOPS 2022

Free preprint: `arxiv.org/html/2206.15139` (journal version paywalled)

### What `data_research.md` claims

> *"New-device/new-location events are rare and histories sparse… Net:
> new-device login ≈ 1–10% of legit logins (true 'new-device binding' nearer
> 1–3%)."* — listed under **WELL-SOURCED / MEASURED**

### What the paper actually reports

**No such statistic appears in the paper.** It reports **re-authentication
trigger rates** under a risk-scoring model (Freeman et al. 2016) at chosen
attacker-blocking thresholds. That is a *model-and-threshold-dependent*
quantity — it conflates genuine feature novelty with wherever the operator sets
the dial. It is not a behavioral base rate.

Actual contents:

- 3.3M users, 31.3M login attempts (12.5M successful / 18.8M failed), Telenor
  Norway SSO, Feb 2020 – Feb 2021. 87 confirmed real ATOs.
- Login frequency is **sparse**: mean 3.8 logins/user, **median 2**, max 5,972.
  **48.3% of users log in less than monthly.**
- Risk model: `S_u(FV) = Π_k [p(FV^k) / p(FV^k | u, legit)] × p(u|attack)/p(u|legit)`
  — likelihood ratio of how rare each feature value is globally vs. for this
  user's own history.
- Calibrated feature weights: **IP 0.6 / ASN 0.3 / country 0.1**; user-agent
  full-string ~0.53–0.55 / browser ~0.27–0.29 / OS ~0.15–0.19 / device-type 0.01.
- Reported numbers are re-auth *rates*: blocking 99.5% of naive attackers →
  average legit user re-authenticates every 2nd login; 99% of VPN/targeted →
  every 2nd–4th login; hardest config (82.76% of "very targeted" replay) →
  every login.

### Consequence for the ATO chain sweep

`data_research.md` built its component decomposition as
`P(new-device) × P(reset) × P(call)`, using the ~1-10% figure for the first
term. **That term was never measured.** The other two are already labeled
INDUSTRY ESTIMATE (self-reported survey data).

**The sweep range 10⁻⁷ – 10⁻⁴ per user-day is unchanged**, but its lower bound
is *less anchored than it appeared* — it is now a product of one unmeasured and
two self-reported quantities. This strengthens rather than weakens the case for
sweeping: there was never a defensible point estimate to prefer.

**Move the "new-device login 1-10%" line from List 1 (WELL-SOURCED) to List 3
(GENUINELY UNAVAILABLE)** in `data_research.md`.

### What the paper *does* give us

**1. A real decay anchor.** They define a "stable setup" as the point where
median re-auth rate falls below 0.5, reached after just **4–8 historical
logins** for most attacker models. Legitimate feature-novelty decays fast with
history size. This is directionally useful for SINK 3's **warm-start** problem:
it says entities need only a handful of historical events before their novelty
features stop being degenerate — a much cheaper burn-in than assuming months of
history are required.

**2. An architecture for E2.** The likelihood-ratio risk score above is a
concrete, published design for exactly what E2 does (`design.md:740`). The
feature weights are calibrated on 31.3M real logins. Arguably more valuable to
us than the base rate I was chasing.

**3. Sparsity realism.** Median 2 logins/user and 48.3% logging in less than
monthly is a caution for SINK 3: if our synthetic population is uniformly
active, per-user history depth will be unrealistically rich and every
history-dependent feature will behave better than it should.

---

## 4. Hawkes process fitting — mechanics for SINK 1 / L3-P1

**Model.** Univariate self-exciting, exponential kernel:

```
λ(t) = μ + Σ_{t_i < t} α · e^{−β(t − t_i)}
```

`μ` baseline rate, `α` jump per event, `β` decay. **Branching ratio
`n = α/β`** = expected children per parent; requires `n < 1` for stability, and
is exactly the parameter the sweep targets (η ∈ [0, 0.6]).

**MLE (Ozaki 1979), closed form:**

```
log L = −μT − (α/β) · Σ_i [1 − e^{−β(T − t_i)}] + Σ_i log(μ + α·A(i))

A(i) = Σ_{j<i} e^{−β(t_i − t_j)}
```

**The O(n) trick (Ogata 1981)** — `A(i)` is naively O(n²), but recurses:

```
A(i) = e^{−β(t_i − t_{i−1})} · (1 + A(i−1)),    A(1) = 0
```

Implement as a running accumulator. This is what makes fitting across thousands
of IEEE-CIS card entities tractable — never compute the double sum directly.

### Recipe

1. Extract chronological event-time sequences per entity from the IEEE-CIS
   **training split** (seconds since the entity's first transaction).
2. **Fit pooled per-archetype, not per-entity.** IEEE-CIS median is 4
   transactions per entity — per-entity fits will be pure noise. Either pool
   into the 6 archetypes, or fit one global `(μ,α,β)` and let archetype
   variation live in a separate baseline-rate scalar.
3. `scipy.optimize.minimize` (L-BFGS-B), reparametrizing `μ,α,β` as `exp()` of
   unconstrained variables to enforce positivity. ~20 lines, no dependencies.
4. Optionally `tick.hawkes.HawkesExpKern`, or `HawkesSumExpKern` for two
   timescales — plausible here: a fast same-session burst plus slower daily
   habit.

> **⚠️ Python 3.13 gotcha.** `tick` 0.8.0.1's fitting classes are reportedly
> broken on 3.13 — the C++ base metaclass rejects instantiation
> (`'HawkesExpKern' object has no settable attribute 'events'`). Simulation
> still works; only the MLE fitters fail. Options: `phawkes` (PyPI, pure
> Python), `tick` on Python ≤3.12, or **just roll the Ozaki likelihood per step
> 3** — given the gotcha, that is probably the cheapest path anyway.

### Goodness of fit — do not skip

**Time-rescaling theorem.** Compute the compensator `Λ(t) = ∫λ(s)ds` between
consecutive events (closed form for the exponential kernel), then transform
each gap:

```
τ_i = Λ(t_i) − Λ(t_{i−1})
```

If the fit is good, `{τ_i}` are i.i.d. **Exponential(1)** — check with a QQ-plot
or K-S test against Exp(1).

**This is the rigorous version of L3-P1**, applied to our own fit rather than a
third-party generator's output. It answers "does my fitted model actually
reproduce real IEEE-CIS burst structure" directly, before any synthetic data
exists. Run it in step 2 of the SINK 12 order of work.

---

## Summary of amendments

| Doc | Change |
|---|---|
| `data_research.md` | Move "new-device login 1-10%" from List 1 → List 3. It is not in Wiefling. |
| `data_requirements.md` SINK 1 | von Mises hour model now serves double duty — benign generator *and* E2/E3 detector feature |
| `design.md:499-511` (`AUTH_ATTEMPT`) | **Add**: (a) compound-conditioned window aggregates per agg2; (b) von Mises periodicity feature; (c) rolling 24h amount sum (the R3 gap from SINK 12) |
| `design.md:766-771` | Add PrecisionRank at a fixed alert budget; cite Dal Pozzolo |
| `design.md:813-815` | Cite Dal Pozzolo's **Forget** strategy for the cumulative-retrain design |
| `design.md:740` (E2) | Consider the Wiefling likelihood-ratio risk score as the architecture |
| SINK 3 warm-start | 4–8 historical events suffices for novelty features to stabilize; also model login/txn sparsity (median 2, 48.3% sub-monthly) |
| `sink12_fidelity_protocol.md` step 2 | Add time-rescaling K-S test as the Hawkes goodness-of-fit gate |
