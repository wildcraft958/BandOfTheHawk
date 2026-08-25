# Closed-Loop GenAI Payment Fraud Red/Blue System
## Component Contract Specification

> **READ THIS FIRST — THE #1 THING PEOPLE GET BACKWARDS**
>
> ```
>   WRONG:   LLM → produces something → RL reads it as state → RL acts
>   RIGHT:   RL decides the action → THEN (only sometimes) LLM renders an artifact for it
> ```
>
> The LLM **never** produces the RL state. The LLM **never** decides anything.
> The RL policy is the brain. The LLM is a printer that the brain occasionally sends a job to.
> On ~16 of 20 actions the LLM is not called at all.
>
> **Direction of every arrow in this system:**
> `RL POLICY → SIMULATOR → (if artifact needed) GENERATIVE TOOL → back to SIMULATOR → resolve → back to RL`

---

# PART A — COMPONENT MAP

Seven components. Nothing else exists.

| # | Component | One-line job |
|---|---|---|
| C1 | **World State** | Holds the entity graph + clock. Passive data. Does not act. |
| C2 | **Simulator Core** | The referee. The ONLY component that reads/writes the graph. |
| C3 | **Attacker Policy (RL)** | Decides what the attacker does next. The brain. |
| C4 | **Generative Tool Layer** | Renders artifacts on demand. A tool, not an agent. |
| C5 | **Benign Population** | Scripted legit users. No learning. |
| C6 | **Defender (MoE)** | Scores events. Outputs risk + mitigation action. |
| C7 | **Training Orchestrator** | Runs rounds, freezes/unfreezes, retrains. |

### Who can touch the graph?

**Only C2.** C3 never sees it. C4 never sees it. C6 never sees it. C5 never sees it.
Everyone else sends C2 a request and gets back a *view* or an *outcome*.

If you break this rule you get information leakage and your results are worthless.

---

# PART B — THE PER-TURN CONTRACT (THE CORE LOOP)

This is one turn. Read it as a chain of function calls. Every arrow is labeled with the exact payload.

```
┌──────────────────────────────────────────────────────────────────┐
│  STEP 1                                                          │
│  C2 SIMULATOR  ──────────────────────────────►  C3 RL POLICY     │
│                     PAYLOAD: AttackerObs                         │
└──────────────────────────────────────────────────────────────────┘

AttackerObs {
    # resources I hold
    n_creds, mean_cred_quality, has_cvv, has_expiry
    n_identities, kyc_status
    n_devices_controlled, n_bound_cards
    voice_artifact_quality      # 0.0 if I haven't cloned a voice yet
    face_artifact_quality
    n_payees_added

    # where I am
    stage_onehot[5]             # NONE/ACQUIRED/BOUND/MONETIZED/TERMINAL
    vertical_onehot[10]
    legal_action_mask[20]       # ← C2 computes this from the stage gate

    # what happened last turn
    last_action_onehot[20]
    last_outcome_onehot[5]      # approved/declined/stepped_up/held/blocked
    last_decline_code
    consecutive_failures, n_stepups_this_episode

    # my ledger
    actions_taken, sim_time_elapsed_h, value_extracted, cost_incurred

    # publicly observable
    hour_of_day, day_of_week, target_card_bin_tier
}

NOT IN THIS PAYLOAD (and never will be):
    ✗ the entity graph
    ✗ the defender's risk score
    ✗ other actors' state
    ✗ ground truth labels
    ✗ anything an LLM produced
```

```
┌──────────────────────────────────────────────────────────────────┐
│  STEP 2                                                          │
│  C3 RL POLICY  ──────────────────────────────►  C2 SIMULATOR     │
│                     PAYLOAD: Action                              │
└──────────────────────────────────────────────────────────────────┘

Action {
    name        : one of 20        # discrete head, masked by legal_action_mask
    amount      : float            # continuous head, log-scaled [1, 5000]
    delay_h     : float            # continuous head, log-scaled [0, 72]
    mcc_cluster : categorical      # continuous-ish head
    channel     : categorical
    target_id   : entity ref
}

THIS IS THE ENTIRETY OF WHAT RL PRODUCES. One action name + its numbers.
The numbers are the whole point — learning "$187 clears, $340 gets stepped up"
lives in the `amount` head.
```

```
┌──────────────────────────────────────────────────────────────────┐
│  STEP 3 — C2 internal: stage gate                                │
└──────────────────────────────────────────────────────────────────┘

if Action.name not in LEGAL_ACTIONS[actor.stage]:
    return Outcome(ILLEGAL, reward=0, graph unchanged)
```

```
┌──────────────────────────────────────────────────────────────────┐
│  STEP 4 — CONDITIONAL. Fires on ~4 of 20 actions.                │
│  C2 SIMULATOR  ──────────────────────────────►  C4 GEN TOOL      │
│                     PAYLOAD: ArtifactRequest                     │
└──────────────────────────────────────────────────────────────────┘

ArtifactRequest {
    tool_name       : clone_voice | deepfake_selfie | write_phish
                      | write_dispute | write_ticket | write_refund_claim
    target_ref      : holder_id / txn_id / merchant_id
    capability_tier : 0 | 1 | 2        # ← the GenAI-uplift knob
    persona_hint    : str
}

*** C4 IS CALLED **BY** C2, **AFTER** C3 HAS ALREADY DECIDED. ***
*** C4 IS NEVER CALLED BEFORE C3. C4 NEVER TALKS TO C3 AT ALL.  ***

Actions that DO call C4 (4 of 20):
    call_ivr_provision   → needs clone_voice
    submit_kyc           → needs deepfake_selfie
    file_dispute         → needs write_dispute
    open_ticket          → needs write_ticket
    (+ phish_holder, request_refund, sim_swap if you implement them)

Actions that DO NOT call C4 (16 of 20):
    attempt_auth, transfer_p2p, cash_out, add_payee, buy_creds,
    make_synth_id, add_device_selfserve, reset_password, escalate_limit,
    complete_3ds, launder_chain, ...
    → these are pure numbers. No LLM involved. Most turns skip C4 entirely.
```

```
┌──────────────────────────────────────────────────────────────────┐
│  STEP 5                                                          │
│  C4 GEN TOOL  ───────────────────────────────►  C2 SIMULATOR     │
│                     PAYLOAD: Artifact                            │
└──────────────────────────────────────────────────────────────────┘

Artifact {
    content       : text | audio_ref | video_ref      # the actual thing
    quality_score : float                             # ← what C2 checks
    aux_scores    : { ... }                           # → become event features
}

Concrete returns:
  clone_voice      → { audio_ref, similarity: 0.87, artifact_detect: 0.12, dur_s: 94 }
  deepfake_selfie  → { video_ref, liveness: 0.91, doc_forensic: 0.78, blink_rate: 0.4 }
  write_dispute    → { text, coherence: 0.90, template_sim: 0.11, sentiment, length }
  write_ticket     → { text, social_pressure: 0.7, inconsistency_count: 2 }
  write_phish      → { text, personalization: 0.83, urgency_markers: 4 }

*** THIS PAYLOAD GOES TO C2 ONLY. IT NEVER GOES TO C3. ***
The RL policy does not see the letter. It only later sees whether the action worked.
```

```
┌──────────────────────────────────────────────────────────────────┐
│  STEP 6 — C2 internal: resolve                                   │
└──────────────────────────────────────────────────────────────────┘

C2 checks Artifact.quality_score against the control:
    IVR:  artifact.similarity (0.87) > policy_config.voice_threshold (0.85)  → PASS
    KYC:  artifact.liveness   (0.91) > policy_config.liveness_threshold      → PASS

C2 also checks graph facts:
    device_age_days, auths_last_60s, is_first_txn_this_merchant, geo_velocity...

→ produces preliminary verdict
```

```
┌──────────────────────────────────────────────────────────────────┐
│  STEP 7                                                          │
│  C2 SIMULATOR  ──────────────────────────────►  C6 DEFENDER      │
│                     PAYLOAD: Event  (NO LABEL ATTACHED)          │
└──────────────────────────────────────────────────────────────────┘

Event {
    event_id, ts, event_type
    <type-specific fields — see PART E>
    <graph-derived features C2 computed: velocities, ratios, ages>
    <artifact aux_scores if any>
}

*** CRITICAL: THIS PAYLOAD CONTAINS NO is_fraud FIELD. ***
C6 is blind at inference. It gets the same struct whether a benign agent
or an attacker produced it. Built by the SAME build_event() function.
```

```
┌──────────────────────────────────────────────────────────────────┐
│  STEP 8                                                          │
│  C6 DEFENDER  ───────────────────────────────►  C2 SIMULATOR     │
│                     PAYLOAD: Verdict                             │
└──────────────────────────────────────────────────────────────────┘

Verdict {
    risk_score      : float [0,1]
    expert_weights  : float[5]        # gate output — for the demo UI
    action          : approve | step_up | hold | decline | block
    mitigation      : None | DELETE_EDGE(a,b) | FREEZE_CARD(id)
                      | BLOCKLIST_DEVICE(id) | REQUIRE_STEPUP(edge)
                      | TIGHTEN_THRESHOLD(control, value)
}
```

```
┌──────────────────────────────────────────────────────────────────┐
│  STEP 9 — C2 internal: mutate the graph                          │
└──────────────────────────────────────────────────────────────────┘

if verdict.action == approve:
    apply_mutation(graph, action)         # e.g. CREATE EDGE card_1 —provisioned→ device_A
    actor.stage = advance(actor.stage)    # ACQUIRED → BOUND

if verdict.mitigation:
    apply_mitigation(graph, verdict.mitigation)
                                          # e.g. DELETE EDGE card_1 —provisioned→ device_A
                                          # → every future auth via device_A now fails
                                          #   at the GRAPH level, not by a rule firing

append Event + verdict to event_log
```

```
┌──────────────────────────────────────────────────────────────────┐
│  STEP 10 — TWO OUTPUTS, TWO CONSUMERS                            │
└──────────────────────────────────────────────────────────────────┘

  C2 ──► C3 (immediate, per-turn)
         Outcome { outcome_code, reward_t, next AttackerObs }
         ← this is what closes the FAST loop

  C2 ──► event_log (deferred, per-round)
         Event row, labeled AFTER the episode closes
         ← this is what feeds the SLOW loop into C6
```

---

# PART C — THE FULL CHAIN, ONE LINE

```
C3 decides ─► C2 gates ─► C2 calls C4 (only if artifact needed) ─►
C4 returns artifact+score ─► C2 resolves vs graph & thresholds ─►
C2 sends Event to C6 ─► C6 returns risk+mitigation ─►
C2 mutates graph ─► C2 returns Outcome to C3  ──┐
     ▲                                          │
     └──────────────── next turn ───────────────┘
```

**C4 sits in the middle of the chain, called by C2, downstream of C3. Not upstream. Never upstream.**

---

# PART D — WORKED TRACE (5 TURNS, EVERY PAYLOAD NAMED)

```
━━━ TURN 1 ━━━
C2 → C3   AttackerObs { stage=NONE, n_creds=0, mask=[phish, buy_creds, make_synth_id, harvest_voice] }
C3 → C2   Action { name=buy_creds, count=1, quality_tier=mid }
C2        artifact needed? NO → C4 not called
C2        graph: attacker now holds (pan, expiry, no cvv, quality=0.6)
C2        emits: nothing bank-visible
C2 → C3   Outcome { SUCCESS, reward=-2 (cost), stage=ACQUIRED }

━━━ TURN 2 ━━━
C2 → C3   AttackerObs { stage=ACQUIRED, n_creds=1, voice_quality=0.0 }
C3 → C2   Action { name=harvest_voice, target=holder_1 }
C2        artifact needed? YES
C2 → C4   ArtifactRequest { tool=clone_voice, target=holder_1, capability_tier=2 }
C4 → C2   Artifact { audio_ref, similarity=0.87, artifact_detect=0.12 }
C2        stores on attacker: voice_artifact_quality=0.87
C2        emits: nothing bank-visible (harvesting happens outside the bank)
C2 → C3   Outcome { SUCCESS, reward=-1, stage=ACQUIRED }
          ↑ note: C3 learns "0.87" only as voice_artifact_quality in the NEXT obs.
            C3 never saw the audio.

━━━ TURN 3 ━━━
C2 → C3   AttackerObs { stage=ACQUIRED, voice_artifact_quality=0.87 }
C3 → C2   Action { name=call_ivr_provision, card=card_1, request=provision_device }
C2        artifact needed? uses the stored voice artifact
C2        resolve: 0.87 > voice_threshold 0.85 → PASS
C2        Event IVR_CALL { voice_similarity=0.87, call_duration_s=94,
                           number_matches_on_file=False, hesitation_count=3,
                           prior_calls_last_30d=0, hour=03 }
C2 → C6   ^ that Event, NO label
C6 → C2   Verdict { risk=0.41, expert_weights=[.05,.85,.05,.03,.02],
                    action=step_up }        ← E2 Binding expert dominates
C2        step-up challenge → attacker has no CVV → fails? here: passes on retry
C2        MUTATE: CREATE EDGE card_1 —provisioned→ device_A
C2 → C3   Outcome { STEPPED_UP→SUCCESS, reward=-3, stage=BOUND }

━━━ TURN 4 ━━━
C2 → C3   AttackerObs { stage=BOUND, n_bound_cards=1, last_outcome=stepped_up }
C3 → C2   Action { name=attempt_auth, merchant=m_44, mcc=5732, amount=40, delay_h=2 }
C2        artifact needed? NO
C2        Event AUTH_ATTEMPT { amount=40, device_age_days=0, device_new_to_card=True,
                               auths_last_60s=0, amount_vs_card_median=0.31,
                               is_first_txn_this_merchant=True }
C2 → C6   ^ Event
C6 → C2   Verdict { risk=0.31, weights=[.02,.20,.75,.02,.01], action=approve }
C2        MUTATE: card_1 —transacts→ m_44
C2 → C3   Outcome { APPROVED, reward=+40, value_extracted=40 }

━━━ TURN 5 ━━━
C2 → C3   AttackerObs { value_extracted=40, last_outcome=approved }
C3 → C2   Action { name=attempt_auth, merchant=m_44, mcc=5732, amount=890, delay_h=0.1 }
C2        Event AUTH_ATTEMPT { amount=890, amount_vs_card_median=6.9,
                               auths_last_60s=1, device_age_days=0 }
C2 → C6   ^ Event
C6 → C2   Verdict { risk=0.88, weights=[.01,.15,.83,.01,.00], action=block,
                    mitigation=DELETE_EDGE(card_1, device_A) + BLOCKLIST_DEVICE(device_A) }
C2        MUTATE: edge deleted, device blocklisted
C2 → C3   Outcome { BLOCKED, reward=-50, episode_value ZEROED, stage=TERMINAL }

━━━ AFTER EPISODE ━━━
C2 → event_log   stamp all 5 events:
                 is_fraud=True, vertical_id=V1, episode_id=e_8812,
                 attacker_succeeded=False,
                 evaded_detection=True on TURNS 3 & 4  ← HARD NEGATIVES

WHAT C3 LEARNED: turn 5 pushed too hard, too fast, too soon after turn 4.
Next episode it tries $200 with delay_h=6, or splits across two MCCs.
Nobody coded "escalate gradually." It emerged from the reward.
```

---

# PART E — COMPONENT SPECS

## C1 — WORLD STATE

**Receives from:** C2 only (read + write)
**Sends to:** C2 only
**Learns:** nothing. Passive data structure.

### Nodes

```
Cardholder  holder_id, home_geo, age_band, income_band, tenure_days
            behavior_archetype {commuter, homebody, traveler, online_heavy, senior, business}
            voice_embedding      ← ground truth reference for IVR checks
            face_embedding       ← ground truth reference for KYC checks

Card        card_id, holder_id, issue_date, credit_line, bin
            median_txn_amount, mcc_histogram
            status {active, frozen, reissued}

Device      device_id, fingerprint_hash, first_seen_ts
            os, browser, app_version, ip_asn, geo_estimate
            is_emulator_flag, reputation_score

Account     account_id, holder_id, balance, open_date
            kyc_level {none, basic, full}
            kyc_passed_via {branch, doc_upload, liveness_video}

Merchant    merchant_id, mcc, avg_ticket, chargeback_rate, risk_tier
            is_high_liquidity    ← gift cards / crypto / prepaid = cash-out friendly

Payee       payee_id, account_id_target, first_added_ts
            is_mule_flag         ← ground truth, NEVER exposed to C6
```

### Edges — the actual attack surface

```
holder  —owns→        card
card    —provisioned→ device    { bind_ts, bind_method, bind_trust, challenge_required }
account —added→       payee     { add_ts, add_method, cooling_off_until }
card    —transacts→   merchant  { first_ts, count, total }
device  —used_by→     account   { count }
```

**Every fraud vertical = "create an edge you shouldn't be able to create."**
**Every mitigation = "delete an edge or raise its cost."**
That symmetry is the whole design.

### Global

```
clock_ts               1-minute ticks
event_log[]            append-only
defender_snapshot      frozen C6 used for inline scoring this round
policy_config          thresholds, step-up rules, blocklists
ground_truth_labels{}  entity_id → is_attacker_controlled
                       ← C2 ONLY. Used to stamp labels AFTER episodes.
                         NEVER in an Event payload.
```

---

## C2 — SIMULATOR CORE

**Receives:** Action (from C3), Action (from C5), Artifact (from C4), Verdict (from C6)
**Sends:** AttackerObs + Outcome (to C3), ArtifactRequest (to C4), Event (to C6), rows (to event_log)
**Learns:** nothing. Pure referee.

```python
def step(actor_id, action) -> Outcome:
    actor = registry[actor_id]

    # 1. STAGE GATE
    if action.name not in LEGAL[actor.stage]:
        return Outcome(ILLEGAL, reward=0)

    # 2. ARTIFACT — C2 calls C4 here, AFTER C3 already decided
    artifact = None
    if action.name in NEEDS_ARTIFACT:
        artifact = C4.generate(ArtifactRequest(
            tool_name       = TOOL_FOR[action.name],
            target_ref      = action.target_id,
            capability_tier = actor.capability_tier))

    # 3. RESOLVE vs graph facts + artifact quality vs thresholds
    verdict = resolve(actor, action, artifact, graph, policy_config)

    # 4. EMIT — intent-blind, SAME builder for C3 and C5 actors
    event = build_event(actor, action, artifact, graph, clock)

    # 5. SCORE — C6 sees no label
    dverdict = C6_snapshot.score(event)
    verdict  = combine(verdict, dverdict)
    event_log.append(event, dverdict)

    # 6. MUTATE
    if verdict.approved:
        apply_mutation(graph, action)
        actor.stage = advance(actor.stage, action.name)
    if verdict.mitigation:
        apply_mitigation(graph, verdict.mitigation)

    # 7. RETURN to C3
    return Outcome(verdict.code, reward(verdict), build_obs(actor))
```

### Stage gate table

| Stage | Legal actions |
|---|---|
| `NONE` | phish_holder, buy_creds, make_synth_id, harvest_voice, harvest_face |
| `ACQUIRED` | call_ivr_provision, submit_kyc, add_device_selfserve, add_payee, sim_swap, reset_password |
| `BOUND` | attempt_auth, transfer_p2p, request_refund, add_payee, escalate_limit |
| `MONETIZED` | cash_out, launder_chain, file_dispute |
| `TERMINAL` | — |

Transitions only on **successful** actions. A failed IVR leaves you in ACQUIRED.

**Why the gate exists:** without it the policy burns 500k episodes learning "you can't spend a card you haven't bound." The gate encodes what's *structurally* true about payments so RL capacity goes to what's *strategically* interesting — timing, sizing, channel, sequencing.

### Action table — who calls C4

| Action | Params | Calls C4? | Emits |
|---|---|---|---|
| phish_holder | target, channel | ✅ write_phish | PHISH_DELIVERY |
| buy_creds | count, quality_tier | ❌ | — |
| make_synth_id | seed_attrs | ❌ | — |
| harvest_voice | target | ✅ clone_voice | — |
| harvest_face | target | ✅ deepfake_selfie | — |
| call_ivr_provision | card, request_type | uses stored voice | IVR_CALL |
| submit_kyc | identity, doc | uses stored face | KYC_SUBMIT |
| add_device_selfserve | card, device | ❌ | DEVICE_BIND |
| sim_swap | holder | ✅ pretext | SIM_CHANGE |
| reset_password | account, channel | ❌ | AUTH_RESET |
| add_payee | account, payee | ❌ | PAYEE_ADD |
| open_ticket | account, intent | ✅ write_ticket | SUPPORT_TICKET |
| escalate_limit | card, target | ❌ | LIMIT_CHANGE |
| **attempt_auth** | card, merchant, **amount**, **delay** | ❌ | AUTH_ATTEMPT |
| complete_3ds | card, challenge | ❌ | THREEDS_RESULT |
| transfer_p2p | account, payee, **amount** | ❌ | TRANSFER |
| request_refund | txn | ✅ write_refund_claim | REFUND_REQUEST |
| file_dispute | txn, reason_code | ✅ write_dispute | DISPUTE_FILED |
| cash_out | channel, **amount** | ❌ | CASHOUT |
| launder_chain | hops | ❌ | TRANSFER×hops |

Bold = continuous params C3 learns to tune. **16 of 20 never touch C4.**

### Event schemas C2 builds

**AUTH_ATTEMPT**
```
event_id, ts, card_id, merchant_id, mcc, amount, currency
entry_mode {chip, contactless, cnp, token}
device_id, device_age_days, device_new_to_card, device_n_cards_seen
ip_asn, geo_distance_from_home_km, geo_velocity_kmh
auths_last_60s, auths_last_1h, auths_last_24h
distinct_mcc_last_1h, distinct_merchants_last_24h
amount_vs_card_median, amount_percentile_for_mcc
declines_last_1h, avs_result, cvv_result
time_since_last_auth_s, is_first_txn_this_merchant
hour_of_day, is_weekend
```

**KYC_SUBMIT**
```
event_id, ts, identity_id, account_id
liveness_score, doc_forensic_score, face_match_score      ← from C4 aux_scores
blink_rate, head_movement_variance, frame_consistency
doc_type, doc_issue_recency_days
submission_attempt_n, time_on_page_s
device_id, device_is_emulator, ip_asn, ip_is_datacenter
identity_credit_history_depth, identity_age_at_first_seen
```

**IVR_CALL**
```
event_id, ts, card_id, caller_number, number_matches_on_file
voice_similarity, voice_artifact_score                     ← from C4 aux_scores
background_noise_profile, call_duration_s
hesitation_count, knowledge_questions_passed
request_type, hour_of_day, prior_calls_last_30d
```

**DISPUTE_FILED / SUPPORT_TICKET / REFUND_REQUEST**
```
event_id, ts, account_id, txn_ref, reason_code
text_embedding[768]                                        ← from C4 content
text_length, coherence_score, template_similarity          ← from C4 aux_scores
sentiment, urgency_markers, entity_consistency_score
time_since_txn_h, prior_disputes_12m, prior_dispute_win_rate
```

**DEVICE_BIND / PAYEE_ADD / SIM_CHANGE / AUTH_RESET / LIMIT_CHANGE**
```
event_id, ts, actor_id, target_id, bind_method, channel
time_since_account_open_d, time_since_last_bind_d
n_binds_last_30d, hour_of_day
preceded_by_password_reset_h, preceded_by_ivr_call_h        ← catches chained ATO
```

### Labels C2 stamps AFTER episode close

```
is_fraud, vertical_id, episode_id
attacker_succeeded    ← reached cash_out?
evaded_detection      ← was fraud, scored below threshold
```

**`evaded_detection=True` rows are the most valuable data in the system. Upweight them in C6 retraining.**

---

## C3 — ATTACKER POLICY (RL)

**Receives:** AttackerObs + Outcome — from C2 only
**Sends:** Action — to C2 only
**Never talks to:** C4, C6, C1
**Learns:** yes, PPO

### Heads

```
discrete    softmax over 20 actions, MASKED by legal_action_mask before softmax
continuous  amount      ∈ [1, 5000]   log-scaled
            delay_h     ∈ [0, 72]     log-scaled
            mcc_cluster ∈ categorical
            channel     ∈ categorical
```

Architecture: MLP 2×256. Nothing exotic needed.

### Reward

```
r_t = + value_extracted_t
      - action_cost[action]              creds cost money, IVR costs time
      - λ_detect · 1[flagged]
      - λ_burn  · 1[resource_burned]     device blocked, card frozen

terminal   + bonus  if reached CASHOUT
           - penalty if TERMINAL by detection

episode value ZEROED if caught before cash_out
```

### Anti-reward-hacking — do all five

1. Hard cap on episode length and total actions
2. Non-zero cost on **every** action, else it spams free ones
3. Cap value per merchant per episode, else it finds one soft merchant and loops forever
4. Jitter C6's thresholds per episode, else it memorizes one exact number
5. **Log the top-10 most frequent action sequences each round and actually read them.** If they look absurd, you have a simulator bug, not a clever attacker

### Multi-vertical coverage

Don't train one policy for all ten — it collapses onto the most profitable one and your attack diversity dies. Either:
- **Conditioned:** vertical ID in the obs, episode initialized into that vertical, reward normalized per-vertical
- **Population:** one policy per vertical family. Simpler, more stable, more compute

Add a **diversity bonus**: hash action sequences, penalize repeats. You're scored on attack diversity — optimize it directly.

### Must train against a FROZEN C6

Moving defender = non-stationary environment = PPO thrashes.

---

## C4 — GENERATIVE TOOL LAYER

**Receives:** ArtifactRequest — from C2 only
**Sends:** Artifact — to C2 only
**Never talks to:** C3, C6, C1
**Learns:** nothing. Stateless. Not an agent.

> **This component is a printer.** C2 sends it a job, it returns a rendered artifact plus a quality score. It has no memory, no goals, no view of the world, and no influence on what action gets chosen. It is called *after* the decision, never before.

### Two implementation modes

**Mode A — LLM-backed, real content.** Actually generate the letter/email/ticket, then score it. **Do this for the text verticals (V4, V6, V10)** — the text becomes features for expert E4, and judges want to see real generated content in the demo.

**Mode B — parametric, score only.** For voice and video, do **not** generate deepfakes. Sample the quality score from a calibrated distribution. Identical decision dynamics, no deepfake media produced. Defensible in the writeup: *"we model the detector-facing signal, not the media."*

### Capability tiers — your GenAI-uplift story

```
tier 0 (pre-GenAI):  voice_sim ~ Beta(2,8)  → mean 0.20
tier 1 (2023 tools): voice_sim ~ Beta(5,5)  → mean 0.50
tier 2 (2026 tools): voice_sim ~ Beta(8,2)  → mean 0.80
```

Run the identical attack at all three tiers, plot success rate. **That one chart is your entire "GenAI changed the threat model" argument, quantified instead of asserted.** Almost nobody else will have it.

---

## C5 — BENIGN POPULATION

**Receives:** BenignObs — from C2
**Sends:** Action — to C2, **through the exact same `step()` as C3**
**Learns:** nothing. Scripted + stochastic.

### Why it exists

1. Without it, FP rate is undefined — and you're scored on it explicitly
2. Without it, C6 has no notion of "normal"
3. **Most important:** legit users must emit the *same event types through the same API*. Real people add devices. Real people file disputes. Real people make an odd $900 purchase abroad. **Separate code paths → C6 learns the code path → 0.99 AUC that means nothing**

### Archetypes

```
commuter      2-4 txn/day, MCC {5814, 5541, 4111}, tight geo, LogNormal(2.8, 0.5)
homebody      1-2 txn/day, MCC {5411, 5912}, very tight geo
online_heavy  3-8 txn/day, MCC {5732, 5942, 7372}, no geo signal, CNP
traveler      bursty, geo jumps, foreign MCC, currency changes
senior        0.5 txn/day, MCC {5912, 8062}, high-value infrequent, IVR-preferring
business      high volume, high ticket, B2B MCC, weekday cycle
```

### MUST produce hard negatives

This separates a real submission from a toy. Deliberately inject:

- **Legit new device** (bought a phone) → DEVICE_BIND with device_age=0
- **Legit travel** → geo jump + foreign MCC + amount spike
- **Legit large purchase** → 10× median at a new merchant
- **Legit dispute** → real merchant error, genuine chargeback
- **Legit account recovery** → password reset + IVR + device rebind within one hour
- **Legit gift-card buy** → high-liquidity MCC, completely innocent
- **Legit velocity burst** → shopping spree, 6 auths in 10 minutes

**Target: 5-8% of benign traffic trips a naive rule engine. Report that number** — it's your evidence the FP metric is real.

### Fidelity validation

Calibrate against IEEE-CIS / Sparkov / PaySim — fit amount distributions, MCC frequencies, hour-of-day curves, inter-arrival times.

Then **train a discriminator: real legit vs synthetic legit.**

```
AUC ≈ 0.50   indistinguishable — excellent
AUC ≈ 0.65   acceptable, note the gaps
AUC ≈ 0.90   your data looks fake, fix it
```

Only objective evidence for the Fidelity criterion. Report it.

---

## C6 — DEFENDER (MoE)

**Receives:** Event (no label) — from C2 at inference; labeled event_log — from C7 at training
**Sends:** Verdict — to C2
**Never sees:** the graph, ground_truth_labels, C3's state, C4's raw content
**Learns:** yes, supervised, between rounds

### Why MoE is architecturally justified

Not "many attack types." The real reason: **event types have structurally different feature spaces.**

`KYC_SUBMIT` has liveness + doc forensics and **no amount** — no money has moved. `AUTH_ATTEMPT` has amount + velocity and **no KYC fields** — onboarding was two years ago. `DISPUTE_FILED` has a 768-dim text embedding appearing nowhere else.

Flatten these into one table → most cells null → the model wastes capacity learning "null in column 14 ⟹ onboarding event." It learns routing anyway, badly, and null-patterns become a leakage shortcut. MoE makes routing explicit.

### Architecture

```
                    Event (no label)
                          │
                ┌─────────┴─────────┐
                │  GATING NETWORK   │ → P(expert | event)
                └─────────┬─────────┘
          ┌───────┬───────┼───────┬───────┐
          ▼       ▼       ▼       ▼       ▼
         E1      E2      E3      E4      E5
      identity binding   txn    text  network
          └───────┴───────┼───────┴───────┘
                          ▼
                weighted risk_score ∈ [0,1]
                          ▼
                ┌──────────────────┐
                │  ACTION POLICY   │ → approve/step_up/hold/decline/block
                └──────────────────┘
                          ▼
                       Verdict → C2
```

### The five experts

| Expert | Handles | Model | Key features |
|---|---|---|---|
| **E1 Identity** | KYC_SUBMIT, onboarding | GBDT | liveness, doc forensics, credit-history depth, emulator flags |
| **E2 Binding** | DEVICE_BIND, IVR_CALL, SIM_CHANGE, AUTH_RESET, PAYEE_ADD | GBDT + sequence feats | voice similarity, time-since-open, bind velocity, **preceded_by chains** |
| **E3 Transaction** | AUTH_ATTEMPT, TRANSFER | XGBoost / LightGBM | amount ratios, velocity windows, MCC novelty, geo velocity |
| **E4 Text** | DISPUTE, TICKET, REFUND, PHISH | Transformer encoder + head | embeddings, coherence, template similarity, entity consistency |
| **E5 Network** | cross-entity | GNN or graph features | device→multi-card fan-out, shared payees, mule rings |

E5 catches what per-event models structurally cannot: a device touching 40 cards is invisible in any single row.

### Gating

**Training:** supervised — C2 hands you event_type and vertical labels for free.
```
gate_loss = CE(gate_output, true_expert_assignment)
```

**Inference:** soft routing. Gate emits a distribution, final score is the weighted combination. Handles events spanning experts (an auth right after a device bind should get both E2 and E3 weight).

```
L = Σ_k w_k · BCE(expert_k, y) + α·CE(gate, expert_label) + β·load_balance
```

### Order of work

1. **Baseline first** — single GBDT on the flat table. Get the AUC. Your fallback and comparison point.
2. **Then MoE** — report the delta. If MoE doesn't beat baseline you still have a working submission and an honest ablation, which judges respect more than an unvalidated fancy architecture.

### Class imbalance

Real fraud is 0.1-1%. **Do not simulate 50/50.**
- Simulate at ~0.5% base rate
- Class weights or focal loss
- **Report PR-AUC, not just ROC-AUC** — ROC-AUC is misleading at this imbalance
- **Report recall @ 1% FPR** — that's what an actual issuer cares about

### Mitigation — detection is a score, mitigation is a graph mutation

| risk_score | action | mitigation sent to C2 |
|---|---|---|
| < 0.3 | approve | none |
| 0.3–0.6 | step_up | `REQUIRE_STEPUP(edge)` — attacker must beat it with its credential quality |
| 0.6–0.8 | hold | `FREEZE_EDGE(edge, hours)` + review cost incurred |
| 0.8–0.95 | decline | action fails, edge unchanged |
| > 0.95 | block | `DELETE_EDGE(a,b)` + `BLOCKLIST_DEVICE(id)` + `FREEZE_CARD(id)` |

Plus adaptive controls C6 can tighten via `TIGHTEN_THRESHOLD`:
- lower velocity threshold on a specific MCC
- raise voice-similarity threshold on IVR provisioning
- extend payee cooling-off period
- force re-KYC on accounts sharing a device fingerprint

**Why this closes the loop:** deleting `card_1 —provisioned→ device_A` means every future auth via device_A fails **at the graph level** — not because a rule fired, but because the capability no longer exists. C3's action space genuinely shrinks and it must spend actions and cost rebuilding.

### Optional: RL on the blue side

The classifier gives a score; choosing the *action* is a cost decision:
```
r = -(fraud_loss) - λ₁·(friction on legit user) - λ₂·(manual review cost)
```
Even grid-searching the band thresholds against this gives you *"we optimize the business cost curve, not just AUC"* — a real-world-feasibility point most teams won't make.

---

## C7 — TRAINING ORCHESTRATOR

**Receives:** event_log, metrics from C3 and C6
**Sends:** freeze/unfreeze signals, retrained weights
**Learns:** nothing. Control loop.

```
ROUND N:
  1. FREEZE C6 → defender_snapshot D_N
  2. TRAIN C3 via PPO against D_N        (~100k episodes)
     C5 runs continuously alongside, same step()
  3. HARVEST all events this round, stamp labels
  4. RETRAIN C6 → D_{N+1} on CUMULATIVE data
     - upweight evaded_detection=True rows
     - keep ALL prior rounds (prevents catastrophic forgetting)
  5. EVALUATE
     - C3 success rate vs D_N and vs D_{N+1}
     - C6 PR-AUC, recall@1%FPR on held-out episodes
  6. LOG newly discovered action sequences
  7. N ← N+1
```
Run 5–10 rounds.

**Why freeze:** two simultaneously-learning agents = non-stationary for both. PPO thrashes, neither converges. Alternating freeze makes each phase a stationary problem.

### The chart that wins it

```
              Attacker success rate
        D_1    D_2    D_3    D_4    D_5
π_1     0.42   0.11   0.06   0.04   0.03
π_2     0.51   0.38   0.14   0.08   0.05
π_3     0.55   0.47   0.35   0.16   0.09
π_4     0.58   0.51   0.44   0.33   0.15
π_5     0.61   0.55   0.49   0.41   0.31
```

**Down a column** = later attackers beat older defenses (C3 is learning).
**Across a row** = later defenses beat older attackers (C6 is learning).
**Non-zero diagonal** = genuine arms race, neither side collapsed.

Direct evidence of the closed loop the brief keeps asking for.

### Also track

- **Attack diversity per round** — distinct sequences, entropy of vertical distribution
- **Zero-shot generalization** — hold out V7 and V9 *entirely* from training; does C6 still catch them? Strongest single result you can produce, and it's your "detects novel attacks" claim
- **Cost curve** — fraud loss + friction per round

---

# PART F — THE 10 VERTICALS

A vertical is a **labeled path through the C2 stage machine**, not a property of a C4 call. The C4 tool is one *step* in the path. The vertical is the whole traversal — which is what you stamp episodes with and what C6's gate learns to route on.

```
V1  Voice-clone provisioning
    buy_creds → harvest_voice → call_ivr_provision → attempt_auth×N → cash_out

V2  Deepfake synthetic onboarding
    make_synth_id → harvest_face → submit_kyc → add_payee → transfer_p2p → launder_chain

V3  Agentic phishing → ATO
    phish_holder → reset_password → add_device_selfserve → attempt_auth×N → cash_out

V4  AI-drafted friendly fraud
    (legit card) → attempt_auth×N → file_dispute(LLM letter)

V5  Card testing at scale
    buy_creds(bulk) → attempt_auth(micro)×many → attempt_auth(large) on survivors

V6  Support-channel social engineering
    buy_creds → open_ticket(LLM persona) → escalate_limit → attempt_auth

V7  SIM-swap → OTP interception
    phish_holder → sim_swap → complete_3ds → attempt_auth

V8  Mule network layering
    (bound) → add_payee(mule) → transfer_p2p → launder_chain

V9  Merchant collusion / bust-out
    make_synth_id → onboard_merchant → self_auth cycle → bust_out

V10 Refund abuse via generated evidence
    attempt_auth → request_refund(LLM claim + fake image)
```

Each gets a page in the writeup: how it works in reality, what GenAI specifically unlocked, which stage it attacks, what it emits.

---

# PART G — METRICS

**Identify pillar**
- Count of distinct verticals mapped (10+), each grounded in a real mechanism
- Taxonomy: stage attacked × channel × GenAI capability used

**Generate pillar**
- Real-vs-synthetic discriminator AUC (target < 0.65)
- KS statistic on amount, inter-arrival, MCC frequency
- % benign tripping naive rules (target 5-8%)
- Attack diversity: distinct sequences, sequence entropy
- Capability-tier ablation: success rate at tier 0 / 1 / 2

**Defend pillar**
- PR-AUC (primary), ROC-AUC
- Recall @ 1% FPR, @ 0.1% FPR
- Per-vertical recall breakdown
- Zero-shot recall on held-out verticals
- MoE vs single-model ablation
- Latency per event (< 100ms for inline scoring — feasibility)
- Cost curve: fraud loss vs friction

---

# PART H — WEB PROTOTYPE

Map each panel to the component it exposes:

| Panel | Shows | Sourced from |
|---|---|---|
| 1. Live event stream | events scrolling, risk score, color by band | C2 → C6 payloads |
| 2. Entity graph view | edge appears on bind, vanishes on block | C1 state |
| 3. Episode replay | AttackerObs → Action → Artifact → Outcome, turn by turn | C3/C2/C4 trace |
| 4. Round dashboard | co-evolution matrix, success curve, PR-AUC curve | C7 metrics |
| 5. Artifact viewer | real LLM dispute letter beside a genuine one, E4's score on each | C4 content + C6 |
| 6. Threshold knob | judge drags it, FP/FN trade off live | C6 policy_config |

Panel 6 is disproportionately effective in a live demo. Panel 3 is what proves the RL→LLM ordering to a skeptical judge.

---

# PART I — BUILD ORDER

```
 1. C1 graph + clock + event_log            nothing works without this
 2. C2 action API + stage gate              the contract everything else uses
 3. C5 benign population + fidelity check   BEFORE attackers — it's your ground truth
 4. C6 baseline single model                get a real AUC number early
 5. Scripted attackers, one per vertical    NO RL yet — validates the API covers all 10
 6. C6 retrained on scripted attacks        proves the label pipeline works
 7. C3 RL policy replaces scripted attacker now the interesting part
 8. C4 generative layer for text verticals
 9. C6 MoE split + ablation vs baseline
10. C6 mitigation write-back to C1
11. C7 co-evolution rounds
12. Web prototype
```

**Steps 1-6 give you a complete, submittable system with zero RL.** Everything after is upside. Build in this order so you always have something that runs.
