---
layout: default
title: How It Works
---

# How the Pipeline Works

This implementation is **inspired by** the Recursive Language Model (RLM)
paradigm ([arXiv:2512.24601](https://arxiv.org/abs/2512.24601)), which treats
the LLM as an orchestrator that interacts with data through a persistent REPL
rather than ingesting it into the context window.

**Important distinction**: In a true RLM, the LLM controls the reasoning
loop -- it decides what code to run, observes results, and iterates. In this
implementation, the **control plane is hardcoded Python**. The 4-phase
pipeline is fixed, the filters are pre-built, and the LLM is called only at
a specific point (Phase 3) for semantic verification. The LLM does not
decide what to filter or when to recurse.

This is an **RLM-inspired orchestration pipeline**, not a true recursive
model. What we borrow from RLM: context folding, symbolic filtering before
LLM invocation, and targeted sub-calls. What differs: all orchestration
logic is deterministic code, not model-driven.

---

## The 4 Phases

### Phase 1: PROBE (0 LLM tokens)

Examine the data structure without any LLM calls. Count users, transactions
per user, time spans, and data shape.

```python
users = group_by_user(context)
for uid, txns in users.items():
    print(f'{uid}: {len(txns)} txns')
time_range = (min(timestamps), max(timestamps))
```

**Output example** (Scenario 1 -- Velocity Attack):
```
Users: {"U_VEL_01": 5}
Time range: 178s (0.0 hours)
Total transactions: 5
```

This costs nothing and tells the system what it is working with.

### Phase 2: FILTER (0 LLM tokens)

Run four **hardcoded** deterministic fraud pattern filters. These are pure
Python functions with fixed thresholds, executing in microseconds with zero
LLM involvement. The thresholds (e.g., 300-second window, z-score > 3.0)
are chosen for the synthetic demo scenarios and would need tuning for
real-world data:

**Velocity Filter**: Flags users with 3+ transactions within a 300-second window.
Catches card testing and bot attacks.

**Amount Anomaly Filter**: Computes leave-one-out z-scores for each transaction.
If removing a transaction from the user's history reveals it is 3+ standard
deviations above the mean, it is flagged. Catches spending spikes.

**Geographic Filter**: Determines each user's "home" location (earliest
transaction location, unless another location is 3x+ more frequent). Flags
transactions from non-home locations within a short time window -- physical
travel would be impossible.

**Device Shift Filter**: Tracks each user's modal device, category, and
spending range. Flags transactions where the device changes AND the spending
pattern shifts simultaneously -- a signature of account takeover.

```python
velocity_flags  = velocity_filter(context, window=300s, min=3)
amount_flags    = amount_anomaly_filter(context, sigma=3.0)
geo_flags       = geo_filter(context, max_sec=600)
device_flags    = device_shift_filter(context)
suspicious_ids  = union(all flagged txn_ids)
```

**Output example** (Scenario 3 -- Amount Spike):
```
velocity_filter:       0 users flagged (none)
amount_anomaly_filter: 1 txns flagged ({"TXN_S3_005": "z=170.61"})
geo_filter:            0 users flagged (none)
device_shift_filter:   1 users flagged ({"U_AMT_01": ["TXN_S3_005"]})
Total suspicious: 1/5 transactions (80% filtered out)
```

Out of 5 transactions, only 1 is flagged. The LLM will only see this one.

### Phase 3: ANALYZE (minimal LLM tokens)

For each flagged user, make a targeted LLM sub-call with **only** that user's
data: their flagged transactions, their baseline transactions, and what the
filters found. This is context folding -- each sub-call gets a fresh, focused
context window instead of a diluted monolithic prompt.

```python
for user in flagged_users:
    user_txns = get_user_context(user)  # flagged + baseline
    result = llm_query(user_txns, filter_findings)
    if result.verdict == 'fraud':
        confirmed.add(result.txn_ids)
```

**Output example** (Scenario 3 -- Amount Spike):
```
1 sub-call (1 per user):
llm_query(user=U_AMT_01, 1 flagged txn + 4 baseline) -> 379 tokens
  Response: {"verdict": "fraud", "reasoning": "The flagged transaction
  TXN_S3_005 shows a significant amount anomaly with $487.50 compared
  to the user's normal amounts ($15-22 range)..."}
-> Confirmed fraud: ['TXN_S3_005']
```

The LLM used 379 tokens. Naive would have used 23,040.

### Phase 4: AGGREGATE (0 LLM tokens)

Merge sub-call results into the final answer. Cross-check: if a transaction
was flagged by 2+ independent filters, confidence increases.

```python
answer['fraudulent_ids'] = merge(confirmed_ids)
answer['reasoning'] = merge(sub_call_reasoning)
answer['ready'] = True
```

---

## Worked Example: Velocity Attack (Scenario 1)

**Input**: 5 transactions from user U_VEL_01, all within 178 seconds.

| Transaction | Amount | Category | Location | Device |
|-------------|--------|----------|----------|--------|
| TXN_S1_001 | $45.99 | electronics | NYC | mobile |
| TXN_S1_002 | $52.30 | electronics | NYC | mobile |
| TXN_S1_003 | $38.75 | clothing | NYC | mobile |
| TXN_S1_004 | $61.20 | electronics | NYC | mobile |
| TXN_S1_005 | $47.85 | electronics | NYC | mobile |

**PROBE**: 1 user, 5 transactions, 178-second time span.

**FILTER**: Velocity filter flags all 5 (5 txns in 178s < 300s threshold).
No other filters trigger. 0% filtered out (all suspicious).

**ANALYZE**: 1 LLM sub-call with all 5 flagged transactions. 387 tokens.
Verdict: fraud -- "high transaction velocity, 5 transactions in 178 seconds,
indicative of potential card testing."

**AGGREGATE**: All 5 confirmed as fraud. Correct (ground truth: all 5 are fraud).

**Comparison**:

| Metric | Naive | RLM |
|--------|-------|-----|
| Tokens | 23,167 | 387 |
| Cost | $0.0036 | $0.0001 |
| Accuracy | 100% | 100% |
| Token savings | -- | **98.3%** |

Same result, 98.3% fewer tokens.

---

## Worked Example: Amount Spike (Scenario 3)

**Input**: 5 transactions from user U_AMT_01 over 4 hours.

| Transaction | Amount | Category | Location | Device | Label |
|-------------|--------|----------|----------|--------|-------|
| TXN_S3_001 | $18.50 | grocery | LA | mobile | legit |
| TXN_S3_002 | $22.30 | gas | LA | mobile | legit |
| TXN_S3_003 | $15.75 | restaurant | LA | mobile | legit |
| TXN_S3_004 | $19.99 | pharmacy | LA | mobile | legit |
| TXN_S3_005 | $487.50 | jewelry | LA | desktop | **fraud** |

**PROBE**: 1 user, 5 transactions, 4-hour span.

**FILTER**: Amount anomaly filter computes leave-one-out z-score for TXN_S3_005:
mean of other 4 = $19.14, std = $2.70, z-score = (487.50 - 19.14) / 2.70 = **170.61**.
Device shift filter also flags TXN_S3_005 (mobile to desktop + category shift).
**80% of transactions filtered out** -- only 1 sent to LLM.

**ANALYZE**: LLM confirms: "significant amount anomaly, $487.50 vs normal $15-22
range, device change from mobile to desktop, category shift to luxury."
379 tokens.

**AGGREGATE**: TXN_S3_005 confirmed as fraud. Correct.

**Comparison**:

| Metric | Naive | RLM |
|--------|-------|-----|
| Tokens | 23,040 | 379 |
| Cost | $0.0035 | $0.0001 |
| Accuracy | 100% | 100% |
| Data sent to LLM | 5 txns + 500 cases | 1 txn + 4 baseline |

RLM sent 98.4% fewer tokens by letting the code filter identify the obvious
anomaly (z=170.61) before involving the LLM.

---

## Why This Beats Monolithic Prompts

**Context rot**: In a 23,000-token prompt, the model's attention is spread
across 500 historical cases and all transactions. Critical signals get diluted.
In Scenario 7 (Mixed Batch), Naive missed 1 of 4 fraud transactions. This
pipeline caught all 4 because each sub-call had fresh, focused context.

**Symbolic vs probabilistic**: Code filters produce deterministic results.
`z-score = 170.61` is a fact, not a probability. The LLM is only asked
"given this evidence, is this fraud?" -- a judgment call it excels at.

**Context folding**: Each sub-call is independent. A 15-transaction batch
with 5 users becomes 2 sub-calls (for the 2 flagged users), each seeing
only their own transactions. No cross-contamination between users.

---

## What This Is Not

This pipeline **borrows principles** from the RLM paper but differs in a
fundamental way: **who controls the loop**.

In a true RLM, the model generates code, executes it in a REPL, observes
results, and decides what to do next. The model is the controller.

In this implementation, every step is predetermined:
- The pipeline always runs 4 phases in order
- The filters are hardcoded Python with fixed thresholds
- The LLM is called at exactly one point (Phase 3) with a fixed prompt template
- The model has no ability to change the pipeline, add filters, or recurse

This is closer to a **rule-gated LLM verifier** -- a valid and practical
design pattern, but distinct from recursive self-directed reasoning.
See [Architecture: How This Differs](architecture#how-this-differs-from-true-rlm-and-tool-calling)
for the full comparison.

---

## Source Code

The complete implementation is in
[`src/agents/rlm_repl_agent.py`](https://github.com/Abivarma/rlm-fraud-detection-course/blob/main/src/agents/rlm_repl_agent.py).

Key methods:
- `_velocity_filter()` -- time-window based card testing detection
- `_amount_anomaly_filter()` -- leave-one-out z-score computation
- `_geo_filter()` -- impossible travel detection with home location logic
- `_device_shift_filter()` -- behavioral profile change detection
- `_llm_query()` -- targeted sub-call with fraud-specific prompt
- `analyze()` -- main entry point, orchestrates the 4-phase loop
- `format_trajectory()` -- renders the full audit trail
