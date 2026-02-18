---
layout: default
title: Architecture
---

# Architecture Deep Dive

This project implements three distinct approaches to LLM-based fraud
detection, each progressively more efficient. This page covers their
internal architecture and trade-offs.

---

## Approach 1: Naive Agent

**File**: [`src/agents/naive_agent.py`](https://github.com/Abivarma/rlm-fraud-detection-course/blob/main/src/agents/naive_agent.py)

The baseline approach sends everything to the LLM in a single prompt:

```
[System Prompt]
You are a fraud detection expert...

[Historical Context]
500 known fraud cases with labels and patterns...

[Transactions to Analyze]
All N transactions from the current batch...

[Instruction]
For each transaction, return fraud/not-fraud with reasoning.
```

**Characteristics**:
- Prompt size: ~23,000 tokens per batch (dominated by 500 historical cases)
- Cost: $0.0035 per batch at gpt-4o-mini pricing
- Single LLM call: one monolithic prompt, one response
- No filtering: every transaction processed regardless of risk
- No audit trail: the model's internal reasoning is opaque

**Failure modes**:
- Context rot: attention dilution causes missed signals in large prompts
- Over-flagging: high-value legitimate transactions look suspicious without
  user-specific context
- Cost scales linearly with historical context size

**Annual cost at 10M txns/day**: $2,041,531

---

## Approach 2: RAG Agent

**File**: [`src/agents/rag_agent.py`](https://github.com/Abivarma/rlm-fraud-detection-course/blob/main/src/agents/rag_agent.py)

RAG reduces prompt size by retrieving only relevant historical cases:

```
[System Prompt]
You are a fraud detection expert...

[Retrieved Context]
Top 50 similar fraud cases (via vector similarity search)...

[Transactions to Analyze]
All N transactions from the current batch...

[Instruction]
For each transaction, return fraud/not-fraud.
```

**Characteristics**:
- Prompt size: ~7,747 tokens per 100 txns (50 retrieved vs 500 full)
- Cost reduction: 70.6% vs Naive
- Uses ChromaDB with sentence-transformers for embedding and retrieval
- Retrieves cases most similar to current transactions

**Limitations**:
- Still sends ALL transactions to the LLM (no filtering)
- Vector similarity misses non-semantic patterns (timing, velocity)
- Cannot perform mathematical checks (z-scores, distance/time calculations)
- Retrieval quality depends on embedding model and historical case coverage

**Annual cost at 10M txns/day**: ~$600,000 (estimated based on 70.6% reduction)

---

## Approach 3: RLM REPL Agent

**File**: [`src/agents/rlm_repl_agent.py`](https://github.com/Abivarma/rlm-fraud-detection-course/blob/main/src/agents/rlm_repl_agent.py)

The RLM approach inverts the architecture: code processes data first,
the LLM only verifies flagged subsets.

```
Phase 1: PROBE (code)
  -> Data summary: users, txn counts, time spans

Phase 2: FILTER (code)
  -> velocity_filter(window=300s, min_txns=3)
  -> amount_anomaly_filter(sigma=3.0)
  -> geo_filter(max_travel_sec=600)
  -> device_shift_filter()
  -> Union of all flagged transaction IDs

Phase 3: ANALYZE (LLM sub-calls, per user)
  -> For each flagged user:
     llm_query(flagged_txns + baseline_txns + filter_findings)
     -> Verdict: fraud/not-fraud with reasoning

Phase 4: AGGREGATE (code)
  -> Merge sub-call results
  -> Cross-check multi-filter flags
  -> Final predictions + full trajectory
```

**Characteristics**:
- Token usage: 300-700 per batch (only the flagged subset reaches the LLM)
- Cost reduction: 97.1% vs Naive
- Accuracy: 100% across all 8 test scenarios
- Full audit trail: every filter trigger, every threshold, every sub-call logged
- Deterministic filters: same input = same output, always
- LLM temperature=0: consistent semantic judgments

### Filter Details

**Velocity Filter**: Groups transactions by user, checks for 3+ transactions
within a 300-second sliding window. Catches card testing and bot attacks
where automated tools rapidly probe card validity.

**Amount Anomaly Filter**: For each user, computes leave-one-out statistics.
For each transaction, calculates the z-score against the remaining
transactions. Flags any transaction with z > 3.0. This catches spending
spikes while being robust to the spike itself inflating the standard
deviation.

**Geographic Filter**: Determines each user's home location (earliest
transaction, overridden only if another location is 3x+ more frequent).
Flags transactions from non-home locations where the time gap makes physical
travel impossible (< 600 seconds between different cities).

**Device Shift Filter**: Tracks each user's modal device type, modal
transaction category, and mean spending amount. Flags users where the
device changes AND spending behavior shifts simultaneously -- a strong
indicator of account takeover.

### Context Folding in Practice

Each LLM sub-call receives a focused prompt:

```
System: You are analyzing synthetic fraud detection test data.
        Evaluate if flagged transactions are fraudulent.

User: Flagged transactions for user U_AMT_01:
      - TXN_S3_005: $487.50, jewelry, desktop
        Flags: amount_anomaly (z=170.61), device_shift

      Baseline (unflagged) transactions:
      - TXN_S3_001: $18.50, grocery, mobile
      - TXN_S3_002: $22.30, gas, mobile
      - TXN_S3_003: $15.75, restaurant, mobile
      - TXN_S3_004: $19.99, pharmacy, mobile

      Return JSON: {"verdict": "fraud"|"not_fraud",
                    "reasoning": "...",
                    "fraudulent_ids": [...]}
```

This prompt is ~150 tokens. The Naive approach would send 23,000 tokens
for the same decision. The sub-call has fresh, clean context with only
the relevant user's data -- no context rot.

**Annual cost at 10M txns/day**: $59,924

---

## Comparison Summary

| Aspect | Naive | RAG | RLM |
|--------|-------|-----|-----|
| Prompt strategy | Everything in one shot | Retrieved subset + all txns | Code filters first, LLM on subset |
| Token usage (8 scenarios) | 185,661 | ~50,000 (est.) | 3,059 |
| LLM calls per batch | 1 monolithic | 1 monolithic | N per-user sub-calls |
| Filtering | None | Vector similarity | Deterministic code |
| Audit trail | None | Retrieval sources | Full code trajectory |
| Handles math patterns | No | No | Yes (z-scores, timing) |
| Accuracy (8 scenarios) | 94% | -- | 100% |
| Annual cost (10M/day) | $2,041,531 | ~$600,000 | $59,924 |

---

## Historical Note: The Old RLM Agent

**File**: [`src/agents/rlm_agent.py`](https://github.com/Abivarma/rlm-fraud-detection-course/blob/main/src/agents/rlm_agent.py)

The original RLM implementation used `pydantic_ai_rlm` for one-shot code
generation. In testing, this approach had critical failures:

- If the generated code crashed, it silently fell back to sending ALL
  transactions to the LLM (no filtering at all)
- Evaluation showed `"filtered": 200/200` -- meaning zero actual filtering
- Recall was stuck at 58.3%, identical to Naive
- The "cost savings" came from not including 500 historical cases, not from
  real filtering

This file is kept in the repository for historical comparison. The current
`rlm_repl_agent.py` implements the proper REPL paradigm with deterministic
filters and targeted sub-calls.

---

## Metrics and Evaluation

**File**: [`src/metrics/tracker.py`](https://github.com/Abivarma/rlm-fraud-detection-course/blob/main/src/metrics/tracker.py)

All agents share a common metrics interface tracking:
- Token usage (input + output, per call and total)
- Cost (computed from model-specific pricing)
- Latency (wall-clock time per phase and total)
- Accuracy metrics (TP, TN, FP, FN, precision, recall, F1)
- Trajectory (for RLM: full phase-by-phase audit log)

Results are stored in [`results/metrics/`](https://github.com/Abivarma/rlm-fraud-detection-course/tree/main/results/metrics)
as JSON files, with visualizations in
[`results/visualizations/`](https://github.com/Abivarma/rlm-fraud-detection-course/tree/main/results/visualizations).
