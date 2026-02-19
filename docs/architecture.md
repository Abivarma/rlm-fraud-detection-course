---
layout: default
title: Architecture
---

# Architecture Deep Dive

This project implements three distinct approaches to LLM-based fraud
detection, each progressively more efficient. This page covers their
internal architecture, trade-offs, and an honest comparison of how this
system relates to tool calling, true RLM, and standard LLM patterns.

---

## Approach 1: Naive Agent

**File**: [`src/agents/naive_agent.py`](https://github.com/Abivarma/smart-llm-fraud-detection/blob/main/src/agents/naive_agent.py)

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

**File**: [`src/agents/rag_agent.py`](https://github.com/Abivarma/smart-llm-fraud-detection/blob/main/src/agents/rag_agent.py)

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

## Approach 3: RLM-Inspired Pipeline

**File**: [`src/agents/rlm_repl_agent.py`](https://github.com/Abivarma/smart-llm-fraud-detection/blob/main/src/agents/rlm_repl_agent.py)

The RLM-inspired approach inverts the architecture: **hardcoded** code
filters process data first, the LLM only verifies flagged subsets. All
orchestration is deterministic Python -- the LLM does not control the
pipeline.

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
- Accuracy: 100% across all 8 synthetic test scenarios (see [Limitations](index#limitations))
- Full audit trail: every filter trigger, every threshold, every sub-call logged
- **Hardcoded filters**: fixed thresholds (300s window, z > 3.0, etc.) -- same input = same output, always
- LLM temperature=0: consistent semantic judgments
- **Code-controlled**: the pipeline is fixed; the LLM does not choose what to run

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

## How This Differs from True RLM and Tool Calling

This section addresses a fair criticism: this system is not a true Recursive
Language Model. Here is an honest comparison of four approaches to
LLM-based systems.

### The Four Approaches

**1. Standard Tool Calling / Function Calling**

The LLM decides which tools to invoke and when. The model is the
orchestrator -- it receives a list of available functions (e.g.,
`search_transactions()`, `check_velocity()`, `flag_fraud()`), decides
which to call based on the prompt, observes results, and decides next steps.

```
User: "Analyze these transactions for fraud"
LLM:  "I'll call check_velocity(user_id='U001')"    <- LLM decides
Tool: {result: "3 txns in 60s"}
LLM:  "That's suspicious. I'll call flag_fraud()"    <- LLM decides
Tool: {result: "flagged"}
LLM:  "Transaction is likely fraud because..."
```

The LLM has **agency** -- it picks which tools to use and in what order.
The code provides tools; the model provides strategy.

**2. This Implementation (RLM-Inspired Pipeline)**

Python code decides everything. The pipeline is **fixed and hardcoded**:
always PROBE, then FILTER (4 specific filters with fixed thresholds), then
ANALYZE (LLM sub-call), then AGGREGATE. The LLM is called at exactly one
point with a fixed prompt template. It cannot change the pipeline, skip
filters, adjust thresholds, or request additional analysis.

```
Code: probe(data)                                    <- code decides
Code: velocity_filter(window=300s)                   <- code decides
Code: amount_anomaly_filter(sigma=3.0)               <- code decides
Code: geo_filter(max_travel=600s)                    <- code decides
Code: device_shift_filter()                          <- code decides
Code: if flagged_users: llm_query(subset)            <- code decides
LLM:  "Given z-score=170.61, this is fraud"          <- LLM confirms
Code: aggregate(results)                             <- code decides
```

The LLM has **no agency**. It is a verifier, not an orchestrator. The code
provides both the tools AND the strategy.

**3. True RLM (from the Paper)**

The LLM operates inside a REPL environment. It generates code, executes it,
observes results, and decides what to do next -- including writing new
filters, adjusting parameters, or recursing with `llm_query` sub-calls.
The model controls the loop.

```
LLM:  "Let me examine the data structure first"
LLM:  >>> df.groupby('user_id').count()              <- LLM generates code
REPL: {U001: 5 txns, U002: 3 txns}
LLM:  "U001 has high volume. Let me check timing"
LLM:  >>> df[df.user_id=='U001'].timestamp.diff()    <- LLM generates code
REPL: [35s, 37s, 58s, 48s]
LLM:  "Rapid succession. I'll use llm_query for deep analysis"
LLM:  >>> llm_query("Analyze U001's 5 transactions")  <- LLM recurses
Sub-LLM: "Velocity pattern indicates card testing"
LLM:  >>> answer["fraudulent"] = ["TXN001", ...]     <- LLM finalizes
```

The LLM has **full agency**. It decides what code to write, what to
investigate, when to recurse, and when to stop.

**4. Agentic Tool Use (e.g., ReAct, LangChain Agents)**

The LLM follows a Thought-Action-Observation loop with predefined tools.
Similar to tool calling, but with explicit reasoning traces and iterative
refinement.

### Comparison Table

| Aspect | Tool Calling | This Project | True RLM | Agentic (ReAct) |
|--------|-------------|-------------|----------|-----------------|
| Who controls the loop | LLM | **Code** | LLM | LLM |
| Who picks which tools | LLM | **Code** | LLM (writes code) | LLM |
| LLM generates code | No | No | **Yes** | No |
| Filter logic | Predefined tools | **Hardcoded filters** | LLM-generated | Predefined tools |
| Thresholds | Fixed or tool params | **Fixed** | LLM-chosen | Fixed or tool params |
| LLM can recurse | Via tool calls | **No** | Yes (llm_query) | Via tool calls |
| Deterministic | No (LLM chooses path) | **Yes** | No | No |
| Reproducible | No | **Yes** | No | No |
| LLM autonomy | Medium | **None** | Full | Medium |

### So What IS This Project?

The most accurate labels:

- **Rule-gated LLM verifier**: Deterministic rules filter data, LLM
  confirms findings
- **Hybrid symbolic + LLM pipeline**: Code handles computation, LLM
  handles semantic judgment
- **RLM-inspired orchestration**: Borrows context folding, sub-calls,
  and symbolic filtering from the RLM paradigm, but with hardcoded
  control flow

### What This Demonstrates

The value is not in the "recursive" label -- it is in the **architectural
principle**: use cheap deterministic computation to reduce what the
expensive LLM sees. This principle is valid regardless of who controls
the loop:

- **98.4% token reduction** comes from code filters, not from recursion
- **100% accuracy** (on synthetic data) comes from focused context, not
  from model self-direction
- **$1.98M/year savings** comes from the filter-then-verify pattern, not
  from the LLM being a controller

Whether you implement this as hardcoded filters (this project), tool
calling, or a true RLM, the cost savings from "don't send everything to
the LLM" remain.

---

## Comparison Summary

| Aspect | Naive | RAG | RLM-Inspired |
|--------|-------|-----|-------------|
| Prompt strategy | Everything in one shot | Retrieved subset + all txns | Hardcoded filters first, LLM on subset |
| Token usage (8 scenarios) | 185,661 | ~50,000 (est.) | 3,059 |
| LLM calls per batch | 1 monolithic | 1 monolithic | N per-user sub-calls |
| Filtering | None | Vector similarity | Deterministic code (fixed thresholds) |
| Control plane | LLM (single call) | LLM (single call) | Python code (hardcoded) |
| Audit trail | None | Retrieval sources | Full code trajectory |
| Handles math patterns | No | No | Yes (z-scores, timing) |
| Accuracy (8 synthetic scenarios) | 94% | -- | 100% |
| Annual cost (10M/day) | $2,041,531 | ~$600,000 | $59,924 |

---

## Historical Note: The Old RLM Agent

**File**: [`src/agents/rlm_agent.py`](https://github.com/Abivarma/smart-llm-fraud-detection/blob/main/src/agents/rlm_agent.py)

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

**File**: [`src/metrics/tracker.py`](https://github.com/Abivarma/smart-llm-fraud-detection/blob/main/src/metrics/tracker.py)

All agents share a common metrics interface tracking:
- Token usage (input + output, per call and total)
- Cost (computed from model-specific pricing)
- Latency (wall-clock time per phase and total)
- Accuracy metrics (TP, TN, FP, FN, precision, recall, F1)
- Trajectory (for RLM: full phase-by-phase audit log)

Results are stored in [`results/metrics/`](https://github.com/Abivarma/smart-llm-fraud-detection/tree/main/results/metrics)
as JSON files, with visualizations in
[`results/visualizations/`](https://github.com/Abivarma/smart-llm-fraud-detection/tree/main/results/visualizations).
