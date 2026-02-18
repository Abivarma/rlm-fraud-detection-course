---
layout: default
title: Home
---

# Recursive Language Models for Fraud Detection

**98.4% fewer tokens. 100% accuracy. $1.98M/year savings at 10M transactions/day.**

This project demonstrates how the Recursive Language Model (RLM) paradigm --
where deterministic code filters handle data processing and the LLM only
verifies flagged subsets -- outperforms traditional approaches to LLM-based
fraud detection on every axis: cost, accuracy, and auditability.

---

## How RLM Works

Traditional fraud detection sends all transactions plus historical context
to the LLM in a single monolithic prompt (~23,000 tokens). RLM replaces this
with a 4-phase pipeline where 3 out of 4 phases use zero LLM tokens:

```
Input (N transactions)
  |
  PROBE       Examine structure, count users/txns           [0 tokens]
  |
  FILTER      Deterministic: velocity, geo, amount,         [0 tokens]
  |           device shift checks
  |
  ANALYZE     LLM sub-calls on flagged subset ONLY          [~300-700 tokens]
  |           (per-user context folding)
  |
  AGGREGATE   Merge verdicts, cross-check                   [0 tokens]
  |
Output (predictions + full audit trail)
```

[Read the full technical walkthrough](how-it-works)

---

## Results Summary

Tested across 8 fraud scenarios (51 transactions) with live gpt-4o-mini API calls:

| Metric | Naive (Baseline) | RLM | Improvement |
|--------|-----------------|-----|-------------|
| Tokens | 185,661 | 3,059 | **98.4% reduction** |
| Cost | $0.0285 | $0.0008 | **97.1% reduction** |
| Accuracy | 94% (48/51) | **100%** (51/51) | +6 points |
| False Positives | 1 | 0 | Eliminated |
| Missed Fraud | 2 | 0 | Eliminated |
| Audit Trail | None | Full code trajectory | Compliance-ready |

The 8 scenarios cover velocity attacks, geographic impossibility, amount
anomalies, account takeover, micro-transaction testing, legitimate high-value
users (false positive test), mixed multi-user batches, and cross-border rapid
transactions.

[See all 8 scenarios with full trajectories](results)

---

## Cost Projections at Enterprise Scale

| Daily Volume | Naive / Year | RLM / Year | Annual Savings |
|-------------|-------------|-----------|----------------|
| 10K txns | $2,042 | $60 | $1,982 |
| 100K txns | $20,415 | $599 | $19,816 |
| 1M txns | $204,153 | $5,992 | $198,161 |
| 10M txns | $2,041,531 | $59,924 | **$1,981,606** |

Projections based on observed per-transaction costs with gpt-4o-mini pricing
($0.15/1M input tokens, $0.60/1M output tokens).

---

## Why RLM Wins

**Cost**: Code filters are computationally free. The LLM only processes the
suspicious subset (typically 10-30% of transactions), reducing token usage
by 98%+.

**Accuracy**: Each LLM sub-call gets fresh, focused context for one user --
no context rot from a monolithic prompt. Deterministic filters catch patterns
(velocity, geo) that attention-based models miss.

**Auditability**: Every decision has an executable code trace showing which
filter triggered, what thresholds were crossed, and what the LLM verified.
This is critical for financial services compliance.

**Scale**: Naive approaches hit context window limits. RLM processes per-user
with context folding -- it can handle millions of transactions by filtering
first, then making targeted sub-calls.

---

## Explore Further

| Page | Description |
|------|-------------|
| [How It Works](how-it-works) | 4-phase REPL loop with real trajectory examples |
| [Results](results) | All 8 scenarios with detailed comparisons |
| [Research](research) | The RLM paradigm from arXiv:2512.24601 |
| [Architecture](architecture) | Technical deep dive into all three agents |
| [Enterprise FAQ](faq) | 16 questions on costs, scaling, production readiness |

---

## About This Project

Built by [Abivarma](https://github.com/Abivarma) as a practical demonstration
of the RLM paradigm described in [arXiv:2512.24601](https://arxiv.org/abs/2512.24601).
The implementation uses a custom lightweight REPL loop with deterministic fraud
filters and OpenAI gpt-4o-mini for semantic verification.

[View the source code on GitHub](https://github.com/Abivarma/rlm-fraud-detection-course)
