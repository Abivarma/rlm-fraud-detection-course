---
layout: default
title: Home
---

# Smart LLM Fraud Detection

**97% fewer LLM costs. 98.4% fewer tokens. $1.98M/year savings at 10M transactions/day.**

A **filter-then-verify pipeline** for LLM-based fraud detection. Instead
of stuffing every transaction into one massive LLM prompt, this system runs
deterministic code filters first and only calls the LLM on the flagged
subset -- cutting API costs by 97% while improving accuracy on synthetic
benchmarks.

> **What this is**: A linear Python pipeline with hardcoded rule-based
> filters + targeted LLM verification. The LLM has zero autonomy -- code
> controls everything. This is not an AI agent, not a recursive model,
> and not tool calling. It is a straightforward sequential pipeline.
> See [how it compares to other approaches](architecture#how-this-differs-from-other-approaches).

---

## How It Works

Traditional fraud detection sends all transactions plus historical context
to the LLM in a single monolithic prompt (~23,000 tokens). This approach
replaces that with a 4-phase pipeline where 3 out of 4 phases use zero LLM
tokens:

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

Tested across 8 **synthetic** fraud scenarios (51 transactions) with live gpt-4o-mini API calls:

| Metric | Naive (Baseline) | Pipeline | Improvement |
|--------|-----------------|----------|-------------|
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

| Daily Volume | Naive / Year | Pipeline / Year | Annual Savings |
|-------------|-------------|----------------|----------------|
| 10K txns | $2,042 | $60 | $1,982 |
| 100K txns | $20,415 | $599 | $19,816 |
| 1M txns | $204,153 | $5,992 | $198,161 |
| 10M txns | $2,041,531 | $59,924 | **$1,981,606** |

Projections based on observed per-transaction costs with gpt-4o-mini pricing
($0.15/1M input tokens, $0.60/1M output tokens).

---

## Why This Approach Wins

**Cost**: Code filters are computationally free. The LLM only processes the
suspicious subset (typically 10-30% of transactions), reducing token usage
by 98%+.

**Accuracy**: Each LLM sub-call gets fresh, focused context for one user --
no context rot from a monolithic prompt. Deterministic filters catch patterns
(velocity, geo) that attention-based models miss.

**Auditability**: Every decision has an executable code trace showing which
filter triggered, what thresholds were crossed, and what the LLM verified.
This is critical for financial services compliance.

**Scale**: Naive approaches hit context window limits. This approach processes
per-user with context folding -- it can handle millions of transactions by
filtering first, then making targeted sub-calls.

---

## Limitations

This is a **proof-of-concept**, not a production system. Be clear about what
it demonstrates and what it does not:

- **Synthetic data only**: 51 hand-crafted transactions across 8 scenarios
  with clear, unambiguous fraud patterns. Real-world fraud is subtler --
  sophisticated card testing, synthetic identities, coordinated mule networks.
  The 100% accuracy validates the architecture, not real-world performance.
- **No class imbalance**: Real fraud rates are ~0.1%. This dataset is ~50%
  fraud by design. Production performance under extreme class imbalance is
  untested.
- **Hardcoded filters**: The four fraud filters (velocity, amount anomaly,
  geographic, device shift) use fixed thresholds. Production deployment
  requires tuning these to your specific fraud patterns, data distribution,
  and risk tolerance.
- **No concept drift handling**: Fraud tactics evolve. This system has no
  retraining pipeline, no threshold adaptation, and no feedback loop from
  outcomes.
- **Linear pipeline**: The LLM does not control the reasoning loop. All
  orchestration is deterministic Python code.
  [See the full comparison](architecture#how-this-differs-from-other-approaches).

What this **does** demonstrate: the principle that deterministic code
filtering before LLM invocation dramatically reduces cost and improves
focus. That principle holds regardless of data source or scale.

---

## Explore Further

| Page | Description |
|------|-------------|
| [How It Works](how-it-works) | 4-phase pipeline with real trajectory examples |
| [Results](results) | All 8 scenarios with detailed comparisons |
| [Research](research) | Context rot, attention dilution, and filter-then-verify principles |
| [Architecture](architecture) | Technical deep dive into all three agents |
| [Enterprise FAQ](faq) | 19 questions on costs, scaling, production readiness |

---

## About This Project

Built by [Abivarma](https://github.com/Abivarma) as a quantified case
study showing how deterministic pre-filtering reduces LLM API costs by 97%.
The implementation uses hardcoded Python filters with targeted OpenAI
gpt-4o-mini sub-calls for semantic verification.

[View the source code on GitHub](https://github.com/Abivarma/smart-llm-fraud-detection)
