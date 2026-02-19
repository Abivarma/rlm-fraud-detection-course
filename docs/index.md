---
layout: default
title: Home
---

# RLM-Inspired Fraud Detection

**98.4% fewer tokens. 100% accuracy on synthetic benchmarks. $1.98M/year savings at 10M transactions/day.**

This project implements an **RLM-inspired orchestration pipeline** for
financial fraud detection. Inspired by the Recursive Language Model paradigm
([arXiv:2512.24601](https://arxiv.org/abs/2512.24601)), it uses deterministic
code filters to process data and targets the LLM only at flagged subsets --
outperforming naive context-stuffing on cost, accuracy, and auditability.

> **Terminology note**: This is not a true Recursive Language Model where the
> LLM controls its own reasoning loop. It is a **code-controlled pipeline**
> that borrows key principles from the RLM paradigm -- context folding,
> symbolic filtering, and targeted sub-calls. The control plane is
> deterministic Python, not the model. See
> [How This Differs from True RLM and Tool Calling](architecture#how-this-differs-from-true-rlm-and-tool-calling)
> for a detailed comparison.

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
- **Not a true RLM**: The LLM does not control the reasoning loop. All
  orchestration is deterministic Python code.
  [See the full comparison](architecture#how-this-differs-from-true-rlm-and-tool-calling).

What this **does** demonstrate: the principle that deterministic code
filtering before LLM invocation dramatically reduces cost and improves
focus. That principle holds regardless of data source or scale.

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

Built by [Abivarma](https://github.com/Abivarma) as a proof-of-concept
applying principles from the RLM paradigm
([arXiv:2512.24601](https://arxiv.org/abs/2512.24601)) to fraud detection.
The implementation uses hardcoded deterministic fraud filters with targeted
OpenAI gpt-4o-mini sub-calls for semantic verification -- an orchestration
pattern, not a true recursive model.

[View the source code on GitHub](https://github.com/Abivarma/rlm-fraud-detection-course)
