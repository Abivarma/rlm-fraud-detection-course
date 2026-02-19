---
layout: default
title: Research
---

# Research Background

This page covers the core problems that motivate the filter-then-verify
approach: context rot in monolithic prompts, the limitations of RAG for
computational patterns, and the principle of symbolic pre-filtering before
LLM invocation.

---

## The Problem with Current Approaches

### Context Rot in Monolithic Prompts

In standard Transformers, self-attention requires every token to attend to
every other token -- scaling quadratically with sequence length. As the
context window grows, the relative attention weight on any single signal
token decreases. This is **context rot**: a single anomalous transaction
among 50,000 benign ones gets drowned out.

Empirically, this manifests as a **58.3% recall ceiling** on complex
filtering tasks. Models that perform well on small contexts degrade
significantly when the same task is embedded in a larger context.

### Why RAG Falls Short

Retrieval-Augmented Generation (RAG) attempts to mitigate context rot by
selecting top-k relevant chunks via vector similarity. However:

- Fraud patterns are often **non-local**: a velocity attack looks normal
  transaction-by-transaction; the signal is in the timing between them.
- Vector similarity misses **temporal and mathematical patterns** that
  require computation, not semantic matching.
- RAG cannot perform **cross-referencing** across disparate events that
  don't share semantic similarity in embedding space.

### Why One-Shot Code Generation Fails

Generating a single filtering script and executing it is fragile:

- If the script has a minor error, the system returns empty results.
- No feedback loop: the model cannot observe and correct its own mistakes.
- Silent failures are worse than explicit errors in production.

---

## The Filter-Then-Verify Principle

The core insight is simple: **use cheap deterministic computation to reduce
what the expensive LLM sees**. This principle is valid regardless of
implementation approach.

### Symbolic Filtering Before LLM Invocation

Instead of sending all data to the LLM, code processes data first:

1. **Deterministic filters** identify suspicious patterns using exact
   computation (z-scores, time windows, location matching)
2. **Only flagged data** reaches the LLM for semantic verification
3. **Each LLM sub-call** gets fresh, focused context for one user

This is the same pattern used at production scale by companies like Stripe
(Radar), PayPal, and Feedzai -- rules and ML handle the bulk, expensive
analysis handles the edge cases.

### Context Folding

When the LLM is called, each sub-call receives only the relevant user's
data. A batch of 15 transactions across 5 users becomes 2 sub-calls
(for the 2 flagged users), each seeing only their own transactions with
fresh context. This prevents attention dilution and cross-contamination.

### The Cost Equation

The savings come from a structural change:

```
Naive:    All N transactions + 500 historical cases → LLM  (~23,000 tokens)
Pipeline: Code filters → flagged subset only → LLM        (~300-700 tokens)
```

The ratio holds regardless of scale because the filter logic is
deterministic and computationally free. At 10M transactions/day, this
translates to $1.98M/year in API cost savings.

---

## Academic Context

The idea of combining symbolic computation with LLM reasoning has been
explored in several research directions:

| Concept | Description | Relevance |
|---------|-------------|-----------|
| Context folding | Fresh per-sub-call context instead of monolithic prompts | Core technique used in this project |
| Symbolic filtering | Code processes data before the LLM sees it | The 4 deterministic filters |
| Targeted sub-calls | LLM only analyzes flagged subset | Per-user sub-calls in Phase 3 |
| Hybrid symbolic + neural | Rules for cheap filtering, models for judgment | The architecture pattern |

Research on Recursive Language Models ([arXiv:2512.24601](https://arxiv.org/abs/2512.24601))
explores a more advanced version of this principle where the LLM itself
controls the reasoning loop, generates code, and decides when to recurse.
This project does **not** implement that level of model autonomy -- the
pipeline is a fixed, code-controlled sequence.

---

## What This Project Demonstrates

The value is in the **architectural principle**: use cheap deterministic
computation to reduce what the expensive LLM sees.

- **98.4% token reduction** comes from code filters, not from model autonomy
- **100% accuracy** (on synthetic data) comes from focused context, not
  from self-directed reasoning
- **$1.98M/year savings** comes from the filter-then-verify pattern

Whether you implement this as hardcoded filters (this project), tool
calling, or a model-controlled system, the cost savings from "don't send
everything to the LLM" remain.

---

## Related Implementations

| Project | Description |
|---------|-------------|
| [Stripe Radar](https://stripe.com/radar) | ML + rules + LLM hybrid, blocks $20B+ fraud annually |
| [PayPal fraud detection](https://www.paypal.com) | LLM for transaction explanation, reduced review time 40% |
| [Feedzai](https://feedzai.com) | LLM + ML ensemble, 60% reduction in false positives |
