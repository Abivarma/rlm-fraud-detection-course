---
layout: default
title: Enterprise FAQ
---

# Enterprise FAQ

19 questions addressing terminology, scope, cost claims, production
readiness, accuracy, technical limitations, and deployment considerations.

---

## Terminology and Scope

### Q1: What exactly is this system?

This is a **rule-gated LLM verifier** -- a linear Python pipeline where
hardcoded deterministic filters process data first, and the LLM only
verifies the flagged subset. The LLM has zero autonomy.

The 4-phase pipeline is fixed: PROBE, FILTER (4 specific filters with
fixed thresholds), ANALYZE (LLM sub-call), AGGREGATE. The LLM is called
at exactly one point with a fixed prompt template. It cannot change the
pipeline, skip filters, adjust thresholds, or request additional analysis.

More accurate labels:
- **Rule-gated LLM verifier**
- **Hybrid symbolic + LLM pipeline**
- **Filter-then-verify pipeline**

### Q2: How is this different from standard tool calling / function calling?

In tool calling (OpenAI function calling, LangChain agents, etc.), the
**LLM decides** which tools to invoke and when. The model has agency --
it picks actions based on its reasoning.

In this system, **code decides** everything:

| Aspect | Tool Calling | This Project |
|--------|-------------|-------------|
| Who picks which function | LLM | Hardcoded Python |
| Who decides execution order | LLM | Fixed 4-phase pipeline |
| Can the LLM skip a step | Yes | No |
| Can the LLM add analysis | Yes | No |
| LLM autonomy | Medium | None |
| Deterministic | No | Yes |
| Reproducible | No | Yes |

The trade-off: this system sacrifices flexibility for **determinism and
reproducibility**. Same input always produces the same filter results.
The LLM's only role is semantic confirmation of pre-computed findings.

Both approaches can achieve the cost savings from "filter before LLM."
The difference is who makes the filtering decisions.

### Q3: If the filters are hardcoded, isn't this just a rule engine with an LLM step?

Essentially, yes -- and that is the point.

The 4 filters (velocity, amount anomaly, geographic, device shift) are
deterministic Python functions with fixed thresholds. They are rules. The
LLM step adds:

1. **Semantic judgment**: The LLM evaluates whether code-detected anomalies
   constitute fraud in context (e.g., a z-score of 170 on a jewelry purchase
   from a new device is different from a z-score of 170 on a recurring bill)
2. **Natural language reasoning**: The audit trail includes human-readable
   explanations, not just threshold triggers
3. **Adaptability**: Changing fraud detection strategy requires updating a
   prompt, not retraining a model

The architecture is: **rules for cheap filtering + LLM for expensive
judgment**. This is a well-established enterprise pattern (Stripe Radar,
PayPal's fraud stack). The contribution is demonstrating the cost impact
of this pattern quantitatively.

---

## Cost and Scalability

### Q4: "98.4% token reduction" -- is this realistic at enterprise scale?

Yes. The numbers are mathematically derived from observed per-transaction
costs across 8 scenarios (51 transactions) with live gpt-4o-mini API calls.

| Approach | Tokens (8 scenarios) | Cost | Per-Transaction |
|----------|---------------------|------|-----------------|
| Naive | 185,661 | $0.0285 | $0.000559 |
| Pipeline | 3,059 | $0.0008 | $0.000016 |

The savings come from a structural change: Naive sends ~23,000 tokens per
batch (500 historical cases + all transactions). The pipeline sends 300-700
tokens (only flagged subset after code filtering). This ratio holds
regardless of scale because the filter logic is deterministic.

At 10M transactions/day: Naive = $2,041,531/year, Pipeline = $59,924/year.

**Caveat**: These are LLM API costs only. Production adds infrastructure
costs (monitoring, compute, storage) of approximately $2,400-5,000/year.

### Q5: What about infrastructure costs beyond API fees?

Production deployment adds:

| Component | Estimated Annual Cost |
|-----------|---------------------|
| Monitoring (Datadog/similar) | $1,200 |
| Compute (API server) | $600-1,200 |
| Vector DB (for RAG, if used) | $0-840 |
| Logging and audit storage | $200-400 |
| **Total infrastructure** | **$2,000-3,640** |

Even with infrastructure, the pipeline at $60 + $3,640 = $3,700/year is
dramatically cheaper than Naive at $2,041,531 + $3,640 = $2,045,171/year
at 10M txns/day.

### Q6: How does cost scale with transaction volume?

Linearly. Each transaction batch is independent.

| Daily Volume | Naive / Year | Pipeline / Year | Savings |
|-------------|-------------|----------------|---------|
| 1K | $204 | $6 | $198 |
| 10K | $2,042 | $60 | $1,982 |
| 100K | $20,415 | $599 | $19,816 |
| 1M | $204,153 | $5,992 | $198,161 |
| 10M | $2,041,531 | $59,924 | $1,981,606 |

No batch processing discounts are assumed. OpenAI's Batch API (50% off)
would further reduce both approaches proportionally.

---

## Production Readiness

### Q7: Is this production-ready as-is?

No. This is a validated proof-of-concept that demonstrates the filter-then-verify
pattern. Production deployment requires:

1. **Error handling**: Retry logic for API failures, circuit breakers
2. **Rate limiting**: OpenAI rate limits require queuing at scale
3. **Monitoring**: Alerting on accuracy drift, cost anomalies, latency spikes
4. **Data masking**: PII must be redacted before sending to external APIs
5. **Failover**: Multi-provider setup (OpenAI + fallback to Anthropic/local)
6. **Testing on real data**: Synthetic scenarios validate the architecture;
   real fraud patterns require domain-specific tuning

The core architecture (deterministic filters + targeted LLM sub-calls) is
production-proven at companies like Stripe and PayPal in similar forms.

### Q8: What about latency? Can this work for real-time decisions?

Pipeline latency per batch: 1.8-25 seconds (depending on number of sub-calls).

| Scenario | Pipeline Time | Naive Time | Sub-calls |
|----------|--------------|-----------|-----------|
| Single user (5 txns) | 1.8-3.2s | 2.9-5.1s | 1 |
| Multi-user (15 txns) | 25.3s | 5.5s | 2 |
| No fraud detected | 0.002s | 4.7s | 0 |

For real-time (<100ms) decisions, use a hybrid approach:
- Layer 1: ML model for instant scoring (<10ms)
- Layer 2: Pipeline for async review of medium-confidence cases (seconds)
- Layer 3: Human review for edge cases

The pipeline excels in **near-real-time** (seconds) and **batch**
(hourly/daily) processing, not sub-second decisions.

---

## Accuracy and Reliability

### Q9: 100% accuracy sounds too good. What's the catch?

It is too good -- for real-world claims. The 100% accuracy is on 8
**synthetic scenarios** (51 transactions) with clear, unambiguous fraud
patterns designed to be detectable by the hardcoded filters. This is not
a generalizable performance metric.

What the synthetic data lacks:
- **Class imbalance**: Real fraud is ~0.1% of transactions. This dataset
  is ~50% fraud by design.
- **Subtle patterns**: Synthetic velocity = 5 txns in 3 minutes (obvious).
  Real card testing uses randomized timing with delays.
- **Adversarial behavior**: Real fraudsters adapt to known detection rules.
- **Distribution shift**: Fraud tactics evolve; fixed thresholds decay.
- **Scale effects**: 51 transactions vs millions with noise.

The accuracy claim validates that the **architecture works correctly on
well-defined scenarios** -- filters detect the intended patterns, LLM
sub-calls confirm correctly, and no false positives or negatives when
patterns are unambiguous.

The pipeline achieved 100% vs Naive's 94% because:
- Context folding prevents attention dilution (Naive missed fraud in Scenarios 7, 8)
- Per-user sub-calls give focused analysis (Naive over-flagged in Scenario 6)
- Deterministic filters catch mathematical patterns LLMs miss (z-scores, timing)

Real-world deployment would require validation on large labeled datasets
with proper cross-validation, class imbalance handling, and threshold tuning.

### Q10: What about the old evaluation showing F1=0.0?

The older evaluation (Phase 1-4 comparison on 10,000 transactions) used a
**broken implementation** (`legacy_agent.py`) that:

- Generated code in a single shot (no iterative loop)
- Silently fell back to sending ALL transactions when code crashed
- Resulted in `"filtered": 200/200` = zero actual filtering

The new implementation (`pipeline_agent.py`) fixes this with:
- Pre-built deterministic filters (no LLM code generation)
- 4-phase pipeline with trajectory logging
- Per-user context folding

The F1=0.0 was a bug, not a fundamental limitation of the approach.

---

## Technical Limitations

### Q11: You use synthetic data. Does this generalize to real fraud?

Synthetic data is a limitation for generalization claims. The 8 scenarios
cover common fraud archetypes (velocity, geographic impossibility, amount
anomaly, account takeover, micro-transaction testing, cross-border rapid),
but real fraud patterns include:

- Multi-step account takeover sequences
- Synthetic identity fraud (long-term)
- Coordinated money mule networks
- BIN attacks with sophisticated evasion
- Friendly fraud (context-dependent)

The **cost optimization principles** generalize regardless of data source:
code filters are always cheaper than LLM token processing. The specific
filter thresholds and patterns would need tuning for real data.

### Q12: What if OpenAI changes pricing or deprecates the model?

Vendor lock-in is a real risk. Mitigation:

| Strategy | Effort | Protection |
|----------|--------|-----------|
| Multi-provider abstraction | Medium | Switch between OpenAI/Anthropic/Google |
| Self-hosted (Llama 3.1) | High | Full control, no vendor dependency |
| Azure OpenAI | Low | Same API, enterprise SLA + compliance |
| OpenAI Batch API | Low | 50% cost reduction, SLA guarantees |

The architecture is **model-agnostic**: the deterministic filters are
pure Python, and the LLM sub-calls can target any provider. Only the
`_llm_query()` method needs changing to switch providers.

### Q13: What about embedding costs for the RAG approach?

Embedding costs are negligible:

```
text-embedding-3-small: $0.020 per 1M tokens
Query embedding: ~50 tokens per query = $0.000001 per query
Annual at 10K queries/day: $0.004/year
```

The RAG annual cost of $46.42/year is dominated by LLM inference, not
embeddings. Vector DB hosting (ChromaDB local = free, Pinecone free tier
covers 500 cases) adds minimal cost.

---

## Comparison with Alternatives

### Q14: Why not use a fine-tuned small model instead?

Fine-tuning is complementary, not competing:

| Approach | Cost/Year (10K/day) | F1 (est.) | Setup |
|----------|-------------------|-----------|-------|
| Pipeline (base gpt-4o-mini) | $60 | 1.00* | 1 day |
| Fine-tuned gpt-4o-mini | $180 | 0.80+ | 1 week |
| Fine-tuned Llama 3.1 8B | $9 | 0.70+ | 2 weeks |

*On synthetic scenarios. Real-world F1 would be lower without fine-tuning.

The optimal production approach: **deterministic filters + fine-tuned sub-model**.
Use code filters for the cheap first pass, then a fine-tuned model for
the semantic verification sub-calls.

### Q15: How does this compare to traditional ML (Random Forest, XGBoost)?

| Factor | XGBoost | LLM (Pipeline) |
|--------|---------|----|
| Inference cost | ~$0 (CPU) | $60/year (API) |
| Latency | <10ms | 1.8-25s |
| F1 Score | 0.80-0.90 | Scenario-dependent |
| Explainability | SHAP values | Natural language |
| Cold start | Needs 10K+ labels | Works with 50 examples |
| Adaptation | Retrain monthly | Prompt update |

LLMs win on: explainability, low-data regimes, unstructured text, rapid
iteration. ML wins on: cost, latency, mature use cases.

**Optimal**: Hybrid -- ML for fast scoring, pipeline for edge case review
and explainable audit trails.

---

## Security and Compliance

### Q16: Is sending transaction data to OpenAI a compliance risk?

Yes, for regulated industries. Mitigation options:

| Approach | GDPR | PCI-DSS | Data Residency |
|----------|------|---------|---------------|
| OpenAI API (standard) | Requires DPA | Redact card data | US only |
| Azure OpenAI | Compliant | Redact card data | Regional |
| Self-hosted (Llama) | Full control | Full control | Full control |

**Recommended for financial services**: Azure OpenAI or self-hosted model.
Standard OpenAI API requires PII masking before sending data.

The pipeline architecture helps compliance: the audit trail shows exactly
which data was sent to the LLM and what it returned, satisfying
explainability requirements.

### Q17: Can we trust LLM fraud decisions? What about hallucinations?

In this implementation, the LLM is **not making autonomous decisions**.
The deterministic code filters identify suspicious patterns using exact
computation (z-scores, time windows, location matching). The LLM only
**confirms** pre-filtered findings.

This architecture reduces hallucination risk because:
1. The LLM receives focused, per-user context (not a massive prompt)
2. Filter findings are included as structured evidence
3. The LLM is asked "given these code-detected anomalies, is this fraud?"
   rather than "analyze these 500 transactions"
4. Structured JSON output with Pydantic validation

For production, add confidence thresholds: auto-approve high-confidence,
human review for medium-confidence, auto-reject low-confidence.

---

## Real-World Deployment

### Q18: Has anyone deployed this pattern in production?

This specific implementation is a proof-of-concept. The underlying patterns
are production-proven:

- **Stripe Radar**: ML + rules + LLM hybrid, blocks $20B+ fraud annually
- **PayPal**: LLM for transaction explanation, reduced review time 40%
- **Feedzai**: LLM + ML ensemble, 60% reduction in false positives

### Q19: What is the ROI timeline?

Depends on what you are replacing:

| Current System | Annual Savings (10M txns/day) | Payback |
|---------------|---------------------------|---------|
| Manual review ($104K/yr) | $34K-100K | <12 months |
| Naive LLM ($2M/yr) | $1.98M | <1 month |
| Traditional ML ($0/yr) | Negative | Never |

Implementation cost: approximately $30,000 (6 weeks engineering).

The pipeline is most valuable when replacing expensive LLM approaches at
scale or supplementing manual review with automated explainable decisions.

---

*Last Updated: February 2026*
*Questions or feedback: [Open an issue](https://github.com/Abivarma/smart-llm-fraud-detection/issues)*
