---
layout: default
title: Research
---

# The RLM Paradigm

This page summarizes the Recursive Language Model (RLM) paradigm as
described in [arXiv:2512.24601](https://arxiv.org/abs/2512.24601) and how
this project implements it for fraud detection.

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

## The RLM Architecture

RLM redefines the relationship between the model and the prompt. Instead of
ingesting the prompt, the model initializes a **persistent REPL environment**
where the prompt is stored as a variable. The model interacts with data
through **symbolic handles**, not attention over raw tokens.

### The REPL Interaction Loop

```
Given prompt P:
  1. Initialize state S_0, store P as variable in REPL
  2. While not answer["ready"]:
     a. Model generates code C_t based on history H_t
     b. REPL executes C_t, produces output O_t
     c. Metadata of O_t appended to H_{t+1}
     d. Model observes results, generates next action
  3. Return answer["content"]
```

The model acts as a **controller** for the REPL. Its context window contains
only orchestration logic and result summaries -- not the raw data.

### Symbolic Recursion: llm_query and llm_batch

The "recursive" in RLM comes from two REPL-available functions:

- **llm_query(prompt)**: Spawns a fresh sub-model instance with a clean
  context to perform semantic analysis on a specific data slice.
- **llm_batch(prompts)**: Fans out multiple sub-calls in parallel.

This enables a pattern where the root model identifies 100 suspicious
segments via code, then analyzes all 100 concurrently via `llm_batch` --
each sub-call with its own clean context.

### Context Folding

When a sub-call returns, only a **self-chosen summary** remains in the root
model's context. This prevents "context saturation" where the reasoning
trace itself causes the model to lose track of its goal.

In fraud detection: the root orchestrator maintains a clear view of the
global strategy while delegating per-user analysis to isolated sub-calls.

### The Diffusion Answer Mechanism

RLM uses an iterative answer refinement process:

```python
answer = {"content": "", "ready": False}
```

Throughout multiple turns, the model edits, refines, and verifies the answer
against raw data. Only when confident does it set `ready: True`. This
enables a "fact-checking" loop before committing to results.

### Error Recovery: Fail Fast, Fail Loud

If a code snippet fails, the model observes the exception and generates
an alternative approach. This iterative refinement handles format variations
and edge cases that break one-shot code generation.

In production, this means the system will not return "No Fraud Found"
because a sub-call failed silently. It signals the failure explicitly.

---

## Benchmark Results from the Paper

| Benchmark | Total Tokens | Base Model | RLM | RLM Cost |
|-----------|-------------|-----------|-----|----------|
| BrowseComp+ (1K) | 8.3M | Degraded | 91.33% accuracy | $0.99 avg |
| CodeQA | 900K | 24.00 | 62.00 | $0.27 |
| S-NIAH | Up to 2^18 | Fails > 2^14 | Consistent | Comparable |
| OOLONG (Dense) | Dense | Low Recall | +10-59% Gain | Comparable |

Key finding: RLMs can be **more cost-effective** than monolithic models.
On BrowseComp+ (1K), ingesting 6-11M tokens costs $1.50-2.75, while the
RLM averaged only $0.99 by filtering through the REPL.

---

## How This Project Maps to the Paper

| Paper Concept | This Implementation |
|--------------|-------------------|
| Persistent REPL | `RLMREPLAgent.analyze()` -- 4-phase loop with state |
| Symbolic filtering | `_velocity_filter()`, `_amount_anomaly_filter()`, `_geo_filter()`, `_device_shift_filter()` |
| llm_query sub-calls | `_llm_query()` -- per-user sub-call with context folding |
| Context folding | Each sub-call gets only 1 user's flagged + baseline txns |
| Diffusion answer | `answer["ready"]` set only after all sub-calls merge |
| Fail Fast, Fail Loud | Exceptions propagate; no silent fallback to monolithic LLM |
| Trajectory logging | `Trajectory` and `TrajectoryStep` dataclasses |

The main departure: instead of the LLM generating arbitrary Python code
at runtime (which proved unreliable in testing -- see below), the filters
are **deterministic and pre-built**. The LLM is used exclusively for
semantic judgment on pre-filtered data.

### Why Pre-Built Filters Over LLM-Generated Code

During development, raw LLM code generation was tested and failed:

- Generated `int + timedelta` (type error crash)
- Wrong standard deviation logic
- Flagged ALL transactions as fraud
- Safety-refused on `llm_query` calls ("I don't have access to personal data")

Pre-built deterministic filters produce the same correct result every run.
The LLM's strength -- semantic judgment -- is used where it excels, not
where code is more reliable.

---

## Open-Source RLM Implementations

| Project | Description | Reference For |
|---------|-------------|--------------|
| [alexzhang13/rlm](https://github.com/alexzhang13/rlm) | Official paper implementation | Core REPL mechanics, error handling |
| [dspy.RLM](https://dspy.ai/api/modules/RLM/) | DSPy library module | Production deployment, signature-based filtering |
| [PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers) | Scalable evaluation framework | Parallel sub-LLM processing, diffusion answers |
| [ysz/recursive-llm](https://github.com/ysz/recursive-llm) | Standalone Python implementation | LiteLLM integration, safety guardrails |
| [manu-mishra/RLMWithStrands](https://github.com/manu-mishra/RLMWithStrands) | AWS/Bedrock implementation | Cloud deployment, observability |
