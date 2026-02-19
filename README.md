# RLM-Inspired Fraud Detection

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![Tests](https://img.shields.io/badge/tests-47%20passing-brightgreen.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**98.4% fewer tokens. 100% accuracy on synthetic benchmarks. $1.98M/year savings at 10M txns/day.**

An **RLM-inspired orchestration pipeline** for financial fraud detection.
Applies principles from the Recursive Language Model paradigm
([arXiv:2512.24601](https://arxiv.org/abs/2512.24601)) -- context folding,
symbolic filtering, targeted sub-calls -- to demonstrate how deterministic
code-based filtering combined with LLM verification can replace brute-force
context stuffing, cutting costs by 97% while improving accuracy.

> **Not a true RLM.** The LLM does not control the reasoning loop. All
> orchestration is hardcoded Python. This is a rule-gated LLM verifier
> that borrows RLM principles, not a recursive model.
> [See the full distinction.](https://abivarma.github.io/rlm-fraud-detection-course/architecture#how-this-differs-from-true-rlm-and-tool-calling)

> [Documentation Site](https://abivarma.github.io/rlm-fraud-detection-course/) |
> Built by [Abivarma](https://github.com/Abivarma) | February 2026

---

## The Problem

LLM-based fraud detection typically sends every transaction plus hundreds of
historical cases to the model in a single prompt. At enterprise scale:

- **10M transactions/day** at ~23K tokens each = **$2M+/year** in API costs
- Context windows fill up, causing **attention dilution** (context rot)
- No audit trail -- decisions are opaque to compliance teams

## The Solution: Filter-Then-Verify Pipeline

Instead of stuffing everything into one prompt, this approach uses a 4-phase
pipeline where **3 out of 4 phases use zero LLM tokens**:

```
Input (N transactions)
  |
  PROBE       Examine data structure, count users/txns       [0 LLM tokens]
  |
  FILTER      Deterministic checks: velocity, geo,           [0 LLM tokens]
  |           amount anomaly, device shift
  |
  ANALYZE     LLM sub-calls on flagged subset ONLY,          [minimal tokens]
  |           per-user context folding
  |
  AGGREGATE   Merge verdicts, cross-check filters            [0 LLM tokens]
  |
Output (fraud predictions + full audit trail)
```

The LLM only sees the suspicious subset (typically 10-30% of transactions),
with fresh per-user context instead of a monolithic prompt.

## Results

Tested across 8 **synthetic** fraud scenarios (51 transactions) with live API calls (gpt-4o-mini).

| Metric | Naive | RLM | Improvement |
|--------|-------|-----|-------------|
| Tokens (total) | 185,661 | 3,059 | **98.4% reduction** |
| Cost (total) | $0.0285 | $0.0008 | **97.1% reduction** |
| Accuracy | 94% (48/51) | **100%** (51/51) | +6 points |
| False Positives | 1 | 0 | Eliminated |
| Missed Fraud | 2 | 0 | Eliminated |
| Audit Trail | None | Full trajectory | Compliance-ready |

### Cost Projections at Scale

| Daily Volume | Naive / Year | RLM / Year | Annual Savings |
|-------------|-------------|-----------|----------------|
| 10K txns | $2,042 | $60 | $1,982 |
| 100K txns | $20,415 | $599 | $19,816 |
| 1M txns | $204,153 | $5,992 | $198,161 |
| 10M txns | $2,041,531 | $59,924 | **$1,981,606** |

Based on gpt-4o-mini pricing ($0.15/1M input, $0.60/1M output tokens).

## Limitations

- **Synthetic data only**: 51 hand-crafted transactions with clear patterns. Real-world fraud is subtler. The 100% accuracy validates the architecture, not real-world performance.
- **Hardcoded filters**: Fixed thresholds (300s velocity window, z > 3.0, etc.) tuned for demo scenarios. Production requires threshold tuning on real data.
- **Not a true RLM**: The LLM does not control the loop. All orchestration is deterministic Python. This is a code-controlled pipeline borrowing RLM principles.
- **No drift handling**: No retraining pipeline, no threshold adaptation, no feedback loop.

See [Enterprise FAQ](https://abivarma.github.io/rlm-fraud-detection-course/faq) for detailed discussion.

## Quick Start

```bash
git clone https://github.com/Abivarma/rlm-fraud-detection-course.git
cd rlm-fraud-detection-course
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the demo (uses cached results, no API key needed):

```bash
python notebooks/rlm_benefits_demo.py --cached
```

Or explore interactively:

```bash
jupyter notebook notebooks/05_rlm_benefits_demo.ipynb
```

## Project Structure

```
.
├── src/agents/
│   ├── rlm_repl_agent.py          # Core: 4-phase pipeline (RLM-inspired)
│   ├── naive_agent.py             # Baseline: 500 cases + all txns -> LLM
│   ├── rag_agent.py               # RAG: vector retrieval + LLM
│   └── rlm_agent.py               # Old one-shot RLM (kept for comparison)
│
├── notebooks/
│   ├── 00_problem_definition.ipynb     # Dataset creation and evaluation framework
│   ├── 01_naive_approach.ipynb         # Phase 1: Baseline ($158/yr)
│   ├── 02_rag_approach.ipynb           # Phase 2: RAG ($46/yr, 70.6% reduction)
│   ├── 03_rlm_approach.ipynb           # Phase 3: RLM ($11/yr, 92.8% reduction)
│   ├── 04_comprehensive_comparison.ipynb  # Phase 4: All approaches compared
│   ├── 05_rlm_benefits_demo.ipynb      # Final showcase: 8 scenarios with results
│   ├── rlm_demo_cache.json             # Cached results from live API runs
│   └── rlm_benefits_demo.py            # CLI demo runner
│
├── data/
│   ├── rlm_demo_examples.csv           # 51 demo transactions (8 scenarios)
│   └── rlm_demo_scenarios.json         # Scenario metadata
│
├── results/
│   ├── visualizations/                 # 13 comparison charts (PNG)
│   └── metrics/                        # Raw metrics (JSON/CSV)
│
├── tests/                              # 47 tests (pytest)
├── docs/                               # GitHub Pages documentation site
└── scripts/                            # Data generation utilities
```

## Documentation

| Page | Description |
|------|-------------|
| [How It Works](https://abivarma.github.io/rlm-fraud-detection-course/how-it-works) | 4-phase REPL loop with real trajectory examples |
| [Results](https://abivarma.github.io/rlm-fraud-detection-course/results) | All 8 scenarios with Naive vs RLM comparison |
| [Research](https://abivarma.github.io/rlm-fraud-detection-course/research) | The RLM paradigm from arXiv:2512.24601 |
| [Architecture](https://abivarma.github.io/rlm-fraud-detection-course/architecture) | Technical deep dive into all three agents |
| [Enterprise FAQ](https://abivarma.github.io/rlm-fraud-detection-course/faq) | 16 questions on costs, scaling, production readiness |

## Key Files

| File | What It Does |
|------|-------------|
| [`src/agents/rlm_repl_agent.py`](src/agents/rlm_repl_agent.py) | Core pipeline: hardcoded filters + LLM sub-calls + trajectory logging |
| [`notebooks/05_rlm_benefits_demo.ipynb`](notebooks/05_rlm_benefits_demo.ipynb) | Interactive results notebook (runs from cached data) |
| [`notebooks/rlm_demo_cache.json`](notebooks/rlm_demo_cache.json) | Cached results: every token count, cost, and trajectory |
| [`research_analysis.md`](research_analysis.md) | Detailed analysis of the RLM research paper |

## Running Tests

```bash
pytest tests/ -v  # 47 tests
```

## Author

**Abivarma** -- [GitHub](https://github.com/Abivarma) | [LinkedIn](https://www.linkedin.com/in/abivarma)

## License

MIT
