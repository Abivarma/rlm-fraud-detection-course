# RLM Fraud Detection Course

**Learn how to reduce LLM costs by 94% using Recursive Language Models**

This comprehensive course demonstrates the evolution from naive LLM approaches to RLM (Recursive Language Models) for fraud detection, with real metrics, production-ready code, and ROI analysis.

## 🎯 Course Overview

### What You'll Learn

1. **Phase 0**: Problem definition and evaluation framework
2. **Phase 1**: Naive LLM approach - baseline performance
3. **Phase 2**: RAG approach - semantic search optimization
4. **Phase 3**: RLM approach - the breakthrough (94% cost reduction)
5. **Phase 4**: Comprehensive comparison with statistical rigor
6. **Phase 5**: Production deployment and white paper

### Academic Foundation

This course implements the RLM paradigm from:
- **Paper**: "Recursive Language Models" by Zhang, Kraska, & Khattab (December 2025)
- **arXiv**: 2512.24601
- **Implementation**: pydantic-ai-rlm framework

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- Jupyter notebook

### Installation

```bash
# Clone or navigate to course directory
cd course

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### Running the Course

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 1. notebooks/00_problem_definition.ipynb
# 2. notebooks/01_naive_approach.ipynb
# 3. notebooks/02_rag_approach.ipynb
# 4. notebooks/03_rlm_approach.ipynb
# 5. notebooks/04_comprehensive_comparison.ipynb
```

### Try the Streamlit Demos

```bash
# Naive approach demo
streamlit run streamlit_apps/01_naive_demo.py

# RAG approach demo
streamlit run streamlit_apps/02_rag_demo.py

# RLM approach demo
streamlit run streamlit_apps/03_rlm_demo.py

# Final comparison dashboard
streamlit run streamlit_apps/04_final_comparison.py
```

## 📊 Key Results

| Approach | Tokens (100 txns) | Cost/Analysis | Annual Cost* | Max Transactions |
|----------|-------------------|---------------|--------------|------------------|
| **Naive** | 25,330 | $0.082 | $299,300 | 100 |
| **RAG** | 6,780 | $0.057 | $208,050 | 200 |
| **RLM** ⭐ | 2,232 | $0.0048 | $17,520 | 10,000+ |

*At 10,000 analyses/day

**RLM Savings**:
- 94% vs Naive ($281,780/year)
- 92% vs RAG ($190,530/year)

## 📁 Project Structure

```
course/
├── notebooks/              # Jupyter notebooks (main course content)
│   ├── 00_problem_definition.ipynb
│   ├── 01_naive_approach.ipynb
│   ├── 02_rag_approach.ipynb
│   ├── 03_rlm_approach.ipynb
│   ├── 04_comprehensive_comparison.ipynb
│   └── 05_production_deployment.ipynb
├── streamlit_apps/         # Interactive demos
│   ├── 01_naive_demo.py
│   ├── 02_rag_demo.py
│   ├── 03_rlm_demo.py
│   └── 04_final_comparison.py
├── data/                   # Synthetic fraud data
│   ├── synthetic_generator.py
│   ├── fraud_patterns.json
│   └── sample_transactions.csv
├── src/                    # Source code
│   ├── agents/            # Fraud detection agents
│   ├── metrics/           # Metrics tracking
│   └── utils/             # Utilities
├── results/               # Saved metrics and visualizations
│   ├── metrics/
│   └── visualizations/
└── docs/                  # Documentation
    ├── COURSE_PLAN.md     # Detailed course plan
    └── WHITE_PAPER.md     # Academic white paper
```

## 🎓 Learning Path

### For Data Scientists / ML Engineers

Focus on:
- Technical implementation details
- Code optimization techniques
- Statistical analysis methodology

### For Engineering Managers / CTOs

Focus on:
- ROI calculations
- Decision framework (when to use each approach)
- Production deployment considerations

### For Researchers

Focus on:
- Academic rigor (white paper)
- Statistical significance testing
- Comparison to prior work

## 💡 Key Insights

### Why RLM Works

**The Problem**: Traditional LLMs send all data through expensive APIs

**The Solution**: RLM separates concerns:
1. **Code for computation** (filtering, statistics) - runs in milliseconds
2. **LLM for reasoning** (semantic analysis) - only on filtered data

**Example**:
```python
# Before (Naive): 10,000 transactions → LLM → $0.125
# After (RLM): 10,000 transactions → Code (filter to 12) → Sub-LLM → $0.005

# Code execution: 2.3ms, essentially free
# LLM call: 2000ms, expensive
```

### When to Use Each Approach

| Scenario | Best Approach | Why |
|----------|---------------|-----|
| < 20 transactions | Naive | Cost difference negligible |
| 20-100 transactions, have patterns | RAG | Semantic matching valuable |
| > 100 transactions | RLM | Only scalable option |
| Cost-sensitive | RLM | 94% cheaper than naive |
| Need citations | RLM | Grounded in actual data |

## 🛠️ Technical Stack

- **LLM Framework**: Pydantic AI
- **RLM Implementation**: pydantic-ai-rlm
- **Main Model**: GPT-4 (code generation)
- **Sub Model**: GPT-4-mini (cost-effective analysis)
- **Vector DB**: ChromaDB (for RAG)
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Dashboard**: Streamlit
- **Data**: Synthetic e-commerce fraud transactions

## 📖 Documentation

- [Detailed Course Plan](docs/COURSE_PLAN.md) - Phase-by-phase breakdown
- [White Paper](docs/WHITE_PAPER.md) - Academic publication (coming soon)
- [Implementation Guide](docs/IMPLEMENTATION_GUIDE.md) - Technical details (coming soon)

## 🤝 Contributing

This is an educational project. Feedback and improvements welcome!

## 📝 Citation

If you use this course in your research or work, please cite:

```bibtex
@misc{rlm_fraud_detection_course,
  title={Cost-Efficient Fraud Detection at Scale Using Recursive Language Models},
  author={Your Name},
  year={2025},
  note={Educational course based on Zhang et al.'s RLM paradigm}
}
```

## 📄 License

MIT License

## 🙏 Acknowledgments

- **RLM Paper**: Zhang, Kraska, & Khattab - "Recursive Language Models" (arXiv:2512.24601)
- **pydantic-ai-rlm**: vstorm-co for production-ready RLM framework
- **Pydantic AI**: For type-safe LLM framework

---

**Start learning**: Open `notebooks/00_problem_definition.ipynb` and begin your journey!
