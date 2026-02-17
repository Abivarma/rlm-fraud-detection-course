"""Streamlit dashboard for Phase 2: RAG Fraud Detection Approach."""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.rag_agent import RAGFraudAgent
from src.metrics.tracker import MetricsTracker

# Page config
st.set_page_config(
    page_title="Phase 2: RAG Approach",
    page_icon="🔎",
    layout="wide"
)

# Title
st.title("🔎 Phase 2: RAG-Based Fraud Detection")
st.markdown("""
This dashboard demonstrates the **RAG (Retrieval-Augmented Generation) approach** where
relevant fraud patterns are retrieved via semantic search before LLM analysis.

**Key Characteristics:**
- ✅ Reduced context size via pattern retrieval
- ✅ Semantic matching with vector embeddings
- ✅ Maintained accuracy vs naive approach
- ⚠️ Retrieval overhead from embedding API calls
- ❌ Still scales linearly with volume
""")

# Load results
@st.cache_data
def load_results():
    """Load pre-computed results from Phase 2 notebook."""
    metrics_dir = Path(__file__).parent.parent / "results" / "metrics"

    # Load scalability data
    scalability_df = pd.read_csv(metrics_dir / "rag_scalability.csv")

    # Load baseline
    with open(metrics_dir / "rag_baseline.json") as f:
        baseline = json.load(f)

    # Load full results
    with open(metrics_dir / "rag_results.json") as f:
        results = json.load(f)

    # Load naive baseline for comparison
    with open(metrics_dir / "naive_baseline.json") as f:
        naive_baseline = json.load(f)

    return scalability_df, baseline, results, naive_baseline

try:
    scalability_df, baseline, results, naive_baseline = load_results()

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cost_delta = baseline['annual_cost'] - naive_baseline['annual_cost']
        st.metric(
            "Annual Cost",
            f"${baseline['annual_cost']:,.0f}",
            delta=f"${cost_delta:,.0f}",
            delta_color="inverse",
            help="Projected annual cost at 10K transactions/day (vs Naive)"
        )

    with col2:
        st.metric(
            "F1 Score",
            f"{baseline['f1_score']:.2f}",
            delta=f"{baseline['f1_score'] - 0.70:.2f}" if baseline['f1_score'] >= 0.70 else f"{baseline['f1_score'] - 0.70:.2f}",
            delta_color="normal" if baseline['f1_score'] >= 0.70 else "inverse",
            help="Target: ≥0.70"
        )

    with col3:
        cost_reduction = ((naive_baseline['cost_per_100'] - baseline['cost_per_100']) / naive_baseline['cost_per_100']) * 100
        st.metric(
            "Cost Reduction",
            f"{cost_reduction:.1f}%",
            delta=f"vs Naive",
            help="Cost reduction compared to naive approach"
        )

    with col4:
        st.metric(
            "Retrieval Latency",
            f"{baseline['retrieval_latency_ms']:.0f}ms",
            help="Additional latency from embedding and vector search"
        )

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Scalability", "💰 Cost Comparison", "🎯 Accuracy", "🔍 How RAG Works"])

    with tab1:
        st.subheader("Scalability Testing Results")

        # Filter successful runs
        success_df = scalability_df[scalability_df['cost_usd'].notna()].copy()

        if len(success_df) > 0:
            # Cost and latency charts
            col1, col2 = st.columns(2)

            with col1:
                st.line_chart(
                    success_df.set_index('batch_size')['cost_usd'],
                    use_container_width=True
                )
                st.caption("💵 Cost vs Batch Size (USD)")

            with col2:
                # Latency breakdown
                latency_data = pd.DataFrame({
                    'batch_size': success_df['batch_size'],
                    'Retrieval': success_df['retrieval_latency_ms'] / 1000,
                    'LLM Analysis': (success_df['latency_ms'] - success_df['retrieval_latency_ms']) / 1000
                }).set_index('batch_size')

                st.bar_chart(latency_data, use_container_width=True)
                st.caption("⏱️ Latency Breakdown (seconds)")

            # Tokens
            st.line_chart(
                success_df.set_index('batch_size')['tokens'],
                use_container_width=True
            )
            st.caption("🔢 Token Usage vs Batch Size")

            # Show data table
            st.dataframe(
                success_df.style.format({
                    'tokens': '{:,.0f}',
                    'cost_usd': '${:.4f}',
                    'latency_ms': '{:.0f}ms',
                    'retrieval_latency_ms': '{:.0f}ms',
                    'f1_score': '{:.3f}',
                    'precision': '{:.3f}',
                    'recall': '{:.3f}',
                    'accuracy': '{:.3f}'
                }),
                use_container_width=True
            )

        # Show failed runs
        failed_df = scalability_df[scalability_df['cost_usd'].isna()]
        if len(failed_df) > 0:
            st.warning(f"⚠️ {len(failed_df)} batch size(s) failed due to rate limits or errors")
            with st.expander("View failed runs"):
                st.dataframe(failed_df, use_container_width=True)

    with tab2:
        st.subheader("Cost Comparison: RAG vs Naive")

        # Side-by-side comparison
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔍 Naive Approach")
            st.markdown(f"""
            - **Per 100 txns**: ${naive_baseline['cost_per_100']:.4f}
            - **Per day**: ${naive_baseline['cost_per_100'] * 100:.2f}
            - **Per year**: ${naive_baseline['annual_cost']:,.2f}

            **How it works:**
            - Send all transactions + all fraud patterns to GPT-4
            - Large context window (8,000+ chars)
            - Higher token usage
            """)

        with col2:
            st.markdown("### 🔎 RAG Approach")
            st.markdown(f"""
            - **Per 100 txns**: ${baseline['cost_per_100']:.4f}
            - **Per day**: ${baseline['cost_per_100'] * 100:.2f}
            - **Per year**: ${baseline['annual_cost']:,.2f}

            **How it works:**
            - Retrieve top-3 relevant patterns via vector search
            - Smaller context window (5,000+ chars)
            - Lower token usage, but embedding overhead
            """)

        # Annual savings
        annual_savings = naive_baseline['annual_cost'] - baseline['annual_cost']
        cost_reduction_pct = ((naive_baseline['annual_cost'] - baseline['annual_cost']) / naive_baseline['annual_cost']) * 100

        st.markdown("---")
        st.markdown(f"""
        ### 💰 Annual Savings

        **Savings**: ${annual_savings:,.2f}/year ({cost_reduction_pct:.1f}% reduction)

        **Why the savings are modest:**
        1. **Embedding costs**: Additional API calls for query and pattern embeddings
        2. **Small dataset**: Only 4 fraud patterns, so retrieval doesn't filter much
        3. **Token overhead**: JSON formatting and structured prompts add tokens

        **When RAG saves more:**
        - Large pattern libraries (100+ patterns)
        - Complex domain knowledge bases
        - Long-form documentation retrieval
        """)

        # Cost breakdown chart
        if len(success_df) > 0:
            st.markdown("### Cost Scaling Comparison")

            # Load naive scalability for comparison
            naive_scalability = pd.read_csv(Path(__file__).parent.parent / "results" / "metrics" / "naive_scalability.csv")
            naive_success = naive_scalability[naive_scalability['cost_usd'].notna()]

            if len(naive_success) > 0:
                comparison_data = pd.DataFrame({
                    'Batch Size': success_df['batch_size'],
                    'RAG': success_df['cost_usd'],
                    'Naive': naive_success['cost_usd'].values[:len(success_df)]
                }).set_index('Batch Size')

                st.line_chart(comparison_data, use_container_width=True)
                st.caption("💵 Cost Comparison: RAG vs Naive")

    with tab3:
        st.subheader("Accuracy Metrics")

        if len(success_df) > 0:
            # Accuracy metrics chart
            accuracy_data = success_df[['batch_size', 'f1_score', 'precision', 'recall']].set_index('batch_size')
            st.line_chart(accuracy_data, use_container_width=True)
            st.caption("📈 Accuracy Metrics vs Batch Size")

            st.markdown("""
            ### Analysis

            ⚠️ **Note:** Similar to naive approach, F1 scores are affected by:
            1. **Low Fraud Rate**: Only 2% of transactions are fraudulent (4% in test batches)
            2. **Small Sample Sizes**: 2-4 fraud cases per batch
            3. **Retrieval Quality**: Performance depends on retrieving the right patterns

            **RAG Accuracy Trade-offs:**
            - ✅ **Maintained accuracy**: F1 scores similar to naive approach
            - ✅ **Focused reasoning**: LLM cites specific retrieved patterns
            - ⚠️ **Retrieval dependency**: Wrong patterns → missed fraud
            """)

            # Show confusion matrix for best result
            best_idx = success_df['f1_score'].idxmax()
            if pd.notna(best_idx):
                best_row = success_df.loc[best_idx]

                st.markdown(f"""
                ### Best Result (Batch Size: {int(best_row['batch_size'])})

                | Metric | Value |
                |--------|-------|
                | F1 Score | {best_row['f1_score']:.3f} |
                | Precision | {best_row['precision']:.3f} |
                | Recall | {best_row['recall']:.3f} |
                | Accuracy | {best_row['accuracy']:.3f} |
                """)

            # Comparison with naive
            st.markdown("### Accuracy: RAG vs Naive")

            comparison_table = pd.DataFrame({
                'Metric': ['F1 Score', 'Precision', 'Recall'],
                'Naive': [naive_baseline['f1_score'], naive_baseline['precision'], naive_baseline['recall']],
                'RAG': [baseline['f1_score'], baseline['precision'], baseline['recall']]
            })

            st.dataframe(
                comparison_table.style.format({
                    'Naive': '{:.3f}',
                    'RAG': '{:.3f}'
                }),
                use_container_width=True,
                hide_index=True
            )

    with tab4:
        st.subheader("🔍 How RAG Works")

        st.markdown("""
        ### Architecture

        ```
        Transaction Batch
            ↓
        1. Create Query
           - Analyze transaction characteristics
           - Build semantic query (e.g., "unusual amounts", "rapid transactions")
            ↓
        2. Embed Query
           - OpenAI text-embedding-3-small
           - Convert to 1536-dim vector
            ↓
        3. Vector Search (ChromaDB)
           - Cosine similarity with fraud pattern embeddings
           - Retrieve top-3 most relevant patterns
            ↓
        4. Build Prompt
           - Retrieved patterns + transactions
           - Smaller context than naive
            ↓
        5. GPT-4 Analysis
           - Analyze with retrieved patterns
           - Generate fraud predictions + reasoning
            ↓
        Results
        ```

        ### Vector Store

        **Fraud patterns embedded:**
        - Velocity attacks (rapid transactions)
        - Amount anomalies (unusual spending)
        - Geographic outliers (location changes)
        - Account takeover (device/behavior changes)

        **Embedding model:**
        - text-embedding-3-small (1536 dimensions)
        - Cost: $0.02 per 1M tokens
        - Latency: ~50-100ms per embedding

        ### Query Generation

        The system analyzes transactions and creates semantic queries:

        | Transaction Pattern | Generated Query |
        |---------------------|-----------------|
        | Multiple rapid txns | "multiple rapid transactions" |
        | High amounts | "unusual transaction amounts" |
        | Many locations | "multiple locations geographic patterns" |
        | Default | "general fraud patterns transaction analysis" |

        ### Retrieval Example

        **Query**: "unusual high amount transactions"

        **Retrieved Patterns**:
        1. Amount Anomaly (cosine similarity: 0.85)
        2. Account Takeover (cosine similarity: 0.62)
        3. Velocity Attack (cosine similarity: 0.41)

        The top-3 patterns are then included in the GPT-4 prompt.
        """)

    # Key Findings
    st.markdown("---")
    st.subheader("🎯 Key Findings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### ✅ Strengths
        - **Reduced token usage**: 20-30% fewer tokens via retrieval
        - **Semantic matching**: Vector search finds relevant patterns
        - **Maintained accuracy**: F1 scores similar to naive
        - **Explainability**: LLM cites specific retrieved patterns
        """)

    with col2:
        st.markdown("""
        ### ❌ Limitations
        - **Modest savings**: Only 0.5-1% cost reduction (embedding overhead)
        - **Retrieval overhead**: Additional 200-500ms latency
        - **Complexity**: Vector database, embeddings, retrieval logic
        - **Still linear scaling**: Cost grows with transaction volume
        """)

    st.markdown(f"""
    ### 🔬 Realistic Outcome

    **Expected**: 20-30% cost reduction

    **Actual**: {cost_reduction_pct:.1f}% reduction

    **Why?** Embedding API calls and overhead offset token savings when:
    - Pattern library is small (only 4 patterns)
    - Context reduction is modest (8K → 5K chars)
    - Retrieval adds latency and complexity

    **When RAG works better:**
    - Large knowledge bases (1000+ documents)
    - Long-form content (documentation, manuals)
    - Diverse queries requiring different context

    ### 🚀 Next Steps

    In **Phase 3 (RLM Approach)**, we'll achieve 70-94% cost reduction by:
    - Using LLM to generate Python code for statistical filtering
    - Executing code to filter transactions BEFORE LLM analysis
    - Sending only filtered subset to LLM (10-30% of original)
    - Dramatically reducing token usage via code generation

    **Target**: 70-94% cost reduction while maintaining F1 > 0.70
    """)

except FileNotFoundError as e:
    st.error(f"❌ Results not found: {e}")
    st.info("💡 Please run the Phase 2 notebook first: `notebooks/02_rag_approach.ipynb`")
except Exception as e:
    st.error(f"❌ Error loading results: {e}")
    st.exception(e)
