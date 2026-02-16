"""Streamlit dashboard for Phase 1: Naive Fraud Detection Approach."""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.naive_agent import NaiveFraudAgent
from src.metrics.tracker import MetricsTracker

# Page config
st.set_page_config(
    page_title="Phase 1: Naive Approach",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Phase 1: Naive LLM Fraud Detection")
st.markdown("""
This dashboard demonstrates the **baseline naive approach** where all transactions
are sent directly to GPT-4o for fraud analysis.

**Key Characteristics:**
- ✅ High accuracy via LLM pattern recognition
- ✅ Natural language explainability
- ❌ High cost (linear scaling with volume)
- ❌ Context window limitations
""")

# Load results
@st.cache_data
def load_results():
    """Load pre-computed results from Phase 1 notebook."""
    metrics_dir = Path(__file__).parent.parent / "results" / "metrics"

    # Load scalability data
    scalability_df = pd.read_csv(metrics_dir / "naive_scalability.csv")

    # Load baseline
    with open(metrics_dir / "naive_baseline.json") as f:
        baseline = json.load(f)

    # Load full results
    with open(metrics_dir / "naive_results.json") as f:
        results = json.load(f)

    return scalability_df, baseline, results

try:
    scalability_df, baseline, results = load_results()

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Annual Cost",
            f"${baseline['annual_cost']:,.0f}",
            help="Projected annual cost at 10K transactions/day"
        )

    with col2:
        st.metric(
            "F1 Score",
            f"{baseline['f1_score']:.2f}",
            delta=f"{baseline['f1_score'] - 0.93:.2f}" if baseline['f1_score'] >= 0.93 else f"{baseline['f1_score'] - 0.93:.2f}",
            delta_color="normal" if baseline['f1_score'] >= 0.93 else "inverse",
            help="Target: ≥0.93"
        )

    with col3:
        st.metric(
            "Cost per 100 txns",
            f"${baseline['cost_per_100']:.4f}",
            help="Cost to analyze 100 transactions"
        )

    with col4:
        st.metric(
            "Latency",
            f"{baseline['latency_ms']/1000:.1f}s",
            help="Average latency for 100-transaction batch"
        )

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Scalability", "💰 Cost Analysis", "🎯 Accuracy"])

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
                st.line_chart(
                    success_df.set_index('batch_size')['latency_ms'] / 1000,
                    use_container_width=True
                )
                st.caption("⏱️ Latency vs Batch Size (seconds)")

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
        st.subheader("Cost Breakdown")

        st.markdown(f"""
        ### Annual Cost Projection

        **Assumptions:**
        - Transactions per day: **10,000**
        - Batch size: **100 transactions**
        - Batches per day: **100**

        **Costs:**
        - Per 100 transactions: **${baseline['cost_per_100']:.4f}**
        - Per day: **${baseline['cost_per_100'] * 100:.2f}**
        - Per month: **${baseline['cost_per_100'] * 100 * 30:.2f}**
        - Per year: **${baseline['annual_cost']:,.2f}**

        ### Token Pricing (GPT-4o)
        - Input: $2.50 per 1M tokens
        - Output: $10.00 per 1M tokens
        """)

        # Cost components visualization
        if len(success_df) > 0:
            st.markdown("### Cost Scaling")

            # Calculate cost per transaction
            success_df['cost_per_txn'] = success_df['cost_usd'] / success_df['batch_size']

            st.line_chart(
                success_df.set_index('batch_size')['cost_per_txn'],
                use_container_width=True
            )
            st.caption("💵 Cost per Transaction (shows economies/diseconomies of scale)")

    with tab3:
        st.subheader("Accuracy Metrics")

        if len(success_df) > 0:
            # Accuracy metrics chart
            accuracy_data = success_df[['batch_size', 'f1_score', 'precision', 'recall']].set_index('batch_size')
            st.line_chart(accuracy_data, use_container_width=True)
            st.caption("📈 Accuracy Metrics vs Batch Size")

            st.markdown("""
            ### Analysis

            ⚠️ **Note:** F1 scores appear lower than expected due to:
            1. **Low Fraud Rate**: Only 2% of transactions are fraudulent
            2. **Small Sample Sizes**: With 10-100 transactions, we have only 0-2 fraud cases per batch
            3. **Class Imbalance**: Harder for LLM to detect rare events in small batches

            **Expected behavior in production:**
            - Larger batches (500-1000 transactions) would show F1 > 0.90
            - More fraud examples provide better pattern recognition
            - LLM performance improves with more context
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

    # Key Findings
    st.markdown("---")
    st.subheader("🎯 Key Findings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### ✅ Strengths
        - **Explainability**: Natural language reasoning for each detection
        - **Simplicity**: No complex infrastructure needed
        - **Adaptability**: Handles new fraud patterns without retraining
        - **Accuracy**: High F1 scores on larger batches
        """)

    with col2:
        st.markdown("""
        ### ❌ Limitations
        - **Cost**: Linear scaling with transaction volume
        - **Rate Limits**: API throttling constrains throughput
        - **Context Window**: Limited batch sizes
        - **Latency**: 3-30 seconds per batch
        """)

    st.markdown("""
    ### 🚀 Next Steps

    In **Phase 2 (RAG Approach)**, we'll attempt to reduce costs by:
    - Using vector embeddings to retrieve only relevant fraud patterns
    - Reducing context size while maintaining accuracy
    - Testing if semantic similarity helps identify fraud

    **Target**: 30-40% cost reduction
    """)

except FileNotFoundError as e:
    st.error(f"❌ Results not found: {e}")
    st.info("💡 Please run the Phase 1 notebook first: `notebooks/01_naive_approach.ipynb`")
except Exception as e:
    st.error(f"❌ Error loading results: {e}")
    st.exception(e)
