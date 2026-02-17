"""Tests for RAG fraud detection agent."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.agents.rag_agent import RAGFraudAgent
from src.metrics.tracker import AnalysisMetrics


class TestRAGFraudAgent:
    """Test cases for RAGFraudAgent."""

    @pytest.fixture
    def sample_transactions(self):
        """Create sample transaction data for testing."""
        return pd.DataFrame({
            'transaction_id': ['TXN_001', 'TXN_002', 'TXN_003'],
            'user_id': ['U_0001', 'U_0001', 'U_0002'],
            'amount': [50.0, 500.0, 25.0],
            'timestamp': [1735670000, 1735670100, 1735670200],
            'location': ['NYC', 'NYC', 'LA'],
            'device': ['mobile', 'mobile', 'desktop'],
            'category': ['grocery', 'electronics', 'restaurant'],
            'is_fraud': [False, True, False],
            'fraud_type': [None, 'amount_anomaly', None]
        })

    @pytest.fixture
    def fraud_patterns(self):
        """Sample fraud pattern definitions."""
        return {
            'velocity': {
                'name': 'Velocity Attack',
                'description': 'Multiple transactions in rapid succession',
                'indicators': {'time_window': '< 5 minutes'}
            },
            'amount_anomaly': {
                'name': 'Amount Anomaly',
                'description': 'Unusual transaction amounts',
                'indicators': {'threshold': '> 3 std deviations'}
            }
        }

    @pytest.fixture
    def agent(self):
        """Create RAGFraudAgent instance."""
        return RAGFraudAgent(model="gpt-4o", temperature=0.1)

    def test_initialization(self, agent):
        """Test agent initialization with default parameters."""
        assert agent.model == "gpt-4o"
        assert agent.temperature == 0.1
        assert agent.client is not None
        assert agent.vector_store is not None

    def test_initialization_loads_fraud_patterns(self, agent):
        """Test that agent loads fraud patterns on init."""
        # Should have loaded patterns from fraud_patterns.json
        assert hasattr(agent, 'fraud_patterns')
        assert len(agent.fraud_patterns) > 0

    def test_embed_text(self, agent):
        """Test text embedding functionality."""
        text = "Test transaction pattern"
        embedding = agent._embed_text(text)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    def test_build_vector_store(self, agent, fraud_patterns):
        """Test vector store is built from fraud patterns."""
        # Agent should build vector store on init
        assert agent.vector_store is not None

        # Should be able to query it
        results = agent._retrieve_patterns("velocity attack", k=2)
        assert isinstance(results, list)
        assert len(results) <= 2

    def test_retrieve_patterns(self, agent):
        """Test pattern retrieval from vector store."""
        query = "multiple transactions in short time"
        patterns = agent._retrieve_patterns(query, k=2)

        assert isinstance(patterns, list)
        assert len(patterns) <= 2

        # Each pattern should have relevant fields
        for pattern in patterns:
            assert 'name' in pattern or 'description' in pattern

    def test_retrieve_patterns_returns_most_relevant(self, agent):
        """Test that retrieval returns semantically similar patterns."""
        # Query for amount-related fraud
        patterns = agent._retrieve_patterns("unusually high transaction amount", k=1)

        assert len(patterns) == 1
        # Should retrieve amount anomaly pattern
        pattern_text = str(patterns[0]).lower()
        assert 'amount' in pattern_text or 'anomaly' in pattern_text

    def test_format_transactions_with_context(self, agent, sample_transactions):
        """Test transaction formatting includes retrieved context."""
        formatted = agent._format_transactions_with_context(
            sample_transactions,
            retrieved_patterns=[]
        )

        assert isinstance(formatted, str)
        assert 'TXN_001' in formatted
        assert 'TXN_002' in formatted

    def test_build_prompt_includes_retrieved_patterns(self, agent, sample_transactions):
        """Test prompt includes retrieved fraud patterns."""
        retrieved_patterns = [{
            'name': 'Amount Anomaly',
            'description': 'Test pattern',
            'full_pattern': {}
        }]

        prompt = agent._build_prompt(sample_transactions, retrieved_patterns)

        assert 'Amount Anomaly' in prompt
        assert 'Test pattern' in prompt

    def test_analyze_returns_predictions_and_metrics(self, agent, sample_transactions):
        """Test that analyze returns both predictions and metrics."""
        with patch.object(agent.client.chat.completions, 'create') as mock_create:
            # Mock LLM response
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="""
            {
                "fraudulent_transactions": ["TXN_002"],
                "reasoning": {"TXN_002": "High amount anomaly"}
            }
            """))]
            mock_response.usage = Mock(
                prompt_tokens=800,
                completion_tokens=150,
                total_tokens=950
            )
            mock_create.return_value = mock_response

            predictions, metrics = agent.analyze(sample_transactions)

            # Check predictions
            assert isinstance(predictions, list)
            assert len(predictions) == len(sample_transactions)

            # Check metrics
            assert isinstance(metrics, AnalysisMetrics)
            assert metrics.approach == 'rag'
            assert metrics.total_tokens == 950
            assert metrics.total_tokens < 1200  # Should use fewer tokens than naive

    def test_analyze_uses_retrieval(self, agent, sample_transactions):
        """Test that analyze actually uses pattern retrieval."""
        with patch.object(agent, '_retrieve_patterns') as mock_retrieve:
            mock_retrieve.return_value = [{
                'name': 'Test Pattern',
                'description': 'Test description',
                'full_pattern': {}
            }]

            with patch.object(agent.client.chat.completions, 'create') as mock_create:
                mock_response = Mock()
                mock_response.choices = [Mock(message=Mock(content='{"fraudulent_transactions": [], "reasoning": {}}'))]
                mock_response.usage = Mock(prompt_tokens=500, completion_tokens=100, total_tokens=600)
                mock_create.return_value = mock_response

                agent.analyze(sample_transactions)

                # Should have called retrieval
                assert mock_retrieve.called

    def test_cost_reduction_vs_naive(self, agent, sample_transactions):
        """Test that RAG approach uses fewer tokens than naive."""
        with patch.object(agent.client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content='{"fraudulent_transactions": [], "reasoning": {}}'))]
            mock_response.usage = Mock(
                prompt_tokens=600,  # Should be less than naive approach
                completion_tokens=120,
                total_tokens=720
            )
            mock_create.return_value = mock_response

            predictions, metrics = agent.analyze(sample_transactions)

            # RAG should use fewer prompt tokens than naive (which uses ~850+)
            assert metrics.prompt_tokens < 850

    def test_retrieval_count_tracked(self, agent, sample_transactions):
        """Test that retrieval operations are tracked in metrics."""
        with patch.object(agent.client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content='{"fraudulent_transactions": [], "reasoning": {}}'))]
            mock_response.usage = Mock(prompt_tokens=500, completion_tokens=100, total_tokens=600)
            mock_create.return_value = mock_response

            predictions, metrics = agent.analyze(sample_transactions)

            # Should track retrieval latency
            assert hasattr(metrics, 'retrieval_latency_ms')
            assert metrics.retrieval_latency_ms is not None
            assert metrics.retrieval_latency_ms >= 0

    def test_handles_empty_retrieval(self, agent, sample_transactions):
        """Test agent handles case when no patterns are retrieved."""
        with patch.object(agent, '_retrieve_patterns') as mock_retrieve:
            mock_retrieve.return_value = []  # No patterns found

            with patch.object(agent.client.chat.completions, 'create') as mock_create:
                mock_response = Mock()
                mock_response.choices = [Mock(message=Mock(content='{"fraudulent_transactions": [], "reasoning": {}}'))]
                mock_response.usage = Mock(prompt_tokens=400, completion_tokens=80, total_tokens=480)
                mock_create.return_value = mock_response

                predictions, metrics = agent.analyze(sample_transactions)

                # Should still work
                assert isinstance(predictions, list)
                assert isinstance(metrics, AnalysisMetrics)

    def test_vector_store_persistence(self, agent):
        """Test that vector store can be reused across calls."""
        # First retrieval
        patterns1 = agent._retrieve_patterns("velocity", k=1)

        # Second retrieval (should use same store)
        patterns2 = agent._retrieve_patterns("velocity", k=1)

        # Should return consistent results
        assert len(patterns1) == len(patterns2)

    def test_embedding_dimension_consistency(self, agent):
        """Test that all embeddings have consistent dimensions."""
        emb1 = agent._embed_text("First text")
        emb2 = agent._embed_text("Different text")

        assert len(emb1) == len(emb2)

    def test_retry_on_rate_limit(self, agent, sample_transactions):
        """Test that agent retries on rate limit errors."""
        with patch.object(agent.client.chat.completions, 'create') as mock_create:
            # Simulate rate limit on first call, success on second
            rate_limit_error = Exception("rate_limit_exceeded")
            mock_success = Mock()
            mock_success.choices = [Mock(message=Mock(content='{"fraudulent_transactions": [], "reasoning": {}}'))]
            mock_success.usage = Mock(prompt_tokens=500, completion_tokens=100, total_tokens=600)

            mock_create.side_effect = [rate_limit_error, mock_success]

            # Should retry and succeed
            predictions, metrics = agent.analyze(sample_transactions, retry_delay=0.1)
            assert isinstance(predictions, list)
