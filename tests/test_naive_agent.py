"""Tests for naive fraud detection agent."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.agents.naive_agent import NaiveFraudAgent
from src.metrics.tracker import AnalysisMetrics


class TestNaiveFraudAgent:
    """Test cases for NaiveFraudAgent."""

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
    def agent(self):
        """Create NaiveFraudAgent instance."""
        return NaiveFraudAgent(model="gpt-4o", temperature=0.1)

    def test_initialization(self, agent):
        """Test agent initialization with default parameters."""
        assert agent.model == "gpt-4o"
        assert agent.temperature == 0.1
        assert agent.client is not None

    def test_initialization_loads_env(self):
        """Test that agent loads API key from environment."""
        with patch('src.agents.naive_agent.os.getenv') as mock_getenv:
            def getenv_side_effect(key, default=None):
                if key == 'OPENAI_API_KEY':
                    return 'test-api-key'
                elif key == 'INPUT_TOKEN_COST':
                    return '2.50'
                elif key == 'OUTPUT_TOKEN_COST':
                    return '10.00'
                return default

            mock_getenv.side_effect = getenv_side_effect
            agent = NaiveFraudAgent()
            assert agent.client is not None

    def test_format_transactions(self, agent, sample_transactions):
        """Test transaction formatting for LLM prompt."""
        formatted = agent._format_transactions(sample_transactions)

        assert isinstance(formatted, str)
        assert 'TXN_001' in formatted
        assert 'TXN_002' in formatted
        assert 'TXN_003' in formatted
        assert '50.0' in formatted or '50.00' in formatted
        assert 'grocery' in formatted

    def test_format_transactions_includes_all_fields(self, agent, sample_transactions):
        """Test that formatting includes all relevant fields."""
        formatted = agent._format_transactions(sample_transactions)

        # Check for key fields
        assert 'user_id' in formatted.lower() or 'U_0001' in formatted
        assert 'amount' in formatted.lower()
        assert 'location' in formatted.lower() or 'NYC' in formatted
        assert 'device' in formatted.lower() or 'mobile' in formatted
        assert 'category' in formatted.lower() or 'grocery' in formatted

    def test_build_prompt(self, agent, sample_transactions):
        """Test prompt construction."""
        prompt = agent._build_prompt(sample_transactions)

        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Should be substantial
        assert 'fraud' in prompt.lower()
        assert 'transaction' in prompt.lower()

    def test_build_prompt_includes_fraud_patterns(self, agent, sample_transactions):
        """Test that prompt includes fraud pattern definitions."""
        prompt = agent._build_prompt(sample_transactions)

        # Should mention key fraud patterns
        assert 'velocity' in prompt.lower() or 'rapid' in prompt.lower()
        assert 'anomaly' in prompt.lower() or 'unusual' in prompt.lower()

    def test_parse_response_valid_json(self, agent):
        """Test parsing valid JSON response from LLM."""
        response_text = """
        {
            "fraudulent_transactions": ["TXN_001", "TXN_003"],
            "reasoning": {
                "TXN_001": "Amount significantly higher than user average",
                "TXN_003": "Geographic velocity impossible"
            }
        }
        """

        result = agent._parse_response(response_text)

        assert 'fraudulent_transactions' in result
        assert 'reasoning' in result
        assert len(result['fraudulent_transactions']) == 2
        assert 'TXN_001' in result['fraudulent_transactions']

    def test_parse_response_handles_markdown(self, agent):
        """Test parsing JSON wrapped in markdown code blocks."""
        response_text = """```json
        {
            "fraudulent_transactions": ["TXN_002"],
            "reasoning": {"TXN_002": "Suspicious pattern"}
        }
        ```"""

        result = agent._parse_response(response_text)

        assert 'fraudulent_transactions' in result
        assert 'TXN_002' in result['fraudulent_transactions']

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
                prompt_tokens=1000,
                completion_tokens=200,
                total_tokens=1200
            )
            mock_create.return_value = mock_response

            predictions, metrics = agent.analyze(sample_transactions)

            # Check predictions
            assert isinstance(predictions, list)
            assert len(predictions) == len(sample_transactions)
            assert all(isinstance(p, bool) for p in predictions)

            # Check metrics
            assert isinstance(metrics, AnalysisMetrics)
            assert metrics.approach == 'naive'
            assert metrics.total_tokens == 1200
            assert metrics.prompt_tokens == 1000
            assert metrics.completion_tokens == 200
            assert metrics.cost_usd > 0
            assert metrics.latency_ms > 0

    def test_analyze_handles_api_errors(self, agent, sample_transactions):
        """Test that analyze handles API errors gracefully."""
        with patch.object(agent.client.chat.completions, 'create') as mock_create:
            mock_create.side_effect = Exception("API Error")

            with pytest.raises(Exception):
                agent.analyze(sample_transactions)

    def test_cost_calculation(self, agent):
        """Test cost calculation with known token counts."""
        cost = agent._calculate_cost(
            prompt_tokens=1000,
            completion_tokens=500
        )

        # GPT-4o pricing: $2.50/1M input, $10.00/1M output
        expected_cost = (1000 * 2.50 / 1_000_000) + (500 * 10.00 / 1_000_000)
        assert abs(cost - expected_cost) < 0.0001

    def test_predictions_match_transaction_count(self, agent, sample_transactions):
        """Test that number of predictions matches number of transactions."""
        with patch.object(agent.client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="""
            {
                "fraudulent_transactions": ["TXN_002"],
                "reasoning": {"TXN_002": "Suspicious"}
            }
            """))]
            mock_response.usage = Mock(
                prompt_tokens=1000,
                completion_tokens=200,
                total_tokens=1200
            )
            mock_create.return_value = mock_response

            predictions, _ = agent.analyze(sample_transactions)

            assert len(predictions) == len(sample_transactions)

    def test_empty_transactions(self, agent):
        """Test handling of empty transaction list."""
        empty_df = pd.DataFrame(columns=['transaction_id', 'user_id', 'amount'])

        with pytest.raises(ValueError):
            agent.analyze(empty_df)

    def test_reasoning_extraction(self, agent, sample_transactions):
        """Test that reasoning is extracted and stored."""
        with patch.object(agent.client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="""
            {
                "fraudulent_transactions": ["TXN_002"],
                "reasoning": {
                    "TXN_002": "Amount $500 is 10x user average of $50"
                }
            }
            """))]
            mock_response.usage = Mock(
                prompt_tokens=1000,
                completion_tokens=200,
                total_tokens=1200
            )
            mock_create.return_value = mock_response

            predictions, metrics = agent.analyze(sample_transactions)

            # Agent should store reasoning internally
            assert hasattr(agent, 'last_reasoning') or hasattr(agent, 'reasoning')
