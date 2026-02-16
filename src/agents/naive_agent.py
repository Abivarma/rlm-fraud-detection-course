"""Naive fraud detection agent - sends all transactions to LLM."""

import os
import json
import time
from typing import List, Tuple, Dict
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from src.metrics.tracker import AnalysisMetrics

# Load environment variables
load_dotenv()


class NaiveFraudAgent:
    """Naive approach: Send all transactions directly to LLM for analysis.

    This is the baseline approach that demonstrates:
    - High accuracy (LLMs are good at pattern recognition)
    - High cost (all data goes to LLM)
    - Limited scalability (context window constraints)
    """

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.1):
        """Initialize the naive fraud detection agent.

        Args:
            model: OpenAI model to use
            temperature: Sampling temperature for LLM
        """
        self.model = model
        self.temperature = temperature

        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=api_key)

        # Cost configuration (per 1M tokens)
        self.input_cost = float(os.getenv('INPUT_TOKEN_COST', '2.50'))
        self.output_cost = float(os.getenv('OUTPUT_TOKEN_COST', '10.00'))

        # Store last analysis reasoning
        self.last_reasoning = {}

    def analyze(self, transactions: pd.DataFrame) -> Tuple[List[bool], AnalysisMetrics]:
        """Analyze transactions for fraud using naive LLM approach.

        Args:
            transactions: DataFrame with transaction data

        Returns:
            Tuple of (predictions, metrics):
                - predictions: List of boolean fraud predictions
                - metrics: AnalysisMetrics with performance data
        """
        if len(transactions) == 0:
            raise ValueError("Cannot analyze empty transaction list")

        # Start timing
        start_time = time.time()

        # Build prompt
        prompt = self._build_prompt(transactions)

        # Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert fraud detection analyst. Analyze transactions and identify fraudulent ones with clear reasoning."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Extract response
        response_text = response.choices[0].message.content
        parsed_response = self._parse_response(response_text)

        # Store reasoning
        self.last_reasoning = parsed_response.get('reasoning', {})

        # Convert to predictions list
        fraudulent_ids = set(parsed_response.get('fraudulent_transactions', []))
        predictions = [
            txn_id in fraudulent_ids
            for txn_id in transactions['transaction_id'].tolist()
        ]

        # Calculate cost
        usage = response.usage
        cost = self._calculate_cost(usage.prompt_tokens, usage.completion_tokens)

        # Create metrics
        metrics = AnalysisMetrics(
            approach='naive',
            total_tokens=usage.total_tokens,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            transactions_analyzed=len(transactions),
            context_size_chars=len(prompt)
        )

        return predictions, metrics

    def _build_prompt(self, transactions: pd.DataFrame) -> str:
        """Build the prompt for LLM analysis.

        Args:
            transactions: DataFrame with transaction data

        Returns:
            Formatted prompt string
        """
        # Format transactions
        transactions_text = self._format_transactions(transactions)

        prompt = f"""Analyze the following transactions for fraud.

## Fraud Patterns to Look For:

1. **Velocity Attack**: Multiple transactions in rapid succession (< 5 minutes) from same user
   - Normal: 1-2 transactions per hour
   - Suspicious: > 8 transactions in short window

2. **Amount Anomaly**: Transaction amount significantly higher than normal
   - Statistical threshold: > 3 standard deviations from mean
   - Context: No prior large purchases in history

3. **Geographic Outlier**: Transactions from impossible locations
   - Pattern: Location velocity > physically possible
   - Example: NYC purchase, then Tokyo 2 hours later

4. **Account Takeover**: Sudden behavioral pattern shift
   - Different device than usual
   - Unusual purchase categories
   - Amount 1.5-3x normal spending

## Transactions to Analyze:

{transactions_text}

## Instructions:

Analyze each transaction and identify which ones are fraudulent. For each fraudulent transaction, provide clear reasoning based on the patterns above.

Return your analysis as JSON with this structure:
{{
    "fraudulent_transactions": ["TXN_001", "TXN_005", ...],
    "reasoning": {{
        "TXN_001": "Specific reason why this is fraudulent",
        "TXN_005": "Specific reason why this is fraudulent"
    }}
}}

Be precise and cite specific data points (amounts, times, locations) in your reasoning."""

        return prompt

    def _format_transactions(self, transactions: pd.DataFrame) -> str:
        """Format transactions for inclusion in prompt.

        Args:
            transactions: DataFrame with transaction data

        Returns:
            Formatted transaction string
        """
        # Select relevant columns
        relevant_cols = [
            'transaction_id', 'user_id', 'amount', 'timestamp',
            'location', 'device', 'category'
        ]

        # Filter to relevant columns that exist
        cols_to_use = [col for col in relevant_cols if col in transactions.columns]
        df_subset = transactions[cols_to_use].copy()

        # Convert timestamp to readable format if it exists
        if 'timestamp' in df_subset.columns:
            df_subset['timestamp'] = pd.to_datetime(df_subset['timestamp'], unit='s')
            df_subset['timestamp'] = df_subset['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Format as structured text
        formatted_lines = []
        for idx, row in df_subset.iterrows():
            line_parts = []
            for col in cols_to_use:
                if col == 'amount':
                    line_parts.append(f"{col}: ${row[col]:.2f}")
                else:
                    line_parts.append(f"{col}: {row[col]}")

            formatted_lines.append(" | ".join(line_parts))

        return "\n".join(formatted_lines)

    def _parse_response(self, response_text: str) -> Dict:
        """Parse LLM response into structured format.

        Args:
            response_text: Raw response from LLM

        Returns:
            Parsed dictionary with fraudulent_transactions and reasoning
        """
        # Remove markdown code blocks if present
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0]

        # Parse JSON
        try:
            parsed = json.loads(response_text.strip())
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}\nResponse: {response_text}")

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost of API call.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        input_cost = (prompt_tokens * self.input_cost) / 1_000_000
        output_cost = (completion_tokens * self.output_cost) / 1_000_000
        return input_cost + output_cost

    def get_reasoning(self, transaction_id: str) -> str:
        """Get reasoning for a specific transaction from last analysis.

        Args:
            transaction_id: Transaction ID to get reasoning for

        Returns:
            Reasoning string, or empty string if not found
        """
        return self.last_reasoning.get(transaction_id, "")
