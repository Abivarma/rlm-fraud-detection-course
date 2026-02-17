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

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1):
        """Initialize the naive fraud detection agent.

        Args:
            model: OpenAI model to use (default: gpt-4o-mini for cost efficiency)
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
        if 'mini' in model.lower():
            # gpt-4o-mini pricing (much cheaper!)
            self.input_cost = 0.150
            self.output_cost = 0.600
        else:
            # gpt-4o pricing (use env vars if available)
            self.input_cost = float(os.getenv('INPUT_TOKEN_COST', '2.50'))
            self.output_cost = float(os.getenv('OUTPUT_TOKEN_COST', '10.00'))

        # Load historical fraud cases
        self.historical_cases = self._load_historical_cases()

        # Store last analysis reasoning
        self.last_reasoning = {}

    def _load_historical_cases(self) -> List[Dict]:
        """Load historical fraud case studies for context.

        Returns:
            List of historical fraud case dictionaries
        """
        cases_path = os.path.join(os.path.dirname(__file__), '../../data/historical_fraud_cases.json')
        with open(cases_path, 'r') as f:
            return json.load(f)

    def analyze(self, transactions: pd.DataFrame, retry_delay: int = 20) -> Tuple[List[bool], AnalysisMetrics]:
        """Analyze transactions for fraud using naive LLM approach.

        Args:
            transactions: DataFrame with transaction data
            retry_delay: Seconds to wait on rate limit (default: 20)

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

        # Call LLM with retry logic
        max_retries = 3
        for attempt in range(max_retries):
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
                break  # Success, exit retry loop
            except Exception as e:
                error_str = str(e)
                if "rate_limit" in error_str.lower() and attempt < max_retries - 1:
                    print(f"⚠️ Rate limit hit. Waiting {retry_delay}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(retry_delay)
                else:
                    raise Exception(f"OpenAI API error: {error_str}")

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

        # Format ALL historical cases (this makes naive expensive!)
        historical_text = self._format_historical_cases(self.historical_cases)

        prompt = f"""Analyze the following transactions for fraud using historical fraud patterns.

## Fraud Pattern Definitions:

1. **Velocity Attack**: Multiple transactions in rapid succession (< 5 minutes) from same user
2. **Amount Anomaly**: Transaction amount significantly higher than normal
3. **Geographic Outlier**: Transactions from impossible locations
4. **Account Takeover**: Sudden behavioral pattern shift

## Historical Fraud Cases (500 examples for reference):

{historical_text}

## New Transactions to Analyze:

{transactions_text}

## Instructions:

Compare the new transactions against the historical fraud cases above. Identify which transactions are fraudulent based on similar patterns.

Return your analysis as JSON:
{{
    "fraudulent_transactions": ["TXN_001", "TXN_005", ...],
    "reasoning": {{
        "TXN_001": "Matches historical case CASE_XXXX: [reason]",
        "TXN_005": "Similar to cases CASE_YYYY, CASE_ZZZZ: [reason]"
    }}
}}

Reference specific historical cases in your reasoning."""

        return prompt

    def _format_historical_cases(self, cases: List[Dict]) -> str:
        """Format historical fraud cases for prompt.

        Args:
            cases: List of historical fraud case dictionaries

        Returns:
            Formatted string with all cases
        """
        formatted_cases = []
        for case in cases:
            case_text = f"{case['case_id']}: {case['summary']} | {case['fraud_type']} | {case['transaction_pattern']}"
            formatted_cases.append(case_text)

        return "\n".join(formatted_cases)

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
