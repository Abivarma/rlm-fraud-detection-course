"""RAG-based fraud detection agent using vector embeddings for pattern retrieval."""

import os
import json
import time
from typing import List, Tuple, Dict, Any
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

from src.metrics.tracker import AnalysisMetrics

# Load environment variables
load_dotenv()


class RAGFraudAgent:
    """RAG approach: Retrieve relevant fraud patterns using vector similarity.

    This approach demonstrates:
    - Reduced context size via semantic search
    - Lower token usage by retrieving only relevant patterns
    - Better than naive but still limited by retrieval quality
    """

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.1):
        """Initialize the RAG fraud detection agent.

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

        # Load fraud patterns
        self.fraud_patterns = self._load_fraud_patterns()

        # Initialize vector store
        self.vector_store = self._build_vector_store()

        # Store last analysis reasoning
        self.last_reasoning = {}

    def _load_fraud_patterns(self) -> Dict:
        """Load fraud pattern definitions from JSON file."""
        patterns_path = os.path.join(os.path.dirname(__file__), '../../data/fraud_patterns.json')
        with open(patterns_path, 'r') as f:
            return json.load(f)

    def _embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def _build_vector_store(self) -> chromadb.Collection:
        """Build vector database from fraud patterns.

        Returns:
            ChromaDB collection with embedded patterns
        """
        # Initialize ChromaDB client (in-memory for simplicity)
        client = chromadb.Client(Settings(anonymized_telemetry=False))

        # Create or get collection
        collection = client.get_or_create_collection(
            name="fraud_patterns",
            metadata={"description": "Fraud detection patterns"}
        )

        # Check if already populated
        if collection.count() > 0:
            return collection

        # Embed and store each fraud pattern
        documents = []
        metadatas = []
        ids = []

        for pattern_id, pattern_data in self.fraud_patterns.items():
            # Create text representation of pattern
            doc_text = f"{pattern_data['name']}: {pattern_data['description']}. "
            doc_text += " ".join([f"{k}: {v}" for k, v in pattern_data['indicators'].items()])

            documents.append(doc_text)
            metadatas.append({
                'pattern_id': pattern_id,
                'name': pattern_data['name'],
                'description': pattern_data['description']
            })
            ids.append(pattern_id)

        # Generate embeddings
        embeddings = [self._embed_text(doc) for doc in documents]

        # Add to collection
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        return collection

    def _retrieve_patterns(self, query: str, k: int = 2) -> List[Dict]:
        """Retrieve most relevant fraud patterns for query.

        Args:
            query: Query text (e.g., transaction description)
            k: Number of patterns to retrieve

        Returns:
            List of relevant fraud patterns
        """
        # Generate query embedding
        query_embedding = self._embed_text(query)

        # Query vector store
        results = self.vector_store.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        # Format results
        retrieved_patterns = []
        if results['metadatas'] and len(results['metadatas']) > 0:
            for metadata, document in zip(results['metadatas'][0], results['documents'][0]):
                pattern_id = metadata['pattern_id']
                retrieved_patterns.append({
                    'id': pattern_id,
                    'name': metadata['name'],
                    'description': metadata['description'],
                    'full_pattern': self.fraud_patterns.get(pattern_id, {})
                })

        return retrieved_patterns

    def analyze(self, transactions: pd.DataFrame, retry_delay: int = 20) -> Tuple[List[bool], AnalysisMetrics]:
        """Analyze transactions for fraud using RAG approach.

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

        # Retrieve relevant fraud patterns
        retrieval_start = time.time()

        # Create query from transaction characteristics
        query = self._create_retrieval_query(transactions)
        retrieved_patterns = self._retrieve_patterns(query, k=3)

        retrieval_latency_ms = (time.time() - retrieval_start) * 1000

        # Build prompt with retrieved context
        prompt = self._build_prompt(transactions, retrieved_patterns)

        # Call LLM with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert fraud detection analyst. Analyze transactions using the provided fraud patterns and identify fraudulent ones with clear reasoning."
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
            approach='rag',
            total_tokens=usage.total_tokens,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            transactions_analyzed=len(transactions),
            context_size_chars=len(prompt),
            retrieval_latency_ms=retrieval_latency_ms
        )

        return predictions, metrics

    def _create_retrieval_query(self, transactions: pd.DataFrame) -> str:
        """Create query for pattern retrieval based on transaction characteristics.

        Args:
            transactions: DataFrame with transactions

        Returns:
            Query string for retrieval
        """
        # Analyze transaction characteristics
        queries = []

        # Check for rapid transactions
        if len(transactions) > 1:
            timestamps = transactions['timestamp'].values
            time_diffs = [(timestamps[i+1] - timestamps[i]) for i in range(len(timestamps)-1)]
            if any(diff < 300 for diff in time_diffs):  # < 5 minutes
                queries.append("multiple rapid transactions")

        # Check for amount anomalies
        amounts = transactions['amount'].values
        if len(amounts) > 2:
            mean_amount = amounts.mean()
            std_amount = amounts.std()
            if any(abs(amt - mean_amount) > 3 * std_amount for amt in amounts):
                queries.append("unusual transaction amounts")

        # Check for location variety
        if transactions['location'].nunique() > 2:
            queries.append("multiple locations geographic patterns")

        # Default query if no specific patterns detected
        if not queries:
            queries.append("general fraud patterns transaction analysis")

        return " ".join(queries)

    def _build_prompt(self, transactions: pd.DataFrame, retrieved_patterns: List[Dict]) -> str:
        """Build prompt with retrieved fraud patterns.

        Args:
            transactions: DataFrame with transaction data
            retrieved_patterns: Retrieved fraud patterns from vector store

        Returns:
            Formatted prompt string
        """
        # Format retrieved patterns
        patterns_text = "## Retrieved Fraud Patterns:\n\n"
        for pattern in retrieved_patterns:
            patterns_text += f"### {pattern['name']}\n"
            patterns_text += f"{pattern['description']}\n\n"
            if 'full_pattern' in pattern and 'indicators' in pattern['full_pattern']:
                patterns_text += "Indicators:\n"
                for key, value in pattern['full_pattern']['indicators'].items():
                    patterns_text += f"- {key}: {value}\n"
                patterns_text += "\n"

        # Format transactions
        transactions_text = self._format_transactions(transactions)

        prompt = f"""Analyze the following transactions for fraud using the retrieved fraud patterns.

{patterns_text}

## Transactions to Analyze:

{transactions_text}

## Instructions:

Based on the fraud patterns above, identify which transactions are fraudulent. For each fraudulent transaction, provide clear reasoning referencing the specific patterns.

Return your analysis as JSON with this structure:
{{
    "fraudulent_transactions": ["TXN_001", "TXN_005", ...],
    "reasoning": {{
        "TXN_001": "Specific reason citing the fraud pattern",
        "TXN_005": "Specific reason citing the fraud pattern"
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

    def _format_transactions_with_context(self, transactions: pd.DataFrame, retrieved_patterns: List[Dict]) -> str:
        """Format transactions with retrieved context (for testing).

        Args:
            transactions: DataFrame with transaction data
            retrieved_patterns: Retrieved patterns

        Returns:
            Formatted string
        """
        return self._format_transactions(transactions)

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
