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
    """RAG approach: Retrieve relevant historical fraud cases using vector similarity.

    This approach demonstrates:
    - Reduced context size via semantic retrieval
    - Lower token usage by retrieving only relevant cases (50 vs 500)
    - 20-30% cost reduction vs naive approach
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1):
        """Initialize the RAG fraud detection agent.

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

        # Load fraud patterns (basic definitions)
        self.fraud_patterns = self._load_fraud_patterns()

        # Load historical fraud cases (500 cases for retrieval)
        self.historical_cases = self._load_historical_cases()

        # Initialize vector store with historical cases
        self.vector_store = self._build_vector_store()

        # Store last analysis reasoning
        self.last_reasoning = {}

    def _load_fraud_patterns(self) -> Dict:
        """Load fraud pattern definitions from JSON file."""
        patterns_path = os.path.join(os.path.dirname(__file__), '../../data/fraud_patterns.json')
        with open(patterns_path, 'r') as f:
            return json.load(f)

    def _load_historical_cases(self) -> List[Dict]:
        """Load historical fraud case studies.

        Returns:
            List of historical fraud case dictionaries
        """
        cases_path = os.path.join(os.path.dirname(__file__), '../../data/historical_fraud_cases.json')
        with open(cases_path, 'r') as f:
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
        """Build vector database from historical fraud cases.

        Returns:
            ChromaDB collection with embedded historical cases
        """
        # Initialize ChromaDB client (in-memory for simplicity)
        client = chromadb.Client(Settings(anonymized_telemetry=False))

        # Create or get collection
        collection = client.get_or_create_collection(
            name="historical_fraud_cases",
            metadata={"description": "Historical fraud case studies for retrieval"}
        )

        # Check if already populated
        if collection.count() > 0:
            return collection

        print(f"📦 Building vector store with {len(self.historical_cases)} historical cases...")

        # Embed and store each historical case
        documents = []
        metadatas = []
        ids = []

        for case in self.historical_cases:
            # Create text representation of case for embedding
            doc_text = f"{case['fraud_type']}: {case['summary']}. "
            doc_text += f"{case['transaction_pattern']}. "
            doc_text += f"Indicators: {', '.join(case['indicators'])}. "
            doc_text += f"Reasoning: {case['reasoning']}"

            documents.append(doc_text)
            metadatas.append({
                'case_id': case['case_id'],
                'fraud_type': case['fraud_type'],
                'summary': case['summary']
            })
            ids.append(case['case_id'])

        # Generate embeddings with rate limiting (100 RPM limit)
        print("⏳ Generating embeddings (this may take ~3 minutes due to rate limits)...")
        embeddings = []
        batch_size = 80  # Stay under 100 RPM limit
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            print(f"   Embedding batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}...")
            batch_embeddings = [self._embed_text(doc) for doc in batch_docs]
            embeddings.extend(batch_embeddings)
            # Wait 60s between batches to respect rate limit
            if i + batch_size < len(documents):
                print(f"   Waiting 60s for rate limit...")
                time.sleep(60)

        # Add to collection
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        print(f"✅ Vector store built with {collection.count()} cases")

        return collection

    def _retrieve_historical_cases(self, query: str, k: int = 50) -> List[Dict]:
        """Retrieve most relevant historical fraud cases for query.

        Args:
            query: Query text describing transactions to analyze
            k: Number of cases to retrieve (default: 50 out of 500)

        Returns:
            List of relevant historical fraud cases
        """
        # Generate query embedding
        query_embedding = self._embed_text(query)

        # Query vector store
        results = self.vector_store.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        # Format results
        retrieved_cases = []
        if results['metadatas'] and len(results['metadatas']) > 0:
            for metadata in results['metadatas'][0]:
                case_id = metadata['case_id']
                # Find full case details
                full_case = next((c for c in self.historical_cases if c['case_id'] == case_id), None)
                if full_case:
                    retrieved_cases.append(full_case)

        return retrieved_cases

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

        # Retrieve relevant historical fraud cases
        retrieval_start = time.time()

        # Create query from transaction characteristics
        query = self._create_retrieval_query(transactions)
        retrieved_cases = self._retrieve_historical_cases(query, k=50)

        retrieval_latency_ms = (time.time() - retrieval_start) * 1000

        # Build prompt with retrieved context
        prompt = self._build_prompt(transactions, retrieved_cases)

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

    def _build_prompt(self, transactions: pd.DataFrame, retrieved_cases: List[Dict]) -> str:
        """Build prompt with retrieved historical fraud cases.

        Args:
            transactions: DataFrame with transaction data
            retrieved_cases: Retrieved historical cases from vector store

        Returns:
            Formatted prompt string
        """
        # Format retrieved historical cases (only top 50 of 500!)
        cases_text = self._format_historical_cases(retrieved_cases)

        # Format transactions
        transactions_text = self._format_transactions(transactions)

        prompt = f"""Analyze the following transactions for fraud using retrieved historical fraud cases.

## Fraud Pattern Definitions:

1. **Velocity Attack**: Multiple transactions in rapid succession
2. **Amount Anomaly**: Transaction amount significantly higher than normal
3. **Geographic Outlier**: Transactions from impossible locations
4. **Account Takeover**: Sudden behavioral pattern shift

## Retrieved Historical Fraud Cases (50 most relevant of 500):

{cases_text}

## New Transactions to Analyze:

{transactions_text}

## Instructions:

Compare the new transactions against the retrieved historical fraud cases. Identify fraudulent transactions based on similar patterns.

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
            Formatted string with cases
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
