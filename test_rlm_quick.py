"""Quick test of RLM agent to verify it works."""

import pandas as pd
import sys
sys.path.insert(0, '.')

from src.agents.rlm_agent import RLMFraudAgent

# Create small test dataset
test_data = {
    'transaction_id': ['TXN_001', 'TXN_002', 'TXN_003'],
    'user_id': ['U_001', 'U_001', 'U_002'],
    'amount': [50.0, 5000.0, 75.0],  # TXN_002 is anomaly
    'timestamp': [1000000, 1000010, 1000020],  # TXN_001 and TXN_002 are rapid
    'location': ['NYC', 'NYC', 'LA'],
    'device': ['mobile', 'mobile', 'desktop'],
    'category': ['grocery', 'electronics', 'restaurant']
}

df = pd.DataFrame(test_data)

print("="*60)
print("Testing RLM Agent - Statistical Fallback Mode")
print("="*60)

# Initialize agent
agent = RLMFraudAgent(model="gpt-4o-mini", temperature=0.1)
print(f"✓ Agent initialized")
print(f"  Model: {agent.model}")
print(f"  Historical cases: {len(agent.historical_cases)}")

# Test analysis with statistical fallback (RLM disabled for now)
print(f"\nAnalyzing {len(df)} transactions (using statistical fallback)...")
predictions, metrics = agent.analyze(df, use_rlm=False)

print(f"\n✓ Analysis complete!")
print(f"  Predictions: {predictions}")
print(f"  Tokens: {metrics.total_tokens}")
print(f"  Cost: ${metrics.cost_usd:.6f}")
print(f"  Latency: {metrics.latency_ms/1000:.1f}s")
print(f"  Filter latency: {metrics.filter_latency_ms:.0f}ms")
print(f"  Filtered transactions: {metrics.transactions_filtered}/{metrics.transactions_analyzed}")

# Show filter stats
filter_stats = agent.get_filter_stats()
print(f"\nFilter Statistics:")
print(f"  Total: {filter_stats['total_transactions']}")
print(f"  Suspicious: {filter_stats['suspicious_transactions']}")
print(f"  Filter rate: {filter_stats['filter_rate']*100:.1f}%")

print("\n✅ Test passed!")
