"""Generate 8 carefully crafted fraud scenarios for pipeline benefits demonstration.

Each scenario is designed to highlight a specific advantage of the pipeline approach
over Naive (500 historical cases) and RAG (50 retrieved cases).
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def generate_scenarios():
    """Generate 8 demo scenarios with clear ground truth."""

    base_ts = int(datetime(2025, 3, 15, 10, 0, 0).timestamp())
    scenarios = []

    # ── SCENARIO 1: Velocity Attack ────────────────────────────
    # 5 transactions in 3 minutes from same user, different electronics
    scenarios.append({
        'id': 1,
        'name': 'Velocity Attack',
        'description': '5 transactions in 3 minutes from same user — card testing pattern',
        'why_pipeline_wins': 'Code detects time_diff < 300s deterministically; Naive sends 22K tokens for same result',
        'transactions': [
            {'transaction_id': 'TXN_S1_001', 'user_id': 'U_VEL_01', 'amount': 45.99,
             'timestamp': base_ts, 'location': 'NYC', 'device': 'mobile',
             'category': 'electronics', 'is_fraud': True, 'fraud_type': 'velocity'},
            {'transaction_id': 'TXN_S1_002', 'user_id': 'U_VEL_01', 'amount': 52.30,
             'timestamp': base_ts + 35, 'location': 'NYC', 'device': 'mobile',
             'category': 'electronics', 'is_fraud': True, 'fraud_type': 'velocity'},
            {'transaction_id': 'TXN_S1_003', 'user_id': 'U_VEL_01', 'amount': 38.75,
             'timestamp': base_ts + 72, 'location': 'NYC', 'device': 'mobile',
             'category': 'clothing', 'is_fraud': True, 'fraud_type': 'velocity'},
            {'transaction_id': 'TXN_S1_004', 'user_id': 'U_VEL_01', 'amount': 61.20,
             'timestamp': base_ts + 130, 'location': 'NYC', 'device': 'mobile',
             'category': 'electronics', 'is_fraud': True, 'fraud_type': 'velocity'},
            {'transaction_id': 'TXN_S1_005', 'user_id': 'U_VEL_01', 'amount': 47.85,
             'timestamp': base_ts + 178, 'location': 'NYC', 'device': 'mobile',
             'category': 'electronics', 'is_fraud': True, 'fraud_type': 'velocity'},
        ],
    })

    # ── SCENARIO 2: Geographic Impossibility ───────────────────
    # User in NYC, then Tokyo 10 minutes later (impossible travel)
    scenarios.append({
        'id': 2,
        'name': 'Geographic Impossibility',
        'description': 'User transacts in NYC, then Tokyo 10 minutes later — impossible travel',
        'why_pipeline_wins': 'Code checks location+time pairs; RAG cannot do distance/time math',
        'transactions': [
            {'transaction_id': 'TXN_S2_001', 'user_id': 'U_GEO_01', 'amount': 28.50,
             'timestamp': base_ts, 'location': 'NYC', 'device': 'mobile',
             'category': 'restaurant', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S2_002', 'user_id': 'U_GEO_01', 'amount': 32.00,
             'timestamp': base_ts + 120, 'location': 'NYC', 'device': 'mobile',
             'category': 'grocery', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S2_003', 'user_id': 'U_GEO_01', 'amount': 89.99,
             'timestamp': base_ts + 500, 'location': 'Tokyo', 'device': 'desktop',
             'category': 'electronics', 'is_fraud': True, 'fraud_type': 'geographic'},
            {'transaction_id': 'TXN_S2_004', 'user_id': 'U_GEO_01', 'amount': 125.00,
             'timestamp': base_ts + 600, 'location': 'Tokyo', 'device': 'desktop',
             'category': 'electronics', 'is_fraud': True, 'fraud_type': 'geographic'},
        ],
    })

    # ── SCENARIO 3: Amount Spike ───────────────────────────────
    # User typically spends $15-25, suddenly $487 on jewelry
    scenarios.append({
        'id': 3,
        'name': 'Amount Spike',
        'description': 'User averages $19 on groceries, suddenly spends $487 on jewelry from desktop',
        'why_pipeline_wins': 'Code computes leave-one-out std dev (z=170); LLM confirms category shift',
        'transactions': [
            {'transaction_id': 'TXN_S3_001', 'user_id': 'U_AMT_01', 'amount': 18.50,
             'timestamp': base_ts, 'location': 'LA', 'device': 'mobile',
             'category': 'grocery', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S3_002', 'user_id': 'U_AMT_01', 'amount': 22.30,
             'timestamp': base_ts + 3600, 'location': 'LA', 'device': 'mobile',
             'category': 'gas', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S3_003', 'user_id': 'U_AMT_01', 'amount': 15.75,
             'timestamp': base_ts + 7200, 'location': 'LA', 'device': 'mobile',
             'category': 'restaurant', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S3_004', 'user_id': 'U_AMT_01', 'amount': 19.99,
             'timestamp': base_ts + 10800, 'location': 'LA', 'device': 'mobile',
             'category': 'pharmacy', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S3_005', 'user_id': 'U_AMT_01', 'amount': 487.50,
             'timestamp': base_ts + 14400, 'location': 'LA', 'device': 'desktop',
             'category': 'jewelry', 'is_fraud': True, 'fraud_type': 'amount_anomaly'},
        ],
    })

    # ── SCENARIO 4: Account Takeover ───────────────────────────
    # User's profile suddenly changes: device, location, category, amount
    scenarios.append({
        'id': 4,
        'name': 'Account Takeover',
        'description': 'User profile shifts: mobile→desktop, LA→Chicago, grocery→gift_cards, $20→$250',
        'why_pipeline_wins': 'Multi-filter cross-check: device_shift + amount catches subtle ATO pattern',
        'transactions': [
            {'transaction_id': 'TXN_S4_001', 'user_id': 'U_ATO_01', 'amount': 22.50,
             'timestamp': base_ts, 'location': 'LA', 'device': 'mobile',
             'category': 'grocery', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S4_002', 'user_id': 'U_ATO_01', 'amount': 18.75,
             'timestamp': base_ts + 3600, 'location': 'LA', 'device': 'mobile',
             'category': 'gas', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S4_003', 'user_id': 'U_ATO_01', 'amount': 25.00,
             'timestamp': base_ts + 7200, 'location': 'LA', 'device': 'mobile',
             'category': 'restaurant', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S4_004', 'user_id': 'U_ATO_01', 'amount': 245.00,
             'timestamp': base_ts + 10800, 'location': 'Chicago', 'device': 'desktop',
             'category': 'gift_cards', 'is_fraud': True, 'fraud_type': 'account_takeover'},
            {'transaction_id': 'TXN_S4_005', 'user_id': 'U_ATO_01', 'amount': 189.99,
             'timestamp': base_ts + 11400, 'location': 'Chicago', 'device': 'desktop',
             'category': 'electronics', 'is_fraud': True, 'fraud_type': 'account_takeover'},
        ],
    })

    # ── SCENARIO 5: Micro-Transaction Testing ──────────────────
    # 8 tiny transactions ($1.01-$1.99) in 2 minutes — bot card testing
    scenarios.append({
        'id': 5,
        'name': 'Micro-Transaction Testing',
        'description': '8 automated $1-2 transactions in 2 minutes — bot card testing pattern',
        'why_pipeline_wins': 'Velocity filter catches count + timing; RAG finds no similar historical cases for $1 txns',
        'transactions': [
            {'transaction_id': f'TXN_S5_{i+1:03d}', 'user_id': 'U_MICRO_01',
             'amount': round(1.0 + i * 0.13, 2),
             'timestamp': base_ts + i * 15, 'location': 'NYC', 'device': 'mobile',
             'category': 'digital_goods', 'is_fraud': True, 'fraud_type': 'velocity'}
            for i in range(8)
        ],
    })

    # ── SCENARIO 6: Legitimate High-Value ──────────────────────
    # Wealthy user consistently spends $200-400 — NO fraud
    scenarios.append({
        'id': 6,
        'name': 'Legitimate High-Value',
        'description': 'Consistent high-value user ($200-400 range) — should NOT be flagged',
        'why_pipeline_wins': 'Filters correctly pass through (no anomaly); Naive may over-flag due to high amounts',
        'transactions': [
            {'transaction_id': 'TXN_S6_001', 'user_id': 'U_LEGIT_01', 'amount': 245.00,
             'timestamp': base_ts, 'location': 'NYC', 'device': 'desktop',
             'category': 'electronics', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S6_002', 'user_id': 'U_LEGIT_01', 'amount': 312.50,
             'timestamp': base_ts + 7200, 'location': 'NYC', 'device': 'desktop',
             'category': 'clothing', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S6_003', 'user_id': 'U_LEGIT_01', 'amount': 189.99,
             'timestamp': base_ts + 14400, 'location': 'NYC', 'device': 'desktop',
             'category': 'electronics', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S6_004', 'user_id': 'U_LEGIT_01', 'amount': 275.00,
             'timestamp': base_ts + 21600, 'location': 'NYC', 'device': 'desktop',
             'category': 'jewelry', 'is_fraud': False, 'fraud_type': ''},
        ],
    })

    # ── SCENARIO 7: Mixed Batch (Multi-User) ──────────────────
    # 5 users, 15 transactions total, 2 users are fraudulent
    scenarios.append({
        'id': 7,
        'name': 'Mixed Batch',
        'description': '5 users, 15 transactions — 2 users have fraud, 3 are legitimate',
        'why_pipeline_wins': 'Per-user sub-calls analyze 5 users in parallel; Naive sends all 15 + 500 cases',
        'transactions': [
            # User A: legitimate
            {'transaction_id': 'TXN_S7_001', 'user_id': 'U_MIX_A', 'amount': 35.00,
             'timestamp': base_ts, 'location': 'LA', 'device': 'mobile',
             'category': 'grocery', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S7_002', 'user_id': 'U_MIX_A', 'amount': 42.50,
             'timestamp': base_ts + 3600, 'location': 'LA', 'device': 'mobile',
             'category': 'gas', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S7_003', 'user_id': 'U_MIX_A', 'amount': 28.00,
             'timestamp': base_ts + 7200, 'location': 'LA', 'device': 'mobile',
             'category': 'restaurant', 'is_fraud': False, 'fraud_type': ''},
            # User B: velocity attack
            {'transaction_id': 'TXN_S7_004', 'user_id': 'U_MIX_B', 'amount': 55.00,
             'timestamp': base_ts, 'location': 'Chicago', 'device': 'mobile',
             'category': 'electronics', 'is_fraud': True, 'fraud_type': 'velocity'},
            {'transaction_id': 'TXN_S7_005', 'user_id': 'U_MIX_B', 'amount': 48.00,
             'timestamp': base_ts + 30, 'location': 'Chicago', 'device': 'mobile',
             'category': 'electronics', 'is_fraud': True, 'fraud_type': 'velocity'},
            {'transaction_id': 'TXN_S7_006', 'user_id': 'U_MIX_B', 'amount': 62.00,
             'timestamp': base_ts + 65, 'location': 'Chicago', 'device': 'mobile',
             'category': 'clothing', 'is_fraud': True, 'fraud_type': 'velocity'},
            # User C: legitimate
            {'transaction_id': 'TXN_S7_007', 'user_id': 'U_MIX_C', 'amount': 15.00,
             'timestamp': base_ts + 1000, 'location': 'Houston', 'device': 'mobile',
             'category': 'pharmacy', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S7_008', 'user_id': 'U_MIX_C', 'amount': 22.00,
             'timestamp': base_ts + 5000, 'location': 'Houston', 'device': 'mobile',
             'category': 'grocery', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S7_009', 'user_id': 'U_MIX_C', 'amount': 18.50,
             'timestamp': base_ts + 9000, 'location': 'Houston', 'device': 'mobile',
             'category': 'restaurant', 'is_fraud': False, 'fraud_type': ''},
            # User D: amount anomaly
            {'transaction_id': 'TXN_S7_010', 'user_id': 'U_MIX_D', 'amount': 20.00,
             'timestamp': base_ts, 'location': 'Phoenix', 'device': 'mobile',
             'category': 'gas', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S7_011', 'user_id': 'U_MIX_D', 'amount': 25.00,
             'timestamp': base_ts + 3600, 'location': 'Phoenix', 'device': 'mobile',
             'category': 'grocery', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S7_012', 'user_id': 'U_MIX_D', 'amount': 18.00,
             'timestamp': base_ts + 7200, 'location': 'Phoenix', 'device': 'mobile',
             'category': 'restaurant', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S7_013', 'user_id': 'U_MIX_D', 'amount': 550.00,
             'timestamp': base_ts + 10800, 'location': 'Phoenix', 'device': 'desktop',
             'category': 'jewelry', 'is_fraud': True, 'fraud_type': 'amount_anomaly'},
            # User E: legitimate
            {'transaction_id': 'TXN_S7_014', 'user_id': 'U_MIX_E', 'amount': 120.00,
             'timestamp': base_ts + 2000, 'location': 'NYC', 'device': 'desktop',
             'category': 'electronics', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S7_015', 'user_id': 'U_MIX_E', 'amount': 95.00,
             'timestamp': base_ts + 6000, 'location': 'NYC', 'device': 'desktop',
             'category': 'clothing', 'is_fraud': False, 'fraud_type': ''},
        ],
    })

    # ── SCENARIO 8: Cross-Border Rapid ─────────────────────────
    # 5 transactions across 4 countries in 30 minutes
    scenarios.append({
        'id': 8,
        'name': 'Cross-Border Rapid',
        'description': '5 transactions across London→Paris→Tokyo→Sydney in 30 minutes',
        'why_pipeline_wins': 'Geo + velocity filters both trigger = high confidence; RAG misses temporal pattern',
        'transactions': [
            {'transaction_id': 'TXN_S8_001', 'user_id': 'U_XBORDER_01', 'amount': 75.00,
             'timestamp': base_ts, 'location': 'London', 'device': 'mobile',
             'category': 'restaurant', 'is_fraud': False, 'fraud_type': ''},
            {'transaction_id': 'TXN_S8_002', 'user_id': 'U_XBORDER_01', 'amount': 120.00,
             'timestamp': base_ts + 300, 'location': 'Paris', 'device': 'mobile',
             'category': 'electronics', 'is_fraud': True, 'fraud_type': 'geographic'},
            {'transaction_id': 'TXN_S8_003', 'user_id': 'U_XBORDER_01', 'amount': 89.50,
             'timestamp': base_ts + 600, 'location': 'Tokyo', 'device': 'desktop',
             'category': 'electronics', 'is_fraud': True, 'fraud_type': 'geographic'},
            {'transaction_id': 'TXN_S8_004', 'user_id': 'U_XBORDER_01', 'amount': 200.00,
             'timestamp': base_ts + 1200, 'location': 'Sydney', 'device': 'desktop',
             'category': 'gift_cards', 'is_fraud': True, 'fraud_type': 'geographic'},
            {'transaction_id': 'TXN_S8_005', 'user_id': 'U_XBORDER_01', 'amount': 150.00,
             'timestamp': base_ts + 1800, 'location': 'Sydney', 'device': 'desktop',
             'category': 'electronics', 'is_fraud': True, 'fraud_type': 'geographic'},
        ],
    })

    return scenarios


def save_scenarios(scenarios, output_dir):
    """Save scenarios to CSV and JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Flatten all transactions
    all_txns = []
    for s in scenarios:
        for t in s['transactions']:
            t['scenario_id'] = s['id']
            t['scenario_name'] = s['name']
            all_txns.append(t)

    # Save CSV
    df = pd.DataFrame(all_txns)
    csv_path = output_dir / 'demo_examples.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} transactions to {csv_path}")

    # Save scenario metadata (without transactions for cleaner JSON)
    meta = []
    for s in scenarios:
        meta.append({
            'id': s['id'],
            'name': s['name'],
            'description': s['description'],
            'why_pipeline_wins': s['why_pipeline_wins'],
            'num_transactions': len(s['transactions']),
            'num_fraud': sum(1 for t in s['transactions'] if t['is_fraud']),
            'fraud_types': list(set(t['fraud_type'] for t in s['transactions'] if t['fraud_type'])),
            'txn_ids': [t['transaction_id'] for t in s['transactions']],
            'fraud_txn_ids': [t['transaction_id'] for t in s['transactions'] if t['is_fraud']],
        })
    json_path = output_dir / 'demo_scenarios.json'
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {len(meta)} scenario metadata to {json_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("DEMO SCENARIOS SUMMARY")
    print(f"{'='*60}")
    for s in meta:
        fraud_pct = s['num_fraud'] / s['num_transactions'] * 100
        print(f"  {s['id']}. {s['name']:<30} "
              f"{s['num_transactions']} txns, {s['num_fraud']} fraud ({fraud_pct:.0f}%)")
        print(f"     → {s['description']}")
    total_txns = sum(s['num_transactions'] for s in meta)
    total_fraud = sum(s['num_fraud'] for s in meta)
    print(f"\n  TOTAL: {total_txns} transactions, {total_fraud} fraud ({total_fraud/total_txns*100:.0f}%)")

    return df


if __name__ == '__main__':
    scenarios = generate_scenarios()
    data_dir = Path(__file__).parent.parent / 'data'
    save_scenarios(scenarios, data_dir)
