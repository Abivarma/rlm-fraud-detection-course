"""Synthetic fraud transaction data generator.

Generates realistic e-commerce transaction data with multiple fraud patterns:
- Velocity attacks: Multiple transactions in short time windows
- Amount anomalies: Unusual transaction amounts (statistical outliers)
- Geographic outliers: Impossible location patterns
- Account takeover: Behavioral pattern shifts
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta


class FraudDataGenerator:
    """Generate synthetic fraud transaction data for testing and training."""

    def __init__(
        self,
        num_transactions: int = 10000,
        fraud_rate: float = 0.02,
        patterns: Optional[List[str]] = None,
        random_state: Optional[int] = None
    ):
        """Initialize the fraud data generator.

        Args:
            num_transactions: Total number of transactions to generate
            fraud_rate: Proportion of fraudulent transactions (0.0 to 1.0)
            patterns: List of fraud patterns to include. Options:
                - 'velocity': Rapid succession of transactions
                - 'amount_anomaly': Unusual transaction amounts
                - 'geographic': Impossible location patterns
                - 'account_takeover': Behavioral shifts
            random_state: Random seed for reproducibility
        """
        self.num_transactions = num_transactions
        self.fraud_rate = fraud_rate
        self.patterns = patterns or ['velocity', 'amount_anomaly', 'geographic', 'account_takeover']
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        # Configuration
        self.num_users = max(100, num_transactions // 50)
        self.categories = ['grocery', 'electronics', 'clothing', 'restaurant', 'gas', 'pharmacy']
        self.locations = ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia']
        self.devices = ['mobile', 'desktop', 'tablet']

    def generate(self) -> pd.DataFrame:
        """Generate synthetic transaction data.

        Returns:
            DataFrame with columns: transaction_id, user_id, amount, timestamp,
            location, device, category, v1-v28 (features), is_fraud, fraud_type
        """
        # Calculate fraud counts
        num_fraud = int(self.num_transactions * self.fraud_rate)
        num_legitimate = self.num_transactions - num_fraud

        # Generate legitimate transactions
        legit_data = self._generate_legitimate_transactions(num_legitimate)

        # Generate fraudulent transactions
        fraud_data = self._generate_fraudulent_transactions(num_fraud)

        # Combine
        df = pd.concat([legit_data, fraud_data], ignore_index=True)

        # Sort by timestamp first to maintain chronological order
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Add transaction IDs after sorting
        df['transaction_id'] = [f'TXN_{i:06d}' for i in range(len(df))]

        return df

    def _generate_legitimate_transactions(self, n: int) -> pd.DataFrame:
        """Generate legitimate (non-fraudulent) transactions."""
        data = {
            'user_id': np.random.choice([f'U_{i:04d}' for i in range(self.num_users)], n),
            'amount': self._generate_normal_amounts(n),
            'timestamp': self._generate_timestamps(n),
            'location': np.random.choice(self.locations, n),
            'device': np.random.choice(self.devices, n),
            'category': np.random.choice(self.categories, n),
            'is_fraud': False,
            'fraud_type': None
        }

        # Add PCA-like features (v1-v28)
        for i in range(1, 29):
            data[f'v{i}'] = np.random.randn(n) * (0.5 + i * 0.01)

        return pd.DataFrame(data)

    def _generate_fraudulent_transactions(self, n: int) -> pd.DataFrame:
        """Generate fraudulent transactions based on specified patterns."""
        if n == 0:
            return pd.DataFrame()

        # Distribute fraud across patterns
        pattern_counts = self._distribute_fraud_patterns(n)

        fraud_dfs = []
        for pattern, count in pattern_counts.items():
            if count > 0:
                df = self._generate_pattern_specific_fraud(pattern, count)
                fraud_dfs.append(df)

        return pd.concat(fraud_dfs, ignore_index=True)

    def _distribute_fraud_patterns(self, total_fraud: int) -> dict:
        """Distribute fraud count across different patterns."""
        num_patterns = len(self.patterns)
        base_count = total_fraud // num_patterns
        remainder = total_fraud % num_patterns

        pattern_counts = {}
        for i, pattern in enumerate(self.patterns):
            pattern_counts[pattern] = base_count + (1 if i < remainder else 0)

        return pattern_counts

    def _generate_pattern_specific_fraud(self, pattern: str, count: int) -> pd.DataFrame:
        """Generate fraud transactions for a specific pattern."""
        if pattern == 'velocity':
            return self._generate_velocity_attacks(count)
        elif pattern == 'amount_anomaly':
            return self._generate_amount_anomalies(count)
        elif pattern == 'geographic':
            return self._generate_geographic_outliers(count)
        elif pattern == 'account_takeover':
            return self._generate_account_takeover(count)
        else:
            # Default to amount anomaly
            return self._generate_amount_anomalies(count)

    def _generate_velocity_attacks(self, n: int) -> pd.DataFrame:
        """Generate velocity attack patterns (rapid transactions)."""
        # Group transactions by user for velocity attacks
        num_attack_users = max(1, n // 8)  # Each user does ~8 rapid transactions
        attack_users = np.random.choice([f'U_VELOCITY_{i:04d}' for i in range(num_attack_users)], n)

        # Generate base timestamps and add small random delays for velocity
        base_times = self._generate_timestamps(num_attack_users)
        timestamps = []

        for i, user in enumerate(attack_users):
            # Find which attack group this transaction belongs to
            user_idx = int(user.split('_')[-1])
            base_time = base_times.iloc[user_idx % len(base_times)] if len(base_times) > 0 else 0

            # Add small random delay (0-300 seconds)
            delay = np.random.uniform(0, 300)
            timestamps.append(base_time + delay)

        data = {
            'user_id': attack_users,
            'amount': self._generate_normal_amounts(n) * np.random.uniform(0.8, 1.5, n),
            'timestamp': timestamps,
            'location': np.random.choice(self.locations, n),
            'device': np.random.choice(self.devices, n),
            'category': np.random.choice(self.categories, n),
            'is_fraud': True,
            'fraud_type': 'velocity'
        }

        # Features with slight anomalies
        for i in range(1, 29):
            data[f'v{i}'] = np.random.randn(n) * (0.6 + i * 0.012)

        return pd.DataFrame(data)

    def _generate_amount_anomalies(self, n: int) -> pd.DataFrame:
        """Generate transactions with anomalous amounts."""
        # Generate high amounts (3-8 standard deviations above normal)
        normal_mean = 50.0
        normal_std = 20.0
        anomalous_amounts = normal_mean + normal_std * np.random.uniform(3, 8, n)

        data = {
            'user_id': np.random.choice([f'U_{i:04d}' for i in range(self.num_users)], n),
            'amount': anomalous_amounts,
            'timestamp': self._generate_timestamps(n),
            'location': np.random.choice(self.locations, n),
            'device': np.random.choice(self.devices, n),
            'category': np.random.choice(['electronics', 'jewelry', 'luxury'], n),
            'is_fraud': True,
            'fraud_type': 'amount_anomaly'
        }

        # Features with distinct patterns
        for i in range(1, 29):
            data[f'v{i}'] = np.random.randn(n) * (0.7 + i * 0.015) + 0.5

        return pd.DataFrame(data)

    def _generate_geographic_outliers(self, n: int) -> pd.DataFrame:
        """Generate transactions with impossible geographic patterns."""
        # Pairs of distant locations
        distant_pairs = [
            ('NYC', 'Tokyo'),
            ('LA', 'London'),
            ('Chicago', 'Sydney'),
            ('Houston', 'Dubai')
        ]

        locations = []
        for _ in range(n):
            pair = distant_pairs[np.random.randint(0, len(distant_pairs))]
            locations.append(np.random.choice(pair))

        # Ensure at least 1 user for small n
        num_geo_users = max(1, n // 2)
        data = {
            'user_id': np.random.choice([f'U_GEO_{i:04d}' for i in range(num_geo_users)], n),
            'amount': self._generate_normal_amounts(n),
            'timestamp': self._generate_timestamps(n),
            'location': locations,
            'device': np.random.choice(self.devices, n),
            'category': np.random.choice(self.categories, n),
            'is_fraud': True,
            'fraud_type': 'geographic'
        }

        # Features
        for i in range(1, 29):
            data[f'v{i}'] = np.random.randn(n) * (0.55 + i * 0.011)

        return pd.DataFrame(data)

    def _generate_account_takeover(self, n: int) -> pd.DataFrame:
        """Generate account takeover pattern (sudden behavior change)."""
        # Ensure at least 1 user for small n
        num_ato_users = max(1, n // 3)
        data = {
            'user_id': np.random.choice([f'U_ATO_{i:04d}' for i in range(num_ato_users)], n),
            'amount': self._generate_normal_amounts(n) * np.random.uniform(1.5, 3.0, n),
            'timestamp': self._generate_timestamps(n),
            'location': np.random.choice(self.locations, n),
            'device': np.random.choice(['mobile', 'desktop'], n),  # Often different device
            'category': np.random.choice(['electronics', 'gift_cards', 'luxury'], n),
            'is_fraud': True,
            'fraud_type': 'account_takeover'
        }

        # Features with behavioral shift patterns
        for i in range(1, 29):
            data[f'v{i}'] = np.random.randn(n) * (0.65 + i * 0.013) + 0.3

        return pd.DataFrame(data)

    def _generate_normal_amounts(self, n: int) -> np.ndarray:
        """Generate normal transaction amounts."""
        # Log-normal distribution for realistic transaction amounts
        amounts = np.random.lognormal(mean=3.5, sigma=0.8, size=n)
        # Clip to reasonable range
        amounts = np.clip(amounts, 5.0, 500.0)
        return amounts

    def _generate_timestamps(self, n: int) -> pd.Series:
        """Generate realistic timestamps spanning 30 days."""
        start_date = datetime(2025, 1, 1)
        end_date = start_date + timedelta(days=30)

        # Generate random timestamps
        timestamp_range = (end_date - start_date).total_seconds()
        timestamps = [
            start_date.timestamp() + np.random.uniform(0, timestamp_range)
            for _ in range(n)
        ]

        return pd.Series(sorted(timestamps))

    def get_pattern_definitions(self) -> dict:
        """Get definitions of fraud patterns for documentation.

        Returns:
            Dictionary with pattern names as keys and definitions as values
        """
        return {
            'velocity': {
                'name': 'Velocity Attack',
                'description': 'Multiple transactions in rapid succession from same user',
                'indicators': {
                    'time_window': '< 5 minutes',
                    'transaction_count': '> 8 transactions',
                    'typical_behavior': 'Normal users: 1-2 transactions per hour'
                }
            },
            'amount_anomaly': {
                'name': 'Amount Anomaly',
                'description': 'Transaction amount significantly higher than normal',
                'indicators': {
                    'statistical_threshold': '> 3 standard deviations from mean',
                    'example': 'User average: $50, transaction: $5,000',
                    'context': 'No prior large purchases in history'
                }
            },
            'geographic': {
                'name': 'Geographic Outlier',
                'description': 'Transactions from impossible locations',
                'indicators': {
                    'pattern': 'Location velocity > physically possible',
                    'example': 'NYC purchase, then Tokyo 2 hours later',
                    'physical_limit': 'Speed of air travel'
                }
            },
            'account_takeover': {
                'name': 'Account Takeover',
                'description': 'Sudden behavioral pattern shift indicating compromised account',
                'indicators': {
                    'device_change': 'Different device than usual',
                    'category_shift': 'Unusual purchase categories',
                    'amount_change': '1.5-3x normal spending',
                    'example': 'Grocery shopper suddenly buying gift cards'
                }
            }
        }
