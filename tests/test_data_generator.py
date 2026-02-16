"""Tests for synthetic fraud data generator."""

import pytest
import pandas as pd
from data.synthetic_generator import FraudDataGenerator


class TestFraudDataGenerator:
    """Test cases for FraudDataGenerator."""

    def test_initialization(self):
        """Test generator initialization with default parameters."""
        generator = FraudDataGenerator(
            num_transactions=1000,
            fraud_rate=0.02,
            patterns=['velocity', 'amount_anomaly']
        )
        assert generator.num_transactions == 1000
        assert generator.fraud_rate == 0.02
        assert len(generator.patterns) == 2

    def test_generate_returns_dataframe(self):
        """Test that generate() returns a pandas DataFrame."""
        generator = FraudDataGenerator(num_transactions=100, fraud_rate=0.05)
        df = generator.generate()
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_has_required_columns(self):
        """Test that generated DataFrame has all required columns."""
        generator = FraudDataGenerator(num_transactions=100, fraud_rate=0.05)
        df = generator.generate()

        required_columns = [
            'transaction_id', 'user_id', 'amount', 'timestamp',
            'location', 'device', 'category', 'is_fraud', 'fraud_type'
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

        # Check for feature columns v1-v28
        for i in range(1, 29):
            assert f'v{i}' in df.columns, f"Missing feature column: v{i}"

    def test_correct_number_of_transactions(self):
        """Test that correct number of transactions are generated."""
        num_trans = 500
        generator = FraudDataGenerator(num_transactions=num_trans, fraud_rate=0.02)
        df = generator.generate()
        assert len(df) == num_trans

    def test_fraud_rate_approximately_correct(self):
        """Test that fraud rate is approximately as specified."""
        num_trans = 1000
        fraud_rate = 0.05
        generator = FraudDataGenerator(num_transactions=num_trans, fraud_rate=fraud_rate)
        df = generator.generate()

        actual_fraud_rate = df['is_fraud'].sum() / len(df)
        # Allow 20% tolerance
        assert abs(actual_fraud_rate - fraud_rate) < fraud_rate * 0.2

    def test_fraud_types_match_patterns(self):
        """Test that fraud_type values match specified patterns."""
        patterns = ['velocity', 'amount_anomaly', 'geographic']
        generator = FraudDataGenerator(
            num_transactions=1000,
            fraud_rate=0.05,
            patterns=patterns
        )
        df = generator.generate()

        fraud_df = df[df['is_fraud'] == True]
        fraud_types = fraud_df['fraud_type'].unique()

        # All fraud types should be from specified patterns
        for fraud_type in fraud_types:
            if fraud_type is not None:  # legitimate transactions have None
                assert fraud_type in patterns

    def test_legitimate_transactions_have_no_fraud_type(self):
        """Test that legitimate transactions have fraud_type as None."""
        generator = FraudDataGenerator(num_transactions=100, fraud_rate=0.05)
        df = generator.generate()

        legitimate_df = df[df['is_fraud'] == False]
        assert legitimate_df['fraud_type'].isna().all() or (legitimate_df['fraud_type'] == '').all()

    def test_timestamps_are_increasing(self):
        """Test that timestamps are in chronological order."""
        generator = FraudDataGenerator(num_transactions=100, fraud_rate=0.02)
        df = generator.generate()

        timestamps = df['timestamp'].values
        # Check if timestamps are sorted
        assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))

    def test_transaction_ids_are_unique(self):
        """Test that all transaction IDs are unique."""
        generator = FraudDataGenerator(num_transactions=500, fraud_rate=0.02)
        df = generator.generate()

        assert df['transaction_id'].nunique() == len(df)

    def test_amounts_are_positive(self):
        """Test that all transaction amounts are positive."""
        generator = FraudDataGenerator(num_transactions=200, fraud_rate=0.03)
        df = generator.generate()

        assert (df['amount'] > 0).all()

    def test_feature_values_are_numeric(self):
        """Test that feature columns (v1-v28) contain numeric values."""
        generator = FraudDataGenerator(num_transactions=100, fraud_rate=0.02)
        df = generator.generate()

        for i in range(1, 29):
            col = f'v{i}'
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"

    def test_seed_reproducibility(self):
        """Test that using same seed produces consistent structure."""
        seed = 42
        gen1 = FraudDataGenerator(num_transactions=100, fraud_rate=0.02, random_state=seed)
        gen2 = FraudDataGenerator(num_transactions=100, fraud_rate=0.02, random_state=seed)

        df1 = gen1.generate()
        df2 = gen2.generate()

        # Test same fraud rate
        assert df1['is_fraud'].sum() == df2['is_fraud'].sum()

        # Test same fraud types distribution
        fraud_types1 = df1[df1['is_fraud']]['fraud_type'].value_counts().to_dict()
        fraud_types2 = df2[df2['is_fraud']]['fraud_type'].value_counts().to_dict()
        assert fraud_types1 == fraud_types2

        # Test same structure
        assert len(df1) == len(df2)
        assert list(df1.columns) == list(df2.columns)

    def test_velocity_attack_pattern(self):
        """Test that velocity attacks create rapid transactions."""
        generator = FraudDataGenerator(
            num_transactions=500,
            fraud_rate=0.1,
            patterns=['velocity']
        )
        df = generator.generate()

        velocity_fraud = df[df['fraud_type'] == 'velocity']
        if len(velocity_fraud) > 1:
            # Check that velocity frauds have close timestamps
            user_groups = velocity_fraud.groupby('user_id')
            for user_id, group in user_groups:
                if len(group) > 1:
                    time_diffs = group['timestamp'].diff().dropna()
                    # At least some should be very close (< 300 seconds)
                    assert (time_diffs < 300).any()

    def test_amount_anomaly_pattern(self):
        """Test that amount anomalies have significantly higher amounts."""
        generator = FraudDataGenerator(
            num_transactions=1000,
            fraud_rate=0.05,
            patterns=['amount_anomaly']
        )
        df = generator.generate()

        legitimate_amounts = df[df['is_fraud'] == False]['amount']
        anomaly_amounts = df[df['fraud_type'] == 'amount_anomaly']['amount']

        if len(anomaly_amounts) > 0:
            mean_legit = legitimate_amounts.mean()
            mean_anomaly = anomaly_amounts.mean()

            # Anomaly amounts should be significantly higher
            assert mean_anomaly > mean_legit
