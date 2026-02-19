"""Metrics tracking for fraud detection approaches."""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass
class AnalysisMetrics:
    """Metrics for a single fraud detection analysis."""

    approach: str  # 'naive', 'rag', 'pipeline'
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    transactions_analyzed: int
    context_size_chars: Optional[int] = None
    retrieval_latency_ms: Optional[float] = None
    filter_latency_ms: Optional[float] = None  # Pipeline: time to filter transactions
    transactions_filtered: Optional[int] = None  # Pipeline: number of transactions after filtering
    code_execution_count: Optional[int] = None
    llm_query_calls: Optional[int] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for fraud detection."""

    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """Calculate precision: TP / (TP + FP)."""
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 0.0

    @property
    def recall(self) -> float:
        """Calculate recall: TP / (TP + FN)."""
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """Calculate F1 score: 2 * (precision * recall) / (precision + recall)."""
        p = self.precision
        r = self.recall
        denominator = p + r
        return 2 * (p * r) / denominator if denominator > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)."""
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary with calculated metrics."""
        return {
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'accuracy': self.accuracy
        }


class MetricsTracker:
    """Track and persist metrics across different fraud detection approaches."""

    def __init__(self, results_dir: str = "results/metrics"):
        """Initialize metrics tracker.

        Args:
            results_dir: Directory to store metric results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history: Dict[str, List[AnalysisMetrics]] = {
            'naive': [],
            'rag': [],
            'pipeline': []
        }

    def record_analysis(self, metrics: AnalysisMetrics):
        """Record metrics from an analysis.

        Args:
            metrics: Analysis metrics to record
        """
        if metrics.approach in self.metrics_history:
            self.metrics_history[metrics.approach].append(metrics)

    def calculate_accuracy(self, y_true: List[bool], y_pred: List[bool]) -> AccuracyMetrics:
        """Calculate accuracy metrics from predictions.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            AccuracyMetrics with calculated values
        """
        tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
        tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
        fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)

        return AccuracyMetrics(
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn
        )

    def save_results(self, approach: str, filename: Optional[str] = None):
        """Save metrics for an approach to JSON file.

        Args:
            approach: Approach name ('naive', 'rag', 'pipeline')
            filename: Optional custom filename
        """
        if filename is None:
            filename = f"{approach}_results.json"

        filepath = self.results_dir / filename

        # Convert metrics to dicts
        metrics_dicts = [m.to_dict() for m in self.metrics_history[approach]]

        # Calculate summary statistics
        if metrics_dicts:
            summary = self._calculate_summary(metrics_dicts)
        else:
            summary = {}

        data = {
            'approach': approach,
            'total_analyses': len(metrics_dicts),
            'summary': summary,
            'history': metrics_dicts
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_results(self, approach: str, filename: Optional[str] = None) -> dict:
        """Load saved metrics for an approach.

        Args:
            approach: Approach name
            filename: Optional custom filename

        Returns:
            Dictionary with metrics data
        """
        if filename is None:
            filename = f"{approach}_results.json"

        filepath = self.results_dir / filename

        if not filepath.exists():
            return {'approach': approach, 'total_analyses': 0, 'summary': {}, 'history': []}

        with open(filepath, 'r') as f:
            return json.load(f)

    def _calculate_summary(self, metrics_dicts: List[dict]) -> dict:
        """Calculate summary statistics from metrics history.

        Args:
            metrics_dicts: List of metrics dictionaries

        Returns:
            Dictionary with summary statistics
        """
        if not metrics_dicts:
            return {}

        import statistics

        # Extract numeric fields
        total_tokens = [m['total_tokens'] for m in metrics_dicts]
        prompt_tokens = [m['prompt_tokens'] for m in metrics_dicts]
        completion_tokens = [m['completion_tokens'] for m in metrics_dicts]
        latencies = [m['latency_ms'] for m in metrics_dicts]
        costs = [m['cost_usd'] for m in metrics_dicts]

        return {
            'tokens': {
                'mean_total': statistics.mean(total_tokens),
                'mean_prompt': statistics.mean(prompt_tokens),
                'mean_completion': statistics.mean(completion_tokens),
                'std_total': statistics.stdev(total_tokens) if len(total_tokens) > 1 else 0,
            },
            'latency_ms': {
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'std': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                'min': min(latencies),
                'max': max(latencies)
            },
            'cost_usd': {
                'mean': statistics.mean(costs),
                'total': sum(costs),
                'std': statistics.stdev(costs) if len(costs) > 1 else 0,
            }
        }

    def get_comparison(self) -> dict:
        """Get comparison across all approaches.

        Returns:
            Dictionary with comparison data
        """
        comparison = {}

        for approach in ['naive', 'rag', 'pipeline']:
            if self.metrics_history[approach]:
                metrics_dicts = [m.to_dict() for m in self.metrics_history[approach]]
                comparison[approach] = self._calculate_summary(metrics_dicts)

        return comparison
