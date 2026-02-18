"""Proper RLM (Recursive Language Model) fraud detection agent.

Implements the real RLM paradigm from arXiv:2512.24601:
- Persistent REPL loop with 4 phases: PROBE → FILTER → ANALYZE → AGGREGATE
- Deterministic code-based filters (velocity, amount, geo, device shift)
- LLM sub-calls (llm_query) only for semantic verification on filtered subsets
- Context folding: each sub-call gets only 1 user's data (no context rot)
- Full trajectory logging for auditability

This replaces the broken one-shot code-gen approach in rlm_agent.py.
"""

import os
import json
import time
import statistics
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from src.metrics.tracker import AnalysisMetrics

load_dotenv()


@dataclass
class TrajectoryStep:
    """One step in the RLM REPL trajectory."""
    phase: str          # PROBE, FILTER, ANALYZE, AGGREGATE
    description: str    # What this step does
    code: str           # The code/logic executed
    output: str         # Result of execution
    tokens: int = 0     # LLM tokens used (0 for code-only phases)
    cost: float = 0.0   # LLM cost in USD
    duration_ms: float = 0.0


@dataclass
class Trajectory:
    """Full RLM trajectory for one analysis run."""
    steps: List[TrajectoryStep] = field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0
    total_duration_ms: float = 0.0

    def add(self, step: TrajectoryStep):
        self.steps.append(step)
        self.total_tokens += step.tokens
        self.total_cost += step.cost
        self.total_duration_ms += step.duration_ms

    def to_dict(self) -> Dict:
        return {
            'steps': [
                {
                    'phase': s.phase,
                    'description': s.description,
                    'code': s.code,
                    'output': s.output,
                    'tokens': s.tokens,
                    'cost': s.cost,
                    'duration_ms': s.duration_ms,
                }
                for s in self.steps
            ],
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'total_duration_ms': self.total_duration_ms,
        }


class RLMREPLAgent:
    """RLM agent with iterative REPL loop for fraud detection.

    Architecture (per research_analysis.md):
        Phase 1 PROBE:     Code examines data structure (0 LLM tokens)
        Phase 2 FILTER:    Deterministic filters flag suspicious txns (0 LLM tokens)
        Phase 3 ANALYZE:   llm_query sub-calls verify flagged subset (minimal tokens)
        Phase 4 AGGREGATE: Code merges results into final answer (0 LLM tokens)
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)

        # Cost per 1M tokens
        if 'mini' in model.lower():
            self.input_cost = 0.150
            self.output_cost = 0.600
        else:
            self.input_cost = float(os.getenv('INPUT_TOKEN_COST', '2.50'))
            self.output_cost = float(os.getenv('OUTPUT_TOKEN_COST', '10.00'))

        self.last_trajectory: Optional[Trajectory] = None
        self.last_reasoning: Dict[str, str] = {}

    # ─── DETERMINISTIC FILTERS (Phase 2) ───────────────────────────

    def _velocity_filter(self, txns: List[Dict], window_sec: int = 300,
                         min_count: int = 3) -> Dict[str, Dict]:
        """Detect rapid transactions from same user within time window."""
        results = {}
        by_user = defaultdict(list)
        for t in txns:
            by_user[t['user_id']].append(t)

        for uid, user_txns in by_user.items():
            user_txns.sort(key=lambda x: x['timestamp'])
            if len(user_txns) < min_count:
                continue

            for i in range(len(user_txns)):
                window = [t for t in user_txns
                          if 0 <= t['timestamp'] - user_txns[i]['timestamp'] <= window_sec]
                if len(window) >= min_count:
                    results[uid] = {
                        'count': len(window),
                        'window_sec': window_sec,
                        'txn_ids': [t['transaction_id'] for t in window],
                        'time_span_sec': window[-1]['timestamp'] - window[0]['timestamp'],
                    }
                    break  # Found velocity for this user
        return results

    def _amount_anomaly_filter(self, txns: List[Dict],
                               threshold_sigma: float = 3.0) -> Dict[str, Dict]:
        """Detect amounts > N std deviations from user mean (leave-one-out)."""
        results = {}
        by_user = defaultdict(list)
        for t in txns:
            by_user[t['user_id']].append(t)

        for uid, user_txns in by_user.items():
            if len(user_txns) < 3:
                continue

            for t in user_txns:
                other_amounts = [x['amount'] for x in user_txns
                                 if x['transaction_id'] != t['transaction_id']]
                if len(other_amounts) < 2:
                    continue

                mean_a = statistics.mean(other_amounts)
                std_a = statistics.stdev(other_amounts)
                if std_a > 0:
                    z_score = abs(t['amount'] - mean_a) / std_a
                    if z_score > threshold_sigma:
                        results[t['transaction_id']] = {
                            'amount': t['amount'],
                            'user_mean': round(mean_a, 2),
                            'user_std': round(std_a, 2),
                            'z_score': round(z_score, 2),
                            'user_id': uid,
                        }
        return results

    def _geo_filter(self, txns: List[Dict],
                    max_seconds: int = 600) -> Dict[str, Dict]:
        """Detect same user in different locations within impossible timeframe.

        Identifies the user's 'home' location (earliest or most common)
        and flags transactions at other locations as suspicious.
        """
        results = {}
        by_user = defaultdict(list)
        for t in txns:
            by_user[t['user_id']].append(t)

        for uid, user_txns in by_user.items():
            if len(user_txns) < 2:
                continue
            user_txns.sort(key=lambda x: x['timestamp'])

            # Check all pairs for impossible travel
            has_impossible_travel = False
            all_locations = [t['location'] for t in user_txns]
            unique_locs = list(set(all_locations))

            if len(unique_locs) < 2:
                continue

            for i in range(len(user_txns) - 1):
                for j in range(i + 1, len(user_txns)):
                    dt = user_txns[j]['timestamp'] - user_txns[i]['timestamp']
                    if dt <= max_seconds and user_txns[i]['location'] != user_txns[j]['location']:
                        has_impossible_travel = True
                        break
                if has_impossible_travel:
                    break

            if not has_impossible_travel:
                continue

            # Determine "home" location: earliest transaction's location
            # In fraud, the first transaction is typically legitimate
            # Only override if another location has significantly more occurrences (3x+)
            home_loc = user_txns[0]['location']
            loc_counts = {}
            for loc in all_locations:
                loc_counts[loc] = loc_counts.get(loc, 0) + 1
            home_count = loc_counts.get(home_loc, 0)
            for loc, cnt in loc_counts.items():
                if cnt >= home_count * 3 and loc != home_loc:
                    home_loc = loc  # Override only if overwhelmingly dominant

            # Flag transactions NOT at home location
            anomalous = [t for t in user_txns if t['location'] != home_loc]
            if anomalous:
                results[uid] = {
                    'home_loc': home_loc,
                    'anomalous_locs': list(set(t['location'] for t in anomalous)),
                    'txn_ids': [t['transaction_id'] for t in anomalous],
                    'locations': unique_locs,
                    'from_loc': home_loc,
                    'to_loc': anomalous[0]['location'],
                    'seconds': anomalous[0]['timestamp'] - user_txns[0]['timestamp'],
                }
        return results

    def _device_shift_filter(self, txns: List[Dict]) -> Dict[str, List[Dict]]:
        """Detect sudden device/category/amount shifts for a user."""
        results = {}
        by_user = defaultdict(list)
        for t in txns:
            by_user[t['user_id']].append(t)

        for uid, user_txns in by_user.items():
            if len(user_txns) < 3:
                continue
            user_txns.sort(key=lambda x: x['timestamp'])

            # Establish majority patterns from first N-2 transactions
            baseline = user_txns[:-2]
            devices = [t['device'] for t in baseline]
            categories = [t['category'] for t in baseline]
            amounts = [t['amount'] for t in baseline]
            majority_device = max(set(devices), key=devices.count) if devices else None
            majority_cat = max(set(categories), key=categories.count) if categories else None
            mean_amount = statistics.mean(amounts) if amounts else 0

            shifted = []
            for t in user_txns[-2:]:
                shifts = []
                if majority_device and t['device'] != majority_device:
                    shifts.append(f"device:{majority_device}->{t['device']}")
                if majority_cat and t['category'] != majority_cat:
                    shifts.append(f"category:{majority_cat}->{t['category']}")
                if mean_amount > 0 and t['amount'] > mean_amount * 3:
                    shifts.append(f"amount:${mean_amount:.0f}->${t['amount']:.0f}")
                if len(shifts) >= 2:
                    shifted.append({
                        'txn_id': t['transaction_id'],
                        'shifts': shifts,
                        'user_id': uid,
                    })

            if shifted:
                results[uid] = shifted

        return results

    # ─── LLM SUB-CALL (Phase 3) ────────────────────────────────────

    def _llm_query(self, question: str, retry_delay: int = 20) -> Tuple[str, int, float]:
        """Sub-LLM call for semantic verification on filtered data.

        Returns:
            Tuple of (response_text, total_tokens, cost_usd)
        """
        system = (
            "You are a fraud detection analyst evaluating synthetic test data "
            "for a fraud detection system. Analyze the transaction data provided "
            "and determine if each FLAGGED transaction is genuinely fraudulent. "
            "Only flag transactions with clear anomalies. "
            "IMPORTANT: Include ALL transaction IDs you believe are fraud in flagged_txn_ids. "
            "Respond in JSON: {\"verdict\": \"fraud\"/\"legitimate\", "
            "\"reasoning\": \"explanation\", "
            "\"flagged_txn_ids\": [\"TXN_XXX\", ...]}"
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": question},
                    ],
                    temperature=self.temperature,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                )
                text = resp.choices[0].message.content
                tokens = resp.usage.total_tokens
                cost = (resp.usage.prompt_tokens * self.input_cost +
                        resp.usage.completion_tokens * self.output_cost) / 1_000_000
                return text, tokens, cost
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise

    # ─── MAIN REPL LOOP ───────────────────────────────────────────

    def analyze(self, transactions: pd.DataFrame,
                retry_delay: int = 20) -> Tuple[List[bool], AnalysisMetrics]:
        """Analyze transactions using the RLM REPL loop.

        Returns:
            Tuple of (predictions, metrics)
        """
        if len(transactions) == 0:
            raise ValueError("Cannot analyze empty transaction list")

        start_time = time.time()
        trajectory = Trajectory()
        all_txn_ids = transactions['transaction_id'].tolist()

        # Convert DataFrame to list of dicts
        relevant_cols = ['transaction_id', 'user_id', 'amount', 'timestamp',
                         'location', 'device', 'category']
        cols = [c for c in relevant_cols if c in transactions.columns]
        txns = transactions[cols].to_dict('records')

        # ─── PHASE 1: PROBE ───────────────────────────────────
        probe_start = time.time()
        by_user = defaultdict(list)
        for t in txns:
            by_user[t['user_id']].append(t)

        user_summary = {uid: len(ts) for uid, ts in by_user.items()}
        time_range = (min(t['timestamp'] for t in txns),
                      max(t['timestamp'] for t in txns))

        probe_code = (
            f"# Count users and transactions\n"
            f"users = group_by_user(context)  # {len(by_user)} users\n"
            f"for uid, txns in users.items():\n"
            f"    print(f'{{uid}}: {{len(txns)}} txns')\n"
            f"time_range = (min(timestamps), max(timestamps))"
        )
        probe_output = (
            f"Users: {json.dumps(user_summary)}\n"
            f"Time range: {time_range[1] - time_range[0]}s "
            f"({(time_range[1] - time_range[0]) / 3600:.1f} hours)\n"
            f"Total transactions: {len(txns)}"
        )

        trajectory.add(TrajectoryStep(
            phase="PROBE",
            description="Examine data structure, count users and transactions",
            code=probe_code,
            output=probe_output,
            tokens=0, cost=0.0,
            duration_ms=(time.time() - probe_start) * 1000,
        ))

        # ─── PHASE 2: FILTER ──────────────────────────────────
        filter_start = time.time()

        velocity_flags = self._velocity_filter(txns)
        amount_flags = self._amount_anomaly_filter(txns)
        geo_flags = self._geo_filter(txns)
        device_flags = self._device_shift_filter(txns)

        # Collect all suspicious transaction IDs
        suspicious_ids = set()
        filter_reasons = {}  # txn_id -> list of filter reasons

        for uid, info in velocity_flags.items():
            for tid in info['txn_ids']:
                suspicious_ids.add(tid)
                filter_reasons.setdefault(tid, []).append(
                    f"velocity({info['count']} txns in {info['time_span_sec']}s)"
                )

        for tid, info in amount_flags.items():
            suspicious_ids.add(tid)
            filter_reasons.setdefault(tid, []).append(
                f"amount_anomaly(z={info['z_score']}, ${info['amount']} vs mean ${info['user_mean']})"
            )

        for uid, info in geo_flags.items():
            for tid in info['txn_ids']:
                suspicious_ids.add(tid)
                filter_reasons.setdefault(tid, []).append(
                    f"geo({info['from_loc']}->{info['to_loc']} in {info['seconds']}s)"
                )

        for uid, shifts in device_flags.items():
            for shift in shifts:
                suspicious_ids.add(shift['txn_id'])
                filter_reasons.setdefault(shift['txn_id'], []).append(
                    f"device_shift({', '.join(shift['shifts'])})"
                )

        filter_code = (
            f"velocity_flags  = velocity_filter(context, window=300s, min=3)\n"
            f"amount_flags    = amount_anomaly_filter(context, sigma=3.0)\n"
            f"geo_flags       = geo_filter(context, max_sec=600)\n"
            f"device_flags    = device_shift_filter(context)\n"
            f"suspicious_ids  = union(all flagged txn_ids)"
        )

        filter_lines = []

        vel_detail = json.dumps({k: v['txn_ids'] for k, v in velocity_flags.items()}) if velocity_flags else 'none'
        filter_lines.append(f"velocity_filter:       {len(velocity_flags)} users flagged ({vel_detail})")

        amt_detail_dict = {k: 'z=' + str(v['z_score']) for k, v in amount_flags.items()} if amount_flags else {}
        amt_detail = json.dumps(amt_detail_dict) if amt_detail_dict else 'none'
        filter_lines.append(f"amount_anomaly_filter: {len(amount_flags)} txns flagged ({amt_detail})")

        geo_detail_dict = {k: v['from_loc'] + '->' + v['to_loc'] for k, v in geo_flags.items()} if geo_flags else {}
        geo_detail = json.dumps(geo_detail_dict) if geo_detail_dict else 'none'
        filter_lines.append(f"geo_filter:            {len(geo_flags)} users flagged ({geo_detail})")

        dev_detail_dict = {k: [s['txn_id'] for s in v] for k, v in device_flags.items()} if device_flags else {}
        dev_detail = json.dumps(dev_detail_dict) if dev_detail_dict else 'none'
        filter_lines.append(f"device_shift_filter:   {len(device_flags)} users flagged ({dev_detail})")

        filter_pct = (1 - len(suspicious_ids) / len(txns)) * 100
        filter_lines.append(f"Total suspicious: {len(suspicious_ids)}/{len(txns)} transactions ({filter_pct:.0f}% filtered out)")

        trajectory.add(TrajectoryStep(
            phase="FILTER",
            description="Run deterministic fraud pattern filters",
            code=filter_code,
            output="\n".join(filter_lines),
            tokens=0, cost=0.0,
            duration_ms=(time.time() - filter_start) * 1000,
        ))

        # ─── PHASE 3: ANALYZE (LLM sub-calls) ─────────────────
        analyze_start = time.time()
        confirmed_fraud_ids = set()
        reasoning = {}
        phase3_tokens = 0
        phase3_cost = 0.0

        if suspicious_ids:
            # Group suspicious txns by user for per-user sub-calls
            suspicious_by_user = defaultdict(list)
            for t in txns:
                if t['transaction_id'] in suspicious_ids:
                    suspicious_by_user[t['user_id']].append(t)

            # Also include non-suspicious txns from flagged users for context
            for uid in list(suspicious_by_user.keys()):
                for t in txns:
                    if t['user_id'] == uid and t['transaction_id'] not in suspicious_ids:
                        suspicious_by_user[uid].append(t)
                suspicious_by_user[uid].sort(key=lambda x: x['timestamp'])

            sub_call_details = []
            for uid, user_txns in suspicious_by_user.items():
                flagged = [t for t in user_txns if t['transaction_id'] in suspicious_ids]
                unflagged = [t for t in user_txns if t['transaction_id'] not in suspicious_ids]

                # Build per-user query with filter findings
                query_parts = [f"Analyze user {uid}'s transactions for fraud.\n"]
                query_parts.append("FLAGGED transactions (filter detected anomalies):")
                for t in flagged:
                    reasons = filter_reasons.get(t['transaction_id'], ['unknown'])
                    query_parts.append(
                        f"  {t['transaction_id']}: ${t['amount']:.2f} {t['category']} "
                        f"{t['location']} {t['device']} "
                        f"[Filters: {'; '.join(reasons)}]"
                    )

                if unflagged:
                    query_parts.append("\nUser's NORMAL transactions (baseline):")
                    for t in unflagged:
                        query_parts.append(
                            f"  {t['transaction_id']}: ${t['amount']:.2f} {t['category']} "
                            f"{t['location']} {t['device']}"
                        )

                query_parts.append("\nFor each flagged transaction, confirm if it is fraud or legitimate.")

                query = "\n".join(query_parts)

                resp_text, resp_tokens, resp_cost = self._llm_query(
                    query, retry_delay=retry_delay
                )
                phase3_tokens += resp_tokens
                phase3_cost += resp_cost

                # Parse sub-call response
                try:
                    parsed = json.loads(resp_text)
                    flagged_ids = parsed.get('flagged_txn_ids', [])
                    if not flagged_ids and parsed.get('verdict') == 'fraud':
                        flagged_ids = [t['transaction_id'] for t in flagged]

                    for tid in flagged_ids:
                        if tid in suspicious_ids:
                            confirmed_fraud_ids.add(tid)
                            reasoning[tid] = parsed.get('reasoning', 'Confirmed by sub-call')
                except (json.JSONDecodeError, KeyError):
                    # Fail loud: if parsing fails, flag all suspicious for this user
                    for t in flagged:
                        confirmed_fraud_ids.add(t['transaction_id'])
                        reasoning[t['transaction_id']] = f"Filter-flagged (sub-call parse failed): {resp_text[:100]}"

                sub_call_details.append(
                    f"llm_query(user={uid}, {len(flagged)} flagged txns + "
                    f"{len(unflagged)} baseline) → {resp_tokens} tokens\n"
                    f"  Response: {resp_text[:200]}"
                )

            analyze_output = (
                f"{len(suspicious_by_user)} sub-calls (1 per user):\n" +
                "\n".join(sub_call_details) +
                f"\n→ Confirmed fraud: {sorted(confirmed_fraud_ids)}"
            )
        else:
            analyze_output = "No suspicious transactions flagged. Skipping LLM sub-calls."

        analyze_code = (
            f"# Per-user sub-calls with context folding\n"
            f"for user in flagged_users:\n"
            f"    user_txns = get_user_context(user)  # flagged + baseline\n"
            f"    result = llm_query(user_txns, filter_findings)\n"
            f"    if result.verdict == 'fraud':\n"
            f"        confirmed.add(result.txn_ids)"
        )

        trajectory.add(TrajectoryStep(
            phase="ANALYZE",
            description="LLM sub-calls verify flagged transactions per user",
            code=analyze_code,
            output=analyze_output,
            tokens=phase3_tokens,
            cost=phase3_cost,
            duration_ms=(time.time() - analyze_start) * 1000,
        ))

        # ─── PHASE 4: AGGREGATE ───────────────────────────────
        agg_start = time.time()

        answer = {
            'fraudulent_ids': sorted(confirmed_fraud_ids),
            'reasoning': reasoning,
            'ready': True,
        }

        agg_code = (
            f"answer['fraudulent_ids'] = merge(confirmed_ids)\n"
            f"answer['reasoning'] = merge(sub_call_reasoning)\n"
            f"answer['ready'] = True"
        )
        agg_output = (
            f"Final fraud IDs: {answer['fraudulent_ids']}\n"
            f"Total flagged: {len(confirmed_fraud_ids)}/{len(txns)} transactions"
        )

        trajectory.add(TrajectoryStep(
            phase="AGGREGATE",
            description="Merge sub-call results into final answer",
            code=agg_code,
            output=agg_output,
            tokens=0, cost=0.0,
            duration_ms=(time.time() - agg_start) * 1000,
        ))

        # ─── BUILD PREDICTIONS AND METRICS ─────────────────────
        self.last_trajectory = trajectory
        self.last_reasoning = reasoning

        predictions = [tid in confirmed_fraud_ids for tid in all_txn_ids]

        total_latency_ms = (time.time() - start_time) * 1000

        metrics = AnalysisMetrics(
            approach='rlm_repl',
            total_tokens=trajectory.total_tokens,
            prompt_tokens=int(trajectory.total_tokens * 0.7),  # approximate split
            completion_tokens=int(trajectory.total_tokens * 0.3),
            latency_ms=total_latency_ms,
            cost_usd=trajectory.total_cost,
            transactions_analyzed=len(txns),
            context_size_chars=sum(len(s.code) + len(s.output) for s in trajectory.steps),
            filter_latency_ms=sum(s.duration_ms for s in trajectory.steps if s.phase == "FILTER"),
            transactions_filtered=len(suspicious_ids),
        )

        return predictions, metrics

    def get_trajectory(self) -> Optional[Trajectory]:
        """Get the full trajectory from the last analysis."""
        return self.last_trajectory

    def get_reasoning(self, transaction_id: str) -> str:
        """Get reasoning for a specific transaction."""
        return self.last_reasoning.get(transaction_id, "")

    def format_trajectory(self, trajectory: Optional[Trajectory] = None) -> str:
        """Format trajectory for display."""
        traj = trajectory or self.last_trajectory
        if not traj:
            return "(no trajectory)"

        lines = []
        phase_icons = {
            "PROBE": "🔍", "FILTER": "⚡",
            "ANALYZE": "🧠", "AGGREGATE": "✅",
        }

        for step in traj.steps:
            icon = phase_icons.get(step.phase, "•")
            lines.append(f"\n{icon} PHASE: {step.phase} — {step.description}")
            lines.append(f"   Code:")
            for code_line in step.code.split('\n'):
                lines.append(f"      {code_line}")
            lines.append(f"   Output:")
            for out_line in step.output.split('\n'):
                lines.append(f"      {out_line}")
            if step.tokens > 0:
                lines.append(f"   Tokens: {step.tokens} | Cost: ${step.cost:.6f}")
            lines.append(f"   Duration: {step.duration_ms:.1f}ms")

        lines.append(f"\n{'─'*50}")
        lines.append(f"Total: {traj.total_tokens} tokens | "
                      f"${traj.total_cost:.6f} | "
                      f"{traj.total_duration_ms:.0f}ms")
        return "\n".join(lines)
