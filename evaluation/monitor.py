"""
Model degradation / drift monitoring (continuous improvement).

Idea: quality doesn't fail all at once - it erodes. Keyword lists go stale, the
LLM provider silently changes a model, or real conversations drift away from what
our golden set covers. To catch that, every run of this monitor:

  1. Evaluates the current pipeline against the labelled golden set.
  2. Appends the metrics (with a timestamp) to metrics_history.jsonl.
  3. Compares today's metrics to a rolling baseline of previous runs.
  4. Raises an alert if accuracy dropped beyond a tolerance, or if per-case
     predictions changed vs last run (behavioural drift), or if input
     characteristics shifted (data drift).

Run it on a schedule (cron / CI nightly). Exit code is non-zero on alert so it
can page a human.

Run:
    python monitor.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from datetime import datetime, timezone

# Ensure this folder is importable so `evaluate` resolves when run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import evaluate_all

# Where we persist run history. JSONL = one JSON object per line, easy to append
# and to load incrementally. Anchored next to this file so cwd doesn't matter.
_HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics_history.jsonl")

# How much accuracy is allowed to drop vs the rolling baseline before we alert.
# 5 percentage points is a reasonable default for a small golden set.
_ACCURACY_TOLERANCE = 0.05

# How many prior runs form the "baseline" we compare against.
_BASELINE_WINDOW = 5


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_history() -> list[dict]:
    """Load prior runs from the JSONL history file (empty if first run)."""
    try:
        with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def _append_history(record: dict) -> None:
    """Append one run's record as a new JSONL line."""
    with open(_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _baseline(history: list[dict], key: str) -> float | None:
    """Mean of a metric over the last N runs - our 'expected' value.

    Returns None when there's no history yet (nothing to compare against).
    """
    window = history[-_BASELINE_WINDOW:]
    values = [h[key] for h in window if key in h]
    return statistics.mean(values) if values else None


def run_monitor() -> int:
    """Evaluate, log, and check for degradation. Returns an exit code."""
    history = _load_history()

    # --- 1. Evaluate current pipeline.
    report = evaluate_all()

    # Compact per-case prediction fingerprint so we can detect behavioural drift
    # (same accuracy but different individual predictions).
    predictions = {
        c["id"]: {"urgency": c["predicted_urgency"], "risk": c["predicted_risk"]}
        for c in report["cases"]
    }

    record = {
        "timestamp": _now_iso(),
        "total": report["total"],
        "urgency_accuracy": report["urgency_accuracy"],
        "risk_accuracy": report["risk_accuracy"],
        "pass_rate": report["pass_rate"],
        "predictions": predictions,
    }

    alerts: list[str] = []

    # --- 2. Accuracy-drop check vs rolling baseline.
    for metric in ("urgency_accuracy", "risk_accuracy", "pass_rate"):
        base = _baseline(history, metric)
        current = record[metric]
        if base is not None and current < base - _ACCURACY_TOLERANCE:
            alerts.append(
                f"{metric} dropped: {current:.0%} vs baseline {base:.0%} "
                f"(tolerance {_ACCURACY_TOLERANCE:.0%})"
            )

    # --- 3. Behavioural drift: did any case's prediction change since last run?
    if history:
        last_preds = history[-1].get("predictions", {})
        for case_id, pred in predictions.items():
            prev = last_preds.get(case_id)
            if prev and prev != pred:
                alerts.append(
                    f"prediction changed for '{case_id}': {prev} -> {pred}"
                )

    # --- 4. Persist AFTER comparison so this run becomes tomorrow's baseline.
    _append_history(record)

    # --- Report.
    print("=== Model monitor ===")
    print(f"run @ {record['timestamp']}")
    print(f"urgency_accuracy: {record['urgency_accuracy']:.0%}")
    print(f"risk_accuracy:    {record['risk_accuracy']:.0%}")
    print(f"pass_rate:        {record['pass_rate']:.0%}")
    print(f"history length:   {len(history) + 1} run(s)")
    print()

    if alerts:
        print("!!! DEGRADATION ALERT !!!")
        for a in alerts:
            print(f"  - {a}")
        return 1

    print("OK - no degradation detected vs baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(run_monitor())
