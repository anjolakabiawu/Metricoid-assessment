"""
Task: test that the pipeline output correctly matches expected urgency and risk.

This is a lightweight, dependency-free test runner (no pytest needed, though it's
pytest-compatible if you have it). It exits non-zero if any case fails, so it can
gate a CI pipeline.

Run:
    python test_pipeline.py
"""

from __future__ import annotations

import os
import sys

# Ensure this folder is importable so `evaluate` resolves when run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import evaluate_all


def test_urgency_and_risk_match_expected() -> None:
    """Assert every golden-set case predicts the correct urgency AND risk flag.

    Written as a `test_` function so pytest can also collect it. When run
    directly (below), we print a readable per-case table first.
    """
    report = evaluate_all()
    failures = [c for c in report["cases"] if not c["passed"]]
    assert not failures, f"{len(failures)} case(s) failed: {[c['id'] for c in failures]}"


def _print_report() -> int:
    """Pretty-print results and return an exit code (0 = all passed)."""
    report = evaluate_all()

    print("=== Pipeline correctness: urgency & risk_flag ===\n")
    header = f"{'case':<20} {'urgency (exp/got)':<28} {'risk (exp/got)':<20} result"
    print(header)
    print("-" * len(header))

    for c in report["cases"]:
        urg = f"{c['expected_urgency']}/{c['predicted_urgency']}"
        risk = f"{c['expected_risk']}/{c['predicted_risk']}"
        mark = "PASS" if c["passed"] else "FAIL"
        # Flag exactly which sub-check failed for fast debugging.
        detail = ""
        if not c["urgency_ok"]:
            detail += " [urgency mismatch]"
        if not c["risk_ok"]:
            detail += " [risk mismatch]"
        print(f"{c['id']:<20} {urg:<28} {risk:<20} {mark}{detail}")

    print()
    print(
        f"Urgency accuracy: {report['urgency_correct']}/{report['total']} "
        f"({report['urgency_accuracy']:.0%})"
    )
    print(
        f"Risk accuracy:    {report['risk_correct']}/{report['total']} "
        f"({report['risk_accuracy']:.0%})"
    )
    print(
        f"Overall pass:     {report['passed']}/{report['total']} "
        f"({report['pass_rate']:.0%})"
    )

    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    # Exit code lets this double as a CI gate.
    sys.exit(_print_report())
