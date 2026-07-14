"""
Shared evaluation core.

Runs the pipeline over the labelled golden set and reports, per case, whether the
predicted urgency and risk_flag match the expected labels. Both test_pipeline.py
(pass/fail gate) and monitor.py (drift tracking over time) import from here so
there is a single source of truth for "how we score the model".
"""

from __future__ import annotations

import json
import os
import sys
from typing import TypedDict

# Make the repo root importable so `src.*` resolves no matter where this script
# is invoked from (repo root, the evaluation/ folder, or a test runner).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.llm_analysis import analyze
from src.preprocess import preprocess, to_transcript, user_text
from src.urgency_scoring import score_urgency

# Golden set lives next to this file; resolve it absolutely so the default works
# from any working directory.
_DEFAULT_GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.json")


class CaseResult(TypedDict):
    id: str
    expected_urgency: str
    predicted_urgency: str
    urgency_ok: bool
    expected_risk: str
    predicted_risk: str
    risk_ok: bool
    passed: bool  # both urgency and risk correct


class EvalReport(TypedDict):
    total: int
    urgency_correct: int
    risk_correct: int
    passed: int
    urgency_accuracy: float
    risk_accuracy: float
    pass_rate: float
    cases: list[CaseResult]


def evaluate_case(case: dict, use_mock: bool = True) -> CaseResult:
    """Run the full pipeline on one labelled case and compare to expectations."""
    turns = preprocess(case["conversation"])
    transcript = to_transcript(turns)
    users_only = user_text(turns)

    # Predictions from the two components under test.
    predicted_risk = analyze(transcript, use_mock=use_mock)["risk_flag"]["value"]
    predicted_urgency = score_urgency(users_only)["level"]

    urgency_ok = predicted_urgency == case["expected_urgency"]
    risk_ok = predicted_risk == case["expected_risk"]

    return CaseResult(
        id=case["id"],
        expected_urgency=case["expected_urgency"],
        predicted_urgency=predicted_urgency,
        urgency_ok=urgency_ok,
        expected_risk=case["expected_risk"],
        predicted_risk=predicted_risk,
        risk_ok=risk_ok,
        passed=urgency_ok and risk_ok,
    )


def evaluate_all(golden_path: str = _DEFAULT_GOLDEN, use_mock: bool = True) -> EvalReport:
    """Evaluate every case in the golden set and aggregate accuracy metrics."""
    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    cases = [evaluate_case(c, use_mock=use_mock) for c in golden["cases"]]
    total = len(cases)

    urgency_correct = sum(c["urgency_ok"] for c in cases)
    risk_correct = sum(c["risk_ok"] for c in cases)
    passed = sum(c["passed"] for c in cases)

    # Guard against an empty golden set so we never divide by zero.
    denom = total or 1

    return EvalReport(
        total=total,
        urgency_correct=urgency_correct,
        risk_correct=risk_correct,
        passed=passed,
        urgency_accuracy=urgency_correct / denom,
        risk_accuracy=risk_correct / denom,
        pass_rate=passed / denom,
        cases=cases,
    )
