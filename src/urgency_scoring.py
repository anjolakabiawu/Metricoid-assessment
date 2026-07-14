"""
Task 3 - Rule-based urgency scoring.

Design decision: we use a transparent, tiered keyword-weighting
scheme rather than an opaque model. Every point in the score is traceable to a
matched keyword, so a clinician/reviewer can see *why* a conversation was flagged
High. That explainability is worth more than marginal accuracy in a
health/safety context.

How it works:
  - Three tiers of keywords with different weights (high/medium/low severity).
  - We sum weights for every keyword hit in the USER's text.
  - The total score maps to Low / Medium / High via thresholds.
  - We also return the matched keywords so the decision is auditable.
"""

from __future__ import annotations

from typing import TypedDict


# --- Keyword tiers. Weights are deliberately spaced so a single critical term
#     (weight 5) can push to High on its own, while milder terms accumulate.
_HIGH_SEVERITY = {
    "severe": 5,
    "urgent": 5,
    "emergency": 5,
    "suicide": 5,
    "self-harm": 5,
    "harm myself": 5,
    "chest pain": 5,
    "overdose": 5,
    "hopeless": 4,
    "unbearable": 4,
    "profusely": 5,
    # --- Breathing emergencies. We match several phrasings because callers
    #     describe the same crisis many ways ("can't breathe", "not breathing",
    #     "stopped breathing", "struggling to breathe").
    "can't breathe": 5,
    "cant breathe": 5,
    "not breathing": 5,
    "stopped breathing": 5,
    "trouble breathing": 5,
    "difficulty breathing": 5,
    "struggling to breathe": 5,
    "choking": 5,
    # --- Loss of consciousness / unresponsiveness (life-threatening).
    "fainted": 5,
    "passed out": 5,
    "unconscious": 5,
    "unresponsive": 5,
    "not responding": 5,
    "not moving": 5,
    "collapsed": 5,
    # --- Other acute medical emergencies.
    "bleeding": 5,
    "seizure": 5,
    "stroke": 5,
    "heart attack": 5,
    "not waking up": 5,
    "won't wake up": 5,
}

_MEDIUM_SEVERITY = {
    "anxious": 2,
    "anxiety": 2,
    "panic": 2,
    "pain": 2,
    "depressed": 2,
    "depression": 2,
    "scared": 2,
    "afraid": 2,
    "worried": 2,
    "trouble sleeping": 2,
    "insomnia": 2,
    "stressed": 2,
    "overwhelmed": 2,
}

_LOW_SEVERITY = {
    "trouble": 1,
    "tired": 1,
    "uncomfortable": 1,
    "concerned": 1,
    "difficult": 1,
    "sad": 1,
}

# Score thresholds -> level. Tuned so:
#   any high-severity term (>=4) => High
#   a couple of medium terms (>=3 total) => Medium
_HIGH_THRESHOLD = 4
_MEDIUM_THRESHOLD = 2


class UrgencyResult(TypedDict):
    level: str  # "Low" | "Medium" | "High"
    score: int
    matched: list[tuple[str, int]]  # (keyword, weight) pairs that fired


def score_urgency(user_text: str) -> UrgencyResult:
    """Compute an urgency level from the user's utterances.

    We scan the (lowercased) user text for each keyword across all tiers and sum
    the weights of every match. Matches are recorded for explainability.
    """
    lower = user_text.lower()
    total = 0
    matched: list[tuple[str, int]] = []

    # Flatten all tiers, then sort by keyword length (longest first). This lets a
    # specific phrase like "trouble sleeping" claim its span before the generic
    # "trouble" can match it, so we don't double-count overlapping keywords.
    all_terms = {**_HIGH_SEVERITY, **_MEDIUM_SEVERITY, **_LOW_SEVERITY}
    consumed = lower  # we blank out matched spans as we go

    for keyword in sorted(all_terms, key=len, reverse=True):
        if keyword in consumed:
            weight = all_terms[keyword]
            total += weight
            matched.append((keyword, weight))
            # Remove the matched text so a shorter substring can't re-match it.
            consumed = consumed.replace(keyword, " ")

    # Present matches strongest-first for readable output.
    matched.sort(key=lambda kw: kw[1], reverse=True)

    # Map the accumulated score to a discrete level.
    if total >= _HIGH_THRESHOLD:
        level = "High"
    elif total >= _MEDIUM_THRESHOLD:
        level = "Medium"
    else:
        level = "Low"

    return UrgencyResult(level=level, score=total, matched=matched)


def explain(result: UrgencyResult) -> str:
    """Human-readable one-liner explaining the score."""
    if not result["matched"]:
        return f"Urgency={result['level']} (score {result['score']}): no keywords matched."
    parts = ", ".join(f"{kw}(+{w})" for kw, w in result["matched"])
    return f"Urgency={result['level']} (score {result['score']}): {parts}"
