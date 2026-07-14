"""
Task 2 - LLM extraction (mock, but shaped like a real API call).

We extract: summary, key_issues, sentiment, and risk_flag.

Design decision: the mock LLM mirrors a real SDK call. The public
function `analyze()` builds the prompt, sends it to either the mock or a real
provider, then parses the returned JSON string. Because both paths take a prompt
and return a JSON string, swapping the mock for Anthropic/OpenAI is a one-line
change (flip `use_mock`). This lets us test the whole pipeline with no API key or
cost, while keeping production-realistic structure.
"""

from __future__ import annotations

import json
import re
from typing import TypedDict

from src.prompts import SYSTEM_PROMPT, build_user_prompt


class RiskFlag(TypedDict):
    value: str  # "Yes" | "No"
    reason: str


class LLMResult(TypedDict):
    summary: str
    key_issues: list[str]
    sentiment: str  # "positive" | "negative" | "neutral"
    risk_flag: RiskFlag


# --- Heuristic keyword banks used only by the MOCK to fake an LLM's judgement.
# A real LLM would infer these; the mock approximates them so the demo produces
# a sensible, deterministic result.

# Terms that suggest emotional/health/safety risk a human should review.
_RISK_TERMS = [
    "suicide", "self-harm", "hopeless", "can't go on", "harm myself",
    "chest pain", "bleeding", "overdose", "severe", "emergency",
    "anxious", "anxiety", "depressed", "depression", "panic",
    "trouble sleeping", "insomnia", "pain",
]

# Negative-sentiment cues from the user.
_NEGATIVE_TERMS = [
    "anxious", "anxiety", "worried", "trouble", "pain", "sad", "depressed",
    "upset", "angry", "frustrated", "scared", "afraid", "can't", "struggling",
    "difficult", "hopeless", "stressed", "overwhelmed",
]

# Positive-sentiment cues.
_POSITIVE_TERMS = [
    "thanks", "thank you", "great", "better", "good", "happy", "resolved",
    "appreciate", "helpful", "relieved",
]


def _mock_llm(system_prompt: str, user_prompt: str, transcript: str) -> str:
    """Stand-in for a real LLM completion.

    Signature intentionally mirrors a chat completion (system + user message).
    Returns a JSON *string* - exactly what a real API would give us in the
    message content - so the caller's parsing path is identical for mock/real.
    """
    lower = transcript.lower()

    def _hits(terms: list[str]) -> list[str]:
        # Return which terms appear, so decisions are explainable.
        return [t for t in terms if t in lower]

    risk_hits = _hits(_RISK_TERMS)
    neg_hits = _hits(_NEGATIVE_TERMS)
    pos_hits = _hits(_POSITIVE_TERMS)

    # --- Sentiment: risk-aware, not a naive keyword count.
    #
    # Two problems a plain count has (both seen in real runs):
    #   1. Risk words like "bleeding" describe a serious state but weren't in the
    #      negative bank, so they didn't push sentiment negative at all.
    #   2. A single polite "thanks" could outweigh a crisis and flip sentiment to
    #      positive ("I'm bleeding profusely, thanks for helping" -> positive).
    #
    # Fixes:
    #   - Risk terms count as strong negative evidence (weight 2 each).
    #   - If the risk flag fired, sentiment is capped at neutral - a user in a
    #     risky state is never "positive", however politely they phrase it.
    negative_weight = len(neg_hits) + 2 * len(risk_hits)
    positive_weight = len(pos_hits)

    if negative_weight > positive_weight:
        sentiment = "negative"
    elif positive_weight > negative_weight:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    # Safety cap: any detected risk overrides an otherwise positive read.
    if risk_hits and sentiment == "positive":
        sentiment = "neutral"

    # --- Key issues: surface the distinct concern terms we detected.
    #     Prefer longer, more specific phrases and drop any shorter term that is
    #     a substring of one we already kept (so "trouble sleeping" suppresses a
    #     bare "trouble"). De-duplicate while keeping a stable order.
    candidates = sorted(set(neg_hits + risk_hits), key=len, reverse=True)
    key_issues: list[str] = []
    for term in candidates:
        if any(term in kept for kept in key_issues):
            continue  # already covered by a more specific phrase
        key_issues.append(term)
    if not key_issues:
        key_issues = ["No specific concerns detected"]

    # --- Risk flag: Yes if any risk term fired, with a grounded reason.
    if risk_hits:
        risk = RiskFlag(
            value="Yes",
            reason=(
                "User expressed "
                + ", ".join(risk_hits)
                + " which may require clinical or human follow-up."
            ),
        )
    else:
        risk = RiskFlag(
            value="No",
            reason="No health, safety, or emotional risk indicators detected.",
        )

    # --- Summary: a short, templated neutral summary. A real LLM would write
    #     free-form; the mock keeps it grounded in detected signals.
    if key_issues and key_issues[0] != "No specific concerns detected":
        summary = (
            "The user contacted the agent and reported "
            f"{', '.join(key_issues[:3])}. "
            "The agent gathered further detail about duration and context."
        )
    else:
        summary = (
            "The user and agent had a general interaction with no major "
            "concerns raised."
        )

    result = LLMResult(
        summary=summary,
        key_issues=key_issues,
        sentiment=sentiment,
        risk_flag=risk,
    )
    # Return a JSON string, mimicking the raw text content of an API response.
    return json.dumps(result)


def _real_llm(system_prompt: str, user_prompt: str) -> str:
    """Real API path - shown for completeness, not called when use_mock=True.

    This is the ONE place that changes to go to production. The anthropic SDK is
    already installed in this environment.
    """
    import os
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    # The model's text content - a JSON string, per our system prompt.
    return message.content[0].text


def _extract_json(raw: str) -> dict:
    """Parse the JSON string returned by the (mock or real) LLM.

    Real models sometimes wrap JSON in prose or ```json fences, so we defensively
    locate the outermost JSON object before parsing instead of trusting the whole
    string.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fall back to grabbing the first {...} block.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"LLM did not return valid JSON:\n{raw}")
        return json.loads(match.group(0))


def analyze(transcript: str, use_mock: bool = True) -> LLMResult:
    """Public entry point for Task 2.

    Builds the prompt, calls the LLM (mock by default), parses and returns the
    structured result. Set use_mock=False to hit the real Anthropic API.
    """
    user_prompt = build_user_prompt(transcript)

    if use_mock:
        raw = _mock_llm(SYSTEM_PROMPT, user_prompt, transcript)
    else:
        raw = _real_llm(SYSTEM_PROMPT, user_prompt)

    parsed = _extract_json(raw)
    return LLMResult(
        summary=parsed["summary"],
        key_issues=parsed["key_issues"],
        sentiment=parsed["sentiment"],
        risk_flag=RiskFlag(**parsed["risk_flag"]),
    )
