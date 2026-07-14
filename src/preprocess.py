"""
Task 1 - Preprocessing.

Goal: take the raw speech-to-text output and turn it into a clean, structured
conversation we can safely feed to the rest of the pipeline.

STT output is messy: smart/curly quotes, stray whitespace, empty turns,
repeated punctuation, and occasionally malformed entries (missing keys, wrong
types). We normalise all of that here so downstream code can assume clean data.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TypedDict


# A single cleaned conversation turn. TypedDict gives us clear structure
# without the overhead of a full class.
class Turn(TypedDict):
    speaker: str
    text: str


# Characters that STT engines commonly emit that we want to fold back to plain
# ASCII equivalents so keyword matching and the LLM see consistent text.
_SMART_QUOTE_MAP = {
    "‘": "'",  # left single curly quote
    "’": "'",  # right single curly quote (the apostrophe in "I've")
    "“": '"',  # left double curly quote
    "”": '"',  # right double curly quote
    "–": "-",  # en dash
    "—": "-",  # em dash
    "…": "...",  # ellipsis character
}

# Speakers we recognise. Anything else is normalised but flagged as "unknown"
# so we never silently drop information.
_KNOWN_SPEAKERS = {"agent", "user"}


def _normalise_text(raw: str) -> str:
    """Clean a single text string coming out of STT.

    Steps, in order:
      1. Unicode-normalise (NFKC) so composed/decomposed forms match.
      2. Replace smart quotes / dashes / ellipsis with ASCII equivalents.
      3. Collapse runs of repeated symbols (e.g. "!!!" -> "!", "..." kept sane).
      4. Collapse repeated whitespace into single spaces.
      5. Strip leading/trailing whitespace.
    """
    # 1. Canonical unicode form.
    text = unicodedata.normalize("NFKC", raw)

    # 2. Fold smart punctuation to ASCII.
    for smart, plain in _SMART_QUOTE_MAP.items():
        text = text.replace(smart, plain)

    # 3. Collapse 3+ repeated punctuation marks down to a single one so
    #    "hello!!!!" and "wait..." don't skew tokenisation. We keep one mark.
    text = re.sub(r"([!?.,])\1{2,}", r"\1", text)

    # 4. Any whitespace run (tabs, newlines, multiple spaces) -> single space.
    text = re.sub(r"\s+", " ", text)

    # 5. Trim the edges.
    return text.strip()


def preprocess(raw_conversation: list[dict]) -> list[Turn]:
    """Clean and structure a raw conversation list.

    Each raw item is expected to look like {"speaker": ..., "text": ...}.
    We defensively handle edge cases rather than trusting the input:
      - non-dict entries are skipped
      - missing/None text becomes "" and is then dropped if empty
      - empty or whitespace-only turns are dropped
      - unknown speakers are normalised to "unknown" (kept, not dropped)
    """
    if not isinstance(raw_conversation, list):
        raise ValueError(
            f"Expected 'conversation' to be a list, got {type(raw_conversation).__name__}"
        )

    cleaned: list[Turn] = []

    for entry in raw_conversation:
        # Edge case: entry isn't even a dict -> skip it, we can't use it.
        if not isinstance(entry, dict):
            continue

        # Pull fields defensively; STT/JSON may omit them.
        raw_speaker = entry.get("speaker")
        raw_text = entry.get("text")

        # Normalise the speaker label: lowercase, trimmed. Unknown -> "unknown".
        speaker = str(raw_speaker).strip().lower() if raw_speaker is not None else "unknown"
        if speaker not in _KNOWN_SPEAKERS:
            speaker = "unknown"

        # Edge case: text missing or not a string -> treat as empty.
        text = _normalise_text(raw_text) if isinstance(raw_text, str) else ""

        # Drop turns that are empty after cleaning - they carry no signal.
        if not text:
            continue

        cleaned.append(Turn(speaker=speaker, text=text))

    return cleaned


def to_transcript(turns: list[Turn]) -> str:
    """Flatten cleaned turns into a single readable transcript string.

    Format: "agent: ...\nuser: ..." - this is what we hand to the LLM prompt
    and to the urgency scorer.
    """
    return "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in turns)


def user_text(turns: list[Turn]) -> str:
    """Return only the user's utterances joined together.

    Urgency and sentiment should be driven by what the *user* says, not the
    agent's scripted prompts, so we expose this helper separately.
    """
    return " ".join(turn["text"] for turn in turns if turn["speaker"] == "user")
