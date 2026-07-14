"""
Voice Insights System - orchestrator.

Wires the four tasks together:
  1. preprocess    - clean & structure the conversation
  2. analyze       - mock LLM -> summary, key_issues, sentiment, risk_flag
  3. score_urgency - rule-based Low/Medium/High
  4. audio (bonus) - prosody features, only when an audio file is supplied

Run:
    python main.py                       # uses sample_input.json, mock LLM
    python main.py --input other.json    # custom input
    python main.py --audio call.wav      # also attach audio_insights
"""

from __future__ import annotations

import argparse
import json
import sys

from src.audio_features import extract_audio_features
from src.llm_analysis import analyze
from src.preprocess import preprocess, to_transcript, user_text
from src.urgency_scoring import explain, score_urgency


def run(input_path: str, output_path: str, audio_path: str | None, use_mock: bool) -> dict:
    """Execute the full pipeline and write the output JSON."""
    # --- Load raw input. Fail loudly if it's missing or malformed.
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Input file not found: {input_path}")
    except json.JSONDecodeError as e:
        sys.exit(f"Input file is not valid JSON: {e}")

    # --- Task 1: preprocess.
    turns = preprocess(raw.get("conversation", []))
    if not turns:
        sys.exit("No usable conversation turns after preprocessing.")
    transcript = to_transcript(turns)
    users_only = user_text(turns)

    print("=== Cleaned transcript ===")
    print(transcript)
    print()

    # --- Task 2: LLM extraction (mock by default).
    llm = analyze(transcript, use_mock=use_mock)

    # --- Task 3: urgency scoring (driven by the user's words).
    urgency = score_urgency(users_only)
    print("=== Urgency reasoning ===")
    print(explain(urgency))  # narratable explanation of the score
    print()

    # --- Assemble output in the exact sample_output.json shape.
    output: dict = {
        "summary": llm["summary"],
        "key_issues": llm["key_issues"],
        "sentiment": llm["sentiment"],
        "risk_flag": {
            "value": llm["risk_flag"]["value"],
            "reason": llm["risk_flag"]["reason"],
        },
        "urgency": urgency["level"],
    }

    # --- Task 4 (bonus): attach audio_insights ONLY when audio is provided,
    #     so the text-only run still matches the sample schema exactly.
    if audio_path:
        try:
            features = extract_audio_features(audio_path)
            output["audio_insights"] = features
            print("=== Audio insights ===")
            print(json.dumps(features, indent=2))
            print()
        except FileNotFoundError as e:
            # Don't crash the whole run over a missing optional input.
            print(f"[warn] Skipping audio insights: {e}")

    # --- Write result.
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"=== Final output (written to {output_path}) ===")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Insights System")
    parser.add_argument("--input", default="data/sample_input.json", help="input JSON path")
    parser.add_argument("--output", default="data/output.json", help="output JSON path")
    parser.add_argument("--audio", default=None, help="optional audio file for prosody features")
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="use the real Anthropic API instead of the mock (needs ANTHROPIC_API_KEY)",
    )
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        audio_path=args.audio,
        use_mock=not args.real_llm,
    )


if __name__ == "__main__":
    main()
