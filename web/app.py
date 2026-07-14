"""
Web interface for the Voice Insights System.

A small Flask app that reuses the existing pipeline in src/. The browser
front-end is a chat: the user talks to a (scripted) agent, then clicks Analyze.

    GET  /            -> the chat demo page
    POST /reply       -> return the scripted agent reply for the next turn
    POST /analyze     -> run the pipeline on the collected turns, return JSON

Run from the repo root:
    python web/app.py
then open http://localhost:5000
"""

from __future__ import annotations

import os
import sys

from flask import Flask, jsonify, render_template, request

# Make the repo root importable so `src.*` resolves when run from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.llm_analysis import analyze
from src.preprocess import preprocess, to_transcript, user_text
from src.urgency_scoring import score_urgency

app = Flask(__name__)

# --- Scripted agent replies (Task-2 mock, chat mode).
# The agent is a *mock*: without a real LLM it can't truly reason about what the
# user said. To avoid the obviously-wrong behaviour of asking "since when?" when
# someone reports a life-threatening emergency, we add a keyword-triggered
# emergency branch on top of the fixed intake script. This is still a lookup
# table, not understanding - a real LLM agent is listed under Improvements.
_AGENT_OPENING = "Hello, how can I help you today?"
_AGENT_FOLLOWUPS = [
    "Since when have you been experiencing this?",
    "Has anything changed recently that might be related?",
    "On a scale of 1 to 10, how would you rate what you're feeling?",
    "Thank you for sharing. Is there anything else you'd like to add?",
]
_AGENT_CLOSING = "Understood. I've noted everything - you can end the chat and analyse it now."

# Crisis keywords that should override the intake script with an emergency reply.
# Kept in sync (conceptually) with the risk/urgency banks in src/.
_EMERGENCY_KEYWORDS = [
    "not breathing", "stopped breathing", "can't breathe", "cant breathe",
    "trouble breathing", "difficulty breathing", "struggling to breathe",
    "choking", "fainted", "passed out", "unconscious", "unresponsive",
    "not responding", "collapsed", "seizure", "stroke", "heart attack",
    "chest pain", "bleeding", "overdose", "suicide", "won't wake up",
    "not waking up",
]
_AGENT_EMERGENCY = (
    "This sounds like a medical emergency. If you're able, please call your local "
    "emergency number right now. Is the person breathing at all, and are they "
    "responsive? I'll stay with you - tell me what's happening."
)


def _agent_reply(user_turn_index: int, user_text: str = "") -> str:
    """Return the agent's reply to the user's latest message.

    Priority:
      1. If the user's message contains an emergency keyword, respond with the
         emergency script regardless of where we are in the intake sequence.
      2. Otherwise step through the fixed intake follow-ups.
      3. Fall back to a closing line so the conversation always ends gracefully.
    """
    lower = user_text.lower()
    if any(keyword in lower for keyword in _EMERGENCY_KEYWORDS):
        return _AGENT_EMERGENCY

    if user_turn_index < len(_AGENT_FOLLOWUPS):
        return _AGENT_FOLLOWUPS[user_turn_index]
    return _AGENT_CLOSING


@app.get("/")
def index():
    """Serve the chat demo page."""
    return render_template("index.html", opening=_AGENT_OPENING)


@app.post("/reply")
def reply_endpoint():
    """Return the scripted agent reply for the user's latest message.

    The browser sends how many user messages have been said so far; we reply with
    the matching step in the scripted intake sequence. No analysis happens here -
    this only drives the conversation.
    """
    payload = request.get_json(silent=True) or {}
    # 0-based index of the user turn we are replying to.
    user_turn_index = int(payload.get("user_turn_index", 0))
    # The user's latest message, so the agent can detect emergencies.
    user_text = str(payload.get("text", ""))
    return jsonify({"reply": _agent_reply(user_turn_index, user_text)})


@app.post("/analyze")
def analyze_endpoint():
    """Run the full pipeline on a posted conversation and return JSON.

    Expects a JSON body: {"conversation": [{"speaker": ..., "text": ...}, ...]}.
    Returns the sample_output schema plus an "urgency_detail" block (score +
    matched keywords) so the UI can explain *why* the urgency was assigned.
    """
    # --- Parse and validate the incoming body defensively.
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    if not isinstance(payload, dict) or "conversation" not in payload:
        return jsonify({"error": "JSON must contain a 'conversation' array."}), 400

    # --- Task 1: preprocess.
    try:
        turns = preprocess(payload["conversation"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not turns:
        return jsonify({"error": "No usable conversation turns after cleaning."}), 400

    transcript = to_transcript(turns)
    users_only = user_text(turns)

    # --- Task 2: LLM extraction (mock).
    llm = analyze(transcript, use_mock=True)

    # --- Task 3: urgency scoring (with the explainable breakdown).
    urgency = score_urgency(users_only)

    # --- Assemble the response: sample schema + extras the UI can visualise.
    result = {
        "summary": llm["summary"],
        "key_issues": llm["key_issues"],
        "sentiment": llm["sentiment"],
        "risk_flag": {
            "value": llm["risk_flag"]["value"],
            "reason": llm["risk_flag"]["reason"],
        },
        "urgency": urgency["level"],
        # Extras (not in sample_output.json) purely for the demo UI:
        "urgency_detail": {
            "score": urgency["score"],
            "matched": [{"keyword": kw, "weight": w} for kw, w in urgency["matched"]],
        },
        "cleaned_transcript": transcript,
    }
    return jsonify(result)


if __name__ == "__main__":
    # debug=True gives auto-reload and readable errors during the demo.
    app.run(host="127.0.0.1", port=5000, debug=True)
