"""
Prompts used for the LLM extraction step (Task 2).

These are kept in their own module so they are easy to review, version, and
reuse. In production these exact strings would be sent to a real LLM
(Anthropic / OpenAI). The mock LLM in llm_analysis.py receives the same rendered
prompt, so the contract (prompt in -> structured JSON out) is identical whether
we run the mock or a real API.
"""

# System prompt: sets the role and, crucially, pins the output to strict JSON.
# Forcing a JSON schema makes the response machine-parseable and keeps the mock
# and the real API interchangeable.
SYSTEM_PROMPT = """You are a clinical/customer-interaction analyst.
You are given a transcribed conversation between an agent and a user.
Analyse it and respond with ONLY a valid JSON object - no prose, no markdown -
using exactly this schema:

{
  "summary": "<2-3 sentence neutral summary of the conversation>",
  "key_issues": ["<short phrase per concern>", "..."],
  "sentiment": "positive" | "negative" | "neutral",
  "risk_flag": {
    "value": "Yes" | "No",
    "reason": "<one sentence explaining the risk decision>"
  }
}

Guidelines:
- sentiment reflects the USER's emotional state, not the agent's.
- risk_flag is "Yes" if there is any health, safety, emotional, or churn risk
  that a human should review; otherwise "No".
- Keep key_issues concise and grounded in what the user actually said."""


def build_user_prompt(transcript: str) -> str:
    """Render the per-conversation user prompt around the cleaned transcript."""
    return (
        "Analyse the following conversation transcript and return the JSON "
        "described in the system instructions.\n\n"
        "TRANSCRIPT:\n"
        f"{transcript}\n"
    )
