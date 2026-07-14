# Voice Insights System

Analyses a transcribed agent/user conversation and produces structured insights:
a summary, key issues, sentiment, a risk flag, and an urgency level. Optionally
extracts prosody (pitch/energy) features when raw audio is supplied.

## Project structure

```text
voice-insights-system/
├── main.py                    # entry point — run from the repo root
├── requirements.txt
│
├── src/                       # the pipeline (Tasks 1–4)
│   ├── preprocess.py          # Task 1 — clean & structure the conversation
│   ├── prompts.py             # Task 2 — system + user prompts
│   ├── llm_analysis.py        # Task 2 — mock LLM (SDK-shaped)
│   ├── urgency_scoring.py     # Task 3 — rule-based urgency
│   └── audio_features.py      # Task 4 (bonus) — pitch/energy extraction
│
├── evaluation/                # testing + continuous improvement
│   ├── golden_set.json        # labelled evaluation set
│   ├── evaluate.py            # shared eval core
│   ├── test_pipeline.py       # correctness gate (urgency & risk)
│   └── monitor.py             # degradation / drift monitor
│
└── data/                      # sample I/O
    ├── sample_input.json
    ├── sample_output.json
    └── output.json            # generated example
```

## Running

All commands are run from the repo root.

```bash
pip install -r requirements.txt          # only needed for the audio bonus / real LLM

python main.py                           # data/sample_input.json -> data/output.json (mock LLM)
python main.py --input data/my.json --output data/my_out.json
python main.py --audio call.wav          # also attaches an "audio_insights" block
python main.py --real-llm                # use the real Anthropic API (needs ANTHROPIC_API_KEY)

python evaluation/test_pipeline.py       # correctness: predicted urgency & risk vs labels
python evaluation/monitor.py             # log metrics & alert on degradation
```

## Approach

1. **Preprocess** (`src/preprocess.py`). Normalise each turn: NFKC unicode, fold
   smart quotes/dashes/ellipsis to ASCII, collapse repeated punctuation and
   whitespace, drop empty/whitespace-only turns, and defensively skip malformed
   entries (non-dict items, missing keys, wrong types). Unknown speakers are kept
   but labelled `unknown` so no information is silently lost.

2. **LLM extraction — mock** (`src/llm_analysis.py`, `src/prompts.py`). The mock
   mirrors a real chat-completion call: it takes the same system/user prompt and
   returns a **JSON string**, exactly like the Anthropic/OpenAI SDK would. The
   caller builds the prompt, gets the string, and parses it — so switching to the
   real API is a one-line change (`--real-llm`). This exercises the whole pipeline
   with no key or cost while keeping production-realistic structure. The mock uses
   keyword heuristics to approximate the model's judgement, and sentiment is
   risk-aware (a detected risk can't read as "positive").

3. **Urgency scoring** (`src/urgency_scoring.py`). A transparent, tiered
   weighted-keyword scheme. Every point is traceable to a matched keyword, and
   longer phrases (e.g. "trouble sleeping") claim their span before generic terms
   ("trouble") to avoid double-counting. Chosen over an opaque classifier because
   in a health/safety context, an auditable "here's *why* it's High" matters more
   than marginal accuracy.

4. **Audio awareness — bonus** (`src/audio_features.py`). `extract_audio_features()`
   uses librosa to compute pitch (F0 via `pyin`), energy (RMS), and a speaking-rate
   proxy. F0 + RMS were chosen over MFCCs/spectrograms because they map directly to
   human distress cues (rising/unstable pitch, loud/erratic energy) and are
   interpretable. See "Audio features & risk" below.

## Output format

Matches `data/sample_output.json` exactly:

```json
{
  "summary": "...",
  "key_issues": ["..."],
  "sentiment": "positive | negative | neutral",
  "risk_flag": { "value": "Yes | No", "reason": "..." },
  "urgency": "Low | Medium | High"
}
```

When `--audio` is given, an extra `audio_insights` object is appended. Without
audio (the text-only sample), the output matches the sample schema exactly.

## Audio features & risk (bonus writeup)

Text loses *how* something is said — two users can say identical words while one
is calm and one is in crisis. Prosody recovers that:

- **Pitch (F0) mean & variability** — fear/panic raise pitch and destabilise it;
  a high `pitch_std` flags a trembling, agitated voice even when the transcript
  looks neutral.
- **Energy (RMS) mean & variability** — shouting, crying, breathlessness show as
  high/erratic energy; sudden spikes mark escalation points.
- **Speaking-rate proxy** — pressured rapid speech is an anxiety cue; slow,
  low-energy speech can indicate depression/fatigue.

Integration: fold these into the urgency score as extra weighted signals
(e.g. `pitch_std` above a per-speaker baseline adds points) and pass a short
prosody summary into the LLM prompt so `risk_flag` can cite tone, not just words.
This catches cases that are calm on paper but alarming in voice.

## Testing & continuous improvement

- **Correctness** — `python evaluation/test_pipeline.py` runs every case in
  `golden_set.json` and checks that predicted `urgency` and `risk_flag` match the
  expected labels, printing a per-case table and overall accuracy. It exits
  non-zero if any case fails, so it works as a CI gate.

- **Detecting degradation over time** — models don't fail all at once; quality
  erodes (stale keywords, a provider silently swapping the model, real traffic
  drifting from the golden set). `python evaluation/monitor.py`:
  1. Evaluates the current pipeline against the golden set.
  2. Appends timestamped metrics to `evaluation/metrics_history.jsonl`.
  3. Compares against a rolling baseline (last N runs) and alerts if accuracy
     drops beyond tolerance (accuracy drift), any case's prediction changes vs the
     last run (behavioural drift), or — extensibly — input characteristics shift
     (data drift).
  4. Exits non-zero on alert, so a scheduled run (nightly cron/CI) can page a human.

- **The loop** — mis-scored production conversations get labelled and added to
  `golden_set.json` → the monitor's baseline tightens → regressions are caught
  earlier → keyword weights / prompts are tuned → re-evaluated. The golden set is
  the asset that compounds. When moving to a real LLM (`--real-llm`), the same
  harness measures whether the swap actually improved accuracy before shipping.

## Assumptions

- Input follows `{"conversation": [{"speaker", "text"}, ...]}`.
- Sentiment, urgency, and risk are driven by the **user's** utterances, not the
  agent's scripted lines.
- The sample is text-only, so `audio_insights` is omitted unless audio is passed.
- "Risk" means any health/safety/emotional/churn concern a human should review.
- Deterministic mock output is acceptable for this assessment in place of a live
  model call.

## Improvements (given more time)

- **Replace mock with a real LLM** — the mock does keyword counting, so nuance
  (sarcasm, mixed tone, implicit risk) is approximate. Sentiment is already made
  risk-aware to avoid obvious errors, but a real model reads context natively; the
  `--real-llm` path is already wired.
- **Per-speaker audio baselines** — pitch/energy are only meaningful relative to a
  speaker's own norm; calibrate against a neutral segment.
- **Negation & intensity handling** in scoring ("no pain", "slightly anxious").
- **STT word-level confidence** to down-weight low-confidence transcribed tokens.
- **Unit tests** for preprocessing edge cases and scoring thresholds.
- **Multi-turn temporal signal** — escalation across turns, not just bag-of-words.

## AI Usage Declaration

This solution was developed with the assistance of an AI coding assistant
(Claude / Claude Code). AI was used to scaffold the module structure, draft the
code and inline comments, and draft this README. All architectural decisions
(mock-LLM-mirrors-real-SDK, transparent keyword scoring, risk-aware sentiment,
output schema, audio feature choice) were reviewed and approved by me, and the
pipeline was run and validated against the provided sample input/output. I
understand and can explain every component.
