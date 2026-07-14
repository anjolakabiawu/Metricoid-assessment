"""
Task 4 (bonus) - Audio awareness.

The provided input is text-only (STT output), so there is no audio to process in
the sample run. This module provides a real, working feature extractor for when
raw audio *is* available, plus a written explanation (see EXPLANATION at the
bottom / README) of how these features sharpen risk detection.

Design decision: we extract pitch (fundamental frequency, F0) and
energy (RMS) with librosa. We chose F0 + RMS over, say, a full spectrogram or
MFCCs because they map directly to human distress cues (rising, unstable pitch
and loud/erratic energy), they're cheap to compute, and they're interpretable -
we can point at a number and say "the voice got higher and louder here".
"""

from __future__ import annotations

import os
from typing import TypedDict

import numpy as np


class AudioFeatures(TypedDict):
    pitch_mean_hz: float      # average fundamental frequency
    pitch_std_hz: float       # pitch variability - instability/agitation cue
    energy_mean: float        # average loudness (RMS)
    energy_std: float         # loudness variability - shouting/crying cue
    speaking_rate_proxy: float  # voiced-frame ratio, a rough tempo proxy


def extract_audio_features(audio_path: str) -> AudioFeatures:
    """Extract pitch and energy features from an audio file.

    Uses librosa:
      - pyin() for a robust probabilistic F0 (pitch) track.
      - rms() for short-time energy.
    Raises FileNotFoundError if the path is missing so we never silently return
    fake numbers (per the "handle errors explicitly" convention).
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Imported lazily so the text-only pipeline doesn't pay librosa's import cost.
    import librosa

    # Load mono audio at its native sample rate.
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # --- Pitch (F0) via probabilistic YIN. We bound the search to a typical
    #     human speech range (~65-400 Hz) to reject spurious estimates.
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=65.0,
        fmax=400.0,
        sr=sr,
    )
    # pyin returns NaN on unvoiced frames; keep only voiced ones for stats.
    voiced_f0 = f0[~np.isnan(f0)]
    pitch_mean = float(np.mean(voiced_f0)) if voiced_f0.size else 0.0
    pitch_std = float(np.std(voiced_f0)) if voiced_f0.size else 0.0

    # --- Energy via short-time RMS.
    rms = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))

    # --- Speaking-rate proxy: fraction of frames that are voiced. A quick,
    #     dependency-free stand-in for words-per-minute.
    speaking_rate = float(np.mean(voiced_flag)) if voiced_flag.size else 0.0

    return AudioFeatures(
        pitch_mean_hz=pitch_mean,
        pitch_std_hz=pitch_std,
        energy_mean=energy_mean,
        energy_std=energy_std,
        speaking_rate_proxy=speaking_rate,
    )


# How these features improve risk detection - kept as a string so it can be
# printed or pulled into the README.
EXPLANATION = """\
How audio features improve risk detection:

Text alone loses *how* something was said. Two users can type the same words
while one is calm and one is in crisis. Prosodic features recover that signal:

- Pitch mean & variability (F0): distress, fear, and panic typically raise pitch
  and make it less stable. A high pitch_std flags an agitated or trembling voice
  even when the transcript looks neutral.
- Energy (RMS) mean & variability: shouting, crying, or breathlessness show up as
  high or erratic energy. Sudden energy spikes can mark an escalation point.
- Speaking-rate proxy: pressured, rapid speech is a known anxiety/mania cue;
  very slow, low-energy speech can indicate depression or fatigue.

Integration: we would fold these into the urgency score as additional weighted
signals (e.g. pitch_std above a per-speaker baseline adds points), and pass a
short prosody summary to the LLM so its risk_flag reasoning can cite tone, not
just words. This catches high-risk cases that are calm on paper but alarming in
voice - exactly the cases text-only systems miss.
"""
