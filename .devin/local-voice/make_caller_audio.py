#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

r"""Synthesize a caller WAV for Chrome's fake microphone.

The file is leading silence followed by one spoken line, so the bot starts
talking first and the caller barges in after ``--delay`` seconds. Chrome loops
the file, so ``--tail`` adds trailing silence before the line repeats.

Run::

    uv run python .devin/local-voice/make_caller_audio.py \\
        --text "Wait, sorry, which pill is the blood pressure one?" \\
        --delay 6 --out /tmp/caller.wav
"""

import argparse
import wave

import numpy as np
from kokoro_onnx import Kokoro

from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR, _ensure_model_files


def main():
    """Write the caller WAV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--delay", type=float, default=6.0, help="Seconds before speaking.")
    parser.add_argument("--tail", type=float, default=30.0, help="Seconds of silence after.")
    parser.add_argument("--voice", default="am_michael")
    parser.add_argument("--out", default="/tmp/caller.wav")
    args = parser.parse_args()

    model_path = KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"
    voices_path = KOKORO_CACHE_DIR / "voices-v1.0.bin"
    _ensure_model_files(model_path, voices_path)
    samples, sample_rate = Kokoro(str(model_path), str(voices_path)).create(
        args.text, voice=args.voice, speed=1.0, lang="en-us"
    )

    lead = np.zeros(int(args.delay * sample_rate), dtype=np.float32)
    tail = np.zeros(int(args.tail * sample_rate), dtype=np.float32)
    audio = np.concatenate([lead, samples.astype(np.float32), tail])
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    with wave.open(args.out, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())
    print(f"wrote {args.out} ({len(pcm) / sample_rate:.1f}s, {sample_rate} Hz)")


if __name__ == "__main__":
    main()
