"""Automatic speech recognition (ASR) interface for Iris Assistant."""

from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel


class SpeechRecognizer:
    """Wrapper around faster-whisper for English-only transcription."""

    def __init__(self, model_dir: str, compute_type: str = "int8", beam_size: int = 1) -> None:
        self.model = WhisperModel(model_dir, device="cpu", compute_type=compute_type)
        self.beam_size = beam_size

    def transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
        """Transcribe PCM audio bytes into text."""
        if not audio_bytes:
            return ""
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio_array,
            language=language,
            beam_size=self.beam_size,
            word_timestamps=False,
        )
        text_parts = [segment.text.strip() for segment in segments if segment.text]
        return " ".join(text_parts).strip()
