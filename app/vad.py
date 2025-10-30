"""Voice activity detection helpers for the Iris Assistant."""

from __future__ import annotations

import collections
from typing import Deque, Iterable, Iterator, List, Tuple

import webrtcvad


class VoiceActivityDetector:
    """WebRTC VAD based speech segmenter for 16 kHz mono PCM audio."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        frame_duration_ms: int = 20,
        padding_ms: int = 300,
        aggressiveness: int = 2,
    ) -> None:
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("WebRTC VAD only supports 8/16/32/48 kHz sample rates")
        if frame_duration_ms not in (10, 20, 30):
            raise ValueError("WebRTC VAD supports frame sizes of 10, 20, or 30 ms")
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # 16-bit mono
        self.padding_frames = int(padding_ms / frame_duration_ms)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_is_complete(self, frame: bytes) -> bool:
        """Return True if the frame has the expected byte length."""
        return len(frame) == self.frame_size

    def _new_window(self) -> Deque[Tuple[bytes, bool]]:
        return collections.deque(maxlen=self.padding_frames)

    def segments(self, frames: Iterable[bytes]) -> Iterator[bytes]:
        """Yield PCM segments containing detected speech."""
        window: Deque[Tuple[bytes, bool]] = self._new_window()
        voiced_frames: List[bytes] = []
        triggered = False

        def voiced_fraction(buffer: Deque[Tuple[bytes, bool]]) -> float:
            if not buffer:
                return 0.0
            return sum(1 for _, speech in buffer if speech) / len(buffer)

        for frame in frames:
            if not self.frame_is_complete(frame):
                continue
            is_voiced = self.vad.is_speech(frame, self.sample_rate)
            if not triggered:
                window.append((frame, is_voiced))
                if voiced_fraction(window) > 0.6:
                    triggered = True
                    voiced_frames.extend(item for item, _ in window)
                    window.clear()
            else:
                voiced_frames.append(frame)
                window.append((frame, is_voiced))
                if voiced_fraction(window) < 0.3:
                    if voiced_frames:
                        yield b"".join(voiced_frames)
                        voiced_frames.clear()
                    window.clear()
                    triggered = False

        if voiced_frames:
            yield b"".join(voiced_frames)

