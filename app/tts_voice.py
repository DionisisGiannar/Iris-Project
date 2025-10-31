"""Piper-based voice for Iris that streams spoken responses."""

from __future__ import annotations

import io
import logging
import shlex
import subprocess
from typing import Iterable, Optional, Tuple, Union
import wave, os
import numpy as np
from piper.voice import AudioChunk, PiperVoice  # type: ignore

logger = logging.getLogger(__name__)


class IrisVoice:
    """Wrapper around Piper that can synthesize and optionally play audio."""

    def __init__(self, model_path: str, playback_command: Optional[str] = None) -> None:
        self.voice = PiperVoice.load(model_path)
        cfg = getattr(self.voice, "config", None)
        sample_rate = None
        sample_width = 2
        channels = 1

        if cfg is not None:
            sample_rate = getattr(cfg, "sample_rate", None)
            audio_cfg = getattr(cfg, "audio", None)
            if not sample_rate and isinstance(audio_cfg, dict):
                sample_rate = audio_cfg.get("sample_rate") or audio_cfg.get("sample_rate_hz")
            if isinstance(audio_cfg, dict):
                sample_width = audio_cfg.get("sample_width", sample_width)
                channels = audio_cfg.get("sample_channels", channels)

        if not sample_rate:
            logger.warning("Unable to read sample rate from Piper config; defaulting to 22050 Hz.")
            sample_rate = 22050

        self.default_sample_rate = int(sample_rate)
        self.default_sample_width = int(sample_width)
        self.default_channels = int(channels)
        self.playback_command = playback_command

        if playback_command:
            logger.info("Playbacks enabled with command template: %s", playback_command)
        else:
            logger.info("No playback command configured; Iris voice will synthesize only.")

    # ------------------------------------------------------------------ #
    def speak(self, text: str) -> Tuple[bytes, int]:
        """Synthesize the provided text and optionally play it."""
        if not text:
            raise ValueError("Cannot synthesize empty text.")

        pcm_bytes, rate, width, channels = self._synthesize(text)
        if not pcm_bytes:
            raise RuntimeError("Piper returned empty audio.")

        self._play(pcm_bytes, rate, width, channels)

        
        os.makedirs("/shared_audio", exist_ok=True)
        path = "/shared_audio/last_output.wav"
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm_bytes)
        print(f"[AUDIO] Saved TTS to {path}")
        
        return pcm_bytes, rate

    def say(self, text: str) -> None:
        """Convenience wrapper matching orchestrator expectations."""
        pcm_bytes, rate = self.speak(text)
        logger.debug("Synthesized %d bytes at %d Hz for text='%s'", len(pcm_bytes), rate, text)

    # ------------------------------------------------------------------ #
    def _synthesize(self, text: str) -> Tuple[bytes, int, int, int]:
        stream = self.voice.synthesize(text)
        return self._collect_pcm(stream)

    def _collect_pcm(
        self, stream: Iterable[Union[bytes, bytearray, AudioChunk]]
    ) -> Tuple[bytes, int, int, int]:
        buf = io.BytesIO()
        detected_rate: Optional[int] = None
        detected_width: Optional[int] = None
        detected_channels: Optional[int] = None

        for chunk in stream:
            if isinstance(chunk, (bytes, bytearray)):
                buf.write(chunk)
                continue

            if isinstance(chunk, AudioChunk):
                detected_rate = detected_rate or getattr(chunk, "sample_rate", None)
                detected_width = detected_width or getattr(chunk, "sample_width", None)
                detected_channels = detected_channels or getattr(chunk, "sample_channels", None)

                for attr in ("audio_int16_bytes", "audio_bytes", "audio"):
                    data = getattr(chunk, attr, None)
                    if data is not None:
                        buf.write(bytes(data))
                        break
                else:
                    samples = getattr(chunk, "samples", None)
                    if samples is None:
                        logger.warning("AudioChunk missing audio data; skipping.")
                        continue
                    if not isinstance(samples, np.ndarray):
                        samples = np.asarray(samples)
                    if samples.dtype.kind == "f":
                        samples = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
                    elif samples.dtype != np.int16:
                        samples = samples.astype(np.int16)
                    buf.write(samples.tobytes(order="C"))
                continue

            logger.warning("Unexpected Piper audio chunk type: %r", type(chunk))

        return (
            buf.getvalue(),
            detected_rate or self.default_sample_rate,
            detected_width or self.default_sample_width,
            detected_channels or self.default_channels,
        )

    def _play(self, pcm_bytes: bytes, sample_rate: int, sample_width: int, channels: int) -> None:
        if not self.playback_command:
            return

        command_str = self.playback_command.format(rate=sample_rate, width=sample_width, channels=channels)
        try:
            process = subprocess.Popen(
                command_str if isinstance(command_str, str) else " ".join(command_str),
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            logger.warning("Playback command not found: %s", command_str)
            return
        except Exception:
            logger.exception("Failed to launch playback command: %s", command_str)
            return

        try:
            assert process.stdin is not None
            process.stdin.write(pcm_bytes)
            process.stdin.close()
        except Exception:
            logger.exception("Failed to stream audio to playback command.")
