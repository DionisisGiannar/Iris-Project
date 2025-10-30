"""Audio ingestion utilities for the Iris Assistant."""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from typing import Generator, Optional

logger = logging.getLogger(__name__)


class AudioIngest:
    """Stream 16 kHz mono audio from an RTSP source using ffmpeg."""

    def __init__(
        self,
        rtsp_url: str,
        sample_rate: int = 16_000,
        frame_duration_ms: int = 20,
        queue_size: int = 200,
        rtsp_transport: str = "tcp",
        analyzeduration: str = "64M",
        probesize: str = "64M",
        stimeout_ms: int = 10_000,
        rw_timeout_ms: int = 10_000,
        enable_stimeout: bool = False,
    ) -> None:
        self.rtsp_url = rtsp_url
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # 16-bit mono
        self._queue: "queue.Queue[bytes]" = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen[bytes]] = None
        self.rtsp_transport = rtsp_transport
        self.analyzeduration = analyzeduration
        self.probesize = probesize
        self.stimeout_us = max(stimeout_ms, 0) * 1000
        self.rw_timeout_us = max(rw_timeout_ms, 0) * 1000
        self.enable_stimeout = enable_stimeout

    def start(self) -> None:
        """Start the background reader thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader, name="AudioIngest", daemon=True)
        self._thread.start()
        logger.debug("Audio ingest thread started.")

    def stop(self) -> None:
        """Stop the ingest thread and terminate ffmpeg."""
        self._stop_event.set()
        self._terminate_process()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        logger.debug("Audio ingest thread stopped.")

    def frames(self) -> Generator[bytes, None, None]:
        """Yield audio frames as bytes."""
        while not self._stop_event.is_set():
            try:
                frame = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._process and self._process.poll() is not None:
                    self._restart_notice()
                continue
            yield frame

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _reader(self) -> None:
        while not self._stop_event.is_set():
            if not self._spawn_process():
                if self._stop_event.is_set():
                    break
                time.sleep(2.0)
                continue

            assert self._process and self._process.stdout  # for MyPy
            stdout = self._process.stdout

            try:
                while not self._stop_event.is_set():
                    chunk = stdout.read(self.frame_size)
                    if not chunk:
                        break
                    if len(chunk) != self.frame_size:
                        continue
                    try:
                        self._queue.put(chunk, timeout=0.5)
                    except queue.Full:
                        logger.warning("Audio frame queue is full; dropping chunk.")
            finally:
                self._terminate_process()
                if self._stop_event.is_set():
                    break
                self._restart_notice()
                time.sleep(1.0)

    def _spawn_process(self) -> bool:
        command = ["ffmpeg", "-loglevel", "error", "-nostdin", "-fflags", "nobuffer", "-flags", "low_delay"]
        if self.rtsp_transport:
            command.extend(["-rtsp_transport", self.rtsp_transport])
        if self.analyzeduration:
            command.extend(["-analyzeduration", self.analyzeduration])
        if self.probesize:
            command.extend(["-probesize", self.probesize])
        if self.enable_stimeout and self.stimeout_us:
            command.extend(["-stimeout", str(self.stimeout_us)])
        command.extend(
            [
                "-i",
                self.rtsp_url,
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(self.sample_rate),
                "-f",
                "s16le",
                "pipe:1",
            ]
        )
        logger.info("Starting ffmpeg audio ingest.")
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError:
            logger.exception("ffmpeg not found. Ensure it is installed in the container.")
            self._stop_event.set()
            return False
        return True

    def _terminate_process(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._process and self._process.stderr:
            try:
                stderr_output = self._process.stderr.read().decode("utf-8", errors="ignore").strip()
                if stderr_output:
                    logger.debug("ffmpeg: %s", stderr_output)
            except Exception:  # pragma: no cover - best effort logging
                pass
        self._process = None

    def _restart_notice(self) -> None:
        logger.info("ffmpeg audio ingest exited; retrying shortly.")
