"""Entry point for the Iris Assistant CLI application."""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time

import cv2
from rich.console import Console
from rich.logging import RichHandler

from audio import AudioIngest
from asr import SpeechRecognizer
from orchestrator import (
    can_speak_now,
    handle_intent,
    note_spoken,
    parse_intent,
    state,
)
from tts_voice import IrisVoice
from utils import env_flag, env_path, ensure_directory
from vad import VoiceActivityDetector
from vision import FrameBuffer, SceneDescriber

console = Console()
logger = logging.getLogger("iris")


def configure_logging() -> None:
    """Configure structured logging with Rich output."""
    log_level = os.getenv("IRIS_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def main() -> None:
    """Run the Iris Assistant main loop."""
    configure_logging()
    console.log("[bold green]Starting Iris Assistant[/bold green]")

    rtsp_url = os.getenv("RTSP_URL")
    if not rtsp_url:
        logger.error("RTSP_URL environment variable is required.")
        sys.exit(1)

    # rtsp_url = os.getenv("RTSP_URL", "0")  # "0" = default webcam
    # rtsp_url = 0
    cap = cv2.VideoCapture(rtsp_url)
    
    whisper_dir = env_path("WHISPER_DIR")
    yolo_weights = env_path("YOLO_WEIGHTS")
    piper_voice = env_path("PIPER_VOICE")
    preview_enabled = env_flag("SHOW_PREVIEW", default=False)
    preview_dir = ensure_directory(os.getenv("PREVIEW_DIR", "/shared_previews"))
    save_previews = env_flag("SAVE_PREVIEWS", default=True)
    describe_frame_count = max(1, int(os.getenv("FRAME_DESCRIBE_COUNT", "3")))
    describe_max_age = float(os.getenv("FRAME_MAX_AGE_SECONDS", "1.5"))
    audio_sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    audio_frame_ms = int(os.getenv("AUDIO_FRAME_MS", "20"))
    audio_queue_size = int(os.getenv("AUDIO_QUEUE_SIZE", "400"))
    ffmpeg_analyzeduration = os.getenv("FFMPEG_ANALYZEDURATION", "64M")
    ffmpeg_probesize = os.getenv("FFMPEG_PROBESIZE", "64M")
    ffmpeg_stimeout_ms = int(os.getenv("FFMPEG_STIMEOUT_MS", "10000"))
    ffmpeg_rw_timeout_ms = int(os.getenv("FFMPEG_RW_TIMEOUT_MS", "10000"))
    rtsp_audio_transport = os.getenv("RTSP_AUDIO_TRANSPORT", "tcp")
    vad_padding_ms = int(os.getenv("VAD_PADDING_MS", "300"))
    vad_aggressiveness = int(os.getenv("VAD_AGGRESSIVENESS", "2"))
    ffmpeg_enable_stimeout = env_flag("FFMPEG_ENABLE_STIMEOUT", default=False)

    recognizer = SpeechRecognizer(str(whisper_dir))
    describer = SceneDescriber(str(yolo_weights))
    iris_voice = IrisVoice(str(piper_voice), playback_command=os.getenv("PLAYBACK_COMMAND"))
    frame_history_seconds = float(os.getenv("FRAME_HISTORY_SECONDS", "1.0"))
    frame_target_fps = float(os.getenv("FRAME_TARGET_FPS", "25.0"))
    frame_buffer = FrameBuffer(rtsp_url, max_seconds=frame_history_seconds, target_fps=frame_target_fps)
    audio_ingest_kwargs = dict(
        rtsp_url=rtsp_url,
        sample_rate=audio_sample_rate,
        frame_duration_ms=audio_frame_ms,
        queue_size=audio_queue_size,
        rtsp_transport=rtsp_audio_transport,
        analyzeduration=ffmpeg_analyzeduration,
        probesize=ffmpeg_probesize,
        stimeout_ms=ffmpeg_stimeout_ms,
        rw_timeout_ms=ffmpeg_rw_timeout_ms,
    )
    if ffmpeg_enable_stimeout:
        audio_ingest_kwargs["enable_stimeout"] = True

    audio_ingest = AudioIngest(**audio_ingest_kwargs)
    vad = VoiceActivityDetector(
        sample_rate=audio_sample_rate,
        frame_duration_ms=audio_frame_ms,
        padding_ms=vad_padding_ms,
        aggressiveness=vad_aggressiveness,
    )
    shutdown = False
    goal_monitor_stop = threading.Event()

    def handle_signal(signum, _frame):
        nonlocal shutdown
        logger.info("Received signal %s; shutting down.", signum)
        shutdown = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    def goal_monitor_loop() -> None:
        """Background loop that keeps persistent goal search alive."""
        logger.debug("Starting goal monitor loop.")
        while not goal_monitor_stop.is_set():
            if shutdown:
                break
            if state.active_goal and not state.paused:
                frame = frame_buffer.latest()
                if frame is not None:
                    try:
                        describer.tick_search_goal(frame, iris_voice)
                    except Exception:
                        logger.exception("Goal monitor tick failed.")
                else:
                    logger.debug("Goal monitor: no latest frame available.")
            time.sleep(0.4)
        logger.debug("Goal monitor loop stopped.")

    goal_thread = threading.Thread(target=goal_monitor_loop, name="GoalMonitor", daemon=True)

    try:
        frame_buffer.start()
        if not frame_buffer.wait_until_ready(timeout=3.0):
            logger.warning("Frame buffer did not deliver frames within 3 seconds; descriptions may lag.")
        audio_ingest.start()
        goal_thread.start()
        preview_active = preview_enabled
        need_preview_frame = preview_enabled or save_previews

        for segment in vad.segments(audio_ingest.frames()):
            if shutdown:
                break
            transcript = recognizer.transcribe(segment)
            if not transcript:
                continue
            console.log(f"[bold blue]You[/bold blue]: {transcript}")
            intent_name, slots, addressed = parse_intent(transcript)
            if not addressed:
                logger.debug("Ignoring utterance without wake word or clear command.")
                continue

            latest_frame = frame_buffer.latest()
            frames = frame_buffer.get_recent(describe_frame_count, max_age=describe_max_age)
            result = handle_intent(
                intent_name,
                slots,
                describer=describer,
                frames=frames,
                preview=need_preview_frame,
                tts=iris_voice,
                latest_frame=latest_frame,
            )
            response_text = result.get("message")
            preview_frame = result.get("preview_frame")

            if response_text:
                console.log(f"[bold magenta]Iris[/bold magenta]: {response_text}")

            if save_previews and preview_frame is not None:
                preview_path = preview_dir / f"last_preview.jpg"
                if not cv2.imwrite(str(preview_path), preview_frame):
                    logger.warning("Failed to write preview image to %s", preview_path)

            if preview_active and preview_frame is not None and preview_enabled:
                try:
                    cv2.imshow("Iris Preview", preview_frame)
                    # Display window briefly; close on key press without blocking
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        shutdown = True
                        break
                except cv2.error:
                    logger.warning(
                        "SHOW_PREVIEW requested, but the OpenCV GUI backend is unavailable in this environment. "
                        "Preview has been disabled."
                    )
                    preview_active = False
        console.log("[bold green]Iris Assistant stopped.[/bold green]")
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting.")
    finally:
        goal_monitor_stop.set()
        audio_ingest.stop()
        frame_buffer.stop()
        if goal_thread.is_alive():
            goal_thread.join(timeout=1.0)
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()
