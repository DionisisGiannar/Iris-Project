# Iris Assistant

Iris is an offline, CPU-only assistant that watches an RTSP stream, listens for short push-to-talk style utterances, and responds with concise scene descriptions. Inside Docker it keeps all models on disk and never downloads assets at runtime.

## Features
- YOLOv8 scene understanding (Ultralytics, small weights) with left/center/right hints.
- Faster-Whisper English ASR (tiny/base c-translate2 models).
- Piper text-to-speech (English voice) via `app/tts_voice.py` for live responses.
- WebRTC VAD for push-to-talk segmentation.
- Entirely offline, CPU-only pipeline designed for Mac-first Docker workflows.

## Repository Layout
```
app/
  audio.py        # ffmpeg-based RTSP audio ingest
  asr.py          # faster-whisper wrapper
  intent.py       # simple intent routing (quit / describe)
  main.py         # orchestration loop
  tts_voice.py    # Piper synthesis + optional playback + preview dumps
  utils.py        # helpers for env + timestamps
  vad.py          # WebRTC VAD segmenter
  vision.py       # YOLO frame buffer + descriptions
models/
  whisper/        # faster-whisper ctranslate2 dir (e.g., tiny.en/)
  yolo/           # yolov8n.pt
  piper/          # Piper voice .onnx + .json
shared_audio/     # synthesized responses (optional)
shared_previews/  # saved preview frames (optional)
```

## Prerequisites
1. Download the models on your host machine and place them under `./models`:
   - **Whisper**: grab a ctranslate2 English model such as `tiny.en` or `base.en` and place the extracted directory inside `models/whisper`.
   - **YOLO**: download `yolov8n.pt` and place it under `models/yolo/`.
   - **Piper**: download an English voice (e.g., `en_US-amy-medium.onnx` plus the matching `.json`) into `models/piper/`.
2. Duplicate the environment file:
   ```bash
   cp .env.example .env
   ```
   Update the paths to match the model directories if you chose different names.
3. Provision an RTSP stream containing audio and video (e.g., via the bundled `mediamtx` service or an existing camera). The assistant expects 48 kHz or 44.1 kHz audio, which it downsamples to 16 kHz.

## Running with Docker Compose
1. Build and start the services:
   ```bash
   docker compose up --build
   ```
2. Optionally bring up the `rtsp` service if you want a local relay:
   ```bash
   docker compose up rtsp
   ```
3. The `iris` service mounts `./models`, `./shared_audio`, and `./app` (into `/app/app`) for iterative development.

### Publish Your Webcam and Mic
If you rely on the bundled `mediamtx` RTSP server, you still need to publish an input stream. On macOS you can push your webcam/microphone pair (replace `video_index:audio_index` as needed; e.g., `0:2` for MacBook Pro camera + mic) with ffmpeg:
```bash
ffmpeg -f avfoundation -framerate 25 -pixel_format uyvy422 -video_size 960x540 \
  -i "0:2" \
  -c:v h264_videotoolbox -b:v 1500k -profile:v baseline -tune zerolatency -preset ultrafast -pix_fmt yuv420p \
  -c:a aac -b:a 128k -ar 44100 -ac 1 -bufsize 1M -max_delay 0 \
  -f rtsp -rtsp_transport tcp rtsp://localhost:8554/iris
```
Keep this publisher running while Iris is active. Adjust device indices, resolution, or bitrate to match your hardware.

For convenience, `scripts/run_iris.sh` automates the sequence:
```bash
chmod +x scripts/run_iris.sh
./scripts/run_iris.sh --build   # optional: rebuild images before starting
AVFOUNDATION_VIDEO=0 AVFOUNDATION_AUDIO=2 ./scripts/run_iris.sh
```
The script starts the RTSP relay, launches `ffmpeg`, and then runs the Iris container. Override variables such as `VIDEO_SIZE`, `VIDEO_BITRATE`, or `RTSP_HOST` as needed.

Set `PLAYBACK_COMMAND` in your `.env` (for example, `ffplay -autoexit -nodisp -f s16le -ac 1 -ar {rate} -`) so the container can stream PCM audio to a player.

### Environment Variables
- `RTSP_URL`: RTSP route to your audio/video stream (default assumes `mediamtx` on `host.docker.internal`).
- `WHISPER_DIR`: Path to the faster-whisper ctranslate2 directory inside the container.
- `YOLO_WEIGHTS`: Path to the YOLO weights file inside the container.
- `PIPER_VOICE`: Path to the Piper ONNX voice file inside the container.
- `SHOW_PREVIEW`: Set to `1` to open an OpenCV window with live detections.
- `AUDIO_SAMPLE_RATE`: Sample rate (Hz) used for ffmpeg output and VAD (default `16000`).
- `AUDIO_FRAME_MS`: Frame duration for audio chunks and VAD (default `20` ms).
- `AUDIO_QUEUE_SIZE`: Buffer capacity (frames) for the ingest queue (default `400`).
- `RTSP_AUDIO_TRANSPORT`: Transport protocol for ffmpeg RTSP ingest (`tcp` by default).
- `FFMPEG_ANALYZEDURATION` / `FFMPEG_PROBESIZE`: Tune ffmpeg probing buffers (defaults `64M` each).
- `FFMPEG_STIMEOUT_MS` / `FFMPEG_RW_TIMEOUT_MS`: Socket timeouts in milliseconds for RTSP ingest (defaults `10000`).
- `FFMPEG_ENABLE_STIMEOUT`: Set to `1` to add the `-stimeout` flag if your ffmpeg build supports it (off by default).
- `VAD_PADDING_MS`: Hangover padding for WebRTC VAD (default `300` ms).
- `VAD_AGGRESSIVENESS`: WebRTC VAD aggressiveness level (0–3, default `2`).
- `PLAYBACK_COMMAND`: Shell command that consumes PCM from stdin and plays audio (e.g., `ffplay -autoexit -nodisp -f s16le -ac 1 -ar {rate} -`). Required to hear live responses.
- `PREVIEW_DIR`: Folder where annotated preview frames are written (`/shared_previews` by default).
- `SAVE_PREVIEWS`: Set to `0` to disable writing preview images (default `1`).
- `FRAME_HISTORY_SECONDS`: How many seconds of frames to retain in the buffer for descriptions (default `1.0`).
- `FRAME_TARGET_FPS`: Target capture FPS for the frame buffer (default `8.0`).
- `FRAME_FLUSH_LIMIT`: Maximum number of extra frames drained from the RTSP buffer each cycle (default `5`).
- `FRAME_DESCRIBE_COUNT`: Number of recent frames to evaluate per description (default `3`).
- `FRAME_MAX_AGE_SECONDS`: Maximum age of frames (seconds) considered for descriptions (default `1.5`).
- `LIVE_PREVIEW_PATH`: File path where the latest raw frame is written every capture (`/shared_previews/live_preview.jpg` by default).
- `FRAME_FLUSH_LIMIT`: Maximum number of extra frames drained from the RTSP buffer each cycle (default `5`).

## How the Loop Works
1. The frame buffer grabs ~10 FPS video frames and keeps the latest few seconds.
2. `ffmpeg` pulls audio, resamples to 16 kHz mono, and the VAD segments speech.
3. On segment end the ASR transcribes English text. Keywords like “quit”, “exit”, or “stop app” end the session; everything else is treated as a describe intent.
4. YOLO runs on the latest frames, merges detections, and composes a ≤25-word summary with left/center/right hints.
5. Piper renders a short spoken response and streams it to your configured playback command.

## Local Development
- Run the CLI app directly (requires the same environment variables):
  ```bash
  python -m app.main
  ```
- Toggle preview mode by setting `SHOW_PREVIEW=1` to see bounding boxes (requires an X server if running inside Docker on macOS).
- The codebase uses Rich logging—set `IRIS_LOG_LEVEL=DEBUG` for verbose output.

## Notes
- Everything runs on CPU—no CUDA or Metal dependencies are required.
- The assistant assumes English-only speech and responses.
- Ensure `ffmpeg` has network access to your RTSP source; if you use the provided `mediamtx` container, publish audio/video to `rtsp://host.docker.internal:8554/iris`.
