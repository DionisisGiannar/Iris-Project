Place your offline model assets in this directory:

- `models/whisper/<model_dir>` – faster-whisper c-translate2 English model (e.g., `tiny.en/`, `base.en/`).
- `models/yolo/yolov8n.pt` – YOLOv8 nano weights.
- `models/piper/<voice>.onnx` and matching `<voice>.json` – Piper English voice.

These assets are mounted into the container at `/models`. No downloads occur at runtime, so ensure the files exist before launching the assistant.
