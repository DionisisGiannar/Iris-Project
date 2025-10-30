"""Computer vision utilities for the Iris Assistant."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

SETTINGS.update({
    "datasets_dir": "/models/yolo",   # set YOLO model/dataset path
    "runs_dir": "/tmp/Ultralytics"    # optional: where logs/results go
})

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Structured detection result."""

    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    sector: str


class FrameBuffer:
    """Background frame reader that keeps the most recent frames from a stream."""

    def __init__(self, rtsp_url: str, max_seconds: float = 1.0, target_fps: float = 8.0) -> None:
        self.rtsp_url = rtsp_url
        self.target_fps = target_fps
        self.capture: Optional[cv2.VideoCapture] = None
        buffer_len = max(1, int(max_seconds * target_fps))
        self._frames: Deque[np.ndarray] = deque(maxlen=buffer_len)
        self._timestamps: Deque[float] = deque(maxlen=buffer_len)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.flush_limit = int(os.getenv("FRAME_FLUSH_LIMIT", "5"))
        self._fps_counter = 0
        self._fps_window_start = time.time()
        self._ready_event = threading.Event()
        self._fps_counter = 0
        self._fps_window_start = time.time()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._reader, name="FrameBuffer", daemon=True)
        self._stop_event.clear()
        self._ready_event.clear()
        self._thread.start()
        logger.debug("Frame buffer thread started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._reset_capture()
        logger.debug("Frame buffer thread stopped.")

    def get_recent(self, count: int, max_age: Optional[float] = None) -> List[np.ndarray]:
        cutoff = None
        if max_age is not None:
            cutoff = time.time() - max_age
        with self._lock:
            frames = list(self._frames)
            timestamps = list(self._timestamps)
        if cutoff is not None:
            filtered = [frame for frame, ts in zip(frames, timestamps) if ts >= cutoff]
        else:
            filtered = frames
        return filtered[-count:]

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._frames[-1].copy() if self._frames else None

    # ------------------------------------------------------------------ #
    def _read_latest_frame(self) -> Optional[np.ndarray]:
        if self.capture is None:
            return None
        ret, frame = self.capture.read()
        if not ret:
            return None
        flush_iters = max(0, self.flush_limit)
        for _ in range(flush_iters):
            if not self.capture.grab():
                break
            ret_retrieve, latest = self.capture.retrieve()
            if not ret_retrieve:
                break
            frame = latest
        return frame

    def _reader(self) -> None:
        delay = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        while not self._stop_event.is_set():
            if not self._ensure_capture():
                logger.debug("Frame buffer waiting for RTSP stream at %s", self.rtsp_url)
                time.sleep(1.0)
                continue
            frame = self._read_latest_frame()
            if frame is None:
                logger.warning("Failed to read frame from RTSP; retrying.")
                self._reset_capture()
                time.sleep(0.5)
                continue
            with self._lock:
                self._frames.append(frame)
                self._timestamps.append(time.time())
                self._fps_counter += 1
                self._ready_event.set()
                now = time.time()
                if now - self._fps_window_start >= 5.0:
                    elapsed = now - self._fps_window_start
                    fps = self._fps_counter / elapsed if elapsed > 0 else 0.0
                    logger.debug(
                        "FrameBuffer capture FPS: %.2f (flush_limit=%d, buffer_size=%d)",
                        fps,
                        self.flush_limit,
                        len(self._frames),
                    )
                    self._fps_counter = 0
                    self._fps_window_start = now
            if delay > 0 and len(self._frames) < self._frames.maxlen:
                time.sleep(delay)

    def _ensure_capture(self) -> bool:
        if self.capture and self.capture.isOpened():
            return True
        self._reset_capture()
        self.capture = cv2.VideoCapture(self.rtsp_url)
        if self.capture and self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.capture.isOpened():
            self._reset_capture()
            return False
        return True

    def _reset_capture(self) -> None:
        if self.capture:
            self.capture.release()
            self.capture = None
        with self._lock:
            self._frames.clear()
            self._timestamps.clear()

    def wait_until_ready(self, timeout: float = 2.0) -> bool:
        return self._ready_event.wait(timeout)


class SceneDescriber:
    """YOLO-powered scene description helper."""

    def __init__(self, weights_path: str, conf_threshold: float = 0.25) -> None:
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        names = getattr(getattr(self.model, "model", None), "names", None) or getattr(self.model, "names", None)
        if isinstance(names, dict):
            self.names = names
        elif isinstance(names, (list, tuple)):
            self.names = {idx: name for idx, name in enumerate(names)}
        else:
            self.names = {}

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO on a frame and return structured detections."""
        results = self.model.predict(frame, verbose=False, conf=self.conf_threshold, device="cpu")
        detections: List[Detection] = []
        height, width = frame.shape[:2]
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = self.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                sector = self._sector_for_bbox(x1, x2, width)
                detections.append(Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2), sector=sector))
        return detections

    def describe(self, frames: Sequence[np.ndarray], preview: bool = False) -> Tuple[str, np.ndarray | None]:
        """Describe the scene based on the provided frames."""
        if not frames:
            return "I cannot see any frames from the camera.", None

        aggregated: Counter[str] = Counter()
        sectors_map: dict[str, Counter[str]] = {}
        preview_frame = None
        boxes_map: Dict[str, List[Tuple[int, int, int, int]]] = {}

        for frame in frames:
            if frame is None:
                continue
            if preview and preview_frame is None:
                preview_frame = frame.copy()
            detections = self.detect(frame)
            for det in detections:
                label_boxes = boxes_map.setdefault(det.label, [])
                if self._is_duplicate(label_boxes, det.bbox):
                    continue
                label_boxes.append(det.bbox)
                aggregated[det.label] += 1
                sectors_map.setdefault(det.label, Counter())[det.sector] += 1
                if preview and preview_frame is not None:
                    cv2.rectangle(preview_frame, (det.bbox[0], det.bbox[1]), (det.bbox[2], det.bbox[3]), (0, 255, 0), 2)
                    cv2.putText(
                        preview_frame,
                        det.label,
                        (det.bbox[0], det.bbox[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )

        if not aggregated:
            return "I do not detect notable objects.", preview_frame

        top_items = aggregated.most_common(100)
        phrases: List[str] = []
        for label, count in top_items:
            sectors = sectors_map.get(label, Counter())
            sector_phrase = self._format_sectors(sectors)
            quantity = self._quantity_word(count)
            noun = self._pluralize(label, count)
            phrase = f"{quantity} {noun}"
            if sector_phrase:
                phrase += f" {sector_phrase}"
            phrases.append(phrase)

        description = self._join_phrases(phrases)
        sentence = f"I see {description}."
        if len(sentence.split()) > 25:
            sentence = self._trim_words(sentence, max_words=25)
        return sentence, preview_frame

    # ------------------------------------------------------------------ #
    @staticmethod
    def _sector_for_bbox(x1: int, x2: int, width: int) -> str:
        center = (x1 + x2) / 2
        if center < width / 3:
            return "left"
        if center > 2 * width / 3:
            return "right"
        return "center"

    @staticmethod
    def _quantity_word(count: int) -> str:
        words = {1: "one", 2: "two", 3: "three", 4: "four"}
        return words.get(count, str(count))

    @staticmethod
    def _pluralize(label: str, count: int) -> str:
        if count == 1:
            return label
        if label.endswith("person"):
            return "people"
        if label.endswith("y"):
            return label[:-1] + "ies"
        if label.endswith("s"):
            return label
        return f"{label}s"

    @staticmethod
    def _format_sectors(sectors: Counter[str]) -> str:
        if not sectors:
            return ""
        ordered = [sector for sector in ("left", "center", "right") if sectors.get(sector)]
        if not ordered:
            return ""
        if len(ordered) == 1:
            return f"in the {ordered[0]}"
        if len(ordered) == 2:
            return f"across the {ordered[0]} and {ordered[1]}"
        return "across the left, center, and right"

    @staticmethod
    def _join_phrases(phrases: List[str]) -> str:
        if not phrases:
            return "nothing notable"
        if len(phrases) == 1:
            return phrases[0]
        return ", ".join(phrases[:-1]) + f", and {phrases[-1]}"

    @staticmethod
    def _trim_words(sentence: str, max_words: int) -> str:
        words = sentence.split()
        return " ".join(words[:max_words]).rstrip(",") + "."

    @staticmethod
    def _is_duplicate(existing: List[Tuple[int, int, int, int]], candidate: Tuple[int, int, int, int], iou_threshold: float = 0.5) -> bool:
        for bbox in existing:
            if SceneDescriber._iou(bbox, candidate) >= iou_threshold:
                return True
        return False

    @staticmethod
    def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union_area = area_a + area_b - inter_area
        if union_area == 0:
            return 0.0
        return inter_area / union_area

    # ------------------------------------------------------------------ #
