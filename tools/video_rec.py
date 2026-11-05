"""Record what the robot's phone camera sees, as it happens, to an .mp4.

    rec = LiveRecorder(camera, path, overlay=None)  camera: phone_cam.PhoneCamera
    rec.start()      asks the phone for a steady stream of frames
    ...              (the robot drives)
    rec.stop()       -> True when path (H.264, plays in a browser) was written

overlay(bgr) -> bgr draws on each frame (the tray, the run's status...). Frames are
saved as JPEGs with the time each one arrived while recording, and the video is
built once at the end, timed by those arrivals so it plays at the real speed;
the Pi does no video encoding while the robot is driving. If the phone page is
too old to stream, its slower preview frames are used instead.
"""
from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

STREAM_FPS = 5
STREAM_WIDTH = 640


class LiveRecorder:
    def __init__(self, camera, path: Path, overlay=None, fps: float = STREAM_FPS, width: int = STREAM_WIDTH):
        self.camera, self.path, self.overlay = camera, Path(path), overlay
        self.fps, self.width = fps, width
        self.frames_dir = self.path.with_name(self.path.stem + "_frames")
        self.queue: queue.Queue = queue.Queue(maxsize=50)
        self.times: list[float] = []
        self.streaming = False
        self.dropped = 0
        self._running = False
        self._thread = None

    def start(self) -> "LiveRecorder":
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._running = True
        self._thread = threading.Thread(target=self._work, daemon=True, name="video-recorder")
        self._thread.start()
        self.camera.subscribe(self._on_frame)
        self.camera.start_stream(self.fps, self.width)
        return self

    def _on_frame(self, jpeg: bytes, t: float, kind: str) -> None:
        # Runs on the camera server's thread: just queue it.
        if kind == "stream":
            self.streaming = True
        elif self.streaming:
            return  # the stream is coming: previews are not needed
        try:
            self.queue.put_nowait((jpeg, t))
        except queue.Full:
            self.dropped += 1

    def _work(self) -> None:
        while self._running or not self.queue.empty():
            try:
                jpeg, t = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            name = self.frames_dir / f"{len(self.times):06d}.jpg"
            if self.overlay is None:
                name.write_bytes(jpeg)
            else:
                bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                try:
                    bgr = self.overlay(bgr)
                except Exception:  # noqa: BLE001  (a drawing problem must not lose the frame)
                    pass
                cv2.imwrite(str(name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.times.append(t)

    def stop(self) -> bool:
        """Stop recording and build the video. False when there were no frames or
        ffmpeg failed (the frames are then kept in <name>_frames for a look)."""
        self.camera.unsubscribe(self._on_frame)
        try:
            self.camera.stop_stream()
        except Exception:  # noqa: BLE001
            pass
        self._running = False
        if self._thread is not None:
            self._thread.join(30)
        if not self.times:
            shutil.rmtree(self.frames_dir, ignore_errors=True)
            return False
        ok = encode(self.frames_dir, self.times, self.path)
        if ok:
            shutil.rmtree(self.frames_dir, ignore_errors=True)
        return ok


def encode(frames_dir: Path, times: list[float], path: Path) -> bool:
    """JPEGs 000000.jpg... shown for as long as they really lasted -> H.264 .mp4."""
    gaps = [b - a for a, b in zip(times, times[1:])]
    typical = sorted(gaps)[len(gaps) // 2] if gaps else 1.0 / STREAM_FPS
    durations = [g if 0 < g < 5 else typical for g in gaps] + [typical]
    lines = ["ffconcat version 1.0"]
    for i, d in enumerate(durations):
        lines += [f"file '{i:06d}.jpg'", f"duration {d:.4f}"]
    lines.append(f"file '{len(durations) - 1:06d}.jpg'")  # concat drops the last duration otherwise
    playlist = frames_dir / "frames.ffconcat"
    playlist.write_text("\n".join(lines) + "\n")
    result = subprocess.run(
        ["nice", "-n", "15", "ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
         "-i", str(playlist), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=10",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)],
        capture_output=True)
    return result.returncode == 0 and path.is_file()


def caption(bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    """White text on a dark band at the top of the frame."""
    h = 26 * len(lines) + 10
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], h), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(bgr, text, (10, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return bgr
