"""The robot's camera is a phone: it streams to the Pi over a WebSocket.

The phone opens https://<pi-ip>:<port> (HTTPS because browsers only allow camera
access on secure pages) and stays on that page, mounted on the robot.

    capture()      asks the phone for a frame taken now, at full resolution, and
                   returns the JPEG bytes. PiHardware.capture() decodes it.
    preview()      the latest small preview frame, for the web app's live view.
    start_stream() asks the phone for a steady stream of frames (for a video);
    subscribe(fn)  fn(jpeg, time, kind) for every frame that arrives (kind
                   "stream" or "preview"); video_rec.py records them.
    status()       connected, resolution and zoom, for the web app.

The page has a zoom slider: the phone's optical zoom when the browser offers it,
otherwise a centre crop. Set the zoom before calibrating and leave it: the
calibration (pixels -> cm) is only valid for that zoom.

The server runs in its own thread with its own event loop, so the Flask app and
the robot code (plain threads) call capture() like any blocking function.
"""
from __future__ import annotations

import asyncio
import json
import struct
import subprocess
import threading
import time
from pathlib import Path

# At module level: FastAPI reads the handlers' type hints from this module's
# names (they are strings because of `from __future__ import annotations`).
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

CERT_DIR = Path(".certs")
# Each frame from the phone: the request id it answers (-1 = preview), then the JPEG.
FRAME_HEADER = struct.Struct("<d")
PREVIEW_ID = -1
STREAM_ID = -2
CONNECTED_S = 5.0  # the phone counts as connected if it was heard this recently

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenWeeder Robot Camera</title>
<style>
  body { margin: 0; padding: 12px; background: #111; color: #eee; font: 17px sans-serif; text-align: center; }
  .view { position: relative; overflow: hidden; margin: 0 auto; max-width: 100%; max-height: 60vh; background: #000; border-radius: 8px; }
  video { display: block; width: 100%; max-height: 60vh; object-fit: contain; transform-origin: center; }
  #status { margin: 10px 0 4px; }
  #info { color: #999; font-size: 14px; font-variant-numeric: tabular-nums; }
  .zoom { display: flex; align-items: center; gap: 10px; max-width: 480px; margin: 14px auto 4px; }
  .zoom input { flex: 1; }
  .zoom button { width: 52px; height: 52px; font-size: 28px; border: 0; border-radius: 8px; background: #333; color: #fff; }
  #zoom-value { font-size: 22px; font-weight: bold; font-variant-numeric: tabular-nums; }
  .hint { color: #999; font-size: 14px; max-width: 480px; margin: 8px auto; }
  body.live #status { color: #81c784; }
</style>
</head>
<body>
<div class="view"><video id="preview" autoplay playsinline muted></video></div>
<div id="status">Starting camera...</div>
<div id="info"></div>
<div class="zoom">
  <button id="zoom-out" aria-label="Zoom out">&minus;</button>
  <input type="range" id="zoom" min="1" max="4" step="0.1" value="1" aria-label="Zoom">
  <button id="zoom-in" aria-label="Zoom in">+</button>
</div>
<div id="zoom-value">1.0&times;</div>
<p class="hint">Mount the phone so the top of the picture points the way the car drives. Set the zoom before calibrating, then leave it. Keep this page open and the screen on.</p>
<script>
const field = (id, value) => { document.getElementById(id).textContent = value; };
const preview = document.getElementById("preview");
const zoomSlider = document.getElementById("zoom");
const canvas = document.createElement("canvas");
const context = canvas.getContext("2d");
const QUALITY = __JPEG_QUALITY__;
const PREVIEW_WIDTH = 480;
const PREVIEW_MS = 700;
let ws;
let track = null;
let nativeZoom = false;
let zoom = 1;
let busy = false;

function sendFrame(id, blob) {
  if (ws && ws.readyState === WebSocket.OPEN && blob) ws.send(new Blob([new Float64Array([id]), blob]));
}

function sendStatus() {
  if (!ws || ws.readyState !== WebSocket.OPEN || !track) return;
  ws.send(JSON.stringify({ status: {
    width: preview.videoWidth, height: preview.videoHeight,
    zoom, zoom_kind: nativeZoom ? "optical" : "digital",
  } }));
}

// The part of the camera image to keep: all of it with optical zoom, the
// centre with digital zoom.
function sourceRect() {
  const w = preview.videoWidth, h = preview.videoHeight;
  if (nativeZoom || zoom <= 1) return [0, 0, w, h];
  const sw = Math.round(w / zoom), sh = Math.round(h / zoom);
  return [Math.round((w - sw) / 2), Math.round((h - sh) / 2), sw, sh];
}

function grab(id, width, quality) {
  if (!preview.videoWidth) return;
  const [sx, sy, sw, sh] = sourceRect();
  const scale = width ? Math.min(1, width / sw) : 1;
  canvas.width = Math.round(sw * scale);
  canvas.height = Math.round(sh * scale);
  context.drawImage(preview, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
  canvas.toBlob((blob) => sendFrame(id, blob), "image/jpeg", quality || (id === -1 ? 0.6 : QUALITY));
}

// A steady stream of frames while the Pi records a video.
let streamTimer = null;
function setStream(cfg) {
  if (streamTimer) clearInterval(streamTimer);
  streamTimer = null;
  if (cfg) streamTimer = setInterval(() => grab(-2, cfg.width, cfg.quality), 1000 / cfg.fps);
  field("info", cfg ? `Recording video: ${cfg.fps} frames per second` : "");
}

function connect() {
  ws = new WebSocket(`wss://${location.host}/ws`);
  ws.onopen = () => { document.body.classList.add("live"); field("status", "Connected to the robot"); sendStatus(); };
  ws.onclose = () => {
    document.body.classList.remove("live");
    field("status", "Not connected, retrying...");
    setTimeout(connect, 1000);
  };
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if ("capture" in message) grab(message.capture, 0);
    if ("stream" in message) setStream(message.stream);
  };
}

async function setZoom(value) {
  zoom = Math.min(Number(zoomSlider.max), Math.max(Number(zoomSlider.min), value));
  zoomSlider.value = zoom;
  field("zoom-value", `${zoom.toFixed(1)}× ${nativeZoom ? "optical" : "digital"}`);
  try { localStorage.setItem("robot-camera-zoom", String(zoom)); } catch (error) { /* ignore */ }
  if (nativeZoom) {
    try { await track.applyConstraints({ advanced: [{ zoom }] }); } catch (error) { /* keep going */ }
    preview.style.transform = "";
  } else {
    preview.style.transform = `scale(${zoom})`;
  }
  sendStatus();
}

async function start() {
  if (!navigator.mediaDevices) {
    field("status", "Camera unavailable: open this page with https://");
    return;
  }
  try {
    preview.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1600 }, height: { ideal: 1200 } },
      audio: false,
    });
  } catch (error) {
    field("status", `Camera error: ${error.message}`);
    return;
  }
  track = preview.srcObject.getVideoTracks()[0];
  const caps = track.getCapabilities ? track.getCapabilities() : {};
  if (caps.zoom && caps.zoom.max > caps.zoom.min) {
    nativeZoom = true;
    zoomSlider.min = caps.zoom.min;
    zoomSlider.max = caps.zoom.max;
    zoomSlider.step = caps.zoom.step || 0.1;
  }
  let saved = NaN;
  try { saved = parseFloat(localStorage.getItem("robot-camera-zoom")); } catch (error) { /* ignore */ }
  await setZoom(Number.isFinite(saved) ? saved : (nativeZoom ? (track.getSettings().zoom || 1) : 1));
  connect();
  setInterval(() => grab(-1, PREVIEW_WIDTH), PREVIEW_MS);
  setInterval(() => {
    const [, , sw, sh] = sourceRect();
    field("info", `${sw}×${sh} per photo`);
    sendStatus();
  }, 2000);
}

zoomSlider.oninput = () => setZoom(Number(zoomSlider.value));
document.getElementById("zoom-in").onclick = () => setZoom(zoom + Number(zoomSlider.step || 0.1) * 5);
document.getElementById("zoom-out").onclick = () => setZoom(zoom - Number(zoomSlider.step || 0.1) * 5);

async function keepAwake() {
  try { if (navigator.wakeLock) await navigator.wakeLock.request("screen"); } catch (error) { /* ignore */ }
}
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") keepAwake(); });

start();
keepAwake();
</script>
</body>
</html>
"""


class CameraError(RuntimeError):
    pass


def local_ips() -> list[str]:
    output = subprocess.run(["hostname", "-I"], capture_output=True, text=True, check=True).stdout
    return [ip for ip in output.split() if ":" not in ip]


def ensure_certificate(ips: list[str], cert_dir: Path = CERT_DIR) -> tuple[Path, Path]:
    """A self-signed certificate for the Pi's addresses, remade if they change."""
    cert, key, names_file = cert_dir / "cert.pem", cert_dir / "key.pem", cert_dir / "names.txt"
    names = ",".join(["DNS:localhost"] + [f"IP:{ip}" for ip in ips])
    if cert.exists() and key.exists() and names_file.exists() and names_file.read_text() == names:
        return cert, key
    cert_dir.mkdir(mode=0o700, exist_ok=True)
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(cert),
         "-days", "825", "-subj", "/CN=OpenWeeder", "-addext", f"subjectAltName={names}"],
        capture_output=True, check=True,
    )
    key.chmod(0o600)
    names_file.write_text(names)
    return cert, key


class PhoneCamera:
    def __init__(self, port: int = 8444, jpeg_quality: float = 0.92, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.jpeg_quality = jpeg_quality
        self.links: list[str] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws = None
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._ids = threading.Lock()
        self._preview: bytes | None = None
        self._status: dict = {}
        self._heard_at = 0.0
        self._started = threading.Event()
        self._server = None
        self._subscribers: list = []
        self._stream: dict | None = None

    # ---- called from any thread ----
    def start(self) -> "PhoneCamera":
        ips = local_ips()
        cert, key = ensure_certificate(ips)
        self.links = [f"https://{ip}:{self.port}" for ip in ips]
        thread = threading.Thread(target=self._serve, args=(cert, key), daemon=True, name="phone-camera")
        thread.start()
        if not self._started.wait(15):
            raise CameraError(f"The phone camera server did not start on port {self.port}")
        return self

    def connected(self) -> bool:
        return self._ws is not None and time.monotonic() - self._heard_at < CONNECTED_S

    def capture(self, timeout: float = 2.0) -> bytes:
        """A JPEG taken now by the phone. Raises CameraError if it does not come."""
        if self._loop is None:
            raise CameraError("The phone camera server is not running")
        with self._ids:
            self._next_id += 1
            request_id = self._next_id
        future = asyncio.run_coroutine_threadsafe(self._request(request_id, timeout), self._loop)
        try:
            return future.result(timeout + 1)
        except CameraError:
            raise
        except Exception as error:  # noqa: BLE001  (timeout, closed socket)
            future.cancel()
            raise CameraError(f"No photo from the phone within {timeout:.1f} s ({type(error).__name__})") from error

    def subscribe(self, fn) -> None:
        """fn(jpeg, monotonic_time, kind) for every frame from the phone; it runs on
        the server's thread, so it must return quickly."""
        self._subscribers.append(fn)

    def unsubscribe(self, fn) -> None:
        if fn in self._subscribers:
            self._subscribers.remove(fn)

    def start_stream(self, fps: float = 5, width: int = 640, quality: float = 0.7) -> None:
        """Ask the phone for fps frames per second (also after it reconnects)."""
        self._stream = {"fps": fps, "width": width, "quality": quality}
        self._send_stream()

    def stop_stream(self) -> None:
        self._stream = None
        self._send_stream()

    def _send_stream(self) -> None:
        ws, loop = self._ws, self._loop
        if ws is not None and loop is not None:
            asyncio.run_coroutine_threadsafe(ws.send_text(json.dumps({"stream": self._stream})), loop)

    def preview(self) -> bytes | None:
        return self._preview if self.connected() else None

    def status(self) -> dict:
        return {"connected": self.connected(), "links": self.links, **self._status}

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True

    # ---- the server (its own event loop) ----
    async def _request(self, request_id: int, timeout: float) -> bytes:
        ws = self._ws
        if ws is None:
            raise CameraError("The phone camera is not connected: open " + " or ".join(self.links) + " on the phone")
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await ws.send_text(json.dumps({"capture": request_id}))
            return await asyncio.wait_for(future, timeout)
        finally:
            self._pending.pop(request_id, None)

    def _app(self) -> FastAPI:
        app = FastAPI()
        page = PAGE.replace("__JPEG_QUALITY__", str(self.jpeg_quality))

        @app.get("/")
        async def phone_page():
            return HTMLResponse(page)

        @app.websocket("/ws")
        async def phone(ws: WebSocket):
            await ws.accept()
            old, self._ws = self._ws, ws  # the newest phone replaces an older one
            self._heard_at = time.monotonic()
            if self._stream is not None:
                await ws.send_text(json.dumps({"stream": self._stream}))
            if old is not None:
                try:
                    await old.close()
                except Exception:  # noqa: BLE001
                    pass
            try:
                while True:
                    message = await ws.receive()
                    if message["type"] == "websocket.disconnect":
                        break
                    self._heard_at = time.monotonic()
                    if message.get("bytes") is not None:
                        data = message["bytes"]
                        if len(data) <= FRAME_HEADER.size:
                            continue
                        (request_id,) = FRAME_HEADER.unpack_from(data)
                        jpeg = data[FRAME_HEADER.size:]
                        if request_id in (PREVIEW_ID, STREAM_ID):
                            if request_id == PREVIEW_ID:
                                self._preview = jpeg
                            kind = "preview" if request_id == PREVIEW_ID else "stream"
                            now = time.monotonic()
                            for fn in list(self._subscribers):
                                try:
                                    fn(jpeg, now, kind)
                                except Exception:  # noqa: BLE001  (a recorder must not break the camera)
                                    pass
                        else:
                            future = self._pending.get(int(request_id))
                            if future is not None and not future.done():
                                future.set_result(jpeg)
                    elif message.get("text") is not None:
                        status = json.loads(message["text"]).get("status")
                        if isinstance(status, dict):
                            self._status = {k: status[k] for k in ("width", "height", "zoom", "zoom_kind") if k in status}
            except (WebSocketDisconnect, RuntimeError, ValueError):
                pass
            finally:
                if self._ws is ws:
                    self._ws = None
                    for future in self._pending.values():
                        if not future.done():
                            future.set_exception(CameraError("The phone camera disconnected"))

        return app

    def _serve(self, cert: Path, key: Path) -> None:
        import uvicorn

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        config = uvicorn.Config(self._app(), host=self.host, port=self.port, log_level="warning", lifespan="off",
                                ssl_certfile=str(cert), ssl_keyfile=str(key))
        self._server = uvicorn.Server(config)

        async def serve():
            task = asyncio.create_task(self._server.serve())
            while not self._server.started and not task.done():
                await asyncio.sleep(0.05)
            if self._server.started:
                self._started.set()
            await task

        loop.run_until_complete(serve())


_SHARED: PhoneCamera | None = None
_SHARED_LOCK = threading.Lock()


def shared(camera_config: dict | None = None) -> PhoneCamera:
    """The one phone camera server of this program, started on first use."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            cfg = camera_config or {}
            _SHARED = PhoneCamera(port=int(cfg.get("port", 8444)),
                                  jpeg_quality=float(cfg.get("jpeg_quality", 0.92))).start()
        return _SHARED


def running() -> PhoneCamera | None:
    return _SHARED
