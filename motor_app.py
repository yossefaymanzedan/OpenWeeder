#!/usr/bin/env python3
"""Drive the car from one phone page that also films with the phone's camera.

The page runs the motors at a set power and shows the wheel's RPM and velocity
from the encoder. Go starts the motors and a recording; STOP stops both.

The phone records the video itself, with its hardware video encoder, so the
video is smooth and full quality whatever the Wi-Fi does. Meanwhile the Pi logs
the car's readings. After STOP the phone uploads the video, and the Pi turns it
into a dataset: a folder in recordings/ with video.mkv, every frame as a JPEG,
frames.csv with the readings at the moment each frame was taken, and
sensors.csv with all the readings.

The page is HTTPS because browsers only allow camera access on secure pages.
The motors stop if the page disconnects or sends nothing for a second.
Requires the pigpio daemon: sudo pigpiod -l
"""

import argparse
import asyncio
import bisect
import csv
import json
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import pigpio
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import pins
from encoder_app import Encoder
from motors import MAX_SPEED, Car
from phone_camera import ensure_certificate, local_ips, now_ms

UPDATE_SECONDS = 0.1
# The page sends a message every 0.25 s. Stop the motors if it goes quiet.
COMMAND_TIMEOUT_SECONDS = 1.0
# How often the car's readings are logged while recording.
SENSOR_INTERVAL_SECONDS = 0.05
# The phone stops filming a moment after the Pi stops the recording, so the
# readings carry on this long to cover the last frames.
SENSOR_TAIL_SECONDS = 1.0
RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"
VIDEO_TYPES = ("mp4", "webm")

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenWeeder Drive</title>
<style>
  body { margin: 0; padding: 16px; background: #111; color: #eee; font: 18px sans-serif; text-align: center; }
  #preview { display: block; margin: 0 auto; max-width: 100%; max-height: 35vh; background: #000; }
  #camera { color: #999; margin: 8px 0 4px; font-size: 15px; }
  .sharp { display: inline-flex; align-items: center; gap: 8px; margin: 4px 0 8px; font-size: 16px; }
  .sharp input { width: 22px; height: 22px; }
  #rec { min-height: 1.4em; color: #999; }
  body.recording #rec { color: #ef5350; font-weight: bold; }
  .readouts { display: flex; justify-content: center; gap: 24px; margin-top: 8px; flex-wrap: wrap; }
  .speed { font-size: 44px; font-weight: bold; font-variant-numeric: tabular-nums; }
  .unit { color: #999; }
  .controls { margin: 20px auto; max-width: 420px; }
  .power { display: flex; gap: 12px; align-items: center; }
  .power input { flex: 1; min-width: 0; font-size: 28px; padding: 10px; border-radius: 8px; border: 0; text-align: center; }
  .buttons { display: flex; gap: 12px; margin-top: 12px; }
  button { flex: 1; font-size: 18px; padding: 16px; border-radius: 8px; border: 0; }
  #go { background: #2e7d32; color: #fff; font-weight: bold; font-size: 24px; }
  #stop { background: #555; color: #fff; font-weight: bold; font-size: 24px; }
  #emergency { width: 100%; margin-top: 16px; padding: 28px; background: #c62828; color: #fff; font-weight: bold; font-size: 28px; }
  #locked { display: none; margin: 16px auto; max-width: 420px; padding: 16px; border-radius: 8px; background: #4a1010; }
  #release { margin-top: 12px; background: #eee; color: #111; }
  body.emergency #locked { display: block; }
  body.emergency .controls input, body.emergency .buttons button { opacity: 0.3; }
  table { margin: 0 auto 16px; border-collapse: collapse; text-align: left; }
  td { padding: 4px 12px; }
  td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
  #reset { padding: 10px 24px; font-size: 16px; }
  #message { min-height: 1.4em; color: #ef9a9a; }
  .hint { color: #999; font-size: 15px; }
  #status { color: #999; }
</style>
</head>
<body>
<video id="preview" autoplay playsinline muted></video>
<div id="camera">Starting camera...</div>
<label class="sharp"><input type="checkbox" id="sharp"> Sharp mode: 1/500 s shutter, needs good light</label>
<div id="rec"></div>
<div class="readouts">
  <div><div class="speed" id="motor">0</div><div class="unit">% motor power</div></div>
  <div><div class="speed" id="rpm">0</div><div class="unit">wheel RPM</div></div>
  <div><div class="speed" id="cms">0.0</div><div class="unit">cm/s</div></div>
</div>
<div class="controls">
  <div class="power">
    <input type="number" id="power" min="0" max="__MAX_SPEED__" step="1" inputmode="numeric" placeholder="0 to __MAX_SPEED__ %">
    <button id="go">Go</button>
  </div>
  <div id="message"></div>
  <div class="buttons">
    <button id="stop">STOP</button>
  </div>
  <div class="hint">Go starts recording. STOP sends the video to the Pi.</div>
  <button id="emergency">EMERGENCY STOP</button>
</div>
<div id="locked">
  Emergency stop. Motors are locked.
  <button id="release">Release</button>
</div>
<table>
  <tr><td>Velocity</td><td id="ms">0.000 m/s</td></tr>
  <tr><td>Velocity</td><td id="kmh">0.00 km/h</td></tr>
  <tr><td>Distance</td><td id="distance">0.00 m</td></tr>
  <tr><td>Pulses</td><td id="count">0</td></tr>
</table>
<button id="reset">Reset distance</button>
<p id="status">Connecting...</p>
<script>
const field = (id, value) => { document.getElementById(id).textContent = value; };
const power = document.getElementById("power");
const preview = document.getElementById("preview");
const sharp = document.getElementById("sharp");
const socketBase = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}`;
const CAMERA = { width: __CAMERA_WIDTH__, height: __CAMERA_HEIGHT__, fps: __CAMERA_FPS__, bitrate: __VIDEO_BITRATE__ };
// Best first. Phones record these with their hardware video encoder.
const VIDEO_FORMATS = [
  { mime: "video/mp4;codecs=avc1", ext: "mp4", name: "H.264 MP4" },
  { mime: "video/mp4", ext: "mp4", name: "MP4" },
  { mime: "video/webm;codecs=h264", ext: "webm", name: "H.264 WebM" },
  { mime: "video/webm;codecs=vp9", ext: "webm", name: "VP9 WebM" },
  { mime: "video/webm", ext: "webm", name: "WebM" },
];
// 1/500 s, in the camera's units of 100 microseconds.
const SHARP_EXPOSURE = 20;
let ws;
let cameraTrack = null;
let cameraNote = "Starting camera...";
let sharpNote = "";
let videoFormat = null;
let recorder = null;
let recordingSession = null;
let uploadNote = "";
// Clock sync: the Pi's clock minus the phone's, from the fastest recent ping.
let clockSamples = [];
let clockOffset = 0;

function send(message) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(message));
}

function go() {
  const value = Number(power.value);
  if (power.value === "" || !Number.isInteger(value) || value < 0 || value > __MAX_SPEED__) {
    field("message", "Enter a whole number from 0 to __MAX_SPEED__");
    return;
  }
  field("message", "");
  send({ speed: value });
}

function showRecording(rec) {
  document.body.classList.toggle("recording", rec.active);
  if (rec.active) {
    const seconds = Math.floor(rec.seconds);
    const clock = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
    field("rec", `● REC ${clock}${recorder ? "" : " · no camera, readings only"}`);
  } else {
    field("rec", uploadNote || rec.last || "");
  }
}

function syncClock(pong) {
  // Standard network time calculation; the Pi's time spent holding the reply
  // is taken out, so only the network delay counts.
  const now = Date.now();
  const roundTrip = (now - pong.t0) - (pong.sent - pong.received);
  const offset = ((pong.received - pong.t0) + (pong.sent - now)) / 2;
  clockSamples = clockSamples.concat([{ roundTrip, offset }]).slice(-8);
  clockOffset = clockSamples.reduce((a, b) => (b.roundTrip < a.roundTrip ? b : a)).offset;
}

function connect() {
  ws = new WebSocket(`${socketBase}/ws`);
  ws.onopen = () => {
    field("status", "Live");
    send({ ping: Date.now() });
  };
  ws.onclose = () => {
    field("status", "Disconnected, motors stopped. Retrying...");
    field("motor", 0);
    stopRecording();
    setTimeout(connect, 1000);
  };
  ws.onmessage = (event) => {
    const r = JSON.parse(event.data);
    if (r.pong) syncClock(r.pong);
    // The Pi decides when a recording runs; the phone follows.
    if (r.recording.active && recordingSession !== r.recording.session) {
      stopRecording();
      startRecording(r.recording.session);
    } else if (!r.recording.active && recordingSession) {
      stopRecording();
    }
    field("motor", r.motor_speed);
    field("rpm", Math.abs(r.rpm).toFixed(1));
    field("cms", Math.abs(r.ms * 100).toFixed(1));
    field("ms", `${Math.abs(r.ms).toFixed(3)} m/s`);
    field("kmh", `${Math.abs(r.kmh).toFixed(2)} km/h`);
    field("distance", `${r.distance_m.toFixed(2)} m`);
    field("count", r.count);
    field("camera", cameraNote + sharpNote);
    showRecording(r.recording);
    document.body.classList.toggle("emergency", r.emergency);
    power.disabled = r.emergency;
  };
}

async function startCamera() {
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    cameraNote = "Camera unavailable: open this page with https:// in Chrome or Safari";
    return;
  }
  try {
    preview.srcObject = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: CAMERA.width },
        height: { ideal: CAMERA.height },
        frameRate: { ideal: CAMERA.fps },
      },
      audio: false,
    });
  } catch (error) {
    cameraNote = `Camera error: ${error.message}. Driving still works.`;
    return;
  }
  cameraTrack = preview.srcObject.getVideoTracks()[0];
  videoFormat = VIDEO_FORMATS.find((format) => MediaRecorder.isTypeSupported(format.mime)) || null;
  const settings = cameraTrack.getSettings();
  const mbps = (CAMERA.bitrate / 1e6).toFixed(0);
  cameraNote = `Camera ready: ${settings.width}×${settings.height}, ${Math.round(settings.frameRate || CAMERA.fps)} fps, records ${videoFormat ? videoFormat.name : "default format"} at ${mbps} Mbps`;
}

function startRecording(session) {
  recordingSession = session;
  if (!cameraTrack) return;
  const chunks = [];
  const format = videoFormat;
  const options = { videoBitsPerSecond: CAMERA.bitrate };
  if (format) options.mimeType = format.mime;
  const mediaRecorder = new MediaRecorder(preview.srcObject, options);
  let startedAt = null;
  mediaRecorder.onstart = () => { startedAt = Date.now() + clockOffset; };
  mediaRecorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
  mediaRecorder.onstop = () => {
    const ext = format ? format.ext : (mediaRecorder.mimeType.includes("mp4") ? "mp4" : "webm");
    upload(session, startedAt, new Blob(chunks, { type: mediaRecorder.mimeType }), ext);
  };
  mediaRecorder.start(1000);
  recorder = mediaRecorder;
}

function stopRecording() {
  if (recorder && recorder.state !== "inactive") recorder.stop();
  recorder = null;
  recordingSession = null;
}

// Sends the video to the Pi, retrying briefly in case the Pi has not yet
// noticed the recording stopped.
function upload(session, startedAt, blob, ext, attempt = 1) {
  const xhr = new XMLHttpRequest();
  xhr.open("POST", `/video/${encodeURIComponent(session)}?start=${startedAt}&ext=${ext}`);
  xhr.upload.onprogress = (event) => {
    uploadNote = `Sending video to the Pi: ${Math.round(100 * event.loaded / event.total)}% of ${(blob.size / 1e6).toFixed(0)} MB`;
  };
  xhr.onload = () => {
    if (xhr.status === 200) {
      uploadNote = "";
    } else if (xhr.status === 404 && attempt < 5) {
      setTimeout(() => upload(session, startedAt, blob, ext, attempt + 1), 1000);
    } else {
      uploadNote = `Video upload failed: ${xhr.responseText}`;
    }
  };
  xhr.onerror = () => {
    if (attempt < 5) {
      setTimeout(() => upload(session, startedAt, blob, ext, attempt + 1), 2000);
    } else {
      uploadNote = "Video upload failed: network error";
    }
  };
  xhr.send(blob);
}

// A short shutter freezes motion, so frames blur less when the car shakes.
// Not every phone lets a web page set it.
async function setSharp(on) {
  sharpNote = "";
  if (!cameraTrack) return;
  const capabilities = cameraTrack.getCapabilities ? cameraTrack.getCapabilities() : {};
  if (!capabilities.exposureTime || !(capabilities.exposureMode || []).includes("manual")) {
    sharpNote = " · Sharp mode is not supported on this phone";
    sharp.checked = false;
    return;
  }
  try {
    if (on) {
      const exposure = Math.min(Math.max(SHARP_EXPOSURE, capabilities.exposureTime.min), capabilities.exposureTime.max);
      await cameraTrack.applyConstraints({ advanced: [{ exposureMode: "manual", exposureTime: exposure }] });
      sharpNote = " · Sharp mode on";
    } else {
      await cameraTrack.applyConstraints({ advanced: [{ exposureMode: "continuous" }] });
    }
  } catch (error) {
    sharpNote = ` · Sharp mode failed: ${error.message}`;
    sharp.checked = false;
  }
}
sharp.onchange = () => setSharp(sharp.checked);

// Keep the phone's screen on, so the camera and controls keep running.
async function keepAwake() {
  try {
    if (navigator.wakeLock) await navigator.wakeLock.request("screen");
  } catch (error) {
    // Not supported or refused; the phone may sleep.
  }
}
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") keepAwake();
});

document.getElementById("go").onclick = go;
power.addEventListener("keydown", (event) => {
  if (event.key === "Enter") go();
});
document.getElementById("stop").onclick = () => send({ speed: 0 });
document.getElementById("reset").onclick = () => send({ reset: true });

function emergencyStop() {
  send({ emergency: true });
}
document.getElementById("emergency").onclick = emergencyStop;
document.getElementById("release").onclick = () => send({ release: true });
// Esc triggers the emergency stop from a keyboard, and so does Space when
// not typing in the power box.
document.addEventListener("keydown", (event) => {
  if (event.code === "Escape" || (event.code === "Space" && event.target !== power)) {
    event.preventDefault();
    emergencyStop();
  }
});

// Tells the Pi the page is still here. Without it, the motors stop.
setInterval(() => send({}), 250);
setInterval(() => send({ ping: Date.now() }), 2000);

connect();
startCamera();
keepAwake();
</script>
</body>
</html>
"""


async def run_command(*command, cwd):
    process = await asyncio.create_subprocess_exec(
        *command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(stderr.decode(errors="replace").strip() or f"{command[0]} failed")
    return stdout.decode()


class Recorder:
    """Logs the car's readings while it drives, then turns the video the
    phone uploads into a dataset matched to those readings."""

    SENSOR_COLUMNS = ["pi_time_ms", "motor_percent", "wheel_rpm", "velocity_cm_s", "distance_m", "pulses"]
    FRAME_COLUMNS = ["frame", "video_time_s", "pi_time_ms"] + SENSOR_COLUMNS[1:]

    def __init__(self, readings):
        self.readings = readings
        self.folder = None
        self.samples = []
        self.logger = None
        self.started_at = 0.0
        self.waiting = {}  # recording name -> (folder, readings, log finished), until its video arrives
        self.saving = set()
        self.last = ""

    @property
    def active(self):
        return self.folder is not None

    async def start(self):
        if self.active:
            return
        self.folder = RECORDINGS_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.samples = []
        self.started_at = time.monotonic()
        self.logger = asyncio.create_task(self._log_readings(self.samples))
        print(f"Recording {self.folder.name}")

    async def _log_readings(self, samples):
        while True:
            r = self.readings()
            samples.append([
                round(now_ms()), r["motor_speed"], round(r["rpm"], 2), round(r["ms"] * 100, 2),
                round(r["distance_m"], 4), r["count"],
            ])
            await asyncio.sleep(SENSOR_INTERVAL_SECONDS)

    async def stop(self):
        if not self.active:
            return
        folder, samples = self.folder, self.samples
        self.folder = None
        finished = asyncio.create_task(self._finish_log(folder, samples, self.logger))
        self.waiting[folder.name] = (folder, samples, finished)
        self.last = f"Stopped {folder.name}, waiting for the video from the phone..."
        print(self.last)

    async def _finish_log(self, folder, samples, logger):
        await asyncio.sleep(SENSOR_TAIL_SECONDS)
        logger.cancel()
        with open(folder / "sensors.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(self.SENSOR_COLUMNS)
            writer.writerows(samples)

    def expects_video(self, name):
        return name in self.waiting

    async def save_video(self, name, started_ms, ext, chunks):
        folder, samples, finished = self.waiting.pop(name)
        upload = folder / f"upload.{ext}"
        with open(upload, "wb") as file:
            async for chunk in chunks:
                await asyncio.to_thread(file.write, chunk)
        await finished
        task = asyncio.create_task(self._build_dataset(folder, samples, upload, started_ms))
        self.saving.add(task)
        task.add_done_callback(self.saving.discard)

    async def _build_dataset(self, folder, samples, upload, started_ms):
        self.last = f"Saving {folder.name}: extracting frames..."
        print(self.last)
        try:
            # Copy the phone's video into an MKV file without re-encoding, which
            # also gives it a proper index for seeking.
            await run_command("ffmpeg", "-loglevel", "error", "-y", "-i", upload.name, "-c", "copy", "video.mkv", cwd=folder)
            upload.unlink()
            (folder / "frames").mkdir(exist_ok=True)
            # Every frame, exactly once, as a high-quality JPEG.
            await run_command(
                "ffmpeg", "-loglevel", "error", "-i", "video.mkv", "-vsync", "passthrough",
                "-q:v", "2", "frames/%06d.jpg", cwd=folder,
            )
            probe = await run_command(
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", "video.mkv", cwd=folder,
            )
        except RuntimeError as error:
            self.last = f"Saving {folder.name} failed: {error}"
            print(self.last)
            return

        frame_times = [float(value) for value in probe.split() if value.strip(",") not in ("", "N/A")]
        names = sorted(path.name for path in (folder / "frames").iterdir())
        if not started_ms and samples:
            started_ms = samples[0][0]
        sample_times = [sample[0] for sample in samples]
        with open(folder / "frames.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(self.FRAME_COLUMNS)
            for name, video_time in zip(names, frame_times):
                taken_ms = started_ms + video_time * 1000
                writer.writerow([name, f"{video_time:.3f}", round(taken_ms)] + self._nearest(samples, sample_times, taken_ms))

        count = min(len(names), len(frame_times))
        seconds = frame_times[count - 1] - frame_times[0] if count > 1 else 0
        fps = (count - 1) / seconds if seconds else 0
        self.last = f"Saved recordings/{folder.name}: {count} frames, {seconds:.1f} s, {fps:.0f} fps"
        print(self.last)

    @staticmethod
    def _nearest(samples, sample_times, taken_ms):
        """The readings logged closest to when a frame was taken."""
        if not samples:
            return [""] * 5
        index = bisect.bisect_left(sample_times, taken_ms)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(samples)]
        best = min(candidates, key=lambda i: abs(sample_times[i] - taken_ms))
        return samples[best][1:]

    async def wait_saved(self):
        pending = list(self.saving) + [finished for _, _, finished in self.waiting.values()]
        if pending:
            await asyncio.gather(*pending)

    def status(self):
        seconds = time.monotonic() - self.started_at if self.active else 0.0
        session = self.folder.name if self.active else None
        return {"active": self.active, "session": session, "seconds": seconds, "last": self.last}


def create_app(car, encoder, recorder, camera_settings):
    page_html = PAGE.replace("__MAX_SPEED__", str(MAX_SPEED))
    for name, value in camera_settings.items():
        page_html = page_html.replace(f"__{name}__", str(value))
    last_message = {"time": time.monotonic()}

    async def set_speed(speed):
        await asyncio.to_thread(car.set_speed, speed)

    async def stop_driving():
        await set_speed(0)
        await recorder.stop()

    async def watchdog():
        while True:
            await asyncio.sleep(0.2)
            quiet = time.monotonic() - last_message["time"] > COMMAND_TIMEOUT_SECONDS
            if quiet and (car.speed or recorder.active):
                await stop_driving()
                print("Page went quiet, motors stopped")

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(watchdog())
        yield
        task.cancel()
        await stop_driving()
        await recorder.wait_saved()

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    async def page():
        return HTMLResponse(page_html)

    @app.post("/video/{name}")
    async def video(name: str, request: Request, start: float = 0, ext: str = ""):
        if ext not in VIDEO_TYPES or not recorder.expects_video(name):
            raise HTTPException(status_code=404, detail="No recording is waiting for this video")
        await recorder.save_video(name, start, ext, request.stream())
        return {"saved": name}

    @app.websocket("/ws")
    async def stream(ws: WebSocket):
        await ws.accept()
        ping = {}

        async def receive_commands():
            async for text in ws.iter_text():
                last_message["time"] = time.monotonic()
                command = json.loads(text)
                if "ping" in command:
                    ping.update(t0=command["ping"], received=now_ms())
                elif command.get("emergency"):
                    await asyncio.to_thread(car.emergency_stop)
                    await recorder.stop()
                    print("EMERGENCY STOP, motors locked until released")
                elif command.get("release"):
                    car.release()
                    print("Emergency stop released")
                elif command.get("reset"):
                    encoder.reset()
                elif "speed" in command:
                    speed = int(command["speed"])
                    if speed == 0:
                        await stop_driving()
                        print("Motors stopped")
                    elif 0 < speed <= MAX_SPEED:
                        await set_speed(speed)
                        # The motors ignore new power after an emergency stop.
                        if car.speed:
                            await recorder.start()
                            print(f"Motors at {speed}%")

        receiver = asyncio.create_task(receive_commands())
        try:
            while not receiver.done():
                message = {
                    **encoder.reading(),
                    "motor_speed": car.speed,
                    "emergency": car.emergency,
                    "recording": recorder.status(),
                }
                if ping:
                    message["pong"] = {**ping, "sent": now_ms()}
                    ping.clear()
                await ws.send_json(message)
                await asyncio.sleep(UPDATE_SECONDS)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            receiver.cancel()
            await stop_driving()

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Drive the car from one phone page that also films with the phone's camera.")
    parser.add_argument("--port", type=int, default=8443, help="HTTPS port (default: 8443)")
    parser.add_argument("--ppr", type=int, default=20, help="encoder pulses per wheel revolution (default: 20)")
    parser.add_argument("--wheel-mm", type=float, default=65, help="wheel diameter in mm (default: 65)")
    parser.add_argument("--glitch-us", type=int, default=100, help="ignore encoder pulses shorter than this (default: 100)")
    parser.add_argument("--camera-width", type=int, default=1920, help="video width to ask the phone for (default: 1920)")
    parser.add_argument("--camera-height", type=int, default=1080, help="video height to ask the phone for (default: 1080)")
    parser.add_argument("--camera-fps", type=int, default=30, help="video frame rate to ask the phone for (default: 30)")
    parser.add_argument("--video-mbps", type=float, default=12, help="video quality in megabits per second (default: 12)")
    parser.add_argument("--reverse-front", action="store_true", help="use if the front wheels spin backward")
    parser.add_argument("--reverse-rear", action="store_true", help="use if the rear wheels spin backward")
    return parser.parse_args()


def main(local_only=False):
    """Serve the page. With local_only, only this Pi can open it."""
    args = parse_args()

    pi = pigpio.pi()
    if not pi.connected:
        sys.exit("pigpio daemon is not running. Start it with: sudo pigpiod -l")

    car = Car(pi, args.reverse_front, args.reverse_rear)
    encoder = Encoder(pi, pins.ENCODER_A, None, args.ppr, args.wheel_mm, args.glitch_us)
    recorder = Recorder(lambda: {**encoder.reading(), "motor_speed": car.speed})
    camera_settings = {
        "CAMERA_WIDTH": args.camera_width,
        "CAMERA_HEIGHT": args.camera_height,
        "CAMERA_FPS": args.camera_fps,
        "VIDEO_BITRATE": round(args.video_mbps * 1e6),
    }
    ips = local_ips()
    cert, key = ensure_certificate(ips)

    # Shut down cleanly, stopping the motors, if the terminal closes or the
    # app is killed. uvicorn re-raises SIGTERM once it has shut down.
    signal.signal(signal.SIGHUP, lambda signum, frame: os.kill(os.getpid(), signal.SIGTERM))
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))

    if local_only:
        host = "127.0.0.1"
        print(f"Open this in a browser on the Pi: https://localhost:{args.port}")
    else:
        host = "0.0.0.0"
        print("Open this one link on your phone (same Wi-Fi as the Pi):")
        for ip in ips:
            print(f"  https://{ip}:{args.port}")
    print("Accept the security warning, then allow the camera.")
    try:
        uvicorn.run(
            create_app(car, encoder, recorder, camera_settings), host=host, port=args.port,
            log_level="warning", ssl_certfile=str(cert), ssl_keyfile=str(key),
        )
    finally:
        car.stop()
        encoder.close()
        pi.stop()


if __name__ == "__main__":
    main()
