#!/usr/bin/env python3
"""Stream a phone camera to the Pi over WebSockets and measure latency.

The phone opens an HTTPS link, because browsers only allow camera access on
secure pages. The Pi views the stream on http://localhost, which is not
reachable from the network.

Latency: the phone syncs to the Pi's clock with ping/pong messages, then stamps
each frame on that clock, so the Pi can subtract timestamps directly.
"""

import argparse
import asyncio
import json
import secrets
import ssl
import struct
import subprocess
import time
import webbrowser
from pathlib import Path

from aiohttp import WSMsgType, web

CERT_DIR = Path(__file__).resolve().parent / ".certs"
MAX_FRAME_BYTES = 5 * 1024 * 1024
STATS_SECONDS = 2

# All times are milliseconds on the Pi's clock.
PHONE_HEADER = struct.Struct("<dd")  # capture time, round trip
VIEWER_HEADER = struct.Struct("<ddd")  # capture time, received time, round trip

PHONE_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenWeeder Camera</title>
<style>
  body { margin: 0; background: #111; color: #eee; font: 16px sans-serif; text-align: center; }
  video { width: 100%; max-height: 80vh; background: #000; }
  p { padding: 0 16px; }
</style>
</head>
<body>
<video id="video" autoplay playsinline muted></video>
<p id="status">Starting...</p>
<script>
const video = document.getElementById("video");
const statusText = document.getElementById("status");
const canvas = document.createElement("canvas");
const context = canvas.getContext("2d");
const key = new URLSearchParams(location.search).get("key");
const SYNC_SAMPLES = 10;
const FRAME_INTERVAL_MS = 66;
const JPEG_QUALITY = 0.6;

let ws;
let samples = [];
let offset = null;
let rtt = 0;
let awaitingAck = false;

function connect() {
  ws = new WebSocket(`wss://${location.host}/ws?key=${key}`);
  ws.onopen = () => { statusText.textContent = "Connected"; };
  ws.onclose = () => {
    statusText.textContent = "Disconnected, retrying...";
    samples = [];
    offset = null;
    awaitingAck = false;
    setTimeout(connect, 2000);
  };
  ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.ack) {
      awaitingAck = false;
    } else if (message.pong) {
      handlePong(message);
    }
  };
}

// The fastest recent ping is the one least delayed by frames ahead of it.
function handlePong(message) {
  const now = Date.now();
  samples.push({
    rtt: now - message.pong,
    offset: message.server - (message.pong + now) / 2,
  });
  samples = samples.slice(-SYNC_SAMPLES);
  const best = samples.reduce((a, b) => (b.rtt < a.rtt ? b : a));
  rtt = best.rtt;
  offset = best.offset;
  statusText.textContent = `Streaming, round trip ${rtt} ms`;
}

function sendPing() {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ ping: Date.now() }));
  }
}

function sendFrame() {
  if (ws.readyState !== WebSocket.OPEN || awaitingAck || offset === null || !video.videoWidth) {
    return;
  }
  awaitingAck = true;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const capturedAt = Date.now() + offset;
  context.drawImage(video, 0, 0);
  canvas.toBlob((jpeg) => {
    ws.send(new Blob([new Float64Array([capturedAt, rtt]), jpeg]));
  }, "image/jpeg", JPEG_QUALITY);
}

async function start() {
  if (!navigator.mediaDevices) {
    statusText.textContent = "Camera is unavailable. Open this page over https.";
    return;
  }
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
  } catch (error) {
    statusText.textContent = `Camera error: ${error.message}`;
    return;
  }
  connect();
  setInterval(sendPing, 500);
  setInterval(sendFrame, FRAME_INTERVAL_MS);
}

start();
</script>
</body>
</html>
"""

VIEWER_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenWeeder Viewer</title>
<style>
  body { margin: 0; background: #111; color: #eee; font: 16px sans-serif; text-align: center; }
  canvas { max-width: 100%; max-height: 80vh; background: #000; }
  table { margin: 12px auto; border-collapse: collapse; text-align: left; }
  td { padding: 2px 12px; }
  td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
<canvas id="view" width="640" height="480"></canvas>
<p id="status">Waiting for phone...</p>
<table>
  <tr><td>Total, capture to display</td><td id="total">-</td></tr>
  <tr><td>Phone to Pi</td><td id="network">-</td></tr>
  <tr><td>Pi to display</td><td id="display">-</td></tr>
  <tr><td>Round trip</td><td id="rtt">-</td></tr>
  <tr><td>Frame rate</td><td id="fps">-</td></tr>
</table>
<script>
const view = document.getElementById("view");
const context = view.getContext("2d");
const statusText = document.getElementById("status");
const field = (id, value) => { document.getElementById(id).textContent = value; };
const WAITING_AFTER_MS = 2000;

let frameTimes = [];
let lastFrame = 0;

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onclose = () => setTimeout(connect, 2000);
  ws.onmessage = async (event) => {
    const [capturedAt, receivedAt, rtt] = new Float64Array(event.data, 0, 3);
    const jpeg = new Blob([event.data.slice(24)], { type: "image/jpeg" });
    const bitmap = await createImageBitmap(jpeg);
    view.width = bitmap.width;
    view.height = bitmap.height;
    context.drawImage(bitmap, 0, 0);
    bitmap.close();

    const now = Date.now();
    lastFrame = now;
    frameTimes = frameTimes.filter((time) => now - time < 1000);
    frameTimes.push(now);
    statusText.textContent = "Streaming";
    field("total", `${Math.round(now - capturedAt)} ms`);
    field("network", `${Math.round(receivedAt - capturedAt)} ms`);
    field("display", `${Math.round(now - receivedAt)} ms`);
    field("rtt", `${Math.round(rtt)} ms`);
    field("fps", `${frameTimes.length}`);
    ws.send("ack");
  };
}

setInterval(() => {
  if (Date.now() - lastFrame > WAITING_AFTER_MS) {
    statusText.textContent = "Waiting for phone...";
  }
}, 1000);

connect();
</script>
</body>
</html>
"""


def now_ms():
    return time.time() * 1000


class LatestFrame:
    """Holds the newest frame, so a slow viewer skips frames instead of lagging."""

    def __init__(self):
        self.condition = asyncio.Condition()
        self.sequence = 0
        self.jpeg = None
        self.captured_at = 0
        self.received_at = 0
        self.rtt = 0
        self.latencies = []

    async def publish(self, jpeg, captured_at, received_at, rtt):
        async with self.condition:
            self.sequence += 1
            self.jpeg = jpeg
            self.captured_at = captured_at
            self.received_at = received_at
            self.rtt = rtt
            self.latencies.append(received_at - captured_at)
            self.condition.notify_all()

    async def wait_after(self, sequence):
        async with self.condition:
            await self.condition.wait_for(lambda: self.sequence != sequence)
            return self.sequence


async def phone_page(request):
    if not secrets.compare_digest(request.query.get("key", ""), request.app["key"]):
        raise web.HTTPForbidden(text="Wrong or missing key")
    return web.Response(text=PHONE_PAGE, content_type="text/html")


async def phone_socket(request):
    if not secrets.compare_digest(request.query.get("key", ""), request.app["key"]):
        raise web.HTTPForbidden(text="Wrong or missing key")

    latest = request.app["latest"]
    ws = web.WebSocketResponse(heartbeat=10, max_msg_size=MAX_FRAME_BYTES)
    await ws.prepare(request)
    print("Phone connected")

    async for message in ws:
        if message.type == WSMsgType.TEXT:
            sent_at = json.loads(message.data)["ping"]
            await ws.send_json({"pong": sent_at, "server": now_ms()})
        elif message.type == WSMsgType.BINARY:
            received_at = now_ms()
            jpeg = message.data[PHONE_HEADER.size:]
            if jpeg.startswith(b"\xff\xd8"):
                captured_at, rtt = PHONE_HEADER.unpack_from(message.data)
                await latest.publish(jpeg, captured_at, received_at, rtt)
            await ws.send_json({"ack": True})

    print("Phone disconnected")
    return ws


async def viewer_page(request):
    return web.Response(text=VIEWER_PAGE, content_type="text/html")


async def viewer_socket(request):
    latest = request.app["latest"]
    ws = web.WebSocketResponse(heartbeat=10)
    await ws.prepare(request)

    sequence = 0
    try:
        while True:
            sequence = await latest.wait_after(sequence)
            header = VIEWER_HEADER.pack(latest.captured_at, latest.received_at, latest.rtt)
            await ws.send_bytes(header + latest.jpeg)
            # The viewer asks for the next frame only after drawing this one.
            if (await ws.receive()).type != WSMsgType.TEXT:
                break
    except ConnectionResetError:
        pass
    return ws


async def print_stats(latest):
    while True:
        await asyncio.sleep(STATS_SECONDS)
        latencies, latest.latencies = latest.latencies, []
        if latencies:
            average = sum(latencies) / len(latencies)
            fps = len(latencies) / STATS_SECONDS
            print(f"{fps:.1f} fps | phone to Pi {average:.0f} ms | round trip {latest.rtt:.0f} ms")


def local_ips():
    output = subprocess.run(["hostname", "-I"], capture_output=True, text=True, check=True).stdout
    return [ip for ip in output.split() if ":" not in ip]


def ensure_certificate(ips):
    """Create a self-signed certificate, replacing it if the Pi's IPs changed."""
    cert = CERT_DIR / "cert.pem"
    key = CERT_DIR / "key.pem"
    names_file = CERT_DIR / "names.txt"
    names = ",".join(["DNS:localhost"] + [f"IP:{ip}" for ip in ips])

    if cert.exists() and key.exists() and names_file.exists() and names_file.read_text() == names:
        return cert, key

    CERT_DIR.mkdir(mode=0o700, exist_ok=True)
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(cert), "-days", "825",
            "-subj", "/CN=OpenWeeder", "-addext", f"subjectAltName={names}",
        ],
        capture_output=True,
        check=True,
    )
    key.chmod(0o600)
    names_file.write_text(names)
    return cert, key


async def start_site(app, host, port, ssl_context=None):
    runner = web.AppRunner(app, shutdown_timeout=1)
    await runner.setup()
    await web.TCPSite(runner, host, port, ssl_context=ssl_context).start()
    return runner


async def run(args):
    latest = LatestFrame()
    ips = local_ips()
    key = secrets.token_hex(4)

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(*ensure_certificate(ips))

    phone_app = web.Application()
    phone_app["latest"] = latest
    phone_app["key"] = key
    phone_app.add_routes([web.get("/", phone_page), web.get("/ws", phone_socket)])

    viewer_app = web.Application()
    viewer_app["latest"] = latest
    viewer_app.add_routes([web.get("/", viewer_page), web.get("/ws", viewer_socket)])

    runners = [
        await start_site(phone_app, "0.0.0.0", args.phone_port, ssl_context),
        await start_site(viewer_app, "127.0.0.1", args.viewer_port),
    ]

    viewer_url = f"http://localhost:{args.viewer_port}"
    print("Open this on your phone (same Wi-Fi as the Pi):")
    for ip in ips:
        print(f"  https://{ip}:{args.phone_port}/?key={key}")
    print(f"View on the Pi: {viewer_url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(viewer_url)

    try:
        await print_stats(latest)
    finally:
        for runner in runners:
            await runner.cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="Stream a phone camera to the Pi.")
    parser.add_argument("--phone-port", type=int, default=8443, help="HTTPS port for the phone (default: 8443)")
    parser.add_argument("--viewer-port", type=int, default=8080, help="local port for the viewer (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the viewer automatically")
    return parser.parse_args()


def main():
    try:
        asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
