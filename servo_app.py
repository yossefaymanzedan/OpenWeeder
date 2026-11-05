#!/usr/bin/env python3
"""Turn the SG90 servo from your phone.

The servo holds its angle until the app quits.
Requires the pigpio daemon: sudo pigpiod -l
"""

import argparse
import asyncio
import json
import os
import signal
import sys

import pigpio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import pins
from phone_camera import local_ips
from servo import MAX_ANGLE, angle_to_pulse

UPDATE_SECONDS = 0.1

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenWeeder Servo</title>
<style>
  body { margin: 0; padding: 16px; background: #111; color: #eee; font: 18px sans-serif; text-align: center; }
  section { margin: 16px auto; max-width: 420px; padding: 16px; border: 1px solid #333; border-radius: 8px; }
  .big { font-size: 56px; font-weight: bold; font-variant-numeric: tabular-nums; }
  .unit { color: #999; }
  input[type=range] { width: 100%; margin: 16px 0; }
  .row { display: flex; gap: 8px; margin-top: 8px; }
  .row input { flex: 1; min-width: 0; font-size: 24px; padding: 8px; border-radius: 8px; border: 0; text-align: center; }
  button { flex: 1; font-size: 18px; padding: 14px 4px; border-radius: 8px; border: 0; }
  #set { background: #1565c0; color: #fff; font-weight: bold; }
  #message { min-height: 1.4em; color: #ef9a9a; }
  #status { color: #999; }
</style>
</head>
<body>
<section>
  <div class="big" id="angle">-</div>
  <div class="unit">degrees</div>
  <input type="range" id="slider" min="0" max="__MAX_ANGLE__" step="1" value="90">
  <div class="row">
    <button data-angle="0">0°</button>
    <button data-angle="45">45°</button>
    <button data-angle="90">90°</button>
    <button data-angle="135">135°</button>
    <button data-angle="180">180°</button>
  </div>
  <div class="row">
    <input type="number" id="exact" min="0" max="__MAX_ANGLE__" step="1" inputmode="numeric" placeholder="0 to __MAX_ANGLE__">
    <button id="set">Set</button>
  </div>
  <div id="message"></div>
</section>
<p id="status">Connecting...</p>
<script>
const field = (id, value) => { document.getElementById(id).textContent = value; };
const slider = document.getElementById("slider");
const exact = document.getElementById("exact");
let ws;
let dragging = false;

function send(message) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(message));
}

function setAngle(angle) {
  slider.value = angle;
  send({ angle });
}

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => field("status", "Live");
  ws.onclose = () => {
    field("status", "Disconnected. Retrying...");
    setTimeout(connect, 1000);
  };
  ws.onmessage = (event) => {
    const r = JSON.parse(event.data);
    field("angle", r.angle === null ? "-" : r.angle);
    if (r.angle !== null && !dragging) slider.value = r.angle;
  };
}

slider.oninput = () => send({ angle: Number(slider.value) });
slider.onpointerdown = () => { dragging = true; };
window.onpointerup = window.onpointercancel = () => { dragging = false; };
document.querySelectorAll("[data-angle]").forEach((button) => {
  button.onclick = () => setAngle(Number(button.dataset.angle));
});

function setExact() {
  const value = Number(exact.value);
  if (exact.value === "" || !Number.isInteger(value) || value < 0 || value > __MAX_ANGLE__) {
    field("message", "Enter a whole number from 0 to __MAX_ANGLE__");
    return;
  }
  field("message", "");
  setAngle(value);
}
document.getElementById("set").onclick = setExact;
exact.addEventListener("keydown", (event) => {
  if (event.key === "Enter") setExact();
});

connect();
</script>
</body>
</html>
"""


class Servo:
    def __init__(self, pi, min_us, max_us):
        self.pi = pi
        self.min_us = min_us
        self.max_us = max_us
        self.angle = None

    def set_angle(self, angle):
        self.pi.set_servo_pulsewidth(pins.SERVO, angle_to_pulse(angle, self.min_us, self.max_us))
        self.angle = angle

    def close(self):
        self.pi.set_servo_pulsewidth(pins.SERVO, 0)


def create_app(servo):
    page_html = PAGE.replace("__MAX_ANGLE__", str(MAX_ANGLE))
    app = FastAPI()

    @app.get("/")
    async def page():
        return HTMLResponse(page_html)

    @app.websocket("/ws")
    async def stream(ws: WebSocket):
        await ws.accept()

        async def receive_commands():
            async for text in ws.iter_text():
                angle = json.loads(text).get("angle")
                if isinstance(angle, (int, float)) and 0 <= angle <= MAX_ANGLE:
                    servo.set_angle(round(angle))

        receiver = asyncio.create_task(receive_commands())
        try:
            while not receiver.done():
                await ws.send_json({"angle": servo.angle})
                await asyncio.sleep(UPDATE_SECONDS)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            receiver.cancel()

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Turn the servo from your phone.")
    parser.add_argument("--port", type=int, default=8002, help="web port (default: 8002)")
    parser.add_argument("--min-us", type=int, default=500, help="pulse width at 0 degrees (default: 500)")
    parser.add_argument("--max-us", type=int, default=2500, help="pulse width at 180 degrees (default: 2500)")
    return parser.parse_args()


def main():
    args = parse_args()

    pi = pigpio.pi()
    if not pi.connected:
        sys.exit("pigpio daemon is not running. Start it with: sudo pigpiod -l")

    servo = Servo(pi, args.min_us, args.max_us)

    # Release the servo cleanly if the terminal closes or the app is killed.
    # uvicorn re-raises SIGTERM once it has shut down.
    signal.signal(signal.SIGHUP, lambda signum, frame: os.kill(os.getpid(), signal.SIGTERM))
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))

    print("Open this on your phone (same Wi-Fi as the Pi):")
    for ip in local_ips():
        print(f"  http://{ip}:{args.port}")
    try:
        uvicorn.run(create_app(servo), host="0.0.0.0", port=args.port, log_level="warning")
    finally:
        servo.close()
        pi.stop()


if __name__ == "__main__":
    main()
