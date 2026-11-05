#!/usr/bin/env python3
"""Show a wheel encoder's speed live on your phone.

Rotate the wheel by hand and watch the speed update. Works with a single-output
speed sensor (such as an LM393 slotted disc) or, with --quadrature, a two-output
encoder that also reports direction.
Requires the pigpio daemon: sudo pigpiod -l
"""

import argparse
import asyncio
import math
import subprocess
import sys
import threading
from collections import deque

import pigpio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import pins

SPEED_WINDOW_US = 500_000
# The wheel counts as stopped when no pulse arrives for this long. Longer
# reads slower speeds but takes longer to show 0 after the wheel stops.
STOP_TIMEOUT_US = 2_000_000
UPDATE_SECONDS = 0.1

# Quadrature state is (A << 1) | B. A leading B counts as forward.
QUADRATURE_STEPS = {
    (0b00, 0b10): 1, (0b10, 0b11): 1, (0b11, 0b01): 1, (0b01, 0b00): 1,
    (0b10, 0b00): -1, (0b11, 0b10): -1, (0b01, 0b11): -1, (0b00, 0b01): -1,
}

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenWeeder Encoder</title>
<style>
  body { margin: 0; padding: 16px; background: #111; color: #eee; font: 18px sans-serif; text-align: center; }
  .rpm { font-size: 72px; font-weight: bold; font-variant-numeric: tabular-nums; margin-top: 24px; }
  .unit { color: #999; margin-bottom: 24px; }
  table { margin: 0 auto 24px; border-collapse: collapse; text-align: left; }
  td { padding: 6px 12px; }
  td:last-child { text-align: right; font-variant-numeric: tabular-nums; }
  button { font-size: 18px; padding: 12px 32px; border-radius: 8px; border: 0; }
  #status { color: #999; }
</style>
</head>
<body>
<div class="rpm" id="rpm">0</div>
<div class="unit">RPM</div>
<table>
  <tr><td>Speed</td><td id="kmh">0.00 km/h</td></tr>
  <tr><td>Speed</td><td id="ms">0.000 m/s</td></tr>
  <tr><td>Direction</td><td id="direction">-</td></tr>
  <tr><td>Distance</td><td id="distance">0.00 m</td></tr>
  <tr><td>Revolutions</td><td id="revolutions">0.00</td></tr>
  <tr><td>Pulses</td><td id="count">0</td></tr>
</table>
<button id="reset">Reset distance</button>
<p id="status">Connecting...</p>
<script>
const field = (id, value) => { document.getElementById(id).textContent = value; };
let ws;

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => field("status", "Live");
  ws.onclose = () => {
    field("status", "Disconnected, retrying...");
    setTimeout(connect, 1000);
  };
  ws.onmessage = (event) => {
    const r = JSON.parse(event.data);
    field("rpm", Math.abs(r.rpm).toFixed(0));
    field("kmh", `${Math.abs(r.kmh).toFixed(2)} km/h`);
    field("ms", `${Math.abs(r.ms).toFixed(3)} m/s`);
    field("direction", r.direction || "-");
    field("distance", `${r.distance_m.toFixed(2)} m`);
    field("revolutions", r.revolutions.toFixed(2));
    field("count", r.count);
  };
}

document.getElementById("reset").onclick = () => {
  if (ws.readyState === WebSocket.OPEN) ws.send("reset");
};

connect();
</script>
</body>
</html>
"""


class Encoder:
    """Counts encoder pulses in the pigpio daemon, timed to the microsecond."""

    def __init__(self, pi, pin_a, pin_b, pulses_per_rev, wheel_mm, glitch_us):
        self.pi = pi
        self.pin_a = pin_a
        self.quadrature = pin_b is not None
        self.counts_per_rev = pulses_per_rev * (4 if self.quadrature else 1)
        self.wheel_m = wheel_mm / 1000
        self.lock = threading.Lock()
        self.count = 0
        self.events = deque()

        pins = [pin_a, pin_b] if self.quadrature else [pin_a]
        for pin in pins:
            pi.set_mode(pin, pigpio.INPUT)
            pi.set_pull_up_down(pin, pigpio.PUD_UP)
            pi.set_glitch_filter(pin, glitch_us)

        if self.quadrature:
            self.state = pi.read(pin_a) << 1 | pi.read(pin_b)
            self.callbacks = [pi.callback(pin, pigpio.EITHER_EDGE, self._on_edge) for pin in pins]
        else:
            self.callbacks = [pi.callback(pin_a, pigpio.RISING_EDGE, self._on_pulse)]

    def _on_pulse(self, gpio, level, tick):
        self._record(tick, 1)

    def _on_edge(self, gpio, level, tick):
        if level > 1:
            return
        if gpio == self.pin_a:
            state = level << 1 | self.state & 0b01
        else:
            state = self.state & 0b10 | level
        step = QUADRATURE_STEPS.get((self.state, state), 0)
        self.state = state
        if step:
            self._record(tick, step)

    def _record(self, tick, step):
        with self.lock:
            self.count += step
            self.events.append((tick, step))

    def reset(self):
        with self.lock:
            self.count = 0

    def counts_per_second(self):
        now = self.pi.get_current_tick()
        with self.lock:
            while self.events and pigpio.tickDiff(self.events[0][0], now) > STOP_TIMEOUT_US:
                self.events.popleft()
            events = list(self.events)

        if len(events) < 2:
            return 0.0

        # Average over the pulses in the last window. When the wheel is slow,
        # reach further back so there are at least two pulses to time.
        last_tick = events[-1][0]
        steps = 0
        for index in range(len(events) - 1, 0, -1):
            steps += events[index][1]
            span = pigpio.tickDiff(events[index - 1][0], last_tick)
            if span >= SPEED_WINDOW_US:
                break
        if span <= 0:
            return 0.0
        rate = steps / (span / 1e6)
        # When the wheel slows, no new pulse arrives to lower the rate, so cap it
        # by the time since the last pulse.
        since_last = pigpio.tickDiff(events[-1][0], now) / 1e6
        if since_last > 0:
            rate = math.copysign(min(abs(rate), 1 / since_last), rate)
        return rate

    def reading(self):
        revs_per_second = self.counts_per_second() / self.counts_per_rev
        meters_per_second = revs_per_second * math.pi * self.wheel_m
        revolutions = self.count / self.counts_per_rev

        direction = None
        if self.quadrature:
            if revs_per_second > 0:
                direction = "forward"
            elif revs_per_second < 0:
                direction = "backward"
            else:
                direction = "stopped"

        return {
            "rpm": revs_per_second * 60,
            "ms": meters_per_second,
            "kmh": meters_per_second * 3.6,
            "direction": direction,
            "revolutions": revolutions,
            "distance_m": revolutions * math.pi * self.wheel_m,
            "count": self.count,
        }

    def close(self):
        for callback in self.callbacks:
            callback.cancel()


def create_app(encoder):
    app = FastAPI()

    @app.get("/")
    async def page():
        return HTMLResponse(PAGE)

    @app.websocket("/ws")
    async def stream(ws: WebSocket):
        await ws.accept()

        async def receive_commands():
            async for text in ws.iter_text():
                if text == "reset":
                    encoder.reset()

        receiver = asyncio.create_task(receive_commands())
        try:
            while not receiver.done():
                await ws.send_json(encoder.reading())
                await asyncio.sleep(UPDATE_SECONDS)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            receiver.cancel()

    return app


def local_ips():
    output = subprocess.run(["hostname", "-I"], capture_output=True, text=True, check=True).stdout
    return [ip for ip in output.split() if ":" not in ip]


def parse_args():
    parser = argparse.ArgumentParser(description="Show a wheel encoder's speed live on your phone.")
    parser.add_argument("--quadrature", action="store_true", help="encoder has two outputs, A and B")
    parser.add_argument("--ppr", type=int, default=20, help="pulses per wheel revolution on one output (default: 20)")
    parser.add_argument("--wheel-mm", type=float, default=65, help="wheel diameter in mm (default: 65)")
    parser.add_argument("--glitch-us", type=int, default=100, help="ignore pulses shorter than this (default: 100)")
    parser.add_argument("--port", type=int, default=8000, help="web port (default: 8000)")
    return parser.parse_args()


def main():
    args = parse_args()

    pi = pigpio.pi()
    if not pi.connected:
        sys.exit("pigpio daemon is not running. Start it with: sudo pigpiod -l")

    encoder = Encoder(pi, pins.ENCODER_A, pins.ENCODER_B if args.quadrature else None, args.ppr, args.wheel_mm, args.glitch_us)

    print("Open this on your phone (same Wi-Fi as the Pi):")
    for ip in local_ips():
        print(f"  http://{ip}:{args.port}")
    try:
        uvicorn.run(create_app(encoder), host="0.0.0.0", port=args.port, log_level="warning")
    finally:
        encoder.close()
        pi.stop()


if __name__ == "__main__":
    main()
