#!/usr/bin/env python3
"""Print every level change on the encoder pin, to check the sensor itself.

Stop encoder_app.py first, then run this and turn the wheel. Press Ctrl+C to stop.
Requires the pigpio daemon: sudo pigpiod -l
"""

import sys
import time

import pigpio

import pins

# Changes closer together than this are almost certainly electrical bounce.
BOUNCE_US = 1000


def main():
    pi = pigpio.pi()
    if not pi.connected:
        sys.exit("pigpio daemon is not running. Start it with: sudo pigpiod -l")

    pin = pins.ENCODER_A
    pi.set_mode(pin, pigpio.INPUT)
    pi.set_pull_up_down(pin, pigpio.PUD_UP)
    pi.set_glitch_filter(pin, 0)

    last_tick = None
    rising = 0
    bounces = 0

    def on_change(gpio, level, tick):
        nonlocal last_tick, rising, bounces
        if level > 1:
            return
        gap = "" if last_tick is None else f"{pigpio.tickDiff(last_tick, tick) / 1000:10.1f} ms since last"
        note = ""
        if last_tick is not None and pigpio.tickDiff(last_tick, tick) < BOUNCE_US:
            bounces += 1
            note = "  <- bounce"
        if level == 1:
            rising += 1
        last_tick = tick
        print(f"{'HIGH' if level else 'LOW '}  {gap}{note}", flush=True)

    print(f"Watching GPIO{pin}, now {'HIGH' if pi.read(pin) else 'LOW'}. Turn the wheel slowly. Ctrl+C to stop.")
    callback = pi.callback(pin, pigpio.EITHER_EDGE, on_change)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        callback.cancel()
        pi.stop()
    print(f"\n{rising} pulses, {bounces} bounces")


if __name__ == "__main__":
    main()
