#!/usr/bin/env python3
"""Control an SG90 180-degree servo from the terminal.

Type an angle from 0 to 180 and press Enter. Type q to quit.
Requires the pigpio daemon: sudo pigpiod -l
"""

import argparse
import sys

import pigpio

import pins

MAX_ANGLE = 180


def angle_to_pulse(angle, min_us, max_us):
    return min_us + (max_us - min_us) * angle / MAX_ANGLE


def parse_args():
    parser = argparse.ArgumentParser(description="Control an SG90 servo from the terminal.")
    parser.add_argument("--pin", type=int, default=pins.SERVO, help=f"BCM GPIO number (default: {pins.SERVO})")
    parser.add_argument("--min-us", type=int, default=500, help="pulse width at 0 degrees (default: 500)")
    parser.add_argument("--max-us", type=int, default=2500, help="pulse width at 180 degrees (default: 2500)")
    return parser.parse_args()


def main():
    args = parse_args()

    pi = pigpio.pi()
    if not pi.connected:
        sys.exit("pigpio daemon is not running. Start it with: sudo pigpiod -l")

    print(f"Servo on GPIO{args.pin}. Enter an angle from 0 to {MAX_ANGLE}, or q to quit.")
    try:
        while True:
            text = input("angle> ").strip().lower()
            if text in ("q", "quit", "exit"):
                break
            if not text:
                continue

            try:
                angle = float(text)
            except ValueError:
                print(f"Not a number: {text}")
                continue
            if not 0 <= angle <= MAX_ANGLE:
                print(f"Angle must be between 0 and {MAX_ANGLE}")
                continue

            pulse = angle_to_pulse(angle, args.min_us, args.max_us)
            pi.set_servo_pulsewidth(args.pin, pulse)
            print(f"Moved to {angle:g} degrees ({pulse:.0f} us)")
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        pi.set_servo_pulsewidth(args.pin, 0)
        pi.stop()


if __name__ == "__main__":
    main()
