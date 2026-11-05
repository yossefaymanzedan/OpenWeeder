#!/usr/bin/env python3
"""Drive a 4WD car forward from the terminal through an L298N motor driver.

Type g to go slowly, or a speed from 0 to 100, and press Enter.
Type s to stop, q to quit.
Requires the pigpio daemon: sudo pigpiod -l
"""

import argparse
import signal
import sys

import pigpio

from motors import MAX_SPEED, SLOW_SPEED, Car


def parse_args():
    parser = argparse.ArgumentParser(description="Drive a 4WD car forward from the terminal.")
    parser.add_argument("--reverse-front", action="store_true", help="use if the front wheels spin backward")
    parser.add_argument("--reverse-rear", action="store_true", help="use if the rear wheels spin backward")
    return parser.parse_args()


def exit_on_signal(signum, frame):
    sys.exit(0)


def main():
    args = parse_args()

    pi = pigpio.pi()
    if not pi.connected:
        sys.exit("pigpio daemon is not running. Start it with: sudo pigpiod -l")

    car = Car(pi, args.reverse_front, args.reverse_rear)

    # Stop the motors if the terminal closes or the script is killed.
    signal.signal(signal.SIGTERM, exit_on_signal)
    signal.signal(signal.SIGHUP, exit_on_signal)

    print(f"Car ready. Enter g to go at {SLOW_SPEED}%, a speed from 0 to {MAX_SPEED}, s to stop, or q to quit.")
    try:
        while True:
            text = input("speed> ").strip().lower()
            if text in ("q", "quit", "exit"):
                break
            if not text:
                continue
            if text in ("s", "stop"):
                car.stop()
                print("Stopped")
                continue
            if text in ("g", "go"):
                text = str(SLOW_SPEED)

            try:
                speed = int(text)
            except ValueError:
                print(f"Not a whole number: {text}")
                continue
            if not 0 <= speed <= MAX_SPEED:
                print(f"Speed must be between 0 and {MAX_SPEED}")
                continue

            car.set_speed(speed)
            print(f"Forward at {speed}%" if speed else "Stopped")
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        car.stop()
        pi.stop()


if __name__ == "__main__":
    main()
