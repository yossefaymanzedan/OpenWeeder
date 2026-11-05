#!/usr/bin/env python3
"""Forward-only control of the 4WD car's motors through an L298N driver.

Run it directly to test the motors from a browser on the Pi:
    python motors.py    then open https://localhost:8443
Requires the pigpio daemon: sudo pigpiod -l
"""

import threading
import time

import pigpio

import pins

MAX_SPEED = 100
SLOW_SPEED = 10
PWM_FREQUENCY = 100

# Motors stall at low power, so a slow start begins with a short burst.
KICK_SPEED = 60
KICK_SECONDS = 0.15

# An emergency stop brakes the motors this long, then lets them go.
BRAKE_SECONDS = 0.5


class Axle:
    def __init__(self, pi, forward, backward, enable):
        self.pi = pi
        self.forward = forward
        self.backward = backward
        self.enable = enable

        for pin in (forward, backward, enable):
            pi.set_mode(pin, pigpio.OUTPUT)
            pi.write(pin, 0)
        pi.set_PWM_frequency(enable, PWM_FREQUENCY)
        pi.set_PWM_range(enable, MAX_SPEED)

    def set_speed(self, speed):
        self.pi.write(self.backward, 0)
        self.pi.write(self.forward, 1 if speed > 0 else 0)
        self.pi.set_PWM_dutycycle(self.enable, speed)

    def brake(self):
        """Both inputs high shorts the motor through the driver, stopping it fast."""
        self.pi.write(self.forward, 1)
        self.pi.write(self.backward, 1)
        self.pi.set_PWM_dutycycle(self.enable, MAX_SPEED)


def build_axle(pi, axle_pins, reverse):
    forward, backward, enable = axle_pins
    if reverse:
        forward, backward = backward, forward
    return Axle(pi, forward, backward, enable)


class Car:
    def __init__(self, pi, reverse_front=False, reverse_rear=False):
        self.axles = [
            build_axle(pi, pins.FRONT_MOTORS, reverse_front),
            build_axle(pi, pins.REAR_MOTORS, reverse_rear),
        ]
        self.speed = 0
        self.emergency = False
        self.lock = threading.Lock()

    def _drive(self, speed):
        for axle in self.axles:
            axle.set_speed(speed)

    def set_speed(self, speed):
        """Set a speed from 0 to MAX_SPEED. Blocks briefly when kick-starting.

        Ignored after an emergency stop, until release() is called.
        """
        with self.lock:
            if self.emergency:
                return
            if self.speed == 0 and 0 < speed < KICK_SPEED:
                self._drive(KICK_SPEED)
                time.sleep(KICK_SECONDS)
            self._drive(speed)
            self.speed = speed

    def stop(self):
        self.set_speed(0)

    def emergency_stop(self):
        """Brake hard, then keep the motors off until release() is called."""
        with self.lock:
            self.emergency = True
            for axle in self.axles:
                axle.brake()
            time.sleep(BRAKE_SECONDS)
            self._drive(0)
            self.speed = 0

    def release(self):
        with self.lock:
            self.emergency = False


if __name__ == "__main__":
    import motor_app

    motor_app.main(local_only=True)
