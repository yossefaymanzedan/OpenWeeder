"""The ONLY way the planner touches the robot.

    Hardware      the interface (drive, stop, ticks, capture, servo_set, pump,
                  laser, now, sleep)
    FakeHardware  the simulated robot: position, encoder, servo, pump, camera.
                  The simulator and the tests use it.
    PiHardware    the real one: pins and limits from calib/hardware.json, the
                  camera is a phone (tools/phone_cam.py).
    robot()       the one PiHardware of this program, made on first use.

planner.py imports nothing but this interface, so the code that runs in the
simulator is the code that will run on the Raspberry Pi.

Units: cm, seconds, degrees. Servo angle: 90 = arm straight ahead, + = left.
"""
from __future__ import annotations

import abc
import json
import math
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

# ---- latencies (every one is a guess until measured on the robot) ----
SETTLE_S = 0.300           # car stopped -> frame/arm steady before spraying
SPRAY_S = 0.150            # pump on time per weed
DRIP_S = 0.100             # wait after the pump stops so drips fall in place
SERVO_S_PER_60DEG = 0.100  # SG90 datasheet: ~0.1 s per 60 degrees
CAPTURE_S = 0.100          # camera exposure + readout
VISION_S = 10.0            # SAM + Muse Spark per frame (measured 7-13 s)
COAST_S = 0.100            # the car keeps rolling this long after stop()
DEFAULT_CM_PER_TICK = 1.0  # ~1 cm per encoder tick (robot.json overrides)


def servo_travel_s(from_deg: float, to_deg: float) -> float:
    """How long a servo needs to turn between two angles."""
    return abs(to_deg - from_deg) / 60.0 * SERVO_S_PER_60DEG


class Hardware(abc.ABC):
    """What the planner may do. Nothing else."""

    @abc.abstractmethod
    def drive(self, speed_cm_s: float) -> None:
        """Start driving; positive = forward. Returns immediately."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop driving (the car may coast a little)."""

    @abc.abstractmethod
    def ticks(self) -> int:
        """Encoder count since power on (forward = up)."""

    @abc.abstractmethod
    def capture(self) -> np.ndarray:
        """One camera frame (H x W x 3, RGB), taken now."""

    @abc.abstractmethod
    def servo_set(self, angle_deg: float) -> None:
        """Start turning the arm servo to angle_deg. Returns immediately."""

    @abc.abstractmethod
    def pump(self, ms: float) -> None:
        """Run the pump for ms milliseconds. Blocks until it is off."""

    @abc.abstractmethod
    def laser(self, on: bool) -> None:
        ...

    @abc.abstractmethod
    def now(self) -> float:
        """Seconds on this robot's clock."""

    @abc.abstractmethod
    def sleep(self, s: float) -> None:
        """Wait s seconds (the car keeps moving if it is driving)."""


class PiHardware(Hardware):
    """The real robot on the Raspberry Pi 4 (PRD R1, R2, R6).

    L298N motors held at a speed by an encoder loop, a single-channel wheel
    encoder, the SG90 arm servo (pigpio hardware-timed PWM), a dummy pump pin,
    and a phone as the camera. Pins and limits: calib/hardware.json; cm_per_tick:
    calib/robot.json.

    Safety: it starts disarmed, so pump(), laser(True) and arm moves other than
    parking refuse until arm(). A watchdog stops the motors, drops the pump pin
    and parks the arm when nothing calls this interface for watchdog_s while the
    car drives; safe_stop() does the same on demand.

    dry_run=True runs the same code with no GPIO: the encoder ticks come from the
    commanded speed. capture() still uses the phone camera (and fails like the
    real robot when the phone is not there); only with no camera at all does a
    dry run return grey frames.

    The encoder has one channel, so it cannot tell direction: each tick counts in
    the direction last commanded, including while the car coasts after stop().
    """

    CONTROL_S = 0.02  # speed loop period

    def __init__(self, config: str | Path = "calib/hardware.json", dry_run: bool = False,
                 camera=None, calib_dir: str | Path = "calib"):
        if os.environ.get("OPENWEEDER_DRY_RUN") == "1":
            dry_run = True  # set by tests/conftest.py: tests never touch the real pins
        self.cfg = json.loads(Path(config).read_text(encoding="utf-8"))
        robot = json.loads((Path(calib_dir) / "robot.json").read_text(encoding="utf-8"))
        self.cm_per_tick = float(robot.get("cm_per_tick", DEFAULT_CM_PER_TICK))
        self.calib_dir = Path(calib_dir)
        self._calib = None
        self.dry_run = dry_run
        self.camera = camera
        lim = self.cfg["limits"]
        self.max_speed = float(lim["max_speed_cm_s"])
        self.pump_max_ms = float(lim["pump_max_ms"])
        self.servo_min, self.servo_max = float(lim["servo_min_deg"]), float(lim["servo_max_deg"])
        self.watchdog_s = float(lim["watchdog_s"])
        self.stall_s = float(lim.get("stall_s", 3.0))
        self.stalled = False
        m = self.cfg["motors"]
        self.axles = [m["front"], m["rear"]]
        self.armed = False
        self.events: list[dict] = []

        self._lock = threading.RLock()
        self._t0 = time.monotonic()
        self._count = 0
        self._direction = 1
        self._tick_times: deque = deque(maxlen=16)
        self._target = 0.0
        self._duty = 0.0
        self._integral = 0.0
        self._brake_until = 0.0
        self._last_call = time.monotonic()
        self._drive_since = 0.0
        self._servo_deg: float | None = None
        self._pins: dict[int, int] = {}  # what each output pin was last set to
        self._sim_pos_cm = 0.0
        self._sim_speed = 0.0
        self._pi = None
        self._callbacks: list = []
        if not dry_run:
            self._setup_gpio()
        else:
            for axle in self.axles:
                for key in ("forward", "backward", "enable"):
                    self._pins[axle[key]] = 0
            self._pins[self.cfg["pump_gpio_bcm"]] = 0

        self._running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True, name="pi-hardware")
        self._thread.start()
        self._log("start", "dry_run" if dry_run else None)
        self.park()

    # ---- the interface ----
    def drive(self, speed_cm_s: float) -> None:
        self._touch()
        speed = float(speed_cm_s)
        if speed == 0:
            self.stop()
            return
        speed = max(-self.max_speed, min(self.max_speed, speed))
        with self._lock:
            direction = 1 if speed > 0 else -1
            if direction != self._direction or self._target == 0:
                self._drive_since = time.monotonic()  # the wheels get stall_s to start turning
            if direction != self._direction:
                self._duty = 0.0
                self._integral = 0.0
            self._direction = direction
            self.stalled = False
            self._target = speed
            self._brake_until = 0.0
        self._log("drive", speed)

    def stop(self) -> None:
        self._touch()
        with self._lock:
            moving = self._target != 0 or self._duty > 0 or self._sim_speed != 0
            self._target = 0.0
            self._integral = 0.0
            self._duty = 0.0
            if moving:
                self._brake_until = time.monotonic() + float(self.cfg["motors"]["brake_s"])
        self._log("stop", None)

    def ticks(self) -> int:
        self._touch()
        with self._lock:
            return self._count

    def capture(self) -> np.ndarray:
        self._touch()
        self._log("capture", None)
        cam = self.cfg["camera"]
        width, height = int(cam["width"]), int(cam["height"])
        if self.camera is None:
            if self.dry_run:  # a bench dry run with no camera at all: grey frames
                return np.full((height, width, 3), 128, np.uint8)
            raise RuntimeError("No camera: the phone camera server is not running")
        import cv2
        # a phone over Wi-Fi sometimes misses one photo: ask again before giving up
        tries = max(1, int(cam.get("capture_retries", 3)))
        for attempt in range(tries):
            try:
                jpeg = self.camera.capture(float(cam.get("capture_timeout_s", 2.0)))
                break
            except Exception as e:  # noqa: BLE001  (CameraError / timeout)
                self._touch()
                self._log("capture_retry", f"{attempt + 1}/{tries}: {type(e).__name__}")
                if attempt == tries - 1:
                    raise
        self._touch()
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("The phone sent a photo that could not be decoded")
        rotate = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        if int(cam.get("rotate_deg", 0)) in rotate:
            bgr = cv2.rotate(bgr, rotate[int(cam["rotate_deg"])])
        return cv2.cvtColor(fit_frame(bgr, width, height), cv2.COLOR_BGR2RGB)

    def servo_set(self, angle_deg: float) -> None:
        self._touch()
        angle = float(angle_deg)
        if not self.servo_min <= angle <= self.servo_max:
            raise ValueError(f"Servo angle {angle:.1f} is outside the safe range "
                             f"[{self.servo_min:.0f}, {self.servo_max:.0f}]")
        if not self.armed and abs(angle - self.park_angle()) > 0.5:
            raise RuntimeError("The robot is disarmed: arm() before moving the arm")
        self._set_servo(angle)
        self._log("servo", angle)

    def pump(self, ms: float) -> None:
        self._touch()
        ms = float(ms)
        if not 0 <= ms <= self.pump_max_ms:
            raise ValueError(f"Pump time {ms:.0f} ms is outside [0, {self.pump_max_ms:.0f}] ms")
        if not self.armed:
            raise RuntimeError("The robot is disarmed: arm() before pumping")
        self._log("pump", ms)
        pin = self.cfg["pump_gpio_bcm"]
        try:
            self._write(pin, 1)
            time.sleep(ms / 1000.0)
        finally:
            self._write(pin, 0)
        self._touch()

    def laser(self, on: bool) -> None:
        self._touch()
        if on and not self.armed:
            raise RuntimeError("The robot is disarmed: arm() before the laser")
        pin = self.cfg.get("laser_gpio_bcm")
        if pin is not None:
            self._write(pin, 1 if on else 0)
        self._log("laser", bool(on))

    def now(self) -> float:
        self._touch()
        return time.monotonic()

    def sleep(self, s: float) -> None:
        # A long sleep is still the planner using the robot, so it keeps the
        # watchdog fed; a crashed planner stops calling anything at all.
        end = time.monotonic() + max(0.0, float(s))
        while True:
            self._touch()
            left = end - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(left, 0.1))

    # ---- safety ----
    def arm(self) -> None:
        self.armed = True
        self._log("arm", True)

    def disarm(self) -> None:
        self.armed = False
        self._write(self.cfg["pump_gpio_bcm"], 0)
        self.park()
        self._log("arm", False)

    def park_angle(self) -> float:
        """Servo angle that parks the arm (calib/servo.json arm.park_deg)."""
        import geometry
        if self._calib is None:
            self._calib = geometry.Calib(calib_dir=self.calib_dir, quiet=True)
        return geometry.arm_to_servo(float(self._calib.servo["arm"]["park_deg"]), self._calib)

    def park(self) -> None:
        angle = min(self.servo_max, max(self.servo_min, self.park_angle()))
        self._set_servo(angle)
        self._log("servo", angle)

    def safe_stop(self, reason: str = "safe stop") -> None:
        """Motors stop, pump pin LOW, arm parked. Safe to call from any thread."""
        with self._lock:
            moving = self._target != 0 or self._duty > 0
            self._target = 0.0
            self._integral = 0.0
            self._duty = 0.0
            if moving:
                self._brake_until = time.monotonic() + float(self.cfg["motors"]["brake_s"])
        self._write(self.cfg["pump_gpio_bcm"], 0)
        self._log("stop", reason)
        try:
            self.park()
        except Exception:  # noqa: BLE001  (parking must not hide the stop)
            pass

    def speed_cm_s(self) -> float:
        """Measured speed from the encoder (not part of Hardware)."""
        with self._lock:
            return self._direction * self._measured_speed(time.monotonic())

    def save_events(self, path: str | Path | None = None, start: int = 0) -> Path:
        """Write the event log (PRD R1) to out/run_<stamp>_events.json. start =
        len(events) when the run began, so the file holds that run only."""
        path = Path(path) if path else Path("out") / f"run_{datetime.now():%Y%m%d_%H%M%S}_events.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.events[start:], indent=1), encoding="utf-8")
        return path

    def close(self) -> None:
        self.safe_stop("close")
        time.sleep(float(self.cfg["motors"]["brake_s"]) + 0.1)
        self._running = False
        self._thread.join(1)
        self._motors(0, 1, brake=False)
        if self._pi is not None:
            for callback in self._callbacks:
                callback.cancel()
            self._pi.set_servo_pulsewidth(self.cfg["arm_servo"]["pin"], 0)
            self._pi.stop()
            self._pi = None

    def __enter__(self):
        return self

    def __exit__(self, kind, error, trace):
        if kind is not None:
            self.safe_stop(f"error: {kind.__name__}")
        return False

    # ---- inside ----
    def _touch(self) -> None:
        self._last_call = time.monotonic()

    def _log(self, kind: str, value) -> None:
        with self._lock:
            self.events.append({"t": round(time.monotonic() - self._t0, 4), "type": kind, "value": value,
                                "tick": self._count,
                                "arm": None if self._servo_deg is None else round(self._servo_deg, 2)})

    def _setup_gpio(self) -> None:
        import pigpio
        pi = pigpio.pi()
        if not pi.connected:
            raise RuntimeError("The pigpio daemon is not running: sudo systemctl start pigpiod")
        self._pi = pi
        hz = int(self.cfg["motors"]["pwm_hz"])
        for axle in self.axles:
            for key in ("forward", "backward", "enable"):
                pi.set_mode(axle[key], pigpio.OUTPUT)
                self._write(axle[key], 0)
            pi.set_PWM_frequency(axle["enable"], hz)
            pi.set_PWM_range(axle["enable"], 100)
        for pin in (self.cfg["pump_gpio_bcm"], self.cfg.get("laser_gpio_bcm")):
            if pin is not None:
                pi.set_mode(pin, pigpio.OUTPUT)
                self._write(pin, 0)
        enc = self.cfg["encoder"]
        pi.set_mode(enc["pin_a"], pigpio.INPUT)
        pi.set_pull_up_down(enc["pin_a"], pigpio.PUD_UP)
        pi.set_glitch_filter(enc["pin_a"], int(enc.get("glitch_us", 100)))
        self._callbacks.append(pi.callback(enc["pin_a"], pigpio.RISING_EDGE, self._on_tick))

    def _on_tick(self, *_args) -> None:
        with self._lock:
            self._count += self._direction
            self._tick_times.append(time.monotonic())

    def _write(self, pin: int, level: int) -> None:
        self._pins[pin] = level
        if self._pi is not None:
            self._pi.write(pin, level)

    def _set_servo(self, angle: float) -> None:
        servo = self.cfg["arm_servo"]
        lo, hi = float(servo["pulse_min_us"]), float(servo["pulse_max_us"])
        if self._pi is not None:
            self._pi.set_servo_pulsewidth(servo["pin"], lo + (hi - lo) * angle / 180.0)
        self._servo_deg = angle

    def _motors(self, duty: float, direction: int, brake: bool) -> None:
        for axle in self.axles:
            forward, backward = axle["forward"], axle["backward"]
            if axle.get("reverse"):
                forward, backward = backward, forward
            if direction < 0:
                forward, backward = backward, forward
            if brake:  # both inputs high with the enable on: the L298N shorts the motor
                self._write(forward, 1)
                self._write(backward, 1)
                level = 100
            else:
                self._write(backward, 0)
                self._write(forward, 1 if duty > 0 else 0)
                level = int(round(duty))
            self._pins[axle["enable"]] = level
            if self._pi is not None:
                self._pi.set_PWM_dutycycle(axle["enable"], level)

    def _measured_speed(self, now: float) -> float:
        """cm/s from the time between ticks, falling toward 0 when ticks stop."""
        times = list(self._tick_times)
        if len(times) < 2 or now - times[-1] > 2.0:
            return 0.0
        recent = [t for t in times if now - t <= 1.0] or times[-2:]
        if len(recent) < 2:
            recent = times[-2:]
        span = recent[-1] - recent[0]
        rate = (len(recent) - 1) / span if span > 0 else 0.0
        rate = min(rate, 1.0 / max(now - times[-1], 1e-6))
        return rate * self.cm_per_tick

    def _control_loop(self) -> None:
        m = self.cfg["motors"]
        last = time.monotonic()
        while self._running:
            time.sleep(self.CONTROL_S)
            now = time.monotonic()
            dt, last = now - last, now
            with self._lock:
                if self._target != 0 and now - self._last_call > self.watchdog_s:
                    watchdog = True
                else:
                    watchdog = False
                last_tick = self._tick_times[-1] if self._tick_times else 0.0
                stalled = (self._target != 0 and not self._brake_until
                           and now - max(self._drive_since, last_tick) > self.stall_s)
                if self._brake_until:
                    braking = now < self._brake_until
                    self._motors(0, self._direction, brake=braking)
                    if not braking:
                        self._brake_until = 0.0
                elif self._target == 0:
                    self._motors(0, self._direction, brake=False)
                else:
                    want_speed = abs(self._target)
                    error = want_speed - self._measured_speed(now)
                    self._integral = max(-20.0, min(20.0, self._integral + error * dt))
                    want = (float(m["min_duty"]) + want_speed * float(m["duty_per_cm_s"])
                            + float(m["kp"]) * error + float(m["ki"]) * self._integral)
                    want = max(0.0, min(100.0, want))
                    step = float(m["ramp_duty_per_s"]) * dt
                    self._duty += max(-step, min(step, want - self._duty))
                    self._motors(self._duty, self._direction, brake=False)
                if self.dry_run:
                    self._simulate(dt)
            if watchdog:
                self.safe_stop("watchdog")
            elif stalled:  # told to drive, but the wheels do not turn
                self.stalled = True
                self.stop()
                self._log("stop", f"stalled: no encoder count for {self.stall_s:g} s")

    def _simulate(self, dt: float) -> None:
        """dry_run: the car follows the commanded speed (with a short ramp) and
        the encoder ticks as it would."""
        want = 0.0 if self._brake_until or self._target == 0 else abs(self._target)
        accel = 20.0  # cm/s per second
        self._sim_speed += max(-accel * dt, min(accel * dt, want - self._sim_speed))
        if self._sim_speed <= 0:
            self._sim_speed = 0.0
            return
        self._sim_pos_cm += self._sim_speed * dt
        while self._sim_pos_cm >= self.cm_per_tick:
            self._sim_pos_cm -= self.cm_per_tick
            self._count += self._direction
            self._tick_times.append(time.monotonic())


def fit_frame(bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Centre-crop to width:height (no stretching), then resize to width x height."""
    import cv2
    h, w = bgr.shape[:2]
    want = width / height
    if w / h > want:
        new_w = int(round(h * want))
        x0 = (w - new_w) // 2
        bgr = bgr[:, x0:x0 + new_w]
    elif w / h < want:
        new_h = int(round(w / want))
        y0 = (h - new_h) // 2
        bgr = bgr[y0:y0 + new_h]
    if bgr.shape[1] != width or bgr.shape[0] != height:
        interp = cv2.INTER_AREA if bgr.shape[1] > width else cv2.INTER_CUBIC
        bgr = cv2.resize(bgr, (width, height), interpolation=interp)
    return bgr


_ROBOT: PiHardware | None = None
_ROBOT_LOCK = threading.Lock()


def robot(config: str | Path = "calib/hardware.json") -> PiHardware:
    """The one PiHardware of this program, with the phone camera, made on first
    use. OPENWEEDER_DRY_RUN=1 makes it a dry run (no GPIO, no motion)."""
    global _ROBOT
    with _ROBOT_LOCK:
        if _ROBOT is None:
            import phone_cam
            cfg = json.loads(Path(config).read_text(encoding="utf-8"))
            camera = phone_cam.shared(cfg["camera"])
            _ROBOT = PiHardware(config, dry_run=os.environ.get("OPENWEEDER_DRY_RUN") == "1", camera=camera)
        return _ROBOT


class FakeHardware(Hardware):
    """A simulated robot on a clock it controls: sleep() IS time passing.

    frame_source(camera_ground_x_cm) -> image   what the camera sees when the
        spot of ground under the lens is at that x (cm). The ground under the
        lens is camera_to_nozzle_cm ahead of the nozzle/pivot.
    True position, encoder, arm angle and the pump log are kept here; the
    planner can only reach them through the Hardware interface. Every action is
    appended to `events`: {t, type, value, tick, arm}.
    """

    def __init__(self, frame_source: Callable[[float], np.ndarray] | None = None,
                 camera_to_nozzle_cm: float = 12.0,
                 cm_per_tick: float = DEFAULT_CM_PER_TICK,
                 true_cm_per_tick: float | None = None,
                 start_cm: float = 0.0, park_deg: float = 90.0,
                 capture_s: float = CAPTURE_S, coast_s: float = COAST_S,
                 servo_s_per_60deg: float = SERVO_S_PER_60DEG):
        self.frame_source = frame_source
        self.D = camera_to_nozzle_cm
        # encoder says cm_per_tick (what the planner believes); the wheels
        # really move true_cm_per_tick per tick (differs if measure_ticks was off)
        self.cm_per_tick = cm_per_tick
        self.true_cm_per_tick = true_cm_per_tick or cm_per_tick
        self.capture_s, self.coast_s = capture_s, coast_s
        self.servo_s = servo_s_per_60deg
        self.t = 0.0
        self.pos = float(start_cm)        # true position of the pivot, cm
        self.speed = 0.0                  # commanded speed, cm/s
        self._coast_left = 0.0            # seconds of rolling after stop()
        self._coast_speed = 0.0
        # servo: moves linearly from _servo_from to _servo_to between t0 and t1
        self._servo_from = self._servo_to = float(park_deg)
        self._servo_t0 = self._servo_t1 = 0.0
        self.laser_on = False
        self.events: list[dict] = []
        self.pumps: list[dict] = []       # {t0, ms, pos, tick}
        self.trace: list[tuple] = []      # (t, pos, arm_servo_deg, driving, pumping)
        self._pumping_until = -1.0
        self._log("start", None)

    # ---- the interface ----
    def drive(self, speed_cm_s: float) -> None:
        self.speed = float(speed_cm_s)
        self._coast_left = 0.0
        self._log("drive", speed_cm_s)

    def stop(self) -> None:
        if self.speed:
            self._coast_speed, self._coast_left = self.speed, self.coast_s
        self.speed = 0.0
        self._log("stop", None)

    def ticks(self) -> int:
        return int(math.floor(self.pos / self.true_cm_per_tick + 1e-9))

    def capture(self) -> np.ndarray:
        """The frame is taken at the START of the call; the exposure then
        takes capture_s (the car keeps moving meanwhile)."""
        cam_x = self.pos + self.D
        self._log("capture", None)
        img = self.frame_source(cam_x) if self.frame_source else np.zeros((4, 4, 3), np.uint8)
        self.sleep(self.capture_s)
        return img

    def servo_set(self, angle_deg: float) -> None:
        cur = self.servo_deg()
        self._servo_from, self._servo_to = cur, float(angle_deg)
        self._servo_t0 = self.t
        self._servo_t1 = self.t + abs(angle_deg - cur) / 60.0 * self.servo_s
        self._log("servo", angle_deg)

    def pump(self, ms: float) -> None:
        self.pumps.append({"t0": self.t, "ms": float(ms), "pos": self.pos,
                           "tick": self.ticks()})
        self._log("pump", ms)
        self._pumping_until = self.t + ms / 1000.0
        self.sleep(ms / 1000.0)

    def laser(self, on: bool) -> None:
        self.laser_on = bool(on)
        self._log("laser", bool(on))

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        """Advance the clock; the car rolls while it is driving or coasting.
        Steps of at most 20 ms so the trace is smooth."""
        left = max(0.0, float(s))
        while left > 1e-12:
            dt = min(left, 0.02)
            move = self.speed * dt
            if self._coast_left > 0:
                c = min(dt, self._coast_left)
                move += self._coast_speed * c
                self._coast_left -= c
            self.pos += move
            self.t += dt
            left -= dt
            self.trace.append((self.t, self.pos, self.servo_deg(),
                               bool(self.speed) or self._coast_left > 0,
                               self.t < self._pumping_until))

    # ---- state the simulator may read (not part of Hardware) ----
    def servo_deg(self, t: float | None = None) -> float:
        """Where the servo horn really is at time t (default: now)."""
        t = self.t if t is None else t
        if t >= self._servo_t1 or self._servo_t1 <= self._servo_t0:
            return self._servo_to
        f = (t - self._servo_t0) / (self._servo_t1 - self._servo_t0)
        return self._servo_from + (self._servo_to - self._servo_from) * f

    def _log(self, kind: str, value) -> None:
        self.events.append({"t": round(self.t, 4), "type": kind, "value": value,
                            "tick": self.ticks(), "arm": round(self.servo_deg(), 2)})
