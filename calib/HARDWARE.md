# calib/hardware.json

Every pin and limit of the real robot, in one place (PRD R10). Pins are BCM GPIO
numbers; the physical header pin is in brackets. `tools/hardware.py` (`PiHardware`)
reads this file; nothing else hard-codes a pin.

| Key | Value | Meaning |
|---|---|---|
| `motors.front` | forward 19 [35], backward 16 [36], enable 6 [31] | L298N IN1, IN2, ENA: front motors on OUT1/OUT2 |
| `motors.rear` | forward 26 [37], backward 20 [38], enable 13 [33] | L298N IN3, IN4, ENB: rear motors on OUT3/OUT4 |
| `motors.*.reverse` | false | Set true if that pair of wheels turns backward |
| `motors.pwm_hz` | 100 | PWM on ENA/ENB (low frequency: more torque at low speed) |
| `motors.brake_s` | 0.3 | `stop()` brakes actively (both inputs high) this long, then lets go |
| `motors.min_duty` | 35 | Power (%) below which the motors stall. **Measure.** |
| `motors.duty_per_cm_s` | 4.0 | Extra power per cm/s (first guess for the speed loop). **Measure.** |
| `motors.kp`, `motors.ki` | 4.0, 3.0 | Speed loop: % power per cm/s of error, and per cm of accumulated error |
| `motors.ramp_duty_per_s` | 120 | How fast the power may change (no slamming) |
| `encoder.pin_a` | 5 [29] | Single-output slot sensor (OUT), powered from 3.3 V [17], GND [30] |
| `encoder.pin_b` | null | No second channel: direction comes from the commanded direction |
| `encoder.slots`, `encoder.wheel_mm` | 20, 65 | **Measure.** `cm_per_tick` itself lives in `calib/robot.json` |
| `encoder.glitch_us` | 100 | Ignore pulses shorter than this (noise) |
| `arm_servo` | pin 21 [40], 500-2400 us | SG90 wiper arm, hardware-timed PWM (pigpio) |
| `tilt_servo.pin` | null | Not fitted (laser mode only) |
| `pump_gpio_bcm` | 17 [11] | **Dummy** output: nothing wired (an LED + resistor for bench checks). The real pump is switched by a push button outside the Pi |
| `laser_gpio_bcm` | null | Not fitted |
| `buttons` | null | Red button is a hardware cut; not sensed |
| `camera.source` | phone | The robot's camera is a phone page: `tools/phone_cam.py` |
| `camera.port` | 8444 | The phone opens `https://<pi-ip>:8444` |
| `camera.width`, `height` | 1296, 972 | What `capture()` returns (cropped to 4:3, then resized) |
| `camera.rotate_deg` | 0 | Turn the phone's picture so the top points forward (0, 90, 180, 270) |
| `camera.jpeg_quality` | 0.92 | Quality of each photo the phone sends |
| `camera.capture_timeout_s` | 2.0 | `capture()` fails if no photo arrives in this time |
| `limits.max_speed_cm_s` | 8 | `drive()` clamps to this |
| `limits.pump_max_ms` | 500 | `pump()` refuses longer |
| `limits.servo_min_deg`, `servo_max_deg` | 10, 170 | `servo_set()` refuses outside |
| `limits.watchdog_s` | 2 | Nothing calls the interface for this long while driving: motors stop, pump pin LOW, arm parks |
| `limits.stall_s` | 3 | The motors are told to drive but the encoder counts nothing for this long: the motors stop (any mode). Field runs and recording notice sooner (2 s, 3 s) and say why |
| `recording.speed_cm_s` | 3 | Speed while recording the dataset |
| `recording.max_cm` | 150 | Recording stops after this distance (so the car cannot run off the rails) |
| `recording.stall_s` | 3 | Recording stops if the encoder sees no movement for this long |
| `recording.crop_to_tray` | true | Save the tray cut out straight and centred (tools/tray.py); the full frame goes to `train/raw/`, the walls to `train/trays.json` |
| `field_run.speed_cm_s` | 3 | Speed in Run on the field (both passes) |
| `field_run.max_cm` | 200 | The most field one run covers, whatever the page asks |
| `field_run.stall_s` | 2 | The car is driving but the encoder has not counted for this long: it reached the wall at the end of the rails (or is stuck). It stops there, takes its last picture, and skips weeds beyond it |
| `field_run.travel_cm` | 90 | Backstop in case a stall is missed: the car never drives further than this from where it starts (rails about 1 m, car 25 cm, so the wall normally comes first at about 75 cm) |
| `field_run.aim_hold_s` | 20 | Calibration: the arm stays aimed at each weed this long before the pump, so you can check where the nozzle points. 0 for normal runs. The page can change it per run |

Do not run `motor_app.py`, `car.py` or `servo_app.py` while the platform drives the robot:
they control the same pins.
