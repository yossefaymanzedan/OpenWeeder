"""GPIO pins for all OpenWeeder hardware, as BCM numbers.

Every signal sits in the bottom block of the header, physical pins 29 to 40.
Only power comes from the top: 3.3V on pin 17 and 5V on pin 4.

  Physical pin               Physical pin
  29  GPIO5   encoder A      30  GND     encoder GND
  31  GPIO6   L298N ENA      32  (empty)
  33  GPIO13  L298N ENB      34  GND     L298N GND
  35  GPIO19  L298N IN1      36  GPIO16  L298N IN2
  37  GPIO26  L298N IN3      38  GPIO20  L298N IN4
  39  GND     servo GND      40  GPIO21  servo signal

  17  3.3V    encoder VCC     4  5V      servo VCC
  11  GPIO17  encoder B, two-output encoders only

The IN pins are pulled low at boot, so the motors stay off while the Pi
starts. Pins used by I2C, SPI, the serial console and 1-Wire are avoided.
"""

SERVO = 21

ENCODER_A = 5
ENCODER_B = 17

# L298N: (forward input, backward input, enable).
FRONT_MOTORS = (19, 16, 6)  # IN1, IN2, ENA
REAR_MOTORS = (26, 20, 13)  # IN3, IN4, ENB
