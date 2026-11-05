"""Tests never touch the real robot: PiHardware always runs as a dry run here."""
import os

os.environ["OPENWEEDER_DRY_RUN"] = "1"
