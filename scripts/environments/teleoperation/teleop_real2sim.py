"""
Script for real-to-sim teleoperation using a joystick.
by exaFLOPS

Contribution:
    1. Teleoperate simulation & real world using a joystick in the same time
    2. Real in Sim(superset)
    3. Cross-embodiement teleoperation
"""
import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentPaser(description="Teleoperation for real2sim")
parser.add_argument(
    "--disable_fabric",
    action = "store_true",
    default = False,
    help="Run the simulation teleoperation",
)
parser.add_argument(
    "--num_envs",
    type = int,
    default = 1,
    help="Number of environments to run",
)
parser.add_argument(
    "--teleop_device",
    type = str,
    default = "oculus_droid",
    help="Device to use for teleoperation",
)
