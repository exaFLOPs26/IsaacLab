# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# LIVESTREAM=2 ./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --teleop_device oculus_droid
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="keyboard", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
    # not the one installed by Isaac Sim pinocchio is required by the Pink IK controllers and the
    # GR1T2 retargeter
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import numpy as np
import torch

import omni.log

if "handtracking" in args_cli.teleop_device.lower():
    from isaacsim.xr.openxr import OpenXRSpec

from isaaclab.devices import OpenXRDevice, Se3Gamepad, Se3Keyboard, Se3SpaceMouse, Oculus_droid, Se3Keyboard_BMM

if args_cli.enable_pinocchio:
    from isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1t2_retargeter import GR1T2Retargeter
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab.devices.openxr.retargeters.manipulator import GripperRetargeter, Se3AbsRetargeter, Se3RelRetargeter
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg
from scipy.spatial.transform import Rotation
import socket
import json


def get_ee_state(env, ee_name, gripper_value=0.0):
    # arm
    ee = env.scene[ee_name].data
    pos = ee.target_pos_source[0, 0]
    rot = ee.target_quat_source[0, 0]

    # euler = torch.from_numpy(
    #     Rotation.from_quat(rot.cpu().numpy()).as_euler('xyz')
    # ).to(dtype=rot.dtype, device=rot.device)
    
    # Gripper
    if ee_name == "ee_L_frame":
        body_pos = env.scene._articulations['robot'].data.body_pos_w[0, -2:]
    else:
        body_pos = env.scene._articulations['robot'].data.body_pos_w[0, -4:-2]
    gripper_dist = torch.norm(body_pos[0] - body_pos[1])*-1*20.8+0.05 # To match [0.05, -1.65] the real robot
    return torch.cat((pos, rot, gripper_dist.unsqueeze(0))).unsqueeze(0)

def pre_process_actions(
    teleop_data: tuple[np.ndarray, bool] | list[tuple[np.ndarray, np.ndarray, np.ndarray]], num_envs: int, device: str
) -> torch.Tensor:
    """Convert teleop data to the format expected by the environment action space.

    Args:
        teleop_data: Data from the teleoperation device.
        num_envs: Number of environments.
        device: Device to create tensors on.

    Returns:
        Processed actions as a tensor.
    """
    # compute actions based on environment
    if "Reach" in args_cli.task:
        delta_pose, gripper_command = teleop_data
        # convert to torch
        delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=device).repeat(num_envs, 1)
        # note: reach is the only one that uses a different action space
        # compute actions
        return delta_pose
    elif "PickPlace-GR1T2" in args_cli.task:
        (left_wrist_pose, right_wrist_pose, hand_joints) = teleop_data[0]
        # Reconstruct actions_arms tensor with converted positions and rotations
        actions = torch.tensor(
            np.concatenate([
                left_wrist_pose,  # left ee pose
                right_wrist_pose,  # right ee pose
                hand_joints,  # hand joint angles
            ]),
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)
        # Concatenate arm poses and hand joint angles
        return actions
    else:
        # resolve gripper command
        delta_pose, gripper_command = teleop_data
        # convert to torch
        print("gripper_command", gripper_command)
        delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=device).repeat(num_envs, 1)
        gripper_vel = torch.zeros((delta_pose.shape[0], 1), dtype=torch.float, device=device)
        # gripper_vel[:] = -1 if gripper_command else 1
        if gripper_command == 1:
            gripper_vel[:] = 1
        elif gripper_command == -1:
            gripper_vel[:] = -1
        print("gripper_vel", gripper_vel)
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)

from isaaclab.sim import SimulationContext                                    # correct import path 
import omni.ui as ui                                                           # Omni UI
import omni.appwindow as appwin                                               
import carb.input                                                             
from carb.input import KeyboardEventType, KeyboardInput                       

class PauseResetController:
    def __init__(self):
        # SimulationContext singleton
        self.sim = SimulationContext.instance()                                
        self.paused = False
        self.win = None

        # Start sim synchronously
        self.sim.reset()                                                       # initialize physics 
        self.sim.play()                                                        # begin stepping 
    
    def _enter_pause(self):
        # Pause the sim
        self.sim.pause()                                                         # 
        self.paused = True

        # Create a simple, floating window (no flags needed)
        self.win = ui.Window("Paused", width=300, height=100)                    
        # Bring it to the front as a modal
        self.win.set_top_modal()                                                 

        # Populate the window
        with self.win.frame:
            with ui.VStack(spacing=10, alignment=ui.Alignment.CENTER):
                ui.Label("Waiting for reset…", alignment=ui.Alignment.CENTER)
                ui.Label("(Press A to reset)", alignment=ui.Alignment.CENTER)

    def _do_reset(self):
        # Hide overlay
        if self.win:
            self.win.visible = False
            self.win = None

        # Reset & resume
        self.sim.reset()                                                         # 
        self.sim.play()                                                          # 
        self.paused = False


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        omni.log.warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance():
        """Reset the environment to its initial state.

        This callback is triggered when the user presses the reset key (typically 'R').
        It's useful when:
        - The robot gets into an undesirable configuration
        - The user wants to start over with the task
        - Objects in the scene need to be reset to their initial positions

        The environment will be reset on the next simulation step.
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    def start_teleoperation():
        """Activate teleoperation control of the robot.

        This callback enables active control of the robot through the input device.
        It's typically triggered by a specific gesture or button press and is used when:
        - Beginning a new teleoperation session
        - Resuming control after temporarily pausing
        - Switching from observation mode to control mode

        While active, all commands from the device will be applied to the robot.
        """
        nonlocal teleoperation_active
        teleoperation_active = True

    def stop_teleoperation():
        """Deactivate teleoperation control of the robot.

        This callback temporarily suspends control of the robot through the input device.
        It's typically triggered by a specific gesture or button press and is used when:
        - Taking a break from controlling the robot
        - Repositioning the input device without moving the robot
        - Pausing to observe the scene without interference

        While inactive, the simulation continues to render but device commands are ignored.
        """
        nonlocal teleoperation_active
        teleoperation_active = False

    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "gamepad":
        teleop_interface = Se3Gamepad(
            pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.1 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "oculus_droid":
        teleop_interface = Oculus_droid()
        pause_reset = PauseResetController()
        teleop_interface.add_callback("B", pause_reset._enter_pause)
        teleop_interface.add_callback("A", pause_reset._do_reset)
        
    elif "dualhandtracking_abs" in args_cli.teleop_device.lower() and "GR1T2" in args_cli.task:
        # Create GR1T2 retargeter with desired configuration
        gr1t2_retargeter = GR1T2Retargeter(
            enable_visualization=True,
            num_open_xr_hand_joints=2 * (int(OpenXRSpec.HandJointEXT.XR_HAND_JOINT_LITTLE_TIP_EXT) + 1),
            device=env.unwrapped.device,
            hand_joint_names=env.scene["robot"].data.joint_names[-22:],
        )

        # Create hand tracking device with retargeter
        teleop_interface = OpenXRDevice(
            env_cfg.xr,
            retargeters=[gr1t2_retargeter],
        )
        teleop_interface.add_callback("RESET", reset_recording_instance)
        teleop_interface.add_callback("START", start_teleoperation)
        teleop_interface.add_callback("STOP", stop_teleoperation)

        # Hand tracking needs explicit start gesture to activate
        teleoperation_active = False

    elif "handtracking" in args_cli.teleop_device.lower():
        # Create EE retargeter with desired configuration
        if "_abs" in args_cli.teleop_device.lower():
            retargeter_device = Se3AbsRetargeter(
                bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT, zero_out_xy_rotation=True
            )
        else:
            retargeter_device = Se3RelRetargeter(
                bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT, zero_out_xy_rotation=True
            )

        grip_retargeter = GripperRetargeter(bound_hand=OpenXRDevice.TrackingTarget.HAND_RIGHT)

        # Create hand tracking device with retargeter (in a list)
        teleop_interface = OpenXRDevice(
            env_cfg.xr,
            retargeters=[retargeter_device, grip_retargeter],
        )

        # Hand tracking needs explicit start gesture to activate
        teleoperation_active = False
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse', 'gamepad',"
            " 'handtracking', 'handtracking_abs'."
        )
    teleop_interface2 = Se3Keyboard_BMM(
            pos_sensitivity=0.005 * args_cli.sensitivity, rot_sensitivity=0.01 * args_cli.sensitivity
        )

    teleop_interface2.add_callback("R", reset_recording_instance)

    # reset environment
    env.reset()
    teleop_interface.reset()
    
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():

            if args_cli.teleop_device.lower() == "oculus_droid":
                # Right arm
                ee_r_state = get_ee_state(env, "ee_frame", gripper_value=0.0)

                obs_dict = {"left_arm": 0 , "right_arm": ee_r_state}
                teleop_data = teleop_interface.advance_onearm(obs_dict)
                print("teleop_data", teleop_data)
            else:
                teleop_data = teleop_interface.advance()
                
            # Only apply teleop commands when active
            if teleoperation_active:
                # compute actions based on environment
                actions = pre_process_actions(teleop_data, env.num_envs, env.device)
                # apply actions
                # receive = start_receiver()
                # print(receive)
                
                env.step(actions)
            else:
                env.sim.render()

            if should_reset_recording_instance:
                env.reset()
                should_reset_recording_instance = False

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
