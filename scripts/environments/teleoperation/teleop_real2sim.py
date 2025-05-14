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
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--teleop_device", type=str, default="oculus_droid", help="Device for interacting with environment")
parser.add_argument("--task", type=str, default="Isaac-Real2sim-Direct-v0", help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# parse the arguments
app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch
import ipdb
import omni.log

from isaaclab.devices import Oculus_droid, Se3Keyboard_BMM
from isaaclab.envs import ViewerCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

def pre_process_actions(delta_pose_L: torch.Tensor, gripper_command_L: bool, delta_pose_R, gripper_command_R: bool, delta_pose_base) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # resolve gripper command
    val_L = -1.0 if gripper_command_L else 1.0
    val_R = -1.0 if gripper_command_R else 1.0

    gripper_vel_L = torch.full((delta_pose_L.shape[0], 1), val_L, device=delta_pose_L.device)
    gripper_vel_R = torch.full((delta_pose_R.shape[0], 1), val_R, device=delta_pose_R.device)

    return torch.cat([delta_pose_L, delta_pose_R, gripper_vel_L, gripper_vel_R, delta_pose_base], dim=1)

def main():
    """Main function for real2sim teleoperation."""
    
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    
    #[TODO] check it is necessary
    # modify configuration
    # env_cfg.terminations.time_out = None
    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    
    if args_cli.teleop_device.lower() == "oculus_droid":
        teleop_interface = Oculus_droid(
            pos_sensitivity=2.15 * args_cli.sensitivity, rot_sensitivity=1.0 * args_cli.sensitivity, base_sensitivity = 0.3 * args_cli.sensitivity
        )
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse''handtracking'."
        )
    
    # add teleoperation key for env reset
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # [TODO] reset button with oclus joystick
    teleop_interface2 = Se3Keyboard_BMM(
            pos_sensitivity=0.005 * args_cli.sensitivity, rot_sensitivity=0.01 * args_cli.sensitivity
        )

    teleop_interface2.add_callback("R", reset_recording_instance)

    # reset environment
    env.reset()
    teleop_interface.reset()
    
    # Simulation environment
    while simulation_app.is_running():
        # run everthing in inference mode
        with torch.inference_mode():
            
            # Get simulation observation of robot
            init_pos = env.scene["ee_L_frame"].data.target_pos_source[0,0]
            init_rot = env.scene["ee_L_frame"].data.target_quat_source[0,0]
            ee_l_state = torch.cat([init_pos, init_rot], dim=0).unsqueeze(0)
        
            init_pos = env.scene["ee_R_frame"].data.target_pos_source[0,0]
            init_rot = env.scene["ee_R_frame"].data.target_quat_source[0,0]
            ee_r_state = torch.cat([init_pos, init_rot], dim=0).unsqueeze(0)
            obs_dict = {"left_arm": ee_l_state, "right_arm": ee_r_state}
            
            # Get teleoperation actions 
            pose_L, gripper_command_L, pose_R, gripper_command_R, delta_pose_base = teleop_interface.advance(obs_dict)

            pose_L = pose_L.astype("float32")
            pose_R = pose_R.astype("float32")
            delta_pose_base = delta_pose_base.astype("float32")
            # convert to torch
            pose_L = torch.tensor(pose_L, device=env.device).repeat(env.num_envs, 1)
            pose_R = torch.tensor(pose_R, device=env.device).repeat(env.num_envs, 1)
            delta_pose_base = torch.tensor(delta_pose_base, device=env.device).repeat(env.num_envs, 1)
            
            actions = pre_process_actions(pose_L, gripper_command_L, pose_R, gripper_command_R, delta_pose_base)
            
            env.step(actions)

            if should_reset_recording_instance:
                env.reset()
                teleop_interface.reset()
                should_reset_recording_instance = False
                
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
