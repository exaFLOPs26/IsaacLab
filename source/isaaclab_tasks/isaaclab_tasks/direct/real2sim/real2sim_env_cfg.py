# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.anubis import ANUBIS_PD_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class Real2simEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    
    action_space = 17
    #+-------------------------------------+
    #|   Active Action Terms (shape: 17)   |
    #+-------+-----------------+-----------+
    #| Index | Name            | Dimension |
    #+-------+-----------------+-----------+
    #|   0   | armL_action     |         6 |
    #|   1   | armR_action     |         6 |
    #|   2   | gripperL_action |         1 |
    #|   3   | gripperR_action |         1 |
    #|   4   | base_action     |         3 |
    #+-------+-----------------+-----------+
    
    observation_space = 55

    #+-----------------------------------------------------+
    #| Active Observation Terms in Group: 'policy' (shape: (55,)) |
    #+--------------+-----------------------+--------------+
    #|    Index     | Name                  |    Shape     |
    #+--------------+-----------------------+--------------+
    #|      0       | joint_pos             |    (19,)     |
    #|      1       | joint_vel             |    (19,)     |
    #|      2       | actions               |    (17,)     |
    #+--------------+-----------------------+--------------+
            
    state_space = 55

    # simulation
    # [TODO] Match dt with the real robot
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = ANUBIS_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=3.0, replicate_physics=False)

    # custom parameters/scales
    # - controllable joint
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    
    # TODO Is this 1 correct?
    # - action scale
    action_scale = 1.0  # [N]
    
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    # - reset states/conditions
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]
    
    """               
    clip = {
                "link1_joint": (-0.523599, 1.91986),
                "link12_joint": (0.174533, 2.79253),
                "link13_joint": (-1.5708, 1.74533),
                "link14_joint": (-1.5708, 1.57085),
                "link15_joint": (-1.74533, 1.74533),
                "arm1_base_joint": (-0.523599, 0.523599),
            }
    """
            