# Environment configuration for the Anubis robot in the Cabinet task for RL training.
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from isaaclab_tasks.manager_based.mobile_manipulation.packing import mdp

from isaaclab_tasks.manager_based.mobile_manipulation.packing.packing_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG,
    PackingEnvCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets.robots.anubis_wheels import ANUBIS_CFG  # isort:skip

@configclass
class AnubisPackingEnvCfg(PackingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set franka as robot
        self.scene.robot = ANUBIS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set Actions for the specific robot type (franka)
        self.actions.armR_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["link1.*", "arm1.*"],
            scale=1.0,
            use_default_offset=True,
            clip = {
                "arm1_base_joint": (-0.523599, 0.523599),
                "link11_joint": (-0.523599, 1.91986),
                "link12_joint": (0.174533, 2.79253),
                "link13_joint": (-1.5708, 1.74533),
                "link14_joint": (-1.5708, 1.57085),
                "link15_joint": (-1.74533, 1.74533),
            }
        )
        self.actions.armL_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["link2.*","arm2.*"],
            scale=1.0,
            use_default_offset=True,
            clip = {
                "link21_joint": (-0.523599, 1.91986),
                "link22_joint": (0.174533, 2.79253),
                "link23_joint": (-1.5708, 1.74533),
                "link24_joint": (-1.5708, 1.57085),
                "link25_joint": (-1.74533, 1.74533),
                "arm2_base_joint": (-0.523599, 0.523599),
            }
        )
        self.actions.gripperR_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper1.*"],
            open_command_expr={"gripper1.*": 0.04},
            close_command_expr={"gripper1.*": 0.0},
        )

        self.actions.gripperL_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper2.*"],
            open_command_expr={"gripper2.*": 0.04},
            close_command_expr={"gripper2.*": 0.0},
        )

        self.actions.base_action = mdp.JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["Omni.*"],
        )
        
        # self.scene.object = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Object",
        #     spawn=sim_utils.MultiAssetSpawnerCfg(
        #         assets_cfg=[
        #             sim_utils.ConeCfg(
        #                 radius=0.3,
        #                 height=0.6,
        #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        #             ),
        #             sim_utils.CuboidCfg(
        #                 size=(0.3, 0.3, 0.3),
        #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        #             ),
        #             sim_utils.SphereCfg(
        #                 radius=0.3,
        #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
        #             ),
        #         ],
        #         random_choice=True,
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #             solver_position_iteration_count=4, solver_velocity_iteration_count=0
        #         ),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(1.3, 0.0, 1.5)),
        # )
        
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[1.3, 2, 1], rot=[0.707, 0, 0, 0.707]),
            spawn=sim_utils.CylinderCfg(
                radius=0.018,
                height=0.35,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15), metallic=1.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="max",
                    restitution_combine_mode="min",
                    static_friction=0.9,
                    dynamic_friction=0.9,
                    restitution=0.0,
                ),
            ),
        )

        # Listens to the required transforms
        # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
        # the other frames are the fingers
        self.scene.ee_R_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/vr_headset_frame",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/RightEndEffectorFrameTransformer_R"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link1",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/gripper1L",
                #     name="tool_leftfinger",
                #     offset=OffsetCfg(
                #         pos=(0.0, 0.0, 0.046),
                #     ),
                # ),
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/gripper1R",
                #     name="tool_rightfinger",
                #     offset=OffsetCfg(
                #         pos=(0.0, 0.0, 0.046),
                #     ),
                # ),
            ],
        )
        self.scene.ee_L_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/vr_headset_frame",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/LeftEndEffectorFrameTransformer_L"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link2",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/gripper2L",
                #     name="tool_leftfinger",
                #     offset=OffsetCfg(
                #         pos=(0.0, 0.0, 0.046),
                #     ),
                # ),
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/gripper2R",
                #     name="tool_rightfinger",
                #     offset=OffsetCfg(
                #         pos=(0.0, 0.0, 0.046),
                #     ),
                # ),
            ],
        )

@configclass
class AnubisPackingEnvCfg_PLAY(AnubisPackingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 6
        # disable randomization for play
        self.observations.policy.enable_corruption = False
