# exaFLOPs

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

ANUBIS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/root/IsaacLab/source/isaaclab_assets/data/Robots/MM/anubis/anubis_v3.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
            max_linear_velocity=12.0,
            max_angular_velocity=12.0,
            max_depenetration_velocity=10.0,
            max_contact_impulse=100.0,
            stabilization_threshold=0.5,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # base
            "dummy_base_prismatic_y_joint": 0.0,
            "dummy_base_prismatic_x_joint": 0.0,
            "dummy_base_revolute_z_joint": 0.0,

            # arm <-> base (rad)
            "arm1_base_link_joint": 0.0,
            "arm2_base_link_joint": 0.0,

            # Right arm
            "link11_joint": -0.69289571,
            "link12_joint": 2.34048653,
            "link13_joint": -0.07679449,
            "link14_joint": 0.52359878,
            "link15_joint": -0.17453293,
            
            # Left arm
            "link21_joint": -0.69289571,
            "link22_joint": 2.34048653,
            "link23_joint": -0.07679449,
            "link24_joint": -0.52359878,
            "link25_joint": 0.17453293,
            
            # finger
            "gripper.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),

    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["dummy_base_.*"],
            effort_limit_sim=1e10,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=1e10,  # tip:: For velocity control of the base with dummy mechanism, we recommend setting high damping gains to the joints. This ensures that the base remains unperturbed from external disturbances, such as an arm mounted on the base.
        ),
        "arm_base": ImplicitActuatorCfg(
            joint_names_expr=["arm.*"],
            effort_limit_sim=1e5,
            velocity_limit_sim=1e4,
            stiffness=1e10,
            damping=1e5,
        ),
        "arm_link": ImplicitActuatorCfg(
            joint_names_expr=["link.*"],
            effort_limit_sim=1e5,
            velocity_limit_sim=1e4,
            stiffness=1e10,
            damping=1e5,
        ),
        "anubis_left_hand": ImplicitActuatorCfg(
            joint_names_expr=["gripper1.*"],
            effort_limit_sim=1e3,
            velocity_limit_sim=0.1,
            stiffness=6e4,
            damping=1e3,
            friction=1.0,
        ),
        "anubis_right_hand": ImplicitActuatorCfg(
            joint_names_expr=["gripper2.*"],
            effort_limit_sim=1e3,
            velocity_limit_sim=0.1,
            stiffness=6e4,
            damping=1e3,
            friction=1.0,
        ),
    },
)



ANUBIS_PD_CFG = ANUBIS_CFG.copy()
# ANUBIS_PD_CFG.spawn.rigid_props.disable_gravity = True
# ANUBIS_PD_CFG.actuators["arm_link"].stiffness = 400.0
# ANUBIS_PD_CFG.actuators["arm_link"].damping = 80.0
# ANUBIS_PD_CFG.actuators["arm_base"].stiffness = 400.0
# ANUBIS_PD_CFG.actuators["arm_base"].damping = 80.0
