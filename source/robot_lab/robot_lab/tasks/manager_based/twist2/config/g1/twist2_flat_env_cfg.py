import os

from isaaclab.utils import configclass

from robot_lab.assets.unitree import UNITREE_G1_29DOF_ACTION_SCALE, UNITREE_G1_29DOF_CFG
from robot_lab.tasks.manager_based.twist2.twist2_env_cfg import Twist2EnvCfg


@configclass
class UnitreeG1Twist2FlatEnvCfg(Twist2EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.actions.joint_pos.scale = UNITREE_G1_29DOF_ACTION_SCALE
        self.commands.motion.motion_file = (
            f"{os.path.dirname(__file__)}/motion/G1_Take_102.bvh_60hz.npz"
        )
        # self.commands.motion.motion_file = f"{os.path.dirname(__file__)}/motion/G1_gangnam_style_V01.bvh_60hz.npz"
        self.commands.motion.root_link_name = "base_link"

        self.commands.motion.body_names = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ]

        # 设置观察值的 scale（从配置文件读取）
        self.obs_scale_lin_vel = 2.0
        self.obs_scale_ang_vel = 0.25
        self.obs_scale_joint_pos = 1.0
        self.obs_scale_joint_vel = 0.05

        # 应用到观察配置中
        self.observations.teacher_policy.base_lin_vel.scale = self.obs_scale_lin_vel
        self.observations.teacher_policy.base_ang_vel.scale = self.obs_scale_ang_vel
        self.observations.teacher_policy.joint_pos.scale = self.obs_scale_joint_pos
        self.observations.teacher_policy.joint_vel.scale = self.obs_scale_joint_vel
