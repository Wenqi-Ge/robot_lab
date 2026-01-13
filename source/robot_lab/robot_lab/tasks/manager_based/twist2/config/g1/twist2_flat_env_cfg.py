import os

from isaaclab.utils import configclass

from robot_lab.assets.unitree import UNITREE_G1_29DOF_ACTION_SCALE, UNITREE_G1_29DOF_CFG
from robot_lab.tasks.manager_based.twist2.twist2_env_cfg import Twist2EnvCfg

TRACKING_ROOT = True
GLOBAL_OBS = False # 使用相对base作为参考系

class UnitreeG1Twist2IsaaclabEnvCfg(Twist2EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = UNITREE_G1_29DOF_ACTION_SCALE
        self.commands.motion.motion_file = \
            f"{os.path.dirname(__file__)}/motion/G1_Take_102.bvh_60hz.npz"
        # self.commands.motion.motion_file = f"{os.path.dirname(__file__)}/motion/G1_gangnam_style_V01.bvh_60hz.npz"
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = []

        