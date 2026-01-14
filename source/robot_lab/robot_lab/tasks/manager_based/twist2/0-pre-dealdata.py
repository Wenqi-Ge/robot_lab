import isaacsim # 要先导入这个包，不然会报错
from omni.isaac.kit import SimulationApp
# simulation_app = SimulationApp({"headless": True})

import omni
print(f"Omni module location: {omni.__path__}")
try:
    import omni.log
    print("omni.log is successfully loaded.")
except ImportError:
    print("omni.log could not be found!")

import sys
import glob
from omegaconf import OmegaConf, DictConfig
import datetime
import os


from robot_lab.robot_lab.tasks.manager_based.twist2.mdp.refmotionDataLoader import PreMotionLoader
from robot_lab.robot_lab.tasks.manager_based.twist2.mdp.assets_name import *  # noqa: F401, F403

data_dir_path = '/home/winky/Documents/code/humanoid/TWIST2_full/OMOMO_g1_GMR_50hz'



pkl_files_glob = glob.glob(os.path.join(data_dir_path, '**/*.pkl'), recursive=True)

# 打印结果
print(f'找到 {len(pkl_files_glob)} 个 .pkl 文件:')


ref_motion_data = PreMotionLoader(
    device='cuda:0',
    motion_files=pkl_files_glob,
    track_body_names=TWIST2_TRACKING_KPS,
    track_joint_names=JOINT_NAMES_ACTION_WITHOUT_ANKLE,
    track_root_link='base_link',
    save_dir=os.getcwd() + '/data/twist2_ref/',
    fps=50,
    dt=1/50,
    rm_data_name=[],
)