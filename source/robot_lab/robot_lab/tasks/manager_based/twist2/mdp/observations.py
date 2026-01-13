"""
@File: observations.py
@Author: Winky GE
@Date: 2026-01-05
@Brief: Twist2 environment observation functions
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING


import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from robot_lab.tasks.manager_based.twist2.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def base_local_linear_vel()



def base_local_angle_vel()



def keypoints_posi_b(env: ManagerBasedRLEnv, 
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    kps_posi_b = math_utils.quat_rotate_inverse(
        asset.data.root_quat_w.unsqueeze(1),
        asset.data.body_pos_w[:, asset_cfg.body_ids] - asset.data.root_pos_w.unsqueeze(1).expand(-1, len(asset_cfg.body_ids), -1)
    )
    return kps_posi_b.reshape(env.num_envs, -1)



def base_local_delate_posi()


def base_local_delate_rot()


