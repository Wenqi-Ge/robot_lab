"""
@File: motion_loader_twist2.py
@Author: Winky GE
@Date: 2026-01-06
@Brief: fit Twist2 motion data to the robot_lab format
"""

from __future__ import annotations

import math
import numpy as np
import pickle
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # numpy 2.x -> 1.x 兼容
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)

def load_pickle_numpy_compat(path: str):
    with open(path, "rb") as f:
        return NumpyCompatUnpickler(f).load()



class MotionLoader:
    def __init__(self, 
                 motion_file: str, 
                 body_indexes: Sequence[int], 
                 motion_decompose: bool = True,
                 device: str = "cuda:0"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        
        self._device = device

        self._motion_decompose = motion_decompose

        
        
        data = load_pickle_numpy_compat(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]