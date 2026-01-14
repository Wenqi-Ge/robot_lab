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
    """处理 NumPy 2.x 保存，NumPy 1.x 加载的情况"""

    def find_class(self, module, name):
        # numpy 2.x -> 1.x 兼容
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core") :]
        return super().find_class(module, name)


def safe_np_load(file_path: str):
    """安全加载可能是由 NumPy 2.0 生成的 npy 文件

    这个函数可以处理 numpy 版本不匹配的情况：
    - 如果文件是用 numpy 2.x 保存的，在 numpy 1.x 环境下也能正常加载
    - 如果文件是用 numpy 1.x 保存的，在 numpy 2.x 环境下也能正常加载

    Args:
        file_path: npy 文件路径

    Returns:
        加载的数据（通常是字典）
    """
    try:
        # 尝试正常加载
        return np.load(file_path, allow_pickle=True).item()
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e):
            # 如果报错，说明是 2.0 格式，手动用字节流读取并使用兼容的 Unpickler
            with open(file_path, "rb") as f:
                # 读取并跳过 npy 文件头
                version = np.lib.format.read_magic(f)
                if version[0] == 1:
                    np.lib.format.read_array_header_1_0(f)
                else:
                    np.lib.format.read_array_header_2_0(f)
                # 此时文件指针停留在 pickle 数据开始处
                data = NumpyCompatUnpickler(f).load()
                # 如果返回的是 0 维 numpy 数组，需要调用 .item() 提取实际对象
                if isinstance(data, np.ndarray) and data.ndim == 0:
                    return data.item()
                return data
        else:
            raise e  # 其他错误直接抛出


def load_pickle_numpy_compat(path: str):
    """兼容的 pickle 加载函数（已废弃，请使用 safe_np_load）"""
    with open(path, "rb") as f:
        return NumpyCompatUnpickler(f).load()


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        body_indexes: Sequence[int],
        motion_decompose: bool = True,
        device: str = "cuda:0",
    ):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"

        self._device = device

        self._motion_decompose = motion_decompose

        data = load_pickle_numpy_compat(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(
            data["joint_pos"], dtype=torch.float32, device=device
        )
        self.joint_vel = torch.tensor(
            data["joint_vel"], dtype=torch.float32, device=device
        )
        self._body_pos_w = torch.tensor(
            data["body_pos_w"], dtype=torch.float32, device=device
        )
        self._body_quat_w = torch.tensor(
            data["body_quat_w"], dtype=torch.float32, device=device
        )
        self._body_lin_vel_w = torch.tensor(
            data["body_lin_vel_w"], dtype=torch.float32, device=device
        )
        self._body_ang_vel_w = torch.tensor(
            data["body_ang_vel_w"], dtype=torch.float32, device=device
        )
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]
