"""
@File: commands.py
@Author: Winky GE
@Date: 2026-01-05
@Brief:  Twist2 environment command definitions
load motion command from file
"""




from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv
from robot_lab.utils import extended_math_utils


class RefMotionCommand(CommandTerm):
    cfg: RefMotionCommandCfg

    def __init__(self, cfg: RefMotionCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.env = env

        # 加载参考动作数据
        if cfg.if_tracking:
            self.ref_motions = MontioLoaderNew(
                device=self.device,
                motion_file=cfg.ref_motion_file,
                track_body_names=cfg.track_body_names if hasattr(cfg, 'track_body_names') else [],
                track_joint_names=cfg.track_joint_names if hasattr(cfg, 'track_joint_names') else [],
                track_root_link=cfg.track_root_link if hasattr(cfg, 'track_root_link') else "base_link"
            )
        # else:
        #     self.ref_motions = None

            self._num_motions = self.ref_motions.ref_motion_weights.shape[0]

            self.control_dt = float(self.env.cfg.decimation * self.env.cfg.sim.dt)  # 0.02 (更稳)
            self.steps_per_bin = int(round(1.0 / self.control_dt))                  # 50

            self.motion_num_frames = self.ref_motions.ref_motion_num_frames          # (M,) long tensor
            
            self._bin_counts = (self.motion_num_frames + self.steps_per_bin - 1) // self.steps_per_bin  # ceil



            self.bin_failed_count = torch.zeros(self._bin_counts.sum(), device=self.device)

            self.motion_failed_count = torch.zeros(self._num_motions, device=self.device)

            self._current_motion_ids = torch.zeros(self.num_envs, 
                                                dtype=torch.long, 
                                                device=self.device)
            self._current_bin_ids = torch.zeros(self.num_envs, 
                                                dtype=torch.long, 
                                                device=self.device)

            # 初始化采样概率（用于第一次采样）
            self._sampling_motion_probs = torch.ones(self._num_motions, device=self.device) / self._num_motions

            self._sampling_bin_probs = [torch.ones(int(self._bin_counts[i].item()), device=self.device) / int(self._bin_counts[i].item()) for i in range(self._num_motions)]

                                            


        # 找到每个关节名称在 joint_names 中的索引
        self._track_ref_jnts_idx = [self.robot.data.joint_names.index(joint_name) for joint_name in TRACKING_UPPER_JOINTS]
        
        self._track_ref_jnts_idx_can_reset = [self.robot.data.joint_names.index(joint_name) for joint_name in TRACKING_UPPER_JOINTS_CAN_RESET]
        # 找到每个身体名称在 body_names 中的索引
        self._track_ref_body_idx = [self.robot.data.body_names.index(body_name) for body_name in TRACKING_KEYPONITS]

        
        self.track_frame_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 获取参考动作数据的形状
        # 使用 TRACKING_KEYPONITS 和 ref_motion_key_joint_pos 的形状
        num_bodies = len(TRACKING_KEYPONITS)
        num_joints = len(TRACKING_UPPER_JOINTS)

        self._ref_key_body_pos_base = torch.zeros(
            self.num_envs, num_bodies, 3, device=self.device
        )
        
        self._ref_key_body_quat_base = torch.zeros(
            self.num_envs, num_bodies, 4, device=self.device
        )

        self._ref_key_body_quat_w = torch.zeros(
            self.num_envs, num_bodies, 4, device=self.device
        )

        self._ref_key_body_pos_w = torch.zeros(
            self.num_envs, num_bodies, 3, device=self.device
        )   

        # command
        self._ref_key_body_tri_pose_b = torch.zeros(
            self.num_envs, num_bodies, 9, device=self.device
        )

        self._ref_key_body_tri_pose_w = torch.zeros(
            self.num_envs, num_bodies, 9, device=self.device
        )
        
        self._ref_key_joint_pos = torch.zeros(
            self.num_envs, num_joints, device=self.device
        )

        
        # from BeyondMimic
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], 
            device=self.device
        )
        
        self.kernel = self.kernel / self.kernel.sum()

        # 找到手部在 TRACKING_KEYPONITS 中的索引（不是全局body索引）
        self._hands_ee_name = ['left_wrist_roll_link', 'right_wrist_roll_link']
        self._hands_ee_idx = [TRACKING_KEYPONITS.index(ee_name) for ee_name in self._hands_ee_name]

        self.metrics["error_ref_key_jnts"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_ref_key_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_ref_key_body_quat"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_ref_key_body_tri_pose"] = torch.zeros(self.num_envs, device=self.device)
        
        # 缓存 max_command_step 避免重复计算
        max_command_time = self.cfg.resampling_time_range[1]
        self._max_command_step = max_command_time / self._env.step_dt

        self.metrics["error_ref_hands_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_ref_hands_rot"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "RefMotionCommand:\n"
        msg += f"\tNum motions: {self._num_motions}\n"
        msg += f"\tMax frames: {self.ref_motions.ref_motion_max_frames}\n"
        msg += f"\tAdaptive kernel size: {self.cfg.adaptive_kernel_size}\n"
        msg += f"\tAdaptive base weight: {self.cfg.adaptive_base_weight}\n"
        return msg


    @property
    def ref_motion_frame_idx(self) -> torch.Tensor:
        return self.track_frame_idx

    @property
    def ref_key_body_pos_base(self) -> torch.Tensor:
        return self._ref_key_body_pos_base

    @property
    def ref_key_body_quat_base(self) -> torch.Tensor:
        return self._ref_key_body_quat_base
    
    @property
    def ref_key_joint_pos(self) -> torch.Tensor:
        return self._ref_key_joint_pos

    @property
    def ref_key_body_tri_pose_b(self) -> torch.Tensor:
        return self._ref_key_body_tri_pose_b
        
    @property
    def robot_current_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self._track_ref_body_idx]
    
    @property
    def robot_current_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self._track_ref_body_idx]

    @property
    def robot_current_body_tri_pose_w(self) -> torch.Tensor:
        return ext_math_utils.triangle_vertices_from_SE3(
            ext_math_utils.pos_quat_to_SE3(self.robot_current_body_pos_w, 
                                           self.robot_current_body_quat_w)
        ).reshape(self.num_envs, -1, 9)
    
    @property
    def robot_current_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self._track_ref_jnts_idx]


    @property
    def ref_command_tri(self) -> torch.Tensor:
        return self.ref_key_body_tri_pose_b

    @property
    def ref_command_jnts(self) -> torch.Tensor:
        return self.ref_key_joint_pos

    @property
    def ref_cmd_quat_pos_w(self):
        return self._ref_key_body_quat_w, self._ref_key_body_pos_w

    @property
    def command(self) -> torch.Tensor:
        """The reference motion command. Returns the concatenated reference joint positions and body triangle poses."""
        return self.ref_key_joint_pos, self.ref_command_tri

    @property
    def ref_command_jnts_err(self) -> torch.Tensor:
        return self.ref_key_joint_pos - self.robot_current_joint_pos

    @property
    def ref_command_tri_err(self) -> torch.Tensor:
        return self._ref_key_body_tri_pose_w.reshape(self.num_envs, -1, 9) - self.robot_current_body_tri_pose_w

    def _update_metrics(self):
        # 使用缓存的 max_command_step
        self.metrics["error_ref_key_jnts"] += torch.abs(self.ref_key_joint_pos - self.robot_current_joint_pos).mean(dim=-1) / self._max_command_step
        self.metrics["error_ref_key_body_pos"] += torch.norm(self._ref_key_body_pos_w - self.robot_current_body_pos_w, dim=-1).mean(dim=-1) / self._max_command_step
        self.metrics["error_ref_key_body_quat"] += quat_error_magnitude(self._ref_key_body_quat_w, self.robot_current_body_quat_w).mean(dim=-1) / self._max_command_step
        self.metrics["error_ref_key_body_tri_pose"] += torch.norm(self._ref_key_body_tri_pose_w - self.robot_current_body_tri_pose_w, dim=-1).mean(dim=-1) / self._max_command_step

        self.metrics["error_ref_hands_pos"] += torch.norm(self.robot_current_body_pos_w[:, self._hands_ee_idx] - self._ref_key_body_pos_w[:, self._hands_ee_idx], dim=-1).mean(dim=-1) / self._max_command_step
        self.metrics["error_ref_hands_rot"] += quat_error_magnitude(self.robot_current_body_quat_w[:, self._hands_ee_idx], self._ref_key_body_quat_w[:, self._hands_ee_idx]).mean(dim=-1) / self._max_command_step
        pass

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """基于 BeyondMimic 的自适应采样策略"""
        # 只检测真正失败的 episode（terminated），不计入超时（truncated）
        # truncated 表示达到时间限制，说明机器人成功运行到了时间限制，不应视为失败
        episode_failed = self.env.termination_manager.terminated[env_ids]

        if torch.any(episode_failed):
            # 获取失败 episode 对应的 motion 和 bin
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            resample_env_ids = env_ids_tensor[episode_failed]
            
            current_failed_motion_ids = self._current_motion_ids[resample_env_ids]
            
            # 无论是否使用 bin 采样，都更新 motion 级别的失败计数
            self.motion_failed_count[current_failed_motion_ids] += 1

            # 仅在使用 bin 采样时更新 bin 级别的失败计数
            if self.cfg.use_bin_sampling:
                current_bin_failed_ids = (self.track_frame_idx[resample_env_ids] // self.steps_per_bin)
                current_bin_failed_ids = torch.minimum(
                    current_bin_failed_ids,
                    self._bin_counts[current_failed_motion_ids] - 1
                )
                # 更新失败计数
                self.bin_failed_count[current_failed_motion_ids, current_bin_failed_ids] += 1

        # 计算 motion 级别的采样概率
        base_p = float(self.cfg.adaptive_base_weight)
        motion_weights = self.motion_failed_count + base_p
        # 乘以 motion 的原始权重（有 NaN 的 motion 权重为 0）
        motion_weights = motion_weights * self.ref_motions.ref_motion_weights
        self._sampling_motion_probs = motion_weights / (motion_weights.sum() + 1e-8)
        
        # 仅在使用 bin 采样时计算 bin 级别的采样概率
        if self.cfg.use_bin_sampling:
            # 计算每个 motion 内部 bin 的采样概率
            # 使用 kernel 卷积平滑失败计数
            kernel_size = self.cfg.adaptive_kernel_size
            if kernel_size > 1:
                x = self.bin_failed_count.unsqueeze(1)                  # (M,1,B)
                x = F.pad(x, (kernel_size - 1, 0))                               # 左 pad
                k = self.kernel.view(1, 1, -1).to(x.dtype)             # (1,1,K)
                smoothed = F.conv1d(x, k).squeeze(1)                   # (M,B)
            else:
                smoothed = self.bin_failed_count
            
            valid = self._bin_mask.to(smoothed.dtype) 
            smoothed = smoothed * valid

            bin_weights = smoothed + base_p * valid
            
            self._sampling_bin_probs = bin_weights / (bin_weights.sum(dim=1, keepdim=True) + 1e-8)

    # reset 的时候会被自动调用？
    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        n_envs = len(env_ids)
        env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)


        if self.cfg.use_uniform_sampling and self.cfg.if_tracking:
            # 完全均匀随机采样模式
            # 1. 均匀随机采样 motion id（只从有效 motion 中采样）
            valid_motion_mask = self.ref_motions.ref_motion_weights > 0
            valid_motion_ids = torch.where(valid_motion_mask)[0]
            
            # 从有效 motion 中均匀随机采样
            random_indices = torch.randint(0, len(valid_motion_ids), (n_envs,), device=self.device)
            sampled_motion_ids = valid_motion_ids[random_indices]
            self._current_motion_ids[env_ids_tensor] = sampled_motion_ids
            
            # 2. 计算起始帧
            motion_lengths = self.motion_num_frames[sampled_motion_ids]
            min_remain = self.cfg.uniform_min_remain_frames
            
            # 如果 motion 大于 min_remain 帧，从 [0, 总帧数-min_remain] 随机采样
            # 否则从 0 开始
            max_start_frames = torch.where(
                motion_lengths > min_remain,
                motion_lengths - min_remain,
                torch.zeros_like(motion_lengths)
            )
            # max_start_frames 为 0 时，起始帧就是 0
            # max_start_frames > 0 时，从 [0, max_start_frames] 随机采样
            random_starts = (torch.rand(n_envs, device=self.device) * (max_start_frames.float() + 1)).long()
            random_starts = torch.minimum(random_starts, max_start_frames)  # 确保不超过上界
            
            sampled_motions_beg_frame_idx = self.ref_motions.ref_motion_beg_frame_idx[sampled_motion_ids]
            self.track_frame_idx[env_ids_tensor] = random_starts + sampled_motions_beg_frame_idx
            
            # bin_ids 设为 0（不使用时无意义）
            self._current_bin_ids[env_ids_tensor] = 0
        elif self.cfg.use_uniform_sampling and not self.cfg.if_tracking:
            pass
        else:
        # 执行自适应采样，更新采样概率
            self._adaptive_sampling(env_ids)

            n_envs = len(env_ids)
            env_ids_tensor = torch.tensor(env_ids, device=self.device, dtype=torch.long)

            # 根据概率采样 motion
            sampled_motion_ids = torch.multinomial(
                self._sampling_motion_probs, 
                num_samples=n_envs, 
                replacement=True
            )
            self._current_motion_ids[env_ids_tensor] = sampled_motion_ids

            if self.cfg.use_bin_sampling:
                # 使用 bin 级别采样（原始 BeyondMimic 方式）
                # 根据概率采样 bin
                # 为每个采样的 motion 采样对应的 bin
                sampled_bin_ids = torch.zeros(n_envs, device=self.device, dtype=torch.long)

                for i in range(n_envs):
                    motion_id = int(sampled_motion_ids[i].item())
                    valid_bins = int(self._bin_counts[motion_id].item())

                    probs = self._sampling_bin_probs[motion_id, :valid_bins]
                    probs = probs / (probs.sum() + 1e-8)
                    sampled_bin_ids[i] = torch.multinomial(probs, num_samples=1)

                self._current_bin_ids[env_ids_tensor] = sampled_bin_ids

                # 计算起始帧（bin_id * steps_per_bin）
                start_frames = sampled_bin_ids * self.steps_per_bin
                remain = self.motion_num_frames[sampled_motion_ids] - start_frames
                # 在 bin 内随机偏移
                max_offsets = torch.minimum(remain, torch.full_like(remain, self.steps_per_bin))
                max_offsets = max_offsets.clamp(min=1)  # 避免上界为0

                # 使用 torch.rand 生成 [0, max_offsets) 范围内的随机整数
                # torch.randint 不支持 tensor 作为 high 参数
                random_offsets = (torch.rand(n_envs, device=self.device) * max_offsets).long()
                self.track_frame_idx[env_ids_tensor] = start_frames + random_offsets
            else:
                # 不使用 bin 采样：从每个 motion 的前 start_frame_ratio 比例的帧中随机选择起始帧
                # 这样可以减少显存使用，同时按 motion 级别的成功率进行自适应采样
                motion_lengths = self.motion_num_frames[sampled_motion_ids]
                # 计算每个 motion 可以选择的最大起始帧（前 start_frame_ratio 比例）
                max_start_frames = (motion_lengths.float() * self.cfg.start_frame_ratio).long()
                max_start_frames = max_start_frames.clamp(min=1)  # 确保至少有1帧可选
                
                # 随机采样起始帧
                random_starts = (torch.rand(n_envs, device=self.device) * max_start_frames.float()).long()
                sampled_motions_beg_frame_idx = self.ref_motions.ref_motion_beg_frame_idx[sampled_motion_ids]
                
                self.track_frame_idx[env_ids_tensor] = random_starts + sampled_motions_beg_frame_idx
                
                # bin_ids 设为 0（不使用 bin 采样时无意义，但保持一致性）
                self._current_bin_ids[env_ids_tensor] = 0

        if self.cfg.if_tracking:
        # 获取参考姿态（此时 track_frame_idx 已经是有效范围内的起始帧）
            frame_ids = self.track_frame_idx[env_ids_tensor]
            
            # 使用 get_frames_batch 获取数据（支持 memmap 和 legacy 模式）
            ref_joint_pos, _, _ = self.ref_motions.get_frames_batch(frame_ids, device=self.device)
        else:
            # 创建固定关节位置的 tensor，shape 为 (n_envs, num_joints)
            # TRACKING_UPPER_JOINTS 有 17 个关节：左臂7个 + 右臂7个 + waist_yaw + waist_roll + waist_pitch
            fixed_joint_values = [-0.39, 0.26, 0, 0, 0, 0, 0,  # left arm: 7 joints
                                  -0.39, -0.26, 0, 0, 0, 0, 0,  # right arm: 7 joints
                                  0, 0, 0]  # waist: yaw, roll, pitch (3 joints)
            # 扩展为 (n_envs, 17) 的 tensor
            ref_joint_pos = torch.tensor(
                [fixed_joint_values] * n_envs,
                device=self.device,
                dtype=torch.float32
            )

        # 写入关节状态（只写入跟踪的关节）
        joint_vel = torch.zeros_like(ref_joint_pos[:,:-2])
        self.robot.write_joint_state_to_sim(
            ref_joint_pos[:,:-2], # 不写入 waist_roll_joint 和 waist_pitch_joint
            joint_vel,
            joint_ids=self._track_ref_jnts_idx_can_reset,
            env_ids=env_ids_tensor
        )

    def _update_command(self):
        if self.cfg.if_tracking:
            # 更新时间步
            self.track_frame_idx += 1

            # 检查哪些环境已经到达动作末尾，需要重采样
            # motion_lengths = self.motion_num_frames[self._current_motion_ids]
            tracking_motion_end_frame_idx = self.ref_motions.ref_motion_end_frame_idx[self._current_motion_ids]

            env_ids = torch.where(self.track_frame_idx >= tracking_motion_end_frame_idx)[0]

            if len(env_ids) > 0:
                self._resample_command(env_ids.tolist())

            # 更新参考姿态缓存（用于 reward 计算）
            # clamp 到每个 motion 的实际帧数范围内
            # motion_lengths_current = self.motion_num_frames[self._current_motion_ids]
            frame_ids = torch.minimum(self.track_frame_idx, tracking_motion_end_frame_idx - 1)
            # 缓存 beg_frame_idx 避免重复索引
            beg_frame_idx = self.ref_motions.ref_motion_beg_frame_idx[self._current_motion_ids]
            frame_ids = frame_ids.clamp(min=beg_frame_idx)  # 确保非负
            
            # 使用 get_frames_batch 获取数据（支持 memmap 和 legacy 模式）
            self._ref_key_joint_pos, self._ref_key_body_pos_base, self._ref_key_body_quat_base = \
                self.ref_motions.get_frames_batch(frame_ids, device=self.device)
            
            # base
            self._ref_key_body_tri_pose_b = ext_math_utils.triangle_vertices_from_SE3(
                ext_math_utils.pos_quat_to_SE3(self._ref_key_body_pos_base, 
                                            self._ref_key_body_quat_base)
            )
            
            # 同时更新 world 坐标系下的参考姿态（用于 ref_command_tri_err）
            _ref_key_body_pos_w, _ref_key_body_quat_w = math_utils.combine_frame_transforms(
                self.robot.data.root_pos_w.unsqueeze(1).expand(-1, len(self._track_ref_body_idx), 3).float(),
                self.robot.data.root_quat_w.unsqueeze(1).expand(-1, len(self._track_ref_body_idx), 4).float(),
                self._ref_key_body_pos_base,
                self._ref_key_body_quat_base
            )
            self._ref_key_body_pos_w = _ref_key_body_pos_w.reshape(self.num_envs, -1, 3)
            self._ref_key_body_quat_w = _ref_key_body_quat_w.reshape(self.num_envs, -1, 4)
            self._ref_key_body_tri_pose_w = ext_math_utils.triangle_vertices_from_SE3(
                ext_math_utils.pos_quat_to_SE3(self._ref_key_body_pos_w, self._ref_key_body_quat_w)
            ).reshape(self.num_envs, -1, 9)

            # 对 bin_failed_count 进行指数衰减（仅在使用 bin 采样时有意义）
            if self.cfg.use_bin_sampling:
                self.bin_failed_count = (
                    self.cfg.adaptive_alpha * self.bin_failed_count
                )
        else:
            pass

            


            self._ref_key_joint_pos = torch.tensor(
                [fixed_joint_values] * self.num_envs,
                device=self.device,
                dtype=torch.float32
            )
            self._ref_key_body_pos_base = torch.tensor(
                [fixed_kps_b] * self.num_envs,
                device=self.device,
                dtype=torch.float32
            )
            self._ref_key_body_quat_base = torch.tensor(
                [fixed_kps_quat_b] * self.num_envs,
                device=self.device,
                dtype=torch.float32
            )
            self._ref_key_body_tri_pose_b = torch.tensor(
                [fixed_kps_tri_b] * self.num_envs,
                device=self.device,
                dtype=torch.float32
            )

            # 同时更新 world 坐标系下的参考姿态（用于 ref_command_tri_err）
            _ref_key_body_pos_w, _ref_key_body_quat_w = math_utils.combine_frame_transforms(
                self.robot.data.root_pos_w.unsqueeze(1).expand(-1, len(self._track_ref_body_idx), 3).float(),
                self.robot.data.root_quat_w.unsqueeze(1).expand(-1, len(self._track_ref_body_idx), 4).float(),
                self._ref_key_body_pos_base,
                self._ref_key_body_quat_base
            )
            self._ref_key_body_pos_w = _ref_key_body_pos_w.reshape(self.num_envs, -1, 3)
            self._ref_key_body_quat_w = _ref_key_body_quat_w.reshape(self.num_envs, -1, 4)
            self._ref_key_body_tri_pose_w = ext_math_utils.triangle_vertices_from_SE3(
                ext_math_utils.pos_quat_to_SE3(self._ref_key_body_pos_w, self._ref_key_body_quat_w)
            ).reshape(self.num_envs, -1, 9)


    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "ref_body_visualizer"):
                self.ref_body_visualizer = VisualizationMarkers(self.cfg.ref_body_visualizer_cfg)
            self.ref_body_visualizer.set_visibility(True)
        else:
            if hasattr(self, "ref_body_visualizer"):
                self.ref_body_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        
        # 可视化参考身体位置
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_quat_w = self.robot.data.root_quat_w.clone()
        
        # 将参考身体位置从 base 坐标系转换到 world 坐标系
        ref_pos_b = self._ref_key_body_pos_base  # (num_envs, num_bodies, 3)
        if ref_pos_b.dim() == 3:
            # 转换到世界坐标系进行可视化
            num_bodies = ref_pos_b.shape[1]
            ref_pos_w = []
            for i in range(num_bodies):
                pos_w = math_utils.quat_rotate(base_quat_w, ref_pos_b[:, i]) + base_pos_w
                ref_pos_w.append(pos_w)
            ref_pos_w = torch.stack(ref_pos_w, dim=1)  # (num_envs, num_bodies, 3)
            
            # 只可视化第一个环境的参考位置
            self.ref_body_visualizer.visualize(ref_pos_w[0])









    
@configclass
class RefMotionCommandCfg(CommandTermCfg):
    """Configuration for the reference motion command."""

    class_type: type = RefMotionCommand

    asset_name: str = MISSING

    ref_motion_file: str = MISSING

    if_tracking: bool = False
    
    # 跟踪的身体名称和关节名称
    track_body_names: list[str] = MISSING
    track_joint_names: list[str] = MISSING
    track_root_link: str = "base_link"
    
    # 自适应采样参数 (BeyondMimic)
    adaptive_kernel_size: int = 5
    adaptive_lambda: float = 0.8
    adaptive_base_weight: float = 0.1  # 0 失败的基础权重，防止遗忘
    adaptive_alpha: float = 0.999  # 衰减系数
    
    # 采样模式控制
    use_bin_sampling: bool = False
    """是否使用 bin 级别采样。True: 按 bin 采样（原始 BeyondMimic）；
    False: 按 motion 成功率采样，从前 start_frame_ratio 的帧中随机起始。"""
    
    start_frame_ratio: float = 0.5
    """当 use_bin_sampling=False 时，从每个 motion 的前 start_frame_ratio 比例的帧中随机选择起始帧。默认 0.5 (50%)。"""

    # 新增：禁用权重采样，使用完全均匀随机采样
    use_uniform_sampling: bool = True
    """禁用权重采样，完全随机采样 motion id。
    当启用时：
    - Motion ID 从所有有效 motion 中均匀随机采样
    - 起始帧：如果 motion 大于 uniform_min_remain_frames 帧，从 [0, 总帧数-uniform_min_remain_frames] 随机采样；
              否则从 0 开始
    """
    
    uniform_min_remain_frames: int = 100
    """使用 uniform sampling 时，保留的最小剩余帧数。默认 100 帧。
    - 如果 motion 总帧数 > uniform_min_remain_frames，起始帧从 [0, 总帧数-uniform_min_remain_frames] 随机采样
    - 如果 motion 总帧数 <= uniform_min_remain_frames，起始帧为 0
    """

    ref_body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/ref_body"
    )
    ref_body_visualizer_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)



    