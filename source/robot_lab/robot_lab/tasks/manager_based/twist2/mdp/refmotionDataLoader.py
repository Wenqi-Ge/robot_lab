import torch
import numpy as np
import isaacsim  # 要先导入这个包，不然会报错
import isaaclab.utils.math as math_utils
import time
from tqdm import tqdm
import os

from robot_lab.utils.motion_loader_twist2 import safe_np_load


class PreMotionLoader:
    def __init__(
        self,
        device,
        motion_files: str | list[str],
        track_body_names,
        track_joint_names,
        track_root_link,
        save_dir: str = "./data/montio_data_twist2_test/",
        fps: int = 50,
        dt: float = 1 / 50,
        rm_data_name=[],
    ):
        self._device = device
        self._motion_files = motion_files
        self._track_body_names = track_body_names
        self._track_joint_names = track_joint_names
        self._track_root_link = track_root_link

        self.ref_data_joints_inx = None
        self.ref_data_body_inx = None

        self.max_frames_num = 1000

        self._save_dir = save_dir

        self._fps = fps
        self._dt = dt

        self._rm_data_name = rm_data_name

        self._load_motions(self._motion_files)

    def _load_motions(self, motion_files: list[str]):
        self._motion_names = []
        self._motion_weights = []  # default 1.0
        self._motion_num_frames = []
        self._motion_lengths = []

        self._motion_beg_frame_idx = []
        self._motion_end_frame_idx = []

        # 只存储 joint positions，keypoints 的 pos/quat 在运行时通过正向运动学计算
        self._motion_key_joint_pos = []
        self._motion_key_body_pos_base = []
        self._motion_key_body_quat_base = []

        self._motion_root_posi_w = []
        self._motion_root_quat_w = [] # x,y,z,w

        # 找到第一个不在排除列表中的 motion 文件用于初始化
        init_idx_motion = None
        for motion_file_i in motion_files:
            motion_name_i = motion_file_i.split("/")[-1].split(".")[0]
            if not any(rm_name in motion_name_i for rm_name in self._rm_data_name):
                init_idx_motion = motion_file_i
                break

        if init_idx_motion is None:
            raise ValueError("All motion files are in _rm_data_name exclusion list!")

        motion_data_raw = safe_np_load(init_idx_motion)  # dict
        all_jnts_name = motion_data_raw["dof_names"]
        all_body_names = motion_data_raw["body_names"]

        self.ref_data_joints_inx = [
            all_jnts_name.index(name) for name in self._track_joint_names
        ]
        self.ref_data_body_inx = [
            all_body_names.index(name) for name in self._track_body_names
        ]

        # 先统计实际会加载的 motion 数量（排除 rm_data_name 中的）
        valid_motion_count = 0
        for motion_file_i in motion_files:
            motion_name_i = motion_file_i.split("/")[-1].split(".")[0]
            if not any(rm_name in motion_name_i for rm_name in self._rm_data_name):
                valid_motion_count += 1

        for i in tqdm(range(len(motion_files))):
            motion_file_i = motion_files[i]
            motion_name_i = motion_file_i.split("/")[-1].split(".")[0]

            # 检查是否在排除列表中
            if any(rm_name in motion_name_i for rm_name in self._rm_data_name):
                print(f"[INFO] Skipping motion: {motion_name_i} (in rm_data_name)")
                continue

            motion_data_raw = safe_np_load(motion_file_i)  # dict

            motion_data_body_states = motion_data_raw["body_states"]
            motion_data_dof_pos_vel = motion_data_raw["dof_pos_vel"]

            motion_num_frames_i = motion_data_dof_pos_vel.shape[0]

            motion_weight_i = 1.0 / valid_motion_count

            if len(self._motion_end_frame_idx) == 0:
                motion_beg_frame_idx_i = 0
                motion_end_frame_idx_i = motion_num_frames_i - 1
            else:
                motion_beg_frame_idx_i = self._motion_end_frame_idx[-1] + 1
                motion_end_frame_idx_i = (
                    motion_beg_frame_idx_i + motion_num_frames_i - 1
                )

            motion_lengths_i = motion_num_frames_i * self._dt

            base_link_root_pos = torch.tensor(
                motion_data_body_states[:, 0, :3], device=self._device
            )

            # in npy q is [x,y,z, w]
            base_link_root_quat = torch.tensor(
                motion_data_body_states[:, 0, 3:7], device=self._device
            )
            # change to [w,x,y,z]
            base_link_root_quat = base_link_root_quat[:, [3, 0, 1, 2]]
            base_link_root_quat = math_utils.quat_unique(base_link_root_quat)

            track_joint_pos = torch.tensor(
                motion_data_dof_pos_vel[:, self.ref_data_joints_inx, 0],
                device=self._device,
                dtype=torch.float32,
            )
            track_body_pos_w = torch.tensor(
                motion_data_body_states[:, self.ref_data_body_inx, :3],
                device=self._device,
            )
            track_body_quat_w = torch.tensor(
                motion_data_body_states[:, self.ref_data_body_inx, 3:7],
                device=self._device,
            )
            track_body_quat_w = track_body_quat_w[:, :, [3, 0, 1, 2]]

            base_link_root_rot_expanded = base_link_root_quat.unsqueeze(1).repeat(
                1, track_body_pos_w.shape[1], 1
            )
            base_link_root_pos_expanded = base_link_root_pos.unsqueeze(1).repeat(
                1, track_body_pos_w.shape[1], 1
            )

            track_body_pos_b = math_utils.quat_apply_inverse(
                (base_link_root_rot_expanded),
                track_body_pos_w - base_link_root_pos_expanded,
            )
            track_body_quat_b = math_utils.quat_mul(
                math_utils.quat_inv(base_link_root_rot_expanded), track_body_quat_w
            )

            self._motion_names.append(motion_name_i)
            self._motion_weights.append(motion_weight_i)

            self._motion_num_frames.append(motion_num_frames_i)
            self._motion_lengths.append(motion_lengths_i)

            self._motion_key_body_pos_base.append(track_body_pos_b)
            self._motion_key_body_quat_base.append(track_body_quat_b)
            self._motion_key_joint_pos.append(track_joint_pos)

            self._motion_beg_frame_idx.append(motion_beg_frame_idx_i)
            self._motion_end_frame_idx.append(motion_end_frame_idx_i)

        self._motion_key_joint_pos = torch.vstack(self._motion_key_joint_pos)
        self._motion_key_body_pos_base = torch.vstack(self._motion_key_body_pos_base)
        self._motion_key_body_quat_base = torch.vstack(self._motion_key_body_quat_base)

        self._save_uniform_motions(self._save_dir)

    def _save_uniform_motions(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        # 检测数据中是否有 NaN
        has_nan = False
        nan_keys = []
        nan_motion_names = []

        data_tensors = {
            "motion_key_body_pos_base": self._motion_key_body_pos_base,
            "motion_key_body_quat_base": self._motion_key_body_quat_base,
            "motion_key_joint_pos": self._motion_key_joint_pos,
        }

        for key, value in data_tensors.items():
            if torch.isnan(value).any():
                has_nan = True
                nan_keys.append(key)
                for i in range(len(self._motion_names)):
                    beg_idx = self._motion_beg_frame_idx[i]
                    end_idx = self._motion_end_frame_idx[i] + 1
                    motion_data = value[beg_idx:end_idx]
                    if torch.isnan(motion_data).any():
                        nan_motion_names.append(self._motion_names[i])

        if has_nan:
            nan_motion_names_unique = list(set(nan_motion_names))
            print(
                f"[WARNING] Found NaN values in saved data! Keys with NaN: {nan_keys}"
            )
            if nan_motion_names_unique:
                print(f"[WARNING] Motion names with NaN: {nan_motion_names_unique}")
        else:
            print(f"[INFO] No NaN values found in saved data.")

        # ===== 保存为 memmap 格式，支持惰性加载 =====
        # 1. 保存元数据（小文件，直接加载到内存）
        metadata = {
            "motion_weights": self._motion_weights,
            "motion_num_frames": self._motion_num_frames,
            "motion_lengths": self._motion_lengths,
            "motion_beg_frame_idx": self._motion_beg_frame_idx,
            "motion_end_frame_idx": self._motion_end_frame_idx,
            # 保存大数据的 shape 信息
            "body_pos_shape": list(self._motion_key_body_pos_base.shape),
            "body_quat_shape": list(self._motion_key_body_quat_base.shape),
            "joint_pos_shape": list(self._motion_key_joint_pos.shape),
        }
        meta_path = os.path.join(save_dir, "motion_metadata.npy")
        np.save(meta_path, metadata)
        print(f"Saved metadata to {meta_path}")

        # 2. 保存大数据为 memmap 文件（可以懒加载）
        body_pos_np = self._motion_key_body_pos_base.cpu().numpy().astype(np.float32)
        body_quat_np = self._motion_key_body_quat_base.cpu().numpy().astype(np.float32)
        joint_pos_np = self._motion_key_joint_pos.cpu().numpy().astype(np.float32)

        body_pos_path = os.path.join(save_dir, "motion_body_pos.dat")
        body_quat_path = os.path.join(save_dir, "motion_body_quat.dat")
        joint_pos_path = os.path.join(save_dir, "motion_joint_pos.dat")

        # 使用 memmap 写入模式保存
        fp_body_pos = np.memmap(
            body_pos_path, dtype=np.float32, mode="w+", shape=body_pos_np.shape
        )
        fp_body_pos[:] = body_pos_np[:]
        fp_body_pos.flush()
        del fp_body_pos

        fp_body_quat = np.memmap(
            body_quat_path, dtype=np.float32, mode="w+", shape=body_quat_np.shape
        )
        fp_body_quat[:] = body_quat_np[:]
        fp_body_quat.flush()
        del fp_body_quat

        fp_joint_pos = np.memmap(
            joint_pos_path, dtype=np.float32, mode="w+", shape=joint_pos_np.shape
        )
        fp_joint_pos[:] = joint_pos_np[:]
        fp_joint_pos.flush()
        del fp_joint_pos

        total_size = (
            os.path.getsize(body_pos_path)
            + os.path.getsize(body_quat_path)
            + os.path.getsize(joint_pos_path)
        ) / 1024**2
        print(f"Saved memmap data to {save_dir}")
        print(f"  - body_pos: {os.path.getsize(body_pos_path)/1024**2:.2f} MB")
        print(f"  - body_quat: {os.path.getsize(body_quat_path)/1024**2:.2f} MB")
        print(f"  - joint_pos: {os.path.getsize(joint_pos_path)/1024**2:.2f} MB")
        print(f"  - Total: {total_size:.2f} MB")
