import glob
import numpy as np
from tqdm import tqdm
from typing import Literal

import torch
from torch.utils.data import Dataset
from torch import nn

class AMASSDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        vposer: nn.Module,
        input_length: int,
        target_length: int,
        split: Literal["train", "test", "all"] = "all",
        device: torch.device = "cuda"
    ) -> None:
        super().__init__()

        self.device = device
        self.split = split

        self.vposer = vposer
        self.vposer.to(self.device)

        self._root_dir = root_dir
        self._file_paths = self._get_file_paths()

        self.input_length = input_length
        self.target_length = target_length

        self._all_motion = self._load_all()

    def __len__(self):
        return len(self._all_motion)

    def _get_file_paths(self, ratio: float = 0.8):
        paths = glob.glob(f"{self._root_dir}/*poses.npz")

        idx = int(len(paths) * ratio)
        return paths[:idx] if self.split == "train" else paths[idx:] if self.split == "test" else paths

    def _preprocess(self, motion_feats):
        N = motion_feats.size(0)

        start = np.random.randint(N - self.input_length  - self.target_length + 1)
        end = start + self.input_length

        motion_input = motion_feats[start:end]
        motion_target = motion_feats[end:end + self.target_length]

        return motion_input.float(), motion_target.float()

    def _load_all(self):
        all_motion = []

        for file_path in tqdm(self._file_paths):
            amass_info = np.load(file_path)

            motion_poses = amass_info["poses"] # 156 joints(all joints of SMPL)
            # [3:66] removes global rotation, hands/fingers, and anything else other than 21 major body joints
            motion_poses = motion_poses[:, 3:66]
            motion_poses = torch.from_numpy(motion_poses).type(torch.float)

            N = motion_poses.size(0)
            if N < self.target_length + self.input_length:
                continue

            # Sample frames
            frame_rate = amass_info["mocap_framerate"]

            sample_rate = int(frame_rate // 25)
            sample_rate = min(sample_rate, N // (self.target_length + self.input_length))
            sampled_index = np.arange(0, N, sample_rate)

            motion_poses = motion_poses[sampled_index]

            # Run Vposer encoder on all frames
            motion_poses = motion_poses.to(self.device)
            motion_embed = self.vposer.encode(motion_poses).mean # (n, 32)

            motion_poses = motion_poses.cpu()
            motion_embed = motion_embed.cpu()

            all_motion.append((motion_poses, motion_embed))

        return all_motion

    def __getitem__(self, index):
        (motion_poses, motion_embed) = self._all_motion[index]

        (motion_poses_input, motion_poses_target) = self._preprocess(motion_poses)
        (motion_embed_input, motion_embed_target) = self._preprocess(motion_embed)

        return motion_poses_input, motion_poses_target, motion_embed_input, motion_embed_target
