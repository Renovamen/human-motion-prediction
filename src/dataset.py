import glob
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch import nn

class AMASSDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        vposer: nn.Module,
        amass_motion_input_length: int,
        amass_motion_target_length: int,
        device: torch.device = "cuda"
    ) -> None:
        super().__init__()

        self.device = device

        self.vposer = vposer
        self.vposer.to(self.device)

        self._root_dir = root_dir
        self._file_paths = self._get_file_paths()

        self.amass_motion_input_length = amass_motion_input_length
        self.amass_motion_target_length = amass_motion_target_length

        self._all_amass_motion_poses = self._load_all()

    def __len__(self):
        return len(self._all_amass_motion_poses)

    def _get_file_paths(self):
        return glob.glob(f"{self._root_dir}/*poses.npz")

    def _preprocess(self, amass_motion_feats):
        N = amass_motion_feats.size(0)

        start = np.random.randint(N - self.amass_motion_input_length  - self.amass_motion_target_length + 1)
        end = start + self.amass_motion_input_length

        amass_motion_input = amass_motion_feats[start:end]
        amass_motion_target = amass_motion_feats[end:end+self.amass_motion_target_length]

        amass_motion = torch.cat([amass_motion_input, amass_motion_target], axis=0)
        return amass_motion

    def _load_all(self):
        all_amass_motion_poses = []

        for file_path in tqdm(self._file_paths):
            amass_info = np.load(file_path)

            amass_motion_poses = amass_info["poses"] # 156 joints(all joints of SMPL)
            # [3:66] removes global rotation, hands/fingers, and anything else other than 21 major body joints
            amass_motion_poses = amass_motion_poses[:, 3:66]
            amass_motion_poses = torch.from_numpy(amass_motion_poses).type(torch.float)

            N = amass_motion_poses.size(0)
            if N < self.amass_motion_target_length + self.amass_motion_input_length:
                continue

            # Sample frames
            frame_rate = amass_info["mocap_framerate"]

            sample_rate = int(frame_rate // 25)
            sample_rate = min(sample_rate, N // (self.amass_motion_target_length + self.amass_motion_input_length))
            sampled_index = np.arange(0, N, sample_rate)

            amass_motion_poses = amass_motion_poses[sampled_index]

            # Run Vposer encoder on all frames
            amass_motion_poses = amass_motion_poses.to(self.device)
            amass_motion_poses = self.vposer.encode(amass_motion_poses).mean # (n, 32)
            amass_motion_poses = amass_motion_poses.cpu()

            all_amass_motion_poses.append(amass_motion_poses)

        return all_amass_motion_poses

    def __getitem__(self, index):
        amass_motion_poses = self._all_amass_motion_poses[index]
        amass_motion = self._preprocess(amass_motion_poses)

        amass_motion_input = amass_motion[:self.amass_motion_input_length].float()
        amass_motion_target = amass_motion[-self.amass_motion_target_length:].float()

        return amass_motion_input, amass_motion_target
