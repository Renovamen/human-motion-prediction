
import torch
from torch.utils.data import DataLoader
from torch import nn
import math
import json
from dataclasses import asdict
from tqdm import tqdm

from src.configs import Configs
from src.utils import get_logger, link_file, ensure_dir

class Evaluator:
    def __init__(
        self,
        configs: Configs,
        model: nn.Module,
        vposer: nn.Module,
        dataloader: DataLoader,
        device: torch.device = "cuda"
    ) -> None:
        self.configs = configs
        self.device = device

        self.model = model
        self.vposer = vposer
        self.vposer.to(self.device)

        self.dataloader = dataloader

        self._setup_logging()

        if configs.eval_model_pth is not None:
            self.load_model(configs.eval_model_pth)

    def _setup_logging(self):
        configs = self.configs

        ensure_dir(configs.snapshot_dir)
        link_file(configs.log_file, configs.link_log_file)

        self.logger = get_logger(configs.log_file, "test")
        self.logger.info(json.dumps(asdict(configs), indent=4, sort_keys=True))

    def load_model(self, model_pth: str):
        state_dict = torch.load(model_pth)
        self.model.load_state_dict(state_dict, strict=True)

        self.logger.info("Loading model path from {} ".format(model_pth))

    @torch.no_grad()
    def generate(self, embed_input):
        all_poses_pred = []
        n_steps = math.ceil(self.configs.target_length_eval / self.configs.target_length_train)

        for _ in range(n_steps):
            embed_pred = self.model(embed_input[:, -self.configs.input_length:]) # (1, n, c)
            poses_pred = self.vposer.decode(embed_pred.squeeze(0))["pose_body"].contiguous().view(-1, 63) # (n, 63)

            all_poses_pred.append(poses_pred)
            embed_input = torch.cat([embed_input, embed_pred], dim=1)

        all_poses_pred = torch.cat(all_poses_pred, dim=0)[:self.configs.target_length_eval]
        return all_poses_pred.unsqueeze(0)

    def eval(self):
        self.model.eval()
        self.model.to(self.device)

        mean_mpjpe = 0.0

        for (_, poses_target, embed_input, embed_target) in tqdm(self.dataloader, desc="Evaluating"):
            embed_input = embed_input.to(self.device)
            embed_target = embed_target.to(self.device)
            poses_target = poses_target.to(self.device)

            poses_pred = self.generate(embed_input)

            poses_target = poses_target.reshape(1, self.configs.target_length_eval, 21, 3)
            poses_pred = poses_pred.reshape(1, self.configs.target_length_eval, 21, 3)

            mpjpe = torch.norm(poses_pred * 1000 - poses_target * 1000, dim=3)
            mpjpe = torch.mean(torch.mean(mpjpe, dim=2), dim=1).squeeze(0)

            mean_mpjpe += mpjpe.item()

        mean_mpjpe /= len(self.dataloader.dataset)

        self.logger.info(f"Mean MPJPE: {mean_mpjpe:.4f}")
        print(f"Mean MPJPE: {mean_mpjpe:.4f}")

        return mean_mpjpe
