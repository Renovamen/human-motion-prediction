
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import json
from dataclasses import asdict
from tqdm import tqdm

from src.configs import Configs
from src.utils import get_logger, link_file, ensure_dir

def update_lr_multistep(
    optimizer: optim.Optimizer,
    nb_iter: int,
    lr_decay_after: int,
    lr_max: float,
    lr_min: float
) -> float:
    if nb_iter > lr_decay_after:
        current_lr = lr_min
    else:
        current_lr = lr_max

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return current_lr

class Trainer:
    def __init__(
        self,
        configs: Configs,
        model: nn.Module,
        optimizer: optim.Optimizer,
        dataloader: DataLoader,
        device: torch.device = "cuda"
    ) -> None:
        self.configs = configs
        self.device = device

        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader

        self._clear_states()
        self._setup_logging()

        if configs.train_model_pth is not None:
            self.load_model(configs.train_model_pth)

    def _clear_states(self):
        self.nb_iter = 0
        self.avg_loss = 0.
        self.avg_lr = 0.

    def _setup_logging(self):
        configs = self.configs

        ensure_dir(configs.snapshot_dir)
        link_file(configs.log_file, configs.link_log_file)

        self.logger = get_logger(configs.log_file, "train")
        self.logger.info(json.dumps(asdict(configs), indent=4, sort_keys=True))

    def load_model(self, model_pth: str):
        state_dict = torch.load(model_pth)
        self.model.load_state_dict(state_dict, strict=True)

        self.logger.info("Loading model path from {} ".format(model_pth))

    def train_step(self, motion_input, motion_target):
        motion_pred = self.model(motion_input)

        motion_pred = motion_pred.reshape(-1, 1)
        motion_target = motion_target.reshape(-1, 1)

        loss = torch.mean(torch.norm(motion_pred - motion_target, 2, 1))
        loss = loss.mean()

        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        current_lr = update_lr_multistep(
            self.optimizer,
            self.nb_iter,
            self.configs.lr_decay_after,
            self.configs.lr_max,
            self.configs.lr_min
        )

        return loss.item(), current_lr

    def train(self):
        self.model.train()
        self.model.to(self.device)

        self._clear_states()

        configs = self.configs

        for epoch in range(configs.total_epochs):
            for (_, _, motion_input, motion_target) in tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{configs.total_epochs}", leave=False):
                motion_input = motion_input.to(self.device)
                motion_target = motion_target.to(self.device)

                loss, current_lr = self.train_step(motion_input, motion_target)

                self.avg_loss += loss
                self.avg_lr += current_lr

                if (self.nb_iter + 1) % configs.print_every ==  0 :
                    self.avg_loss /= configs.print_every
                    self.avg_lr /= configs.print_every

                    self.logger.info("Iter {} Summary: ".format(self.nb_iter + 1))
                    self.logger.info(f"\t lr: {self.avg_lr} \t Training loss: {self.avg_loss}")

                    self.avg_loss = 0
                    self.avg_lr = 0

                if (self.nb_iter + 1) % configs.save_every ==  0 :
                    torch.save(self.model.state_dict(), f"{configs.snapshot_dir}/model-iter-{self.nb_iter + 1}.pth")

                self.nb_iter += 1

        return self.avg_loss, self.avg_lr
