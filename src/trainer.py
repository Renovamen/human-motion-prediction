
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import json
from dataclasses import asdict
from tqdm import tqdm

from src.configs import Configs
from src.utils import get_logger, link_file, ensure_dir


def update_lr_multistep(nb_iter: int, lr_max: float, lr_min: float, optimizer) :
    if nb_iter > 100000:
        current_lr = lr_min
    else:
        current_lr = lr_max

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr


class Trainer:
    def __init__(
        self,
        configs: Configs,
        model: nn.Module,
        optimizer: optim.Optimizer,
        dataloader: DataLoader
    ) -> None:
        self.configs = configs

        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader

        self._setup_logging()

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

    def train_step(
        self,
        amass_motion_input,
        amass_motion_target,
        nb_iter: int
    ) :
        motion_pred = self.model(amass_motion_input)

        motion_pred = motion_pred.reshape(-1, 1)
        amass_motion_target = amass_motion_target.reshape(-1, 1)

        loss = torch.mean(torch.norm(motion_pred - amass_motion_target, 2, 1))
        loss = loss.mean()

        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        self.optimizer, current_lr = update_lr_multistep(
            nb_iter,
            self.configs.lr_max,
            self.configs.lr_min,
            self.optimizer
        )

        return loss.item(), current_lr

    def train(self):
        self.model.train()

        configs = self.configs

        nb_iter = 0
        avg_loss = 0.
        avg_lr = 0.

        for epoch in range(configs.total_epochs):
            for (amass_motion_input, amass_motion_target) in tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{configs.total_epochs}", leave=False):
                amass_motion_input = amass_motion_input.to("cuda")
                amass_motion_target = amass_motion_target.to("cuda")

                loss, current_lr = self.train_step(
                    amass_motion_input,
                    amass_motion_target,
                    nb_iter
                )

                avg_loss += loss
                avg_lr += current_lr

                if (nb_iter + 1) % configs.print_every ==  0 :
                    avg_loss = avg_loss / configs.print_every
                    avg_lr = avg_lr / configs.print_every

                    self.logger.info("Iter {} Summary: ".format(nb_iter + 1))
                    self.logger.info(f"\t lr: {avg_lr} \t Training loss: {avg_loss}")

                    avg_loss = 0
                    avg_lr = 0

                if (nb_iter + 1) % configs.save_every ==  0 :
                    torch.save(self.model.state_dict(), configs.snapshot_dir + "/model-iter-" + str(nb_iter + 1) + ".pth")

                nb_iter += 1

        return avg_loss, avg_lr
