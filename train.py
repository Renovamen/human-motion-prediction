import torch
from torch.utils.data import DataLoader
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from os import path as osp

from src.configs import Configs
from src.models import MotionPrediction
from src.dataset import AMASSDataset
from src.trainer import Trainer

def build_trainer():
    configs = Configs()
    torch.manual_seed(configs.seed)

    model = MotionPrediction(configs)

    support_dir = "./VPoserModelFiles/"
    expr_dir = osp.join(support_dir, "vposer_v2_05/")

    vposer, ps = load_model(
        expr_dir,
        model_code=VPoser,
        remove_words_in_model_weights="vp_model.",
        disable_grad=True
    )

    dataset = AMASSDataset(
        root_dir=configs.root_dir,
        vposer=vposer,
        amass_motion_input_length=configs.amass_input_length,
        amass_motion_target_length=configs.amass_target_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=configs.batch_size,
        num_workers=configs.num_workers,
        drop_last=True,
        shuffle=True,
        pin_memory=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=configs.lr_max,
        weight_decay=configs.weight_decay
    )

    trainer = Trainer(
        configs=configs,
        model=model,
        optimizer=optimizer,
        dataloader=dataloader
    )

    return trainer

if __name__ == "__main__":
    trainer = build_trainer()
    trainer.train()
