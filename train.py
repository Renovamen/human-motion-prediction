import torch
from torch.utils.data import DataLoader

from src.configs import Configs
from src.models import build_model, load_vposer
from src.dataset import AMASSDataset
from src.trainer import Trainer
from src.utils import set_seed

def build_trainer(configs: Configs) -> Trainer:
    set_seed(configs.seed)

    model = build_model(configs)
    vposer = load_vposer(configs)

    dataset = AMASSDataset(
        root_dir=configs.root_dir,
        vposer=vposer,
        input_length=configs.input_length,
        target_length=configs.target_length_train,
        split="train"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=configs.batch_size_train,
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
    configs = Configs()

    trainer = build_trainer(configs)
    trainer.train()
