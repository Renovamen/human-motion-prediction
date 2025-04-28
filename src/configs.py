from dataclasses import dataclass
from typing import Literal
import time

exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

@dataclass
class Configs:
    seed: int = 888

    # Dirs
    snapshot_dir: str = "/home/xfz5266/586/snapshot"
    vposer_dir: str = "/home/xfz5266/586/VPoserModelFiles"
    log_file: str = f"/home/xfz5266/586/log/log_{exp_time}.log"
    link_log_file: str = f"/home/xfz5266/586/log/log_last.log"

    # Dataset
    root_dir: str = "/home/xfz5266/586/AMASS_CMUsubset"
    input_length: int = 50
    target_length_train: int = 25
    target_length_eval: int = 50
    batch_size_train: int = 8
    num_workers: int = 8

    # Training
    lr_max: float = 3e-4
    lr_min: float = 1e-4
    lr_decay_after: int = 200
    total_epochs: int = 150
    weight_decay: float = 1e-4
    train_model_pth: str = None

    # Evaluation
    eval_model_pth: str = "./snapshot/model-iter-1500.pth"

    # Model
    motion_dim: int = 32
    model_name: Literal["mlp", "transformer"] = "mlp"

    mlp_dim: int = 32
    mlp_num_layers: int = 48
    mlp_layer_norm: Literal["spatial", "temporal", "all", False] = "spatial"
    mlp_spatial_fc: bool = False

    transformer_num_layers: int = 16
    transformer_num_heads: int = 4
    transformer_dim: int = 32

    # Logging
    print_every: int = 20
    save_every: int = 100
