from dataclasses import dataclass
import time

exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

@dataclass
class Configs:
    # Dirs
    snapshot_dir: str = "/home/xfz5266/586/snapshot"
    log_file: str = f"/home/xfz5266/586/log/log_{exp_time}.log"
    link_log_file: str = f"/home/xfz5266/586/log/log_last.log"

    # Dataset
    root_dir: str = "/home/xfz5266/586/AMASS_CMUsubset"
    amass_input_length: int = 50
    amass_target_length: int = 25
    batch_size: int = 8
    num_workers: int = 8

    # Training
    lr_max: float = 3e-4
    lr_min: float = 1e-5
    total_epochs: int = 200
    weight_decay: float = 1e-4
    model_pth: str = None
    seed: int = 888

    # Model
    motion_dim: int = 32
    mlp_dim: int = 32
    mlp_num_layers: int = 48
    mlp_with_normalization: bool = True
    mlp_spatial_fc_only: bool = False
    mlp_norm_axis: str = "spatial"

    # Logging
    print_every: int = 20
    save_every: int = 100
