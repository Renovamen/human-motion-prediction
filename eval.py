from torch.utils.data import DataLoader

from src.configs import Configs
from src.models import build_model, load_vposer
from src.dataset import AMASSDataset
from src.evaluator import Evaluator
from src.utils import set_seed

def build_evaluator(configs: Configs) -> Evaluator:
    set_seed(configs.seed)

    model = build_model(configs)
    vposer = load_vposer(configs)

    dataset = AMASSDataset(
        root_dir=configs.root_dir,
        vposer=vposer,
        input_length=configs.input_length,
        target_length=configs.target_length_eval,
        split="test"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        drop_last=False,
        shuffle=False,
        pin_memory=True
    )

    evaluator = Evaluator(
        configs=configs,
        model=model,
        vposer=vposer,
        dataloader=dataloader
    )

    return evaluator

if __name__ == "__main__":
    configs = Configs()

    evaluator = build_evaluator(configs)
    evaluator.eval()
