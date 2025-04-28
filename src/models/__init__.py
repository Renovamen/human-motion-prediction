from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from os import path as osp

from .mlp import MotionMLP
from .transformer import MotionTransformer

def build_model(configs):
    model = MotionMLP(configs) if configs.model_name == "mlp" else MotionTransformer(configs)
    return model

def load_vposer(configs):
    vposer, ps = load_model(
        osp.join(configs.vposer_dir, "vposer_v2_05/"),
        model_code=VPoser,
        remove_words_in_model_weights="vp_model.",
        disable_grad=True
    )

    return vposer
