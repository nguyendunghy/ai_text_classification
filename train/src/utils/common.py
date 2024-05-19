import torch
from pytorch_lightning import seed_everything


def seed_everything_deterministic(seed):
    seed_everything(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
