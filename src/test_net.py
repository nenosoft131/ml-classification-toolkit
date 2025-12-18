import autoroot

import numpy as np
import hydra
import os
import sys
import signal
from rich import print
from omegaconf import DictConfig

import torch
import torch.utils.data as td
import torch.optim
from torchvision.transforms import v2

from src.utils.util import init_random
from src.data.raman_dataset import RamanDataset
from src.training.multilabel_classifier import MultilabelTraining


@hydra.main(version_base=None, config_path="../configs", config_name="test.yaml")
def main(cfg: DictConfig):

    # Disable traceback on Ctrl+c
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    if cfg.extras.print_config:
        print(cfg)

    np.set_printoptions(linewidth=None)
    torch.set_printoptions(sci_mode=False)

    if cfg.seed is not None:
        init_random(cfg.seed)

    torch.backends.cudnn.benchmark = True

    transform = v2.Compose([v2.ToDtype(torch.float32)])

    dataset = RamanDataset(
        data_csv=cfg.test.data.data_csv,
        meta_csv=cfg.test.data.meta_csv,
        transform=transform,
        target_species=cfg.target_species
    )

    dataloader = td.DataLoader(
        dataset,
        batch_size=cfg.test.data.batchsize,
        num_workers=cfg.test.data.workers,
        shuffle=cfg.test.data.shuffle
    )

    MultilabelTraining(dataloaders=dict(test=dataloader), args=cfg).test()


if __name__ == '__main__':
    main()
