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
from src.utils import log
from src.data.raman_dataset import RamanDataset
from src.training.multilabel_classifier import MultilabelTraining


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):

    # Disable traceback on Ctrl+c
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    if cfg.extras.print_config:
        print(cfg)

    np.set_printoptions(linewidth=None)
    torch.set_printoptions(sci_mode=False)

    if cfg.seed is not None:
        init_random(cfg.seed)

    if cfg.exp_name is None:
        if cfg.resume:
            modelname = os.path.split(cfg.resume)[0]
            cfg.exp_name = modelname
        else:
            cfg.exp_name = 'debug'

    torch.backends.cudnn.benchmark = True

    # from src.data import transforms as t
    transform = v2.Compose([
        # t.DespikeWhitaker(),
        # t.SmoothSavitzkiGoilay(),
        # t.BaselineCorrect(),
        # t.Normalize(),
        v2.ToDtype(torch.float32)
    ])

    dataset_train = RamanDataset(
        data_csv=cfg.training.data.data_csv,
        meta_csv=cfg.training.data.meta_csv,
        transform=transform,
        target_species=cfg.target_species
    )
    dataset_val = RamanDataset(
        data_csv=cfg.validation.data.data_csv,
        meta_csv=cfg.validation.data.meta_csv,
        transform=transform,
        target_species=cfg.target_species
    )

    dataloader_train = td.DataLoader(
        dataset_train,
        batch_size=cfg.training.data.batchsize,
        num_workers=cfg.training.data.workers,
        shuffle=cfg.training.data.shuffle
    )
    dataloader_val = td.DataLoader(
        dataset_val,
        batch_size=cfg.validation.data.batchsize,
        num_workers=cfg.validation.data.workers,
        shuffle=cfg.validation.data.shuffle
    )

    dataloaders = {
        'train': dataloader_train,
        'val': dataloader_val,
    }

    MultilabelTraining(dataloaders, cfg).train()


if __name__ == '__main__':
    main()
