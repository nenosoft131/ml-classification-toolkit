import autoroot

import numpy as np
import hydra
import os
import sys
import signal
from rich import print
from omegaconf import DictConfig
import matplotlib.pyplot as plt

import torch
import torch.utils.data as td
import torch.optim
from torchvision.transforms import v2

from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, \
    MultilabelConfusionMatrix

from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

from src.utils.util import init_random
from src.utils import log
from src.data.raman_dataset import RamanDataset
from src.data import preprocessing
import classification
import constants as NC


class SVMClassification():

    def __init__(self, dataloaders: dict[str, td.DataLoader], args):
        self.args = args
        self.session_name = args.exp_name
        self.dataloaders = dataloaders
        num_labels = len(self.args.target_species)
        self.metrics = MetricCollection([
            MultilabelAccuracy(num_labels=num_labels),
            MultilabelPrecision(num_labels=num_labels),
            MultilabelRecall(num_labels=num_labels),
        ])
        self.confmat = MultilabelConfusionMatrix(num_labels=num_labels)

        model = classification.get_model(NC.METHOD_SVM, params={'C': 10.0})
        self.clf = MultiOutputClassifier(model)

    def _preprocess(self, X):
        X = preprocessing.despike_whitaker(X)
        X = preprocessing.smooth_savitzky_golay(X)
        X = preprocessing.baseline_correct(X)
        X = preprocessing.normalize(X)
        return X

    def train(self):
        self.metrics.reset()
        self.confmat.reset()

        print("\nLoading training data...")
        data = next(iter(self.dataloaders['train']))
        X = data['spectrum'].numpy()[:, 0]
        y = data['species_labels'].numpy().astype(int)[:, 0]

        print("Preprocessing data...")
        X = self._preprocess(X)

        print("Fitting model...")
        self.clf.fit(X, y)
        print("Done.")

        print("\nTraining performance:")
        y_pred = self.clf.predict(X)
        metrics = self.metrics(torch.tensor(y_pred), torch.tensor(y))
        print(metrics)
        self.confmat(torch.tensor(y_pred), torch.tensor(y))
        self.confmat.plot(labels=self.args.target_species)
        plt.show()

    def validate(self):
        self.metrics.reset()
        self.confmat.reset()

        print("\nLoading validation data...")
        data = next(iter(self.dataloaders['val']))
        X = data['spectrum'].numpy()[:, 0]
        y = data['species_labels'].numpy().astype(int)[:, 0]

        print("Preprocessing data...")
        X = self._preprocess(X)

        print("Running prediction...")
        y_pred = self.clf.predict(X)

        print("\nValidation performance:")
        metrics = self.metrics(torch.tensor(y_pred), torch.tensor(y))
        print(metrics)
        self.confmat(torch.tensor(y_pred), torch.tensor(y))
        self.confmat.plot(labels=self.args.target_species)
        plt.show()


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
        batch_size=len(dataset_train),
        num_workers=0,
        shuffle=False
    )
    dataloader_val = td.DataLoader(
        dataset_val,
        batch_size=len(dataset_val),
        num_workers=0,
        shuffle=False
    )

    dataloaders = {
        'train': dataloader_train,
        'val': dataloader_val,
    }

    task = SVMClassification(dataloaders, cfg)
    task.train()
    task.validate()


if __name__ == '__main__':
    main()
