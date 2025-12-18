import pandas as pd
import time
import datetime

import torch
import torch.utils.data as td
import torch.optim
from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, \
    MultilabelConfusionMatrix

from src.utils import log
from src.models.mlp import MLP1D
from src.training import base_training


class MultilabelTraining(base_training.Training):

    def __init__(self, dataloaders: dict[str, td.DataLoader], args, **kwargs):
        super().__init__(dataloaders, args, **kwargs)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.training.optimizer.lr)
        num_labels = len(self.args.target_species)
        threshold = self.args.threshold
        self.metrics = MetricCollection([
            MultilabelAccuracy(num_labels=num_labels),
            MultilabelPrecision(num_labels=num_labels),
            MultilabelRecall(num_labels=num_labels),
            MultilabelConfusionMatrix(num_labels=num_labels)
        ])
        if self.args.gpu:
            self.metrics = self.metrics.cuda()

    def _get_network(self, pretrained):
        return MLP1D(self.args.num_frequencies, num_classes=len(self.args.target_species), hidden_dim=1000)

    def _print_iter_stats(self, stats):
        means = pd.DataFrame(stats)
        means = means.mean().to_dict()
        current = stats[-1]
        str_stats = ['[{ep}][{i}/{iters_per_epoch}] '
                     'loss={avg_loss:.5f} '
                     '{t_data:.2f}/{t_proc:.2f}/{t:.2f}s ({total_iter:06d} {total_time})'][0]
        log.info(str_stats.format(
            ep=current['epoch'] + 1, i=current['iter'] + 1, iters_per_epoch=self.iters_per_epoch,
            avg_loss=means.get('loss', -1),
            t=means['iter_time'],
            t_data=means['time_dataloading'],
            t_proc=means['time_processing'],
            total_iter=self.total_iter + 1, total_time=str(datetime.timedelta(seconds=self._training_time()))
        ))

    def _print_epoch_summary(self, epoch_stats, epoch_starttime, is_eval=False):
        means = pd.DataFrame(epoch_stats)
        means = means.mean().to_dict()

        duration = int(time.time() - epoch_starttime)
        log.info("{}".format('-' * 100))
        str_stats = ['           loss={avg_loss:.5f} \tT: {time_epoch}'][0]
        log.info(str_stats.format(
            iters_per_epoch=self.iters_per_epoch,
            avg_loss=means.get('loss', -1),
            t=means['iter_time'],
            t_data=means['time_dataloading'],
            t_proc=means['time_processing'],
            total_iter=self.total_iter + 1, total_time=str(datetime.timedelta(seconds=self._training_time())),
            time_epoch=str(datetime.timedelta(seconds=duration))))

    def _run_batch(self, batch, is_eval=False, ds=None):
        iter_stats = {}

        self.net.zero_grad()
        self.net.train(not is_eval)

        inputs: torch.Tensor = batch['spectrum']
        targets: torch.Tensor = batch['species_labels'].squeeze(1).squeeze(1)

        if self.args.gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        with torch.set_grad_enabled(not is_eval):
            outputs = self.net(inputs)

        loss = torch.nn.functional.binary_cross_entropy(outputs, targets)

        metrics = self.metrics(outputs, targets.long())

        iter_stats['loss'] = loss.item()
        iter_stats['acc'] = metrics['MultilabelAccuracy'].item()

        if not is_eval:
            loss.backward()
            self.optimizer.step()

        return iter_stats

    def _evaluate_metrics(self):
        metrics = self.metrics.compute()
        log.info("")
        for metric_name, value in metrics.items():
            if value.shape == torch.Size():  # is scalar?
                log.info(f"{metric_name}: {value.item():.2f}")
        log.info("")
