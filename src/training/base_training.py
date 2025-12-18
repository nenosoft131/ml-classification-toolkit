from __future__ import annotations

import time

import datetime
import json
import os
import matplotlib.pyplot as plt

import torch
import torch.utils.data as td
import torch.nn.modules.distance
from torchmetrics import MetricCollection

from src.utils import nn, io_utils
from src.utils.nn import to_numpy, count_parameters
import src.utils.log as log

eps = 1e-8


# save some samples to visualize the training progress
def get_fixed_samples(dl, num):
    assert isinstance(dl, td.DataLoader)
    dl = td.DataLoader(dl.dataset, batch_size=num, shuffle=False, num_workers=0)
    return next(iter(dl))


def __reduce(errs, reduction):
    if reduction == 'mean':
        return errs.mean()
    elif reduction == 'sum':
        return errs.sum()
    elif reduction == 'none':
        return errs
    else:
        raise ValueError("Invalid parameter reduction={}".format(reduction))


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.xavier_uniform(m)
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)


class Training(object):

    def __init__(self, dataloaders, args, **kwargs):

        self.args = args
        self.session_name = args.exp_name
        self.dataloaders = dataloaders
        self.net = self._get_network(pretrained=self.args.model.pretrained)
        log.info("Total model params: {:,}\n".format(count_parameters(self.net)))

        self.snapshot_dir = args.paths.snapshot_dir
        self.max_epochs = args.training.max_epochs
        self.total_iter = 0
        self.total_items = 0
        self.iter_in_epoch = 0
        self.epoch = 0
        self.best_score = 999
        self.epoch_stats = []
        self.figures = {}

        self.total_training_time_previous = 0
        self.time_start_training = time.time()

        self.metrics = MetricCollection([])


    def _get_network(self, pretrained):
        raise NotImplementedError

    def _run_batch(self, data, is_eval):
        raise NotImplementedError

    def _print_iter_stats(self, stats):
        raise NotImplementedError

    def _print_epoch_summary(self, epoch_stats, epoch_starttime):
        raise NotImplementedError

    def _init_metrics(self):
        self.metrics.reset()
        self.epoch_stats = []

    def _evaluate_metrics(self):
        pass

    def train(self):
        log.info("Learning rate: {}".format(self.args.training.optimizer.lr))
        log.info("Batch size: {}".format(self.args.training.data.batchsize))
        log.info("Workers: {}".format(self.args.training.data.workers))

        snapshot = self.args.resume
        if snapshot is not None:
            log.info("Resuming session {} from snapshot {}...".format(self.session_name, snapshot))
            self._load_snapshot(snapshot)

        if self.args.gpu:
            self.net = self.net.cuda()

        log.info("")
        log.info("Training '{}'...".format(self.session_name))
        # log.info("")

        while self.max_epochs is None or self.epoch < self.max_epochs:
            log.info('')
            log.info('Epoch {}/{}'.format(self.epoch + 1, self.max_epochs))
            log.info('=' * 10)

            self._init_metrics()
            epoch_starttime = time.time()
            self._run_epoch(self.dataloaders['train'])

            # save model every few epochs
            if self.args.training.save_freq > 0 and (self.epoch + 1) % self.args.training.save_freq == 0:
                self._save_snapshot(is_best=False)

            # print average loss and accuracy over epoch
            self._print_epoch_summary(self.epoch_stats, epoch_starttime)
            self._evaluate_metrics()

            if self._is_eval_epoch():
                self.validate()

            self.epoch += 1

        time_elapsed = time.time() - self.time_start_training
        log.info('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def validate(self):
        log.info("")
        log.info("Validating '{}'...".format(self.session_name))

        epoch_starttime = time.time()
        self._init_metrics()
        self.net.eval()

        self._run_epoch(self.dataloaders['val'], is_eval=True)

        # print average loss and accuracy over epoch
        self._print_epoch_summary(self.epoch_stats, epoch_starttime)

        self._evaluate_metrics()
        return self.epoch_stats

    def test(self):
        log.info("Batch size: {}".format(self.args.test.data.batchsize))
        log.info("Workers: {}".format(self.args.test.data.workers))

        snapshot = self.args.resume
        if snapshot is not None:
            log.info("Loading snapshot {}...".format(snapshot))
            self._load_snapshot(snapshot)

        if self.args.gpu:
            self.net = self.net.cuda()

        log.info("")
        log.info("Testing '{}'...".format(self.session_name))

        epoch_starttime = time.time()
        self._init_metrics()
        self.net.eval()

        self._run_epoch(self.dataloaders['test'], is_eval=True)

        # print average loss and accuracy over epoch
        self._print_epoch_summary(self.epoch_stats, epoch_starttime)

        self._evaluate_metrics()
        return self.epoch_stats

    def _run_epoch(self, dataloader: td.DataLoader, is_eval=False):
        iter_endtime = time.time()

        self.iters_per_epoch = len(dataloader.dataset) // dataloader.batch_size
        self.iter_in_epoch = 0

        for data in dataloader:
            time_dataloading = time.time() - iter_endtime

            # run prediction and backprop
            t_proc_start = time.time()
            iter_stats = self._run_batch(data, is_eval=is_eval)
            time_processing = time.time() - t_proc_start

            # statistics
            self.total_items += dataloader.batch_size
            iter_stats.update({
                'epoch': self.epoch,
                'iter': self.iter_in_epoch,
                'total_iter': self.total_iter,
                'timestamp': time.time(),
                'time_dataloading': time_dataloading,
                'time_processing': time_processing,
                'iter_time': time.time() - iter_endtime,
            })
            self.epoch_stats.append(iter_stats)

            # print stats every N batches
            if self._is_printout_iter(is_eval):
                self._print_iter_stats(self.epoch_stats[-self._print_interval(is_eval):])

            self.total_iter += 1
            self.net.total_iter = self.total_iter
            self.iter_in_epoch += 1

            iter_endtime = time.time()

    def _save_snapshot(self, is_best=False):
        def write_file(filepath, model):
            meta=dict(
                epoch=self.epoch + 1,
                total_iter=self.total_iter,
                total_time=self.total_training_time(),
                best_score=self.best_score,
                seed=self.args.seed,
                experiment_name=self.args.exp_name
            )
            snapshot=dict(
                arch=type(model).__name__,
                state_dict=model.state_dict(),
                meta=meta
            )
            io_utils.makedirs(filepath)
            torch.save(snapshot, filepath)

        snapshot_name = os.path.join('epoch_{:05d}.pth'.format(self.epoch+1))
        output_dir =  os.path.join(self.snapshot_dir, snapshot_name)
        write_file(output_dir, self.net)
        log.info(f"*** saved checkpoint {output_dir} *** ")

        # save a copy of this snapshot as the best one so far
        # if is_best:
        #     io_utils.copy_files(src_dir=model_snap_dir, dst_dir=model_data_dir, pattern='*.mdl')

    def _load_snapshot(self, filename):

        if os.path.isabs(filename):
            filepath = filename
        else:
            items = os.path.split(filename)
            if items[0] == '':
                filepath = os.path.join(self.args.paths.snapshot_dir, filename)
            else:
                filepath = os.path.join(self.args.paths.log_dir, filename)

        snapshot = torch.load(filepath, weights_only=True)

        try:
            self.net.load_state_dict(snapshot['state_dict'], strict=False)
        except RuntimeError as e:
            print(e)

        meta = snapshot['meta']
        self.epoch = meta['epoch']
        self.total_iter = meta['total_iter']
        self.total_training_time_previous = meta.get('total_time', 0)
        self.total_items = meta.get('total_items', 0)
        self.best_score = meta['best_score']
        self.net.total_iter = self.total_iter
        str_training_time = str(datetime.timedelta(seconds=self.total_training_time()))

        log.info("Model {} trained for {} iterations ({}).".format(
            filename, self.total_iter, str_training_time)
        )

    def _is_snapshot_iter(self):
        return (self.total_iter+1) % self.args.snapshot_interval == 0 and (self.total_iter+1) > 0

    def _print_interval(self, eval):
        return self.args.validation.print_freq if eval else self.args.training.print_freq

    def _is_printout_iter(self, eval):
        return (self.iter_in_epoch+1) % self._print_interval(eval) == 0

    def _is_eval_epoch(self):
        return (self.epoch+1) % self.args.training.eval_freq == 0 and 'val' in self.dataloaders

    def _training_time(self):
        return int(time.time() - self.time_start_training)

    def total_training_time(self):
        return self.total_training_time_previous + self._training_time()


def bool_str(x):
    return str(x).lower() in ['True', 'true', '1']


def add_arguments(parser, defaults=None):

    if defaults is None:
        defaults = {}

    # model params
    parser.add_argument('-s', '--sessionname',  default=defaults.get('sessionname'), type=str, help='output filename (without ext)')
    parser.add_argument('-r', '--resume', default=defaults.get('resume'), type=str, metavar='PATH', help='path to snapshot (default: None)')
    parser.add_argument('-i','--input-size', default=defaults.get('input_size', (256, 256)), type=int, nargs='+', help='CNN input size')
    parser.add_argument('--pretrained', default=defaults.get('pretrained', True), type=int,
                        help='Initialize with pretrained weights if available (only for ResNet etc.)')
    # parser.add_argument('--full-size', default=defaults.get('full_size', 512), type=int, help='full image size')

    # training
    parser.add_argument('--seed', type=int, default=defaults.get('seed', None))
    parser.add_argument('-e', '--epochs', default=None, type=int, metavar='N', help='maximum epoch count')
    parser.add_argument('-b', '--batchsize', default=defaults.get('batchsize', 50), type=int, metavar='N', help='batch size')
    parser.add_argument('--eval', default=False, action='store_true',  help='run evaluation instead of training')
    parser.add_argument('--phases', default=['train', 'val'], nargs='+')
    parser.add_argument('--reset', default=False, action='store_true', help='reset the discriminator')
    parser.add_argument('--lr', default=defaults.get('lr', 0.0001), type=float, help='learning rate for autoencoder')
    parser.add_argument('--beta1', default=0.0, type=float, help='Adam beta 1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam beta 2')
    parser.add_argument('--gpu', default=True, action='store_true')

    # reporting
    parser.add_argument('--save-freq', default=defaults.get('save_freq', 1), type=int, metavar='N', help='save snapshot every N epochs')
    parser.add_argument('--print-freq', '-p', default=defaults.get('print_freq', 20), type=int, metavar='N', help='print every N steps')
    parser.add_argument('--print-freq-eval', default=defaults.get('print_freq_eval', 1), type=int, metavar='N', help='print every N steps')
    parser.add_argument('--eval-freq', default=defaults.get('eval_freq', 50), type=int, metavar='N', help='evaluate every N steps')
    parser.add_argument('--batchsize-eval', default=defaults.get('batchsize_eval', 10), type=int, metavar='N', help='batch size for evaluation')

    # data
    parser.add_argument('--use-cache', type=bool_str, default=True, help='use cached crops')
    parser.add_argument('--train-count', default=defaults.get('train_count', None), type=int, help='number of training images per dataset')
    parser.add_argument('--train-count-multi', default=None, type=int, help='number of total training images for training using multiple datasets')
    parser.add_argument('--val-count',  default=None, type=int, help='number of test images')
    parser.add_argument('-j', '--workers', default=defaults.get('workers'), type=int, metavar='N', help=f"number of data loading workers (default: {defaults.get('workers')})")
    parser.add_argument('--workers_eval', default=6, type=int, metavar='N', help='number of data loading workers (default: 0)')

    # visualization
    parser.add_argument('--show', type=bool_str, default=True, help='visualize training')
    parser.add_argument('--wait', default=10, type=int)

    parser.add_argument('--debug', type=bool_str, default=False, help='set workers count to zero and print every iteration')
