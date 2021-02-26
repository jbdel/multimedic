import numpy as np
import torch
import random
import json
import copy

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tensorboardX import SummaryWriter

from torchnmt.networks import *
from torchnmt.datasets import *
from torchnmt.datasets import BucketBatchSampler
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler

from .utils import print_args
import time


class EpochSkipper(Exception):
    pass


class IterationSkipper(Exception):
    pass


class Executor(object):
    def __init__(self, opts):
        self.opts = opts
        print_args(opts)

    def create_writer(self, split):
        return SummaryWriter('ckpt/{}/runs/{}'.format(self.opts.name, split))

    def create_model(self, state_dict=None):
        opts = self.opts.model
        arch = opts.proto

        opts = vars(copy.deepcopy(opts))
        model = eval(arch)(**opts)

        if state_dict is not None:
            model.load_state_dict(torch.load(state_dict))
            print(state_dict, 'loaded.')
        return model

    def create_dataset(self, split):
        opts = self.opts.dataset
        dataset = opts.proto

        opts = vars(copy.deepcopy(opts))
        return eval(dataset)(split=split, **opts)

    def create_data_loader(self, split):
        dataset = self.create_dataset(split)
        if hasattr(dataset, 'get_collate_fn'):
            collate_fn = dataset.get_collate_fn()
        else:
            collate_fn = default_collate

        if not hasattr(self.opts.dataset, 'use_bucket'):
            self.opts.dataset.use_bucket = False

        if split == 'train':
            print('\033[1m\033[91mDataLoader \033[0m')
            if self.opts.dataset.use_bucket:
                sort_lens = dataset.lengths
                sampler = BucketBatchSampler(
                    batch_size=self.opts.batch_size,
                    sort_lens=sort_lens,
                    max_len=dataset.src_len)
            else:
                sampler = BatchSampler(
                    RandomSampler(dataset),
                    batch_size=self.opts.batch_size, drop_last=False)
            print('Using \033[1m\033[94m' + type(sampler.sampler).__name__ + '\033[0m')

        else:  # eval or test
            sampler = BatchSampler(
                SequentialSampler(dataset),
                batch_size=self.opts.batch_size, drop_last=False)

        return DataLoader(dataset,
                          # num_workers=4,
                          collate_fn=collate_fn,
                          batch_sampler=sampler)

    def set_seed(self, seed=None):
        if seed is None:
            seed = time.time()

        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        return seed

    def start(self):
        for _ in self:
            pass

    def __iter__(self):
        for _ in self.epoch_iter():
            try:
                self.on_epoch_start()
                for _ in self.iteration_iter():
                    try:
                        self.on_iteration_start()
                        self.update()
                        self.on_iteration_end()
                        yield
                    except IterationSkipper:
                        continue
                self.on_epoch_end()
            except EpochSkipper:
                continue

    def skip_epoch(self):
        """
        Skip the current epoch.
        """
        raise EpochSkipper()

    def skip_iteration(self):
        """
        Skip the current iteration.
        """
        raise IterationSkipper()

    def epoch_iter(self):
        raise NotImplementedError()

    def iteration_iter(self):
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()

    def on_epoch_start(self):
        return

    def on_epoch_end(self):
        return

    def on_iteration_start(self):
        return

    def on_iteration_end(self):
        return
