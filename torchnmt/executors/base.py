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

from .utils import CheckpointSaver
from .utils import print_args


class EpochSkipper(Exception):
    pass


class IterationSkipper(Exception):
    pass


class Executor(object):
    def __init__(self, opts, random_seed=7):
        self.opts = opts
        print_args(opts)
        self.saver = CheckpointSaver('ckpt/{}'.format(self.opts.name))
        self.set_seed(random_seed)

    def create_writer(self, split):
        return SummaryWriter('ckpt/{}/runs/{}'.format(self.opts.name, split))

    def create_model(self, state_dict=None):
        opts = self.opts.model
        arch = opts.proto

        opts = vars(copy.deepcopy(opts))
        model = eval(arch)(**opts)

        if state_dict is not None:
            state_dict = torch.load(state_dict)
            model.load_state_dict(state_dict)
            print(state_dict, 'loaded.')
        else:
            print('Model {} created.'.format(type(model).__name__))
        return model

    def create_dataset(self, split):
        opts = self.opts.dataset
        dataset = opts.proto

        opts = vars(copy.deepcopy(opts))
        return eval(dataset)(split=split, **opts)

    def create_data_loader(self, split, shuffle=False):
        dataset = self.create_dataset(split)

        if hasattr(dataset, 'get_collate_fn'):
            collate_fn = dataset.get_collate_fn()
        else:
            collate_fn = default_collate

        return DataLoader(dataset, self.opts.batch_size,
                          shuffle=shuffle,
                          num_workers=4,
                          collate_fn=collate_fn)

    def set_seed(self, seed):
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed(seed)  # gpu
        np.random.seed(seed)  # numpy
        random.seed(seed)  # random and transforms
        torch.backends.cudnn.deterministic = True  # cudnn

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
