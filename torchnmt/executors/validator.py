import os
import tqdm
import numpy as np

import torch

from .base import Executor
from torchnmt.scorers.scores import compute_scores

from torchnmt.networks.rnn.beam import evaluation as rnn_evaluation
from torchnmt.networks.huggingface.beam import evaluation as transformerhug_evaluation
from torchnmt.networks.vqa.beam import evaluation as vqa_evaluation


# from torchnmt.networks.rnn.beam import eval as rnn_eval


class Validator(Executor):
    def __init__(self, opts):
        super().__init__(opts)

    def iteration_iter(self):
        yield

    def on_iteration_start(self):
        self.skip_iteration()


class NMTValidator(Validator):
    def __init__(self, models, metrics, opts, seed=0):
        super().__init__(opts)
        self.models = models
        self.metrics = metrics
        self.epoch = 0
        self.best_rouge = 0.0
        self.seed = seed

        self.out_dir = os.path.join('ckpt',
                                    self.opts.name)
        os.makedirs(self.out_dir, exist_ok=True)

    def epoch_iter(self):
        assert isinstance(self.models, list)

        splits = [(split,
                   self.create_data_loader(split),
                   )
                  for split in self.opts.splits]

        self.scores = []

        for split, dl in splits:
            print('Running split: {} by ensembling {} models. '
                  'Using {}.'.format(split,
                                     len(self.models),
                                     type(dl.batch_sampler.sampler).__name__,
                                     ))
            self.split = split
            self.dl = dl
            yield

    def on_epoch_start(self):
        self.models = [m.eval() for m in self.models]
        eval_func = type(self.models[0]).__name__.lower() + '_evaluation'
        with torch.no_grad():
            self.losses, self.refs, self.hyps = eval(eval_func)(self.models, self.opts, self.dl)

    def on_epoch_end(self):
        # Handle loss
        loss = np.mean(self.losses)
        ppl = np.exp(loss)

        print('{}:\tloss: {:.4g}, ppl: {:.4g}'.format(self.split, loss, ppl))

        refs = self.refs
        hyps = self.hyps
        # Handle scores
        base = os.path.join(self.out_dir, '{}_{}_{}'.format(self.split, self.seed, '{}'))
        scores = compute_scores(refs, hyps, base, self.metrics)
        print(scores)
        with open(base.format('metrics.txt'), 'a+') as f:
            f.write(str({
                'split': self.split,
                'epoch': self.epoch,
                'scores': scores,
                'loss': loss,
                'ppl': ppl,
            }) + '\n')

        self.scores.append(scores)
        self.models = [m.train() for m in self.models]
