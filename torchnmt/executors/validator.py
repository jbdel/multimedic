import os
import tqdm
import numpy as np

import torch

from .base import Executor
from torchnmt.scorers.scores import compute_scores

from torchnmt.networks.transformer.beam import evaluation as transformer_evaluation
from torchnmt.networks.rnn.beam import evaluation as rnn_evaluation


# from torchnmt.networks.rnn.beam import eval as rnn_eval


class Validator(Executor):
    def __init__(self, opts):
        super().__init__(opts)


    def iteration_iter(self):
        yield

    def on_iteration_start(self):
        self.skip_iteration()


class NMTValidator(Validator):
    def __init__(self, models, opts):
        super().__init__(opts)
        self.models = models
        self.epoch = 0
        self.best_rouge = 0.0

    def on_epoch_start(self):
        self.out_dir = os.path.join('ckpt',
                                    self.opts.name)
        os.makedirs(self.out_dir, exist_ok=True)

        assert isinstance(self.models, list)
        self.models = [m.eval() for m in self.models]

        eval_func = type(self.models[0]).__name__.lower() + '_evaluation'
        with torch.no_grad():
            self.losses, self.refs, self.hyps = eval(eval_func)(self.models, self.opts, self.dl)


    def epoch_iter(self):
        splits = [(split,
                   self.create_data_loader(split),
                   self.create_writer(split))
                  for split in self.opts.splits]

        self.scores = []

        for split, dl, writer in splits:
            print('Running split: {} by ensembling {} models'.format(split, len(self.models)))
            self.split = split
            self.dl = dl
            self.writer = writer
            yield

    def on_epoch_end(self):
        # Handle loss
        loss = np.mean(self.losses)
        ppl = np.exp(loss)

        print('{}:\tloss: {:.4g}, ppl: {:.4g}'.format(self.split, loss, ppl))

        refs = self.refs
        hyps = self.hyps
        # Handle scores
        assert len(refs) == len(hyps)

        refs = list(map(' '.join, refs))
        hyps = list(map(' '.join, hyps))

        assert len(refs) == len(hyps)

        base = os.path.join(self.out_dir, '{}')
        scores = compute_scores(refs, hyps, base)
        print(scores)

        with open(base.format('metrics.txt'), 'a+') as f:
            f.write(str({
                'split': self.split,
                'epoch': self.epoch,
                'scores': self.scores,
                'loss': loss,
                'ppl': ppl,
            }) + '\n')

        self.writer.add_scalar(self.split + ' loss', loss, self.epoch)
        self.writer.add_scalar(self.split + ' ppl', ppl, self.epoch)
        self.writer.add_scalar(self.split + ' BLEU', scores['BLEU'], self.epoch)
        self.writer.add_scalar(self.split + ' ROUGE', scores['ROUGE'], self.epoch)
        self.writer.add_scalar(self.split + ' METEOR', scores['METEOR'], self.epoch)
        # self.writer.flush()  # requires: https://github.com/lanpa/tensorboardX/pull/451
        self.scores.append(scores)
        self.models = [m.train() for m in self.models]
