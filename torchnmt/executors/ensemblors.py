import os
from .validator import NMTValidator
import glob


def get_n_best(mode):
    n = 1
    # checking if args is formatted as best-n
    if '-' in mode:
        n = int(mode.split('-')[-1])
    return n


class NMTEnsemblor(NMTValidator):
    def __init__(self, train_opts, ensemble_opts):
        super().__init__(None, ensemble_opts)
        self.best_metric = 0.0
        self.models = []

        ckpts = glob.glob(os.path.join('ckpt',
                                       self.opts.name,
                                       '*.pth'))

        # Sort by score
        ckpts = sorted(ckpts, reverse=True)

        # Getting n-best models
        if 'best' in self.opts.mode:
            n = get_n_best(self.opts.mode)
            ckpts = ckpts[:n]

        for ckpt in ckpts:
            self.models.append(self.create_model(state_dict=ckpt).cuda().eval())

        # create train dataset to set up the right dataset static options
        self.create_data_loader("train")
