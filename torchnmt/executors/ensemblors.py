import os
from .validator import NMTValidator
import glob


class NMTEnsemblor(NMTValidator):
    def __init__(self, train_opts, ensemble_opts):
        super().__init__(None, ensemble_opts)
        self.best_metric = 0.0
        self.models = []

        ckpts = glob.glob(os.path.join('ckpt',
                                       self.opts.name,
                                       '*.pth'))

        if self.opts.mode == 'best':
            ckpts = [sorted(ckpts, reverse=True)[0]]

        for ckpt in ckpts:
            self.models.append(self.create_model(state_dict=ckpt).cuda().eval())
