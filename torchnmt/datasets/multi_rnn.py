import os
import torch
from torch.nn.utils.rnn import pad_sequence
from .text_rnn import TextDatasetRNN
import numpy as np


class MultiDatasetRNN(TextDatasetRNN):
    def __init__(self, npy, **kwargs):
        super().__init__(**kwargs)
        self.file = npy
        self.features = np.load(os.path.join(self.root, self.split + '_' + self.file))
        assert self.__len__() == super().__len__(), 'len(MultiDatasetRNN) != len(TextDatasetRNN)'

    def __getitem__(self, index):
        ret = super().__getitem__(index)
        ret['feats'] = self.features[index]
        return ret

    def __len__(self):
        return len(self.features)

    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {}
            collated['src'] = pad_sequence(
                [s['src'] for s in batch], batch_first=False)
            collated['tgt'] = pad_sequence(
                [s['tgt'] for s in batch], batch_first=False)

            # Features
            v = torch.from_numpy(np.array(
                [s['feats'] for s in batch], dtype='float32'))
            collated['feats'] = v.view(*v.size()[:2], -1).permute(2, 0, 1)

            return collated

        return collate_fn
