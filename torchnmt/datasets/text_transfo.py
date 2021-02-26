from torch.nn.utils.rnn import pack_sequence
from .text_rnn import TextDatasetRNN, TextDatasetRNNStatic
from torch.nn.utils.rnn import pad_sequence
import os

class TextDatasetTransfo(TextDatasetRNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {'src': pad_sequence(
                [s['src'] for s in batch], batch_first=True), 'tgt': pad_sequence(
                [s['tgt'] for s in batch], batch_first=True)}

            return collated

        return collate_fn
