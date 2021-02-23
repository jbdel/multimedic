from torch.nn.utils.rnn import pack_sequence
from .text_rnn import TextDatasetRNN


class TextDatasetTransfo(TextDatasetRNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {}
            # pack first will make the training faster since it is done by multi workers
            collated['src'] = pack_sequence([s['src'] for s in batch],
                                            enforce_sorted=False)
            collated['tgt'] = pack_sequence([s['tgt'] for s in batch],
                                            enforce_sorted=False)
            return collated

        return collate_fn
