import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .utils import Vocab


class TextDatasetRNN(Dataset):
    def __init__(self, root, split, src, tgt, vocab_share=False, max_len=80, **kwargs):
        samples = self.make_samples(root, 'train', src, tgt)

        if vocab_share:
            self.src_vocab = Vocab([*map(lambda x: x[0], samples),
                                    *map(lambda x: x[1], samples)])
            self.tgt_vocab = self.src_vocab
        else:
            self.src_vocab = Vocab(map(lambda x: x[0], samples))
            self.tgt_vocab = Vocab(map(lambda x: x[1], samples))

        if split == 'train':
            print('src_vocab', self.src_vocab)
            print('tgt_vocab', self.tgt_vocab)
            self.samples = samples
            self.max_len = max_len

        else:
            self.samples = self.make_samples(root, split, src, tgt)
            self.max_len = 999

        self.processed_samples = []
        for idx in range(len(self.samples)):
            src, tgt = self.samples[idx]
            src = ['<s>'] + src[:self.max_len] + ['</s>']
            tgt = ['<s>'] + tgt[:self.max_len] + ['</s>']
            self.processed_samples.append((
                torch.tensor(self.src_vocab.words2idxs(src)).long(),
                torch.tensor(self.tgt_vocab.words2idxs(tgt)).long())
            )

    def load_file(self, path):
        """Default loading function, which loads nth sentence at line n.
        """
        with open(path, 'r') as f:
            content = f.read().strip()
        return [s.strip().split() for s in content.split('\n')]

    def make_samples(self, root, split, src, tgt):
        src = self.load_file(os.path.join(root, split + '.' + src))
        tgt = self.load_file(os.path.join(root, split + '.' + tgt))
        return list(zip(src, tgt))

    def __getitem__(self, index):
        src, tgt = self.processed_samples[index]
        return {
            'src': src,
            'tgt': tgt,
        }

    def __len__(self):
        return len(self.samples)

    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {}
            collated['src'] = pad_sequence(
                [s['src'] for s in batch], batch_first=False)
            collated['tgt'] = pad_sequence(
                [s['tgt'] for s in batch], batch_first=False)

            return collated

        return collate_fn
