import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from .utils import Vocab


class TextDatasetRNNStatic:
    max_len = None
    src_vocab = None
    tgt_vocab = None


class TextDatasetRNN(Dataset):
    def __init__(self, root, split, src, tgt, vocab_share=False, max_len=80, eval_max_len=200, **kwargs):
        self.root = root
        self.split = split
        self.samples = self.make_samples(root, split, src, tgt)

        if split == 'train':
            # Create vocab
            if vocab_share:
                TextDatasetRNNStatic.src_vocab = Vocab([*map(lambda x: x[0], self.samples),
                                                        *map(lambda x: x[1], self.samples)])
                TextDatasetRNNStatic.tgt_vocab = TextDatasetRNNStatic.src_vocab
            else:
                TextDatasetRNNStatic.src_vocab = Vocab(map(lambda x: x[0], self.samples))
                TextDatasetRNNStatic.tgt_vocab = Vocab(map(lambda x: x[1], self.samples))
                TextDatasetRNNStatic.src_vocab.dump(os.path.join(root, "vocab.src"))
                TextDatasetRNNStatic.tgt_vocab.dump(os.path.join(root, "vocab.tgt"))

            TextDatasetRNNStatic.max_len = max_len
            self.src_len, self.tgt_len = max_len, max_len

            print('src_vocab', TextDatasetRNNStatic.src_vocab)
            print('tgt_vocab', TextDatasetRNNStatic.tgt_vocab)
            print('train_max_len', TextDatasetRNNStatic.max_len)

        else:
            self.src_len, self.tgt_len = TextDatasetRNNStatic.max_len, eval_max_len

        self.src_vocab = TextDatasetRNNStatic.src_vocab
        self.tgt_vocab = TextDatasetRNNStatic.tgt_vocab
        self.processed_samples = []
        self.lengths = []

        # self.excluded = 0
        for idx in range(len(self.samples)):
            src, tgt = self.samples[idx]

            src = src + ['[SEP]']
            tgt = ['[CLS]'] + tgt + ['[SEP]']
            self.processed_samples.append((
                torch.tensor(self.src_vocab.words2idxs(src)).long(),
                torch.tensor(self.tgt_vocab.words2idxs(tgt)).long())
            )
            self.lengths.append(len(src))

        # print("Rejected", self.excluded, "samples at len >", self.src_len)

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
        return len(self.processed_samples)

    def get_collate_fn(self):
        def collate_fn(batch):
            collated = {
                'src': pad_sequence([s['src'] for s in batch], batch_first=False),
                'tgt': pad_sequence([s['tgt'] for s in batch], batch_first=False)}
            return collated

        return collate_fn
