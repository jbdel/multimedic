import os
from torch.utils.data import Dataset
from transformers import BertTokenizer
import tqdm
import torch

extra = [
    "[PAD]",
    "[SEP]",
    "[CLS]",
    "[UNK]",
    "[MASK]"]


class TextDatasetHugStatic:
    max_len = None


class TextDatasetHug(Dataset):
    def __init__(self, root, split, src, tgt, vocab_share=False, max_len=80, **kwargs):
        self.root = root
        self.split = split
        self.samples = self.make_samples(root, split, src, tgt)
        src_vocab_file = os.path.join(self.root, 'vocab_src.txt')
        tgt_vocab_file = os.path.join(self.root, 'vocab_tgt.txt')

        if split == 'train':
            src_vocab = self.get_vocab(map(lambda x: x[0], self.samples))
            tgt_vocab = self.get_vocab(map(lambda x: x[1], self.samples))

            open(src_vocab_file, "w").write("\n".join(str(word) for word in extra + src_vocab))
            open(tgt_vocab_file, "w").write("\n".join(str(word) for word in extra + tgt_vocab))

            TextDatasetHugStatic.max_len = max_len
            self.src_len, self.tgt_len = max_len, 999

            print("Vocab(#vocab={}, #extra={}, src_train_len={})".format(len(src_vocab), len(extra), self.src_len))
            print("Vocab(#vocab={}, #extra={}, tgt_train_len_len={})".format(len(tgt_vocab), len(extra), self.tgt_len))

        else:
            self.src_len, self.tgt_len = TextDatasetHugStatic.max_len, 999

        self.src_tokenizer = BertTokenizer(vocab_file=src_vocab_file, do_basic_tokenize=False)
        self.tgt_tokenizer = BertTokenizer(vocab_file=tgt_vocab_file, do_basic_tokenize=False)

        # self.processed_samples = []
        # for idx in tqdm.tqdm(range(len(self.samples))):
        #     src, tgt = self.samples[idx]
        #     self.processed_samples.append((self.src_tokenizer(' '.join(src)), self.tgt_tokenizer(' '.join(tgt))))

    def __getitem__(self, index):
        # src, tgt = self.processed_samples[index]
        # print(self.samples[index])
        src, tgt = self.samples[index]
        return {
            'src': src,
            'tgt': tgt,
        }

    def get_collate_fn(self):
        def collate_fn(batch):
            # src = self.src_tokenizer([s['src'] for s in batch], padding=True, truncation=True, return_tensors="pt",
            #                          max_length=self.src_len)
            # tgt = self.tgt_tokenizer([s['tgt'] for s in batch], padding=True, truncation=True, return_tensors="pt",
            #                          max_length=self.tgt_len)
            # v = batch[0]["tgt"]
            # print(len(v))
            # print(v)
            src = self.src_tokenizer([s['src'] for s in batch], padding=True, return_tensors="pt")
            tgt = self.tgt_tokenizer([s['tgt'] for s in batch], padding=True, return_tensors="pt")
            # print(src)
            # print(tgt)
            # sys.exit()
            # print("####")
            # print(tgt["input_ids"].shape)
            # print(tgt["input_ids"][0])
            # print(len(tgt["input_ids"][0]))
            # print(self.tgt_tokenizer.decode(tgt["input_ids"][0], skip_special_tokens=False, clean_up_tokenization_spaces=False))
            collated = {'input_ids': src.input_ids,
                        'attention_mask': src.attention_mask,
                        'decoder_input_ids': tgt.input_ids,
                        'decoder_attention_mask': tgt.attention_mask}

            return collated

        return collate_fn

    def make_samples(self, root, split, src, tgt):
        src = self.load_file(os.path.join(root, split + '.' + src))
        tgt = self.load_file(os.path.join(root, split + '.' + tgt))
        return list(zip(src, tgt))

    def load_file(self, path):
        """Default loading function, which loads nth sentence at line n.
        """
        with open(path, 'r') as f:
            content = f.read().strip()
        return [s.strip() for s in content.split('\n')]

    def get_vocab(self, sentences):
        vocab = []
        for sentence in sentences:
            vocab += sentence.split()
        return sorted(set(vocab))

    def __len__(self):
        return len(self.samples)
