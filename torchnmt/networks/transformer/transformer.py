import copy
import random
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

from .encoders import TransformerEncoder
from .decoders import TransformerDecoder
from .utils import padding_mask, subsequent_mask, get_n_params
from torchnmt.datasets.utils import Vocab


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, vocab_share=False, **kwargs):
        super().__init__()
        assert encoder.model_dim == decoder.model_dim
        encoder_func = encoder.proto
        decoder_func = decoder.proto

        self.enc = eval(encoder_func)(**vars(encoder))
        self.dec = eval(decoder_func)(**vars(decoder))
        self.pad = Vocab.extra2idx('<pad>')

        if vocab_share:
            self.dec.embed = self.enc.embed

    def encode(self, src, **kwargs):
        # src, src_len = pad_packed_sequence(src, True, self.pad)
        print(src.shape)
        # src_mask = padding_mask(src_len)
        # print(src_mask)
        sys.exit()

        src = src.cuda()
        src_mask = src_mask.cuda()
        mem = self.enc(src, src_mask)
        return {'enc': (mem, src_mask)}

    def decode(self, tgt, ctx_dict):
        mem, src_mask = ctx_dict['enc']
        logp, loss = 0., 0.
        if tgt is not None:
            tgt, tgt_len = pad_packed_sequence(tgt, True, self.pad)
            tgt_mask = padding_mask(tgt_len) & subsequent_mask(tgt_len)

            tgt = tgt.cuda()
            tgt_mask = tgt_mask.cuda()

            outputs = self.dec(tgt, mem, src_mask, tgt_mask)
            logp = F.log_softmax(outputs, dim=-1)

            chopped_outputs = outputs[:, :-1].reshape(-1, outputs.shape[-1])
            shifted_targets = tgt[:, 1:].reshape(-1)

            loss = F.cross_entropy(chopped_outputs,
                                   shifted_targets,
                                   ignore_index=self.pad)
        return logp, loss

    def forward(self, src, tgt=None, **_):
        """
        Args:
            src: packed sequence (*, input_dim)
            tgt: packed sequence (*, output_dim)
        """
        print(tgt)
        logp, loss = self.decode(tgt, self.encode(src))
        return {'logp': logp, 'loss': loss}

    def __repr__(self):
        # s = super().__repr__() + '\n'
        s = ""
        s += "{}\n".format(get_n_params(self))
        return s
