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

        self.encoder = eval(encoder_func)(**vars(encoder))
        self.decoder = eval(decoder_func)(**vars(decoder))

        if vocab_share:
            self.decoder.embed = self.encoder.embed

    def forward(self, src, tgt=None, **_):
        """
        Args:
            src: packed sequence (*, input_dim)
            tgt: packed sequence (*, output_dim)
        """
        pad = Vocab.extra2idx('<pad>')
        src, src_len = pad_packed_sequence(src, True, pad)
        src_mask = padding_mask(src_len)
        src = src.cuda()
        src_mask = src_mask.cuda()
        mem = self.encoder(src, src_mask)

        ret = {'loss': 0.0}

        if tgt is not None:
            tgt, tgt_len = pad_packed_sequence(tgt, True, pad)
            tgt_mask = padding_mask(tgt_len) & subsequent_mask(tgt_len)

            tgt = tgt.cuda()
            tgt_mask = tgt_mask.cuda()

            outputs = self.decoder(tgt, mem, src_mask, tgt_mask)
            logp = F.log_softmax(outputs, dim=-1)

            chopped_outputs = outputs[:, :-1].reshape(-1, outputs.shape[-1])
            shifted_targets = tgt[:, 1:].reshape(-1)

            loss = F.cross_entropy(chopped_outputs,
                                   shifted_targets,
                                   ignore_index=pad)
            ret.update({
                'logp': logp,
                'loss': loss
            })

        if not self.training:
            # Return for possible beam search and model ensembling
            ret.update({
                'mem': mem,
                'src_mask': src_mask
            })

        return ret

    def __repr__(self):
        # s = super().__repr__() + '\n'
        s = ""
        s += "{}\n".format(get_n_params(self))
        return s