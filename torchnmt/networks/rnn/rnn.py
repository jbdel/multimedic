import torch.nn as nn
from .textencoder import TextEncoder
from .decoder import ConditionalDecoder
import torch
from .utils import get_n_params


class RNN(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__()

        encoder_func = encoder.proto
        decoder_func = decoder.proto

        self.enc = eval(encoder_func)(**vars(encoder))
        self.dec = eval(decoder_func)(**vars(decoder), encoder_size=self.enc.ctx_size)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            # Skip 1-d biases and scalars
            if param.requires_grad and param.dim() > 1:
                nn.init.kaiming_normal_(param.data)
        # Reset padding embedding to 0
        if hasattr(self, 'enc') and hasattr(self.enc, 'emb'):
            with torch.no_grad():
                self.enc.emb.weight.data[0].fill_(0)

    def encode(self, src, **kwargs):
        d = {'enc': self.enc(src)}
        # if 'feats' in kwargs: # todo
        #     d['feats'] = (kwargs[''], None)
        return d

    def forward(self, src, tgt, **kwargs):
        # Get loss dict
        src = src.cuda()
        tgt = tgt.cuda()
        result = self.dec(self.encode(src, **kwargs), tgt)
        result['n_items'] = torch.nonzero(tgt[1:]).shape[0]
        result['loss'] = result['loss'] / result['n_items']

        return result

    def __repr__(self):
        s = super().__repr__() + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
