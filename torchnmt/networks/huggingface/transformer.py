import torch.nn as nn
from ..rnn.utils import get_n_params

# v4.3.2
from transformers.models.bert_generation import BertGenerationEncoder, BertGenerationConfig, BertGenerationDecoder
from transformers import EncoderDecoderModel, EncoderDecoderConfig
from types import MethodType
from .beam_helpers import generate


class TransformerHug(nn.Module):
    def __init__(self, encoder, decoder, vocab_share=False, **kwargs):
        super().__init__()
        encoder = vars(encoder)
        decoder = vars(decoder)
        encoder_func = encoder.pop('proto')
        decoder_func = decoder.pop('proto')

        configuration = BertGenerationConfig(**encoder)
        self.enc = BertGenerationEncoder(configuration)

        configuration.update(decoder)
        configuration.is_decoder = True
        configuration.add_cross_attention = True

        # config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config=config_encoder, decoder_config=config_decoder)


        self.dec = BertGenerationDecoder(configuration)
        self.enc_dec = EncoderDecoderModel(encoder=self.enc, decoder=self.dec)

        self.bos_token_id = configuration.bos_token_id
        self.eos_token_id = configuration.eos_token_id
        self.pad_token_id = configuration.pad_token_id

    def forward(self, input_ids, decoder_input_ids):
        input_ids = input_ids.cuda()
        decoder_input_ids = decoder_input_ids.cuda()

        out = self.enc_dec(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=decoder_input_ids)
        out = vars(out)
        return out

    def __repr__(self):
        # s = super().__repr__() + '\n'
        s = ""
        s += "{}\n".format(get_n_params(self))
        return s
