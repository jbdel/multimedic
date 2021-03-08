from .transformer import EncoderDecoderModel
from .beam_helpers import generate
import tqdm
import torch


def evaluation(models, opts, dl):
    model = models[0]  # get model attributes

    encdecs = [model.enc_dec for model in models]
    encdec: EncoderDecoderModel = encdecs[0]

    encdecs = [encdec, encdec]
    tgt_tokenizer = dl.dataset.tgt_tokenizer
    ref_list = []
    hyp_list = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dl):
            refs = batch['decoder_input_ids']

            hyps = generate(encdec,
                            encdecs,
                            input_ids=batch['input_ids'].cuda(),
                            num_return_sequences=1,
                            max_length=dl.dataset.tgt_len,
                            num_beams=opts.beam_width,
                            early_stopping=True,
                            bos_token_id=model.bos_token_id,
                            eos_token_id=model.eos_token_id,
                            pad_token_id=model.pad_token_id
                            )

            for h, r in zip(hyps, refs):
                hyp_list.append(tgt_tokenizer.decode(h, skip_special_tokens=True, clean_up_tokenization_spaces=False))
                ref_list.append(tgt_tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=False))

        return 0.0, ref_list, hyp_list
