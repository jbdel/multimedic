from torch.nn.utils.rnn import pad_packed_sequence
from .utils import padding_mask, subsequent_mask
from torchnmt.datasets.utils import Vocab
import torch
import numpy as np
import tqdm


def unpack_packed_sequence(sequence):
    seqs, lens = pad_packed_sequence(sequence, True)
    return [seq[:l] for seq, l in zip(seqs, lens)]


def greedy_inference(models, mems, src_masks, max_len, **kwargs):
    """
    Args:
        mem: (bs, src_len, model_dim)
    Outputs:
        tgt_output: [(tgt_len,)]
    """
    bos = Vocab.extra2idx('<s>')
    eos = Vocab.extra2idx('</s>')

    dummy_model = models[0]
    dummy_mem = mems[0]
    batch_size = len(dummy_mem)

    batch_idx = torch.arange(batch_size)
    running = torch.full((batch_size, 1), bos).long().to(dummy_model.device)
    finished = []

    for l in range(1, max_len):
        tgt_mask = subsequent_mask([l]).to(dummy_model.device)

        # average logps
        logps = torch.stack([model.decoder(running, mem, src_mask, tgt_mask)
                             for model, mem, src_mask in zip(models, mems, src_masks)])
        logps = logps.mean(0)

        # then argmax
        logps = logps[:, -1].argmax(dim=-1)  # (bs,)
        running = torch.cat([running, logps[:, None]], dim=-1)

        running_idx = torch.nonzero(logps != eos).squeeze(1)
        finished_idx = torch.nonzero(logps == eos).squeeze(1)

        finished += list(zip(batch_idx[finished_idx],
                             running[finished_idx].tolist()))

        running = running[running_idx]
        batch_idx = batch_idx[running_idx]

        mems = [mem[running_idx] for mem in mems]
        src_masks = [src_mask[running_idx] for src_mask in src_masks]

        if len(running) == 0:
            break

    finished += list(zip(batch_idx, running.tolist()))
    finished = [x[1] for x in sorted(finished, key=lambda x: x[0])]

    return finished


def beam_search_inference(models, mems, src_masks, max_len, beam_width, **kwargs):
    batch_hyps = []

    bos = Vocab.extra2idx('<s>')
    eos = Vocab.extra2idx('</s>')

    dummy_model = models[0]

    mems = torch.stack(mems)  # torch.Size([num_models, bs, len, feature_dim])
    src_masks = torch.stack(src_masks)

    memories = mems.expand(beam_width, *mems.shape)  # torch.Size([beam_size, num_models, bs, len, feature_dim])
    mem_masks = src_masks.expand(beam_width, *src_masks.shape)

    memories = memories.permute(2, 1, 0, 3, 4)  # torch.Size([bs, num_models, beam_size, len, feature_dim])
    mem_masks = mem_masks.permute(2, 1, 0, 3, 4)

    # per sample beam search
    for memory, mem_mask in zip(memories, mem_masks):
        logps = torch.full((beam_width,), -np.inf).to(dummy_model.device)
        logps[0] = 0
        hyps = torch.full((beam_width, 1), bos).long().to(dummy_model.device)
        finished = []

        for l in range(1, max_len):
            k = len(logps)
            tgt_mask = subsequent_mask([l]).to(dummy_model.device)

            outputs = [model.decoder(hyps,
                                     mem[:k],
                                     mask[:k],
                                     tgt_mask) for model, mem, mask in zip(models, memory, mem_mask)]

            # Averaging per model
            outputs = torch.stack(outputs).mean(0)
            outputs = torch.log_softmax(outputs[:, -1], dim=-1)
            # for each beam, calculate top k
            tmp_logps, tmp_idxs = torch.topk(outputs, k)

            # calculate accumulated logps
            tmp_logps += logps[:, None]
            # calculate new top k
            tmp_logps = tmp_logps.view(-1)
            tmp_idxs = tmp_idxs.view(-1)

            logps, idxs = torch.topk(tmp_logps, k)

            words = tmp_idxs[idxs]
            hyps_idxs = idxs // k

            hyps = torch.cat([hyps[hyps_idxs], words[:, None]], dim=1)

            finished_idx = torch.nonzero(words == eos).squeeze(1)
            running_idx = torch.nonzero(words != eos).squeeze(1)

            finished += list(zip(logps[finished_idx], hyps[finished_idx]))

            logps = logps[running_idx]
            hyps = hyps[running_idx]

            if len(logps) <= 0:
                break

        finished = finished + list(zip(logps, hyps))

        hyp = max(finished, key=lambda t: t[0])[1]
        batch_hyps.append(hyp)

    return batch_hyps


def evaluation(models, opts, dl):
    ret_losses = []
    ret_refs = []
    ret_hyps = []
    pbar = tqdm.tqdm(dl, total=len(dl))
    for batch in pbar:
        refs = unpack_packed_sequence(batch['tgt'])
        losses = []
        mems = []
        src_masks = []
        with torch.no_grad():
            # encode
            for m in models:
                assert not m.training
                out = m(**batch, **vars(opts))
                losses.append(out['loss'].item())
                mems.append(out['mem'])
                src_masks.append(out['src_mask'])

            # decode
            if opts.beam_width > 1:
                hyps = beam_search_inference(models, mems, src_masks, **vars(opts))
            else:
                hyps = greedy_inference(models, mems, src_masks, **vars(opts))

        ret_losses.append(np.mean(losses))
        ret_refs += refs
        ret_hyps += hyps

    vocab = dl.dataset.tgt_vocab
    ret_refs = [vocab.strip_beos_w(vocab.idxs2words(ref))
                for ref in ret_refs]
    ret_hyps = [vocab.strip_beos_w(vocab.idxs2words(hyp))
                for hyp in ret_hyps]

    return ret_losses, ret_refs, ret_hyps
