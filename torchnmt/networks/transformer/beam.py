from torch.nn.utils.rnn import pad_packed_sequence
from .utils import padding_mask, subsequent_mask
from torchnmt.datasets.utils import Vocab
import torch
import numpy as np
import tqdm


def evaluation(models, opts, dl, lp_alpha=0.0):
    def tile_ctx_dict(ctx_dict, idxs):
        """Returns dict of 3D tensors repeatedly indexed along the sample axis."""
        # 1st: tensor, 2nd optional mask
        return {
            k: (t[:, idxs], None if mask is None else mask[:, idxs])
            for k, (t, mask) in ctx_dict.items()
        }

    def check_context_ndims(ctx_dict):
        for name, (ctx, mask) in ctx_dict.items():
            assert ctx.dim() == 3, \
                f"{name}'s 1st dim should always be a time dimension."

    # This is the batch-size requested by the user but with sorted
    # batches, efficient batch-size will be <= max_batch_size
    max_batch_size = opts.batch_size
    k = opts.beam_width
    inf = -1000
    max_len = dl.dataset.src_len

    results = []

    # For classical models that have single encoder, decoder and
    # target vocabulary
    decs = [m.decode for m in models]

    # Common parts
    encoders = [m.encode for m in models]
    eos = Vocab.extra2idx('</s>')
    unk = Vocab.extra2idx('<unk>')
    bos = Vocab.extra2idx('<s>')

    vocab = dl.dataset.tgt_vocab
    n_vocab = len(vocab)

    # Tensorized beam that will shrink and grow up to max_batch_size
    beam_storage = torch.zeros(
        max_len, max_batch_size, k, dtype=torch.long).cuda()
    mask = torch.arange(max_batch_size * k).cuda()
    nll_storage = torch.zeros(max_batch_size).cuda()

    pbar = tqdm.tqdm(dl, total=len(dl))
    for batch in pbar:
        refs = unpack_packed_sequence(batch['tgt'])
        batch_size = len(refs)
        batch = {k: v.cuda() for k, v in batch.items()}

        # Always use the initial storage
        beam = beam_storage.narrow(1, 0, batch_size).zero_()

        # Mask to apply to pdxs.view(-1) to fix indices
        nk_mask = mask.narrow(0, 0, batch_size * k)

        # nll: batch_size x 1 (will get expanded further)
        nll = nll_storage.narrow(0, 0, batch_size).unsqueeze(1)

        # Tile indices to use in the loop to expand first dim
        tile = range(batch_size)

        # Encode source

        ctx_dicts = [encoder(**batch) for encoder in encoders]
        # Sanity check one of the context dictionaries for dimensions
        check_context_ndims(ctx_dicts[0])

        # we always have <bos> tokens except that the returned embeddings
        # may differ from one model to another.

        idxs = torch.LongTensor(batch_size).fill_(bos).cuda()
        print(idxs)
        for tstep in range(max_len):
            # Select correct positions from source context
            ctx_dicts = [tile_ctx_dict(cd, tile) for cd in ctx_dicts]

            # Get log probabilities and next state
            # log_p: batch_size x vocab_size (t = 0)
            #        batch_size*beam_size x vocab_size (t > 0)
            # NOTE: get_emb does not exist in some models, fix this.
            log_ps, _ = zip(
                *[dec(idxs, cd) for
                  dec, cd in zip(decs, ctx_dicts)])
            print(log_ps)
            print(print(len(log_ps)))
            print(log_ps.shape)
            sys.exit()
            # Do the actual averaging of log-probabilities
            log_p = sum(log_ps).data

            # Detect <eos>'d hyps
            idxs = torch.nonzero(idxs == eos)
            if idxs.numel():
                if idxs.numel() == batch_size * k:
                    break
                idxs.squeeze_(-1)
                # Unfavor all candidates
                log_p.index_fill_(0, idxs, inf)
                # Favor <eos> so that it gets selected
                log_p.view(-1).index_fill_(0, idxs * n_vocab + 2, 0)

            # Expand to 3D, cross-sum scores and reduce back to 2D
            # log_p: batch_size x vocab_size ( t = 0 )
            #   nll: batch_size x beam_size (x 1)
            # nll becomes: batch_size x beam_size*vocab_size here
            # Reduce (N, K*V) to k-best
            nll, beam[tstep] = nll.unsqueeze_(2).add(log_p.view(
                batch_size, -1, n_vocab)).view(batch_size, -1).topk(
                k, sorted=False, largest=True)

            # previous indices into the beam and current token indices
            pdxs = beam[tstep] / n_vocab
            beam[tstep].remainder_(n_vocab)
            idxs = beam[tstep].view(-1)

            # Compute correct previous indices
            # Mask is needed since we're in flattened regime
            tile = pdxs.view(-1) + (nk_mask / k) * (k if tstep else 1)

            if tstep > 0:
                # Permute all hypothesis history according to new order
                beam[:tstep] = beam[:tstep].gather(2, pdxs.repeat(tstep, 1, 1))

        # Put an explicit <eos> to make idxs_to_sent happy
        beam[max_len - 1] = eos

        # Find lengths by summing tokens not in (pad,bos,eos)
        len_penalty = beam.gt(2).float().sum(0).clamp(min=1)

        if lp_alpha > 0.:
            len_penalty = ((5 + len_penalty) ** lp_alpha) / 6 ** lp_alpha

        # Apply length normalization
        nll.div_(len_penalty)

        # Get best-1 hypotheses
        top_hyps = nll.topk(1, sorted=False, largest=True)[1].squeeze(1)
        hyps = beam[:, range(batch_size), top_hyps].t().to('cpu')
        results.extend(hyps.tolist())

    # Recover order of the samples if necessary
    if getattr(dl.batch_sampler, 'store_indices', False):
        results = [results[i] for i, j in sorted(
            enumerate(dl.batch_sampler.orig_idxs), key=lambda k: k[1])]

    ret_refs = []
    for i in range(len(dl.dataset)):
        _, tgt = dl.dataset.samples[i]
        ret_refs.append(tgt)

    ret_hyps = [vocab.strip_beos_w(vocab.idxs2words(hyp))
                for hyp in results]

    return 0.0, ret_refs, ret_hyps




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

#
# def evaluation(models, opts, dl):
#     ret_losses = []
#     ret_refs = []
#     ret_hyps = []
#     max_len = dl.dataset.src_len
#     pbar = tqdm.tqdm(dl, total=len(dl))
#     for batch in pbar:
#         refs = unpack_packed_sequence(batch['tgt'])
#         losses = []
#         mems = []
#         src_masks = []
#         with torch.no_grad():
#             # encode
#             for m in models:
#                 assert not m.training
#                 out = m(**batch, **vars(opts))
#                 losses.append(out['loss'].item())
#                 mems.append(out['mem'])
#                 src_masks.append(out['src_mask'])
#
#             # decode
#             if opts.beam_width > 1:
#                 hyps = beam_search_inference(models, mems, src_masks, max_len, **vars(opts))
#             else:
#                 hyps = greedy_inference(models, mems, src_masks, max_len, **vars(opts))
#
#         ret_losses.append(np.mean(losses))
#         ret_refs += refs
#         ret_hyps += hyps
#
#     vocab = dl.dataset.tgt_vocab
#     ret_refs = [vocab.strip_beos_w(vocab.idxs2words(ref))
#                 for ref in ret_refs]
#     ret_hyps = [vocab.strip_beos_w(vocab.idxs2words(hyp))
#                 for hyp in ret_hyps]
#
#     return ret_losses, ret_refs, ret_hyps
