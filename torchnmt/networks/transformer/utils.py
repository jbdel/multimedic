import torch
import numpy as np


def padding_mask(lens):
    """Mask out the blank (padding) values
    Args:
        lens: (bs,)
    Return:
        mask: (bs, 1, max_len)
    """
    bs, max_len = len(lens), max(lens)
    mask = torch.zeros(bs, 1, max_len)
    for i, l in enumerate(lens):
        mask[i, :, :l] = 1
    mask = mask > 0
    return mask


def subsequent_mask(lens):
    """Mask out future word
    Args:
        lens: (bs,)
    Return:
        mask: (bs, max_len, max_len)
    """
    bs, max_len = len(lens), max(lens)
    mask = torch.ones([bs, max_len, max_len]).tril_(0)
    mask = mask > 0
    return mask

def readable_size(n):
    """Return a readable size string."""
    sizes = ['K', 'M', 'G']
    fmt = ''
    size = n
    for i, s in enumerate(sizes):
        nn = n / (1000 ** (i + 1))
        if nn >= 1:
            size = nn
            fmt = sizes[i]
        else:
            break
    return '%.2f%s' % (size, fmt)

def get_n_params(module):
    n_param_learnable = 0
    n_param_frozen = 0

    for param in module.parameters():
        if param.requires_grad:
            n_param_learnable += np.cumprod(param.data.size())[-1]
        else:
            n_param_frozen += np.cumprod(param.data.size())[-1]

    n_param_all = n_param_learnable + n_param_frozen
    return "# parameters: {} ({} learnable)".format(
        readable_size(n_param_all), readable_size(n_param_learnable))


