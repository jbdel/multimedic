import os
import glob
import torch
from typing import Dict, Tuple
import json
import argparse


def todict(opts):
    d = dict()
    namespace = vars(opts)
    for k, v in namespace.items():
        if isinstance(v, argparse.Namespace):
            d[k] = todict(v)
        else:
            d.update({k: v})
    return d


def print_args(opts):
    classname = opts.proto
    d = todict(opts)
    if 'train' not in classname.lower():
        d.pop('model')
    print('\033[1m\033[91m' + classname + '\033[0m')
    print(json.dumps(d, indent=4, sort_keys=True))


class CheckpointSaver(object):
    def __init__(self, root, seed):
        self.root = root
        self.seed = seed
        self.current_tag = None
        self.current_step = None
        os.makedirs(os.path.dirname(self.root), exist_ok=True)

    def save(self, model, tag, current_step):
        if self.current_tag is not None:
            old_ckpt = os.path.join(self.root, '{}_{}_{}.pth'.format(self.current_tag, self.current_step, self.seed))
            assert os.path.exists(old_ckpt)
            os.remove(old_ckpt)

        path = os.path.join(self.root, '{}_{}_{}.pth'.format(tag, current_step, self.seed))
        torch.save(model, path)
        print('{} saved.'.format(path))

        self.current_tag = tag
        self.current_step = current_step
