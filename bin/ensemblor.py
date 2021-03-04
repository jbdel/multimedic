import os
import sys
import argparse
import copy

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import parse_config, _parse_value
from torchnmt.executors import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args, others = parser.parse_known_args()
    return args, others


def get(opts, mode):
    opts = copy.deepcopy(opts)
    exec_opts = getattr(opts, mode)
    exec_opts.name = opts.name
    exec_opts.dataset = opts.dataset
    exec_opts.model = opts.model
    return exec_opts


def override(opts, others):
    # switch to dict
    for conf in others:
        try:
            op = opts
            key, value = conf.split(':')
            keys = key.split('.')
            for k in keys[:-1]:
                op = getattr(op, k)
            setattr(op, keys[-1], _parse_value(value))
        except ValueError:
            print(conf, 'badly formated')
            raise
    return opts

def main():
    args, others = get_args()
    opts = parse_config(args.config)
    opts = override(opts, others)

    train_opts = get(opts, 'train')
    beam_opts = get(opts, 'ensemblor')

    beamer = eval(beam_opts.proto)(train_opts=train_opts, ensemble_opts=beam_opts)
    beamer.start()


if __name__ == "__main__":
    main()
