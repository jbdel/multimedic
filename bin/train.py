import os
import sys
import argparse
import copy

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import parse_config, override
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


def main():
    args, others = get_args()
    opts = parse_config(args.config)
    opts = override(opts, others)

    train_opts = get(opts, 'train')
    val_opts = get(opts, 'validator')

    trainer = eval(train_opts.proto)(opts=train_opts, val_opts=val_opts)
    trainer.start()


if __name__ == "__main__":
    main()
