import os
import sys
import argparse
import copy

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import parse_config
from torchnmt.executors import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    return args

def get(opts, mode):
    opts = copy.deepcopy(opts)
    exec_opts = getattr(opts, mode)
    exec_opts.name = opts.name
    exec_opts.dataset = opts.dataset
    exec_opts.model = opts.model
    return exec_opts

def main():
    args = get_args()
    opts = parse_config(args.config)

    train_opts = get(opts, 'train')
    beam_opts = get(opts, 'ensemblor')

    beamer = eval(beam_opts.proto)(train_opts=train_opts, ensemble_opts=beam_opts)
    beamer.start()


if __name__ == "__main__":
    main()
