import argparse
import yaml

def make_namespace(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = make_namespace(d[key])
    return argparse.Namespace(**d)


def parse_config(p):
    with open(p, 'r') as stream:
        d = yaml.load(stream, Loader=yaml.FullLoader)
    ns = make_namespace(d)
    return ns
