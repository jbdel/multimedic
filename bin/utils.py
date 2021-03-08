import argparse
import yaml
from ast import literal_eval


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


def _parse_value(value):
    """Automatic type conversion for configuration values.

    Arguments:
        value(str): A string to parse.
    """

    # Check for boolean or None
    if str(value).capitalize().startswith(('False', 'True', 'None')):
        return eval(str(value).capitalize(), {}, {})

    # Detect strings, floats and ints
    try:
        # If this fails, this is a string
        result = literal_eval(value)
    except Exception:
        result = value

    return result


def override(opts, others):
    # switch to dict
    for conf in others:
        try:
            op = opts
            key, value = conf.split(':')
            if "." not in key:
                setattr(op, key, _parse_value(value))
                continue
            keys = key.split('.')
            for k in keys[:-1]:
                op = getattr(op, k)
            setattr(op, keys[-1], _parse_value(value))
        except ValueError:
            print(conf, 'badly formated')
            raise
    return opts
