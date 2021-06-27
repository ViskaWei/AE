import argparse


def get_args(default=None):
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default=default,
        help='The Configuration file')
    args = argparser.parse_args()
    return args
