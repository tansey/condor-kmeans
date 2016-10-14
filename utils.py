import os
import sys
import csv
import numpy as np


def make_directory(base, subdir):
    if not base.endswith('/'):
        base += '/'
    directory = base + subdir
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not directory.endswith('/'):
        directory = directory + '/'
    return directory

def pretty_str(p, decimal_places=2, ignore=None, label_columns=False):
    '''Pretty-print a matrix or vector.'''
    if len(p.shape) == 1:
        return vector_str(p, decimal_places, ignore)
    if len(p.shape) == 2:
        return matrix_str(p, decimal_places, ignore, label_columns)
    raise Exception('Invalid array with shape {0}'.format(p.shape))

def matrix_str(p, decimal_places=2, ignore=None, label_columns=False):
    '''Pretty-print the matrix.'''
    return '[{0}]'.format("\n  ".join([(str(i) if label_columns else '') + vector_str(a, decimal_places, ignore) for i, a in enumerate(p)]))

def vector_str(p, decimal_places=2, ignore=None):
    '''Pretty-print the vector values.'''
    style = '{0:.' + str(decimal_places) + 'f}'
    return '[{0}]'.format(", ".join([' ' if ((hasattr(ignore, "__len__") and a in ignore) or a == ignore) else style.format(a) for a in p]))

