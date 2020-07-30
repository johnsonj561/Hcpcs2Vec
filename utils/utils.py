import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import pandas as pd
import sys
import os
import time
from io import StringIO
from datetime import datetime

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


def file_ts():
    '''Returns formatted timestamp for identifying files'''
    ts = datetime.now()
    return ts.strftime('%m-%d-%y-%H-%M-%S')


def args_to_dict(args, skip_filename=True):
    if skip_filename:
        args = args[1:]
    result = {}
    for arg in args:
        key, value = arg.split('=')
        result[key] = value
    return result


def get_minority_ratio(col, posLabel=1, negLabel=0):
    neg_count, pos_count = len(col[col == negLabel]), len(col[col == posLabel])
    positive_ratio = (pos_count / (pos_count + neg_count)) * 100
    return positive_ratio, 1 - positive_ratio


def model_summary_to_string(model):
    '''Converts Tensorflow model summary to string for IO.'''
    # keep track of the original sys.stdout
    origStdout = sys.stdout
    # replace sys.stdout temporarily with our own buffer
    outputBuf = StringIO()
    sys.stdout = outputBuf
    # print the model summary
    model.summary()
    # put back the original stdout
    sys.stdout = origStdout
    # get the model summary as a string
    return 'Model Summary:\n' + outputBuf.getvalue()


def rounded_str(num, precision=6):
    if type(num) == str:
        return num
    return str(round(num, precision))


class Timer():
    def __init__(self):
        self.times = []
        self.reset()

    def reset(self):
        self.t0 = time.time()

    def lap(self):
        interval = time.time() - self.t0
        self.t0 = time.time()
        self.times.append(interval)
        return interval

    def write_to_file(self, out):
        with open(out, 'a') as fout:
            times = [rounded_str(t) for t in self.times]
            fout.write(','.join([*times, '\n']))
