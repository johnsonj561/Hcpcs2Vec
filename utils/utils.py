import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import sys
import time
from io import StringIO
from datetime import datetime

warnings.filterwarnings(action="ignore", category=UndefinedMetricWarning)


def file_ts():
    """Returns formatted timestamp for identifying files"""
    ts = datetime.now()
    return ts.strftime("%m%d%y%H%M%S")


def get_vocab_size(corpus):
    uniques = {}
    for row in corpus:
        for val in row:
            if val != np.float:
                uniques[val] = 1

    return len(uniques.keys())
