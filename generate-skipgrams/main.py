import pandas as pd
import numpy as np
import os
import sys
import pickle
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
make_sampling_table = tf.keras.preprocessing.sequence.make_sampling_table
skipgrams = tf.keras.preprocessing.sequence.skipgrams
# proj_dir = '/Users/jujohnson/git/Hcpcs2Vec/'
proj_dir = '/home/jjohn273/git/Hcpcs2Vec/'
sys.path.append(proj_dir)
from utils import args_to_dict, file_ts, Timer  # NOQA: E402
from medicare import load_data  # NOQA: E402

# parse cli args
filename = sys.argv[0].replace('.py', '')
cli_args = args_to_dict(sys.argv)
window_size = cli_args.get('window_size')
if window_size != None:
    window_size = int(window_size)
debug = cli_args.get('debug') == 'true'
print(f'Using debug: {debug}, window size: {window_size}')


# config
data_dir = os.environ['CMS_RAW']
ts = file_ts()


# output files
partb_output = os.path.join(proj_dir, 'data', 'partb-2012-2015.csv.gz')
hcpcs_map_output = os.path.join(
    proj_dir, 'data', f'hcpcs-labelencoding.{ts}.pickle')
skipgram_x_output = os.path.join(
    proj_dir, 'data', f'skipgrams-x-w{window_size}-{ts}.npy')
skipgram_y_output = os.path.join(
    proj_dir, 'data', f'skipgrams-y-w{window_size}-{ts}.npy')


# load Medicare Data
timer = Timer()
data = load_data(data_dir, partb_output, debug)
print(f'Loaded data in {timer.lap()}')


# generate and save HCPCS ID Mapping
le = LabelEncoder()
data['hcpcs_id'] = le.fit_transform(data['hcpcs'].astype(str))
if not debug:
    with open(hcpcs_map_output, 'wb') as fout:
        pickle.dump(le.classes_, fout)
print(f'Generated HCPCS-ID mapping in {timer.lap()}')


# generate HCPCS sequence corpus
# each item in corpus is a list of HCPCS provided
# by a provider in a given year
corpus = []
for npi, group in data.groupby(by='npi'):
    group.sort_values(by='count', inplace=True)
    hcpcs_set = np.asarray(group['hcpcs_id'], dtype='int16')
    corpus.append(hcpcs_set)
corpus_length = len(corpus)
print(f'Created corpus with length: {len(corpus)} in {timer.lap()}')


# Reduce the max sequence length
quantile = 0.98
lengths = np.array(list(map(lambda x: len(x), corpus)))
max_seq_length = np.quantile(lengths, quantile)
corpus = np.array(list(filter(lambda x: len(x) <= max_seq_length, corpus)))
print(f'Removed longest hcpcs seq from {quantile}+ quantile in {timer.lap()}')
print(f'Updated corpus length {len(corpus)}')


# Get vocab size
vocab_size = data['hcpcs_id'].max() + 1
print(f'Using vocab_size: {vocab_size}')


# Free up some memory
del data


# Create skip-gram pairs and negative Ssmples (thx Keras)
timer.reset()
print(f'Using vocab_size {vocab_size}')
sampling_table = make_sampling_table(vocab_size)
x, y = [], []
for seq in corpus:
    # if window size is undefined, treat entire sequence as window
    if window_size == None:
        window_size = len(corpus) / 2
    pairs, labels = skipgrams(
        seq, vocab_size, window_size=window_size, sampling_table=sampling_table)
    x.extend(pairs)
    y.extend(labels)
x, y = np.array(x, dtype='int16'), np.array(y, dtype='int8')
print(f'Created skip-gram pairs with shape: {x.shape} in {timer.lap()}')


# Save results
if not debug:
    np.save(skipgram_x_output, x)
    np.save(skipgram_y_output, y)
    print(f'Saved skipgrams to {skipgram_x_output}, {skipgram_y_output}')
