from models import skipgram_model
import pandas as pd
import numpy as np
import pickle
import os
import sys
import math
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
Keras = tf.keras
Model, Layers = Keras.models.Model, Keras.layers
make_sampling_table = Keras.preprocessing.sequence.make_sampling_table
skipgrams = Keras.preprocessing.sequence.skipgrams
proj_dir = '/Users/jujohnson/git/Hcpcs2Vec/'
sys.path.append(proj_dir)
from utils import file_ts, Timer, args_to_dict  # NOQA: E402
from medicare import load_data, load_data_sample  # NOQA: E402


################################
# Config
################################
data_dir = os.environ['CMS_RAW']
filename = sys.argv[0].replace('.py', '')
cli_args = args_to_dict(sys.argv)
debug = cli_args.get('debug') == 'true'
epochs = int(cli_args.get('epochs', 100))
embedding_size = int(cli_args.get('embedding_size', 300))
window_size = int(cli_args.get('window_size', 5))
hcpcs_map_filename = os.path.join(
    proj_dir, 'data', 'hcpcs-labelencoding.pickle')


################################
# Load Medicare Data
################################
timer = Timer()
data = load_data_sample(data_dir)
print(f'Loaded data in {timer.lap()}')


################################
# Create HCPCS <--> ID Mapping
################################
le = LabelEncoder()
data['hcpcs_id'] = le.fit_transform(data['hcpcs'])
with open(hcpcs_map_filename, 'wb') as fout:
    pickle.dump(le.classes_, fout)
print(f'Saved HCPCS ID mapping to {hcpcsIdFile} in {timer.lap()}')


################################
# Extract HCPCS Corpus from the
# Medicare data
################################
corpus = []
for npi, group in data.groupby(by='npi'):
    group.sort_values(by='count', inplace=True)
    hcpcs_set = np.asarray(group['hcpcs_id'], dtype='int16')
    corpus.append(hcpcs_set)
corpus_length = len(corpus)
print(f'Created corpus with length: {corpus_length} in {timer.lap()}')


################################
# Reduce the Max Sequence Length
################################
quantile = 0.98
lengths = np.array(list(map(lambda x: len(x), corpus)))
max_seq_length = np.quantile(lengths, quantile)
corpus = np.array(list(filter(lambda x: len(x) <= max_seq_length, corpus)))
print(f'Removed {corpus_length - len(corpus)} samples greater than {quantile} quantile in {timer.lap()}')
corpus_length = len(corpus)
print(f'Updated corpus length: {corpus_length}')


################################
# Get Vocab Size
################################
vocab_size = data['hcpcs_id'].max() + 1
# don't need original DF anymore
del data
print(f'Using vocab_size: {vocab_size}')


################################
# Create Skip-Gram Pairs and
# Negative Samples (thx Keras)
################################
timer.reset()
sampling_table = make_sampling_table(vocab_size)
x, y = [], []

for seq in corpus:
    couples, labels = skipgrams(
        seq, vocab_size, window_size=window_size, sampling_table=sampling_table)
    x.extend(couples)
    y.extend(labels)

x = np.array(x, dtype='int16')
word_target, word_context = x[:, 0], x[:, 1]
y = np.array(y, dtype='int8')

print(
    f'Created skip-gram pairs for training with shape: {x.shape} in {timer.lap()}')


################################
# Load and Train Model
################################
model = skipgram_model(vocab_size, embedding_size)
print(f'Loaded model in {timer.lap()}')

history = model.fit(x=[word_target, word_context], y=y,
                    epochs=epochs, batch_size=batch_size)
print(f'Trained model for {epochs} in {timer.lap()}')


################################
# Save Weights
################################
model_name = f'epochs-{epochs}:embedding_size-{embedding_size}:{file_ts()}.model'
model.save(model_name)
