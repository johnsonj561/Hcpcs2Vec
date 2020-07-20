import pandas as pd
import numpy as np
import pickle
import os
import sys
import math
proj_dir = '/Users/jujohnson/git/Hcpcs2Vec/'
sys.path.append(proj_dir)
from medicare import get_medicare_skipgrams  # NOQA: E402
from utils import file_ts, Timer, args_to_dict  # NOQA: E402


# model settings
data_dir = os.environ['CMS_RAW']
filename = sys.argv[0].replace('.py', '')
cli_args = args_to_dict(sys.argv)
debug = cli_args.get('debug') == 'true'

window_size = int(cli_args.get('window_size', 5))
save_skipgrams = cli_args.get('save_skipgrams') == 'true'
x_skipgrams = cli_args.get('x_skipgrams')
y_skipgrams = cli_args.get('y_skipgrams')
preloaded_skipgrams = x_skipgrams != None

epochs = int(cli_args.get('epochs', 100))
batch_size = int(cli_args.get('batch_size', 128))
embedding_size = int(cli_args.get('embedding_size', 300))


# output files
ts = file_ts()
partb_filename = os.path.join(proj_dir, 'data', 'partb-2012-2017.csv.gz')
hcpcs_map_filename = os.path.join(
    proj_dir, 'data', f'hcpcs-labelencoding.{ts}.pickle')
skipgram_x_filename = os.path.join(proj_dir, 'data', f'skipgrams-x-{ts}.npy')
skipgram_y_filename = os.path.join(proj_dir, 'data', f'skipgrams-y-{ts}.npy')


# load Medicare data and create skipgrams
if not preloaded_skipgrams:
    x, y = get_medicare_skipgrams(
        data_dir,
        partb_output=partb_filename,
        hcpcs_id_output=hcpcs_map_filename,
        debug=debug)


# or load existing skipgrams
if preloaded_skipgrams:
    print(f'Loading skipgrams from disk')
    x = np.load(skipgram_x_filename, allow_pickle=True)
    y = np.load(skipgram_y_filename, allow_pickle=True)


# Get vocab size
vocab_size = x.flatten().max() + 1


# Load and train model
word_target, word_context = x[:, 0], x[:, 1]
timer = Timer()
model = skipgram_model(vocab_size, embedding_size)
print(f'Loaded model in {timer.lap()}')

history = model.fit(x=[word_target, word_context], y=y,
                    epochs=epochs, batch_size=batch_size)
print(f'Trained model for {epochs} in {timer.lap()}')


# Save model and weights
model_name = f'epochs-{epochs}:embedding_size-{embedding_size}:{file_ts()}.model'
model.save(model_name)
