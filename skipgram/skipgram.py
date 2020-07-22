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
from models import skipgram_model  # NOQA: E402

# parse cli args
data_dir = os.environ['CMS_RAW']
filename = sys.argv[0].replace('.py', '')
cli_args = args_to_dict(sys.argv)


# IO settings
save_skipgrams = cli_args.get('save_skipgrams') == 'true'
x_skipgrams_input = cli_args.get('x_skipgrams_input')
y_skipgrams_input = cli_args.get('y_skipgrams_input')


# model settings
debug = cli_args.get('debug') == 'true'
window_size = int(cli_args.get('window_size', 5))
epochs = int(cli_args.get('epochs', 100))
batch_size = int(cli_args.get('batch_size', 128))
embedding_size = int(cli_args.get('embedding_size', 300))


# input files
preloaded_skipgrams = x_skipgrams_input != None
if preloaded_skipgrams:
    x_skipgrams_input = os.path.join(proj_dir, 'data', x_skipgrams_input)
    y_skipgrams_input = os.path.join(proj_dir, 'data', y_skipgrams_input)


# output files
ts = file_ts()
partb_output = os.path.join(proj_dir, 'data', 'partb-2012-2017.csv.gz')
hcpcs_map_output = os.path.join(
    proj_dir, 'data', f'hcpcs-labelencoding.{ts}.pickle')
skipgram_x_output = os.path.join(proj_dir, 'data', f'skipgrams-x-{ts}.npy')
skipgram_y_output = os.path.join(proj_dir, 'data', f'skipgrams-y-{ts}.npy')


# load Medicare data and create skipgrams
if not preloaded_skipgrams:
    x, y = get_medicare_skipgrams(
        data_dir,
        partb_output=partb_output,
        hcpcs_id_output=hcpcs_map_output,
        window_size=window_size,
        debug=debug)


# save skipgrams
if save_skipgrams:
    np.save(skipgram_x_output, x)
    np.save(skipgram_y_output, y)


# or load existing skipgrams
if preloaded_skipgrams:
    print(f'Loading skipgrams from disk')
    x = np.load(x_skipgrams_input, allow_pickle=True)
    y = np.load(y_skipgrams_input, allow_pickle=True)


# Get vocab size
vocab_size = x.flatten().max() + 1
print(f'Using vocab size {vocab_size}')


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
print(f'Model saved to {model_name}')
