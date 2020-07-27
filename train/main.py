import pandas as pd
import numpy as np
import pickle
import os
import sys
import math
proj_dir = '/home/jjohn273/git/Hcpcs2Vec/'
# proj_dir = '/Users/jujohnson/git/Hcpcs2Vec/'
sys.path.append(proj_dir)
from medicare import get_medicare_skipgrams  # NOQA: E402
from models import skipgram_model, skipgram_callbacks    # NOQA: E402
from utils import file_ts, Timer, args_to_dict, model_summary_to_string  # NOQA: E402


# parse cli args
filename = sys.argv[0].replace('.py', '')
cli_args = args_to_dict(sys.argv)


# inputs
x_skipgrams_input = cli_args.get('x_skipgrams_input')
x_skipgrams_input = os.path.join(proj_dir, 'data', x_skipgrams_input)
y_skipgrams_input = cli_args.get('y_skipgrams_input')
y_skipgrams_input = os.path.join(proj_dir, 'data', y_skipgrams_input)


# model settings
epochs = int(cli_args.get('epochs', 1000))  # early stopping will end earlier
batch_size = int(cli_args.get('batch_size', 256))
embedding_size = int(cli_args.get('embedding_size', 300))
model_config = f'bs{batch_size}-es{embedding_size}'
print(f'Using model config: {model_config}')


# outputs
ts = file_ts()
model_output = os.path.join(proj_dir, 'models', f'model-{model_config}-{ts}')
callback_output = f'train-{ts}.csv'


# load skipgrams
x = np.load(x_skipgrams_input, allow_pickle=True)
y = np.load(y_skipgrams_input, allow_pickle=True)
print(f'Loaded skipgrams {x_skipgrams_input} and {y_skipgrams_input}')
print(f'x shape: {x.shape}')
print(f'y shape: {y.shape}')


# compute vocab size
vocab_size = x.flatten().max() + 1
print(f'Using vocab size: {vocab_size}')


# load model
word_target, word_context = x[:, 0], x[:, 1]
timer = Timer()
model = skipgram_model(vocab_size, embedding_size)
print(f'Loaded model in {timer.lap()}')
print(model_summary_to_string(model))


# train model
callbacks = skipgram_callbacks(callback_output)
model.fit(x=[word_target, word_context], y=y, verbose=0,
          epochs=epochs, batch_size=batch_size, callbacks=callbacks)
print(f'Training completed in {timer.lap()}')


# Save model and weights
model.save(model_output)
print(f'Model saved to {model_output}')


print('Job complete :)')
