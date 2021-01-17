import sys
import os
import numpy as np
from gensim.models import Word2Vec, KeyedVectors

proj_dir = '/home/jjohn273/git/Hcpcs2Vec/'
# proj_dir = '/Users/jujohnson/git/Hcpcs2Vec/'
sys.path.append(proj_dir)
from utils.callbacks import GensimEpochCallback  # NOQA: E402
from utils.utils import get_vocab_size, file_ts, Timer, args_to_dict  # NOQA: E402
from utils.data import load_hcpcs_corpus  # NOQA: E402


# parse cli args
filename = sys.argv[0].replace('.py', '')
cli_args = args_to_dict(sys.argv)
debug = cli_args.get('debug') == 'true'
ts = file_ts()

# model config
window_size = int(cli_args.get('window_size', 5))
min_seq_length = int(cli_args.get('min_seq_length', 2))
embedding_size = int(cli_args.get('embedding_size', 300))
iters = int(cli_args.get('iters', 5))
desc = f'e{embedding_size}-w{window_size}-i{iters}-t{ts}'

# I/O
data_dir = os.environ['CMS_RAW']
curr_dir = os.path.join(proj_dir, 'cbow')
embeddings_output = os.path.join(
    proj_dir, 'embeddings', f'cbow-{desc}.kv')
loss_output = os.path.join(curr_dir, 'logs', f'train-loss-{desc}.csv')
time_output = os.path.join(curr_dir, 'logs', f'train-time-{desc}.csv')


# load corpus
timer = Timer()
corpus = load_hcpcs_corpus(debug)
print(f'Loaded corpus with length {len(corpus)} in {timer.lap()}')


# use sample for debug
if debug:
    corpus = corpus[:500000]
    print(f'Using sample of corpus with length {len(corpus)}')


# vocab size
vocab_size = get_vocab_size(corpus)


# loss and timing callback
callback = GensimEpochCallback(loss_output, time_output)


# train model
timer.reset()
model = Word2Vec(
    sentences=corpus,
    window=window_size,
    size=embedding_size,
    min_count=min_seq_length,
    workers=10,
    sg=0,  # use cbow
    callbacks=[callback],
    compute_loss=True,
    iter=iters
)

print(f'Training completed in {timer.lap()}')


# save embeddings
model.wv.save(embeddings_output)
