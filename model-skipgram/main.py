import sys
import os
import numpy as np
from gensim.models import Word2Vec, KeyedVectors

# proj_dir = '/home/jjohn273/git/Hcpcs2Vec/'
proj_dir = '/Users/jujohnson/git/Hcpcs2Vec/'
sys.path.append(proj_dir)
from utils.callbacks import GensimEpochCallback  # NOQA: E402
from utils.utils import file_ts, Timer, args_to_dict  # NOQA: E402
from utils.data import load_hcpcs_corpus  # NOQA: E402


# parse cli args
filename = sys.argv[0].replace('.py', '')
cli_args = args_to_dict(sys.argv)
debug = cli_args.get('debug') == 'true'


# model config
window_size = int(cli_args.get('window_size', 5))
min_seq_length = int(cli_args.get('min_seq_length', 2))


# I/O
data_dir = os.environ['CMS_RAW']
curr_dir = os.path.join(proj_dir, 'model-skipgram')
ts = file_ts()
model_output = os.path.join(curr_dir, 'saved-models', f'sg-{ts}.model')
embeddings_output = os.path.join(
    curr_dir, 'saved-models', f'hcpcs-embeddings-{ts}.kv')
loss_output = os.path.join(curr_dir, 'logs', f'train-loss-{ts}.csv')
time_output = os.path.join(curr_dir, 'logs', f'train-time-{ts}.csv')


# load corpus
timer = Timer()
corpus = load_hcpcs_corpus(debug=True)
print(f'Loaded corpus with length {len(corpus)} in {timer.lap()}')


# use sample for debug
if debug:
    corpus = corpus[:500000]
    print(f'Using sample of corpus with length {len(corpus)}')


# vocab size
vocab_size = len(np.unique(corpus.flatten()))


# loss and timing callback
callback = GensimEpochCallback(loss_output, time_output)


# train model
timer.reset()
model = Word2Vec(
    sentences=corpus,
    window=window_size,
    size=300,
    min_count=min_seq_length,
    workers=10,
    sg=1,  # use skipgram
    hs=0,  # use negative sampling,
    callbacks=[callback],
    compute_loss=True,
)

print(f'Training completed in {timer.lap()}')


# save embeddings
model.wv.save(embeddings_output)
model.save(model_output)
