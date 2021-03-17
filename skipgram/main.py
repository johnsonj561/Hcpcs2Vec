import argparse
import logging
import os
import sys
import time

from gensim.models import Word2Vec

# proj_dir = "/home/jjohn273/git/Drug2Vec/"
proj_dir = "/Users/jujohnson/git/Drug2Vec/"
sys.path.append(proj_dir)
from utils.callbacks import GensimEpochCallback  # NOQA: E402
from utils.data import load_drug_name_corpus  # NOQA: E402
from utils.utils import file_ts, get_vocab_size  # NOQA: E402

logging.basicConfig(level=logging.INFO)

# Parse cli argurments
parser = argparse.ArgumentParser(
    description="Train an ALS Collab Filtering model on transactional data"
)
parser.add_argument("--iters", type=int, help="Number of training iterations.")
parser.add_argument(
    "--embedding-size", type=int, help="Number of dimensions for embeddings."
)
parser.add_argument("--window-size", type=int, help="Window size for skipgram model")
parser.add_argument(
    "--debug",
    type=bool,
    help="Enabling debug mode runs job with reduced data set",
    default=False,
)
args = parser.parse_args()

# I/O
ts = file_ts()
min_seq_length = 2
desc = f"e{args.embedding_size}-w{args.window_size}-i{args.iters}-t{ts}"
embeddings_output = os.path.join(proj_dir, "embeddings", f"skipgram-{desc}.kv")
loss_output = os.path.join("logs", f"train-loss-{desc}.csv")
time_output = os.path.join("logs", f"train-time-{desc}.csv")


# Load Part D data
t0 = time.time()
corpus = load_drug_name_corpus(args.debug)
logging.info(f"Loaded corpus with length {len(corpus)} in {time.time() - t0}")

# Get vocab size
vocab_size = get_vocab_size(corpus)

# Create loss and timing callback
callback = GensimEpochCallback(loss_output, time_output)

# train model
logging.info(f"Training model for {args.iters} iterations")
t0 = time.time()
model = Word2Vec(
    sentences=corpus,
    window=args.window_size,
    size=args.embedding_size,
    min_count=min_seq_length,
    workers=10,
    sg=1,  # use skipgram
    hs=0,  # use negative sampling,
    callbacks=[callback],
    compute_loss=True,
    iter=args.iters,
)

logging.info(f"Training completed in {time.time() - t0}")


# save embeddings
model.wv.save(embeddings_output)
logging.info(f"Embeddings saved to {embeddings_output}")
