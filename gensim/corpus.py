import pandas as pd
import numpy as np
import os
import sys
import pickle
proj_dir = '/Users/jujohnson/git/Hcpcs2Vec/'
# proj_dir = '/home/jjohn273/git/Hcpcs2Vec/'
sys.path.append(proj_dir)
from utils import args_to_dict, file_ts, Timer  # NOQA: E402
from medicare import load_data  # NOQA: E402


# parse cli args
filename = sys.argv[0].replace('.py', '')
cli_args = args_to_dict(sys.argv)
debug = cli_args.get('debug') == 'true'
print(f'Using debug: {debug}')


# config
data_dir = os.environ['CMS_RAW']
ts = file_ts()


# output files
partb_output = os.path.join(proj_dir, 'data', 'partb-2012-2015.csv.gz')
corpus_file = f'corpus-{ts}.pickle'
corpus_output = os.path.join(proj_dir, 'gensim', corpus_file)


# load Medicare Data
timer = Timer()
data = load_data(data_dir, partb_output, debug)
print(f'Loaded data in {timer.lap()}')


# generate sequences of HCPCS codes
# that occur in the same context
grouped_hcpcs = data \
  .sort_values(by='count') \
  .groupby(by=['year', 'npi'])['hcpcs'] \
  .agg(list)
grouped_hcpcs = pd.DataFrame(grouped_hcpcs)
print(f'Generated hcpcs sequences in {timer.lap()}')


# drop short sequences
min_seq_length = 3
grouped_hcpcs['seq_length'] = grouped_hcpcs['hcpcs'].apply(len)
grouped_hcpcs = grouped_hcpcs.loc[grouped_hcpcs['seq_length'] > min_seq_length]
print(f'Removed sequences shorter than {min_seq_length}')


# drop top 1 percent longest sequences
quantile = 0.99
max_seq_length = grouped_hcpcs['seq_length'].quantile(quantile)
grouped_hcpcs = grouped_hcpcs.loc[grouped_hcpcs['seq_length'] <= max_seq_length]
print(f'Removed sequences longer than {max_seq_length}')


# save corpus
np.save(corpus_output, grouped_hcpcs['hcpcs'].values)
print(f'Job complete')
