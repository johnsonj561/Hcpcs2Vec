import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import LabelEncoder
import pickle
import tensorflow as tf
Keras = tf.keras
Model, Layers = Keras.models.Model, Keras.layers
make_sampling_table = Keras.preprocessing.sequence.make_sampling_table
skipgrams = Keras.preprocessing.sequence.skipgrams
proj_dir = '/Users/jujohnson/git/Hcpcs2Vec/'
sys.path.append(proj_dir)
from utils import Timer  # NOQA: E402


def load_data_sample(data_dir, nrows):
    data_file = os.path.join(
        data_dir,
        '2012',
        'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_CY2012.csv.gz')
    print(f'Loading sample of Medicare Part B 2012 data from {data_file}')
    columns = {
        'National Provider Identifier': 'npi',
        'HCPCS Code': 'hcpcs',
        'HCPCS Description': 'description',
        'Number of Services': 'count',
    }
    df = pd.read_csv(data_file, usecols=list(columns.keys()))
    df = df.sample(nrows)
    df.rename(columns=columns, inplace=True)
    print(f'Loaded data with shape: {df.shape}')
    return df


def load_data(data_dir, output_path=None, debug=False):
    if debug:
        return load_data_sample(data_dir, 2000000)

    if os.path.isfile(output_path):
        return pd.read_csv(output_path)

    # load 2012 Part B
    columns = {
        'National Provider Identifier': 'npi',
        'HCPCS Code': 'hcpcs',
        'HCPCS Description': 'description',
        'Number of Services': 'count'
    }
    path = os.path.join(
        data_dir,
        '2012',
        'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_CY2012.csv.gz')
    df_2012 = pd.read_csv(path, usecols=columns.keys()) \
        .rename(columns=columns)
    df_2012['year'] = 2012
    print(f'Loaded 2012 with shape {df_2012.shape}')

    # Load 2013 Part B
    columns = {
        'National Provider Identifier ': 'npi',
        'HCPCS_CODE': 'hcpcs',
        'HCPCS_DESCRIPTION': 'description',
        'LINE_SRVC_CNT': 'count'
    }
    path = os.path.join(
        data_dir,
        '2013',
        'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_CY2013.csv.gz')
    df_2013 = pd.read_csv(path, usecols=columns.keys()) \
        .rename(columns=columns)
    df_2013['year'] = 2013
    print(f'Loaded 2013 with shape {df_2013.shape}')

    # Load 2014 Part B
    columns = {
        'National Provider Identifier': 'npi',
        'HCPCS Code': 'hcpcs',
        'HCPCS Description': 'description',
        'Number of Services': 'count'
    }
    path = os.path.join(
        data_dir,
        '2014',
        'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_PUF_CY2014.csv.gz')
    df_2014 = pd.read_csv(path, usecols=columns.keys()) \
        .rename(columns=columns)
    df_2014['year'] = 2014
    print(f'Loaded 2014 with shape {df_2014.shape}')

    # 2015 Part B
    columns = {
        'National Provider Identifier': 'npi',
        'HCPCS Code': 'hcpcs',
        'HCPCS Description': 'description',
        'Number of Services': 'count'
    }
    path = os.path.join(
        data_dir,
        '2015',
        'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_PUF_CY2015.csv.gz')
    df_2015 = pd.read_csv(path, usecols=columns.keys()) \
        .rename(columns=columns)
    df_2015['year'] = 2015
    print(f'Loaded 2015 with shape {df_2015.shape}')

    # 2016 Part B
    # reuses columns from 2015
    # path = os.path.join(
    #     data_dir,
    #     '2016',
    #     'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_PUF_CY2016.csv.gz')
    # df_2016 = pd.read_csv(path, usecols=columns.keys()) \
    #     .rename(columns=columns)
    # df_2016['year'] = 2016
    # print(f'Loaded 2016 with shape {df_2016.shape}')

    # 2017 Part B
    # Reuses columns from 2015
    # path = os.path.join(
    #     data_dir,
    #     '2017',
    #     'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_PUF_CY2017.csv.gz')
    # df_2017 = pd.read_csv(path, usecols=columns.keys()) \
    #     .rename(columns=columns)
    # df_2017['year'] = 2017
    # print(f'Loaded 2017 with shape {df_2017.shape}')

    # Concatenate All Years
    df = pd.concat([df_2012, df_2013, df_2014, df_2015])
    print(f'All years concatenated, final shape is {df.shape}')
    print(f'Data info: \n{df.info()}')

    # Save Results
    if output_path != None:
        df.to_csv(output_path, compression='gzip')
        print(f'Results saved to {output_path}')

    return df


def get_hcpcs_ids(data, output_file):
    le = LabelEncoder()
    data['hcpcs_id'] = le.fit_transform(data['hcpcs'].astype(str))
    with open(output_file, 'wb') as fout:
        pickle.dump(le.classes_, fout)
    return data


def get_hcpcs_corpus(data):
    corpus = []
    for npi, group in data.groupby(by='npi'):
        group.sort_values(by='count', inplace=True)
        hcpcs_set = np.asarray(group['hcpcs_id'], dtype='int16')
        corpus.append(hcpcs_set)
    corpus_length = len(corpus)
    return corpus


def set_max_hcpcs_seq_length(corpus, quantile):
    lengths = np.array(list(map(lambda x: len(x), corpus)))
    max_seq_length = np.quantile(lengths, quantile)
    corpus = np.array(list(filter(lambda x: len(x) <= max_seq_length, corpus)))
    return corpus


def get_hcpcs_skipgrams(corpus, vocab_size, window_size):
    print(f'Using vocab_size {vocab_size}')
    sampling_table = make_sampling_table(vocab_size)
    x, y = [], []
    for seq in corpus:
        pairs, labels = skipgrams(
            seq, vocab_size, window_size=window_size, sampling_table=sampling_table)
        x.extend(pairs)
        y.extend(labels)
    return np.array(x, dtype='int16'), np.array(y, dtype='int8')


def get_medicare_skipgrams(data_dir, partb_output, hcpcs_id_output, window_size, debug):
    timer = Timer()
    # Load Medicare Data
    data = load_data(data_dir, partb_output, debug)
    print(f'Loaded data in {timer.lap()}')

    # Create HCPCS <--> ID mapping
    data = get_hcpcs_ids(data, hcpcs_id_output)
    print(f'Created HCPCS ID mapping in {timer.lap()}')

    # Extract HCPCS corpus from the Medicare data
    corpus = get_hcpcs_corpus(data)
    print(f'Created corpus with length: {len(corpus)} in {timer.lap()}')

    # Reduce the max sequence length
    quantile = 0.98
    corpus = set_max_hcpcs_seq_length(corpus, quantile)
    print(
        f'Removed the longest hcpcs sequences from {quantile}+ quantile in {timer.lap()}')
    print(f'Updated corpus length {len(corpus)}')

    # Get vocab size
    vocab_size = data['hcpcs_id'].max() + 1
    print(f'Using vocab_size: {vocab_size}')

    # Free up some memory
    del data

    # Create skip-gram pairs and negative Ssmples (thx Keras)
    timer.reset()
    x, y = get_hcpcs_skipgrams(corpus, vocab_size, window_size)
    print(f'Created skip-gram pairs with shape: {x.shape} in {timer.lap()}')

    return x, y
