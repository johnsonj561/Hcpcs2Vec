from utils.utils import Timer
import pandas as pd
import os
import numpy as np
import sys

proj_dir = '/home/jjohn273/git/Hcpcs2Vec/'
# proj_dir = '/Users/jujohnson/git/Hcpcs2Vec/'
sys.path.append(proj_dir)


# path to raw cms data
data_dir = os.environ['CMS_RAW']


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
    df['year'] = 2012
    print(f'Loaded data with shape: {df.shape}')
    return df


def load_data(data_dir, output_path=None, debug=False):
    if debug:
        return load_data_sample(data_dir, 1000000)

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
    path = os.path.join(
        data_dir,
        '2016',
        'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_PUF_CY2016.csv.gz')
    df_2016 = pd.read_csv(path, usecols=columns.keys()) \
        .rename(columns=columns)
    df_2016['year'] = 2016
    print(f'Loaded 2016 with shape {df_2016.shape}')

    # 2017 Part B
    path = os.path.join(
        data_dir,
        '2017',
        'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_PUF_CY2017.csv.gz')
    df_2017 = pd.read_csv(path, usecols=columns.keys()) \
        .rename(columns=columns)
    df_2017['year'] = 2017
    print(f'Loaded 2017 with shape {df_2017.shape}')

    # 2018 Part B
    path = os.path.join(
        data_dir,
        '2018',
        'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_PUF_CY2018.csv.gz')
    df_2018 = pd.read_csv(path, usecols=columns.keys()) \
        .rename(columns=columns)
    df_2018['year'] = 2018
    print(f'Loaded 2018 with shape {df_2018.shape}')

    # Concatenate All Years
    df = pd.concat([df_2012, df_2013, df_2014, df_2016, df_2017, df_2018])
    print(f'All years concatenated, final shape is {df.shape}')
    print(f'Data info: \n{df.info()}')

    # Save Results
    if output_path != None:
        df.to_csv(output_path, compression='gzip')
        print(f'Results saved to {output_path}')

    return df
      


def load_hcpcs_corpus(debug=False):
    corpus_file = 'debug-corpus.npy' if debug else 'corpus.npy'
    corpus_output = os.path.join(proj_dir, 'data', corpus_file)
    partb_file = 'partb-2012.csv.gz' if debug else 'partb-2012-2018.csv.gz'
    partb_output = os.path.join(proj_dir, 'data', partb_file)

    # load from disk if exists
    if os.path.isfile(corpus_output):
        print(f'Loading corpus from disk {corpus_output}')
        corpus = np.load(corpus_output, allow_pickle=True)
        return corpus

    # load Medicare Data
    timer = Timer()
    data = load_data(data_dir, partb_output, debug)
    print(f'Loaded data in {timer.lap()}')

    # clean missing values
    data.dropna(subset=['hcpcs','count'], inplace=True)

    # generate sequences of HCPCS codes
    # that occur in the same context
    grouped_hcpcs = data \
        .sort_values(by='count') \
        .groupby(by=['year', 'npi'])['hcpcs'] \
        .agg(list)
    grouped_hcpcs = pd.DataFrame(grouped_hcpcs)
    print(f'Generated hcpcs sequences in {timer.lap()}')

    # drop top 1 percent longest sequences
    quantile = 0.99
    grouped_hcpcs['seq_length'] = grouped_hcpcs['hcpcs'].agg(len)
    max_seq_length = grouped_hcpcs['seq_length'].quantile(quantile)
    grouped_hcpcs = grouped_hcpcs.loc[grouped_hcpcs['seq_length']
                                      <= max_seq_length]
    print(f'Removed sequences longer than {max_seq_length}')

    # save corpus
    np.save(corpus_output, grouped_hcpcs['hcpcs'].values)

    return grouped_hcpcs['hcpcs'].values

  
def load_dmepos_data(data_dir, output_path=None, sample_size=False):
    if os.path.isfile(output_path):
        return pd.read_csv(output_path)
    data_file = os.path.join(data_dir, 'medicare-dmepos-2013-2018.csv.gz')
    columns = ['npi', 'hcpcs_code', 'number_of_supplier_claims']
    df = pd.read_csv(dmepos_file, usecols=columns, nrows=sample_size)
    return df

  
def load_dmepos_hcpcs_corpus(sample_size=None):
    corpus_file = os.path.join(proj_dir, 'data', 'corpus.npy')
    dmepos_file = '/Users/jujohnson/cms-data/raw/medicare-dmepos-2013-2018.csv.gz'
    dmepos_cols = ['npi', 'year', 'hcpcs_code', 'number_of_supplier_claims']
    
    # load corpus from disk if exists
    if os.path.isfile(corpus_file):
        print(f'Loading corpus from disk {corpus_file}')
        corpus = np.load(corpus_file, allow_pickle=True)
        return corpus

    # load Medicare Data
    timer = Timer()
    data = pd.read_csv(dmepos_file, usecols=dmepos_cols, nrows=sample_size)
    print(f'Loaded data in {timer.lap()}')

    # clean missing values
    data.dropna(subset=['hcpcs_code','number_of_supplier_claims'], inplace=True)

    # generate sequences of HCPCS codes
    # that occur in the same context
    grouped_hcpcs = data \
        .sort_values(by='count') \
        .groupby(by=['year', 'npi'])['hcpcs_code'] \
        .agg(list)
    grouped_hcpcs = pd.DataFrame(grouped_hcpcs)
    print(f'Generated hcpcs sequences in {timer.lap()}')

    # drop top 1 percent longest sequences
    quantile = 0.99
    grouped_hcpcs['seq_length'] = grouped_hcpcs['hcpcs_code'].agg(len)
    max_seq_length = grouped_hcpcs['seq_length'].quantile(quantile)
    grouped_hcpcs = grouped_hcpcs.loc[grouped_hcpcs['seq_length']
                                      <= max_seq_length]
    print(f'Removed sequences longer than {max_seq_length}')

    # save corpus
    np.save(corpus_output, grouped_hcpcs['hcpcs_code'].values)

    return grouped_hcpcs['hcpcs_code'].values
  