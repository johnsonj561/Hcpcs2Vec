import pandas as pd
import os


def load_data_sample(data_dir, nrows):
    data_file = os.path.join(
        data_dir,
        '2012',
        'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_CY2012.csv')
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
        'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_CY2012.csv')
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
        'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_CY2013.csv')
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
        'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_PUF_CY2014.csv')
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
        'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_PUF_CY2015.csv')
    df_2015 = pd.read_csv(path, usecols=columns.keys()) \
        .rename(columns=columns)
    df_2015['year'] = 2015
    print(f'Loaded 2015 with shape {df_2015.shape}')

    # Concatenate All Years
    df = pd.concat([df_2012, df_2013, df_2014, df_2015])
    print(f'All years concatenated, final shape is {df.shape}')
    print(f'Data info: \n{df.info()}')

    # Save Results
    if output_path != None:
        df.to_csv(output_path, compression='gzip')
        print(f'Results saved to {output_path}')

    return df
