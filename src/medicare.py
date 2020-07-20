import pandas as pd


def load_data_sample(data_dir, nrows):
    '''Returns a sample of Medicare Part B data from 2012.
      Expects 2012 data to exist at <data_dir>/2012/<filename>.csv.gz to exist.

      Keyword arguments:

      data_dir -- root directory of Medicare data

      nrows -- number of rows to sample
    '''
    data_file = os.path.join(
        data_dir,
        '2012',
        'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_CY2012.csv.gz')
    print(f'Loading sample of Medicare Part B 2012 data from {data_file}')
    columns = {
        'National Provider Identifier': 'npi',
        'HCPCS Code': 'hcpcs',
        'Number of Services': 'count',
    }
    df = pd.read_csv(data_file, usecols=list(columns.keys()))
    df = df.sample(nrows)
    df.rename(columns=columns, inplace=True)
    print(f'Loaded data with shape: {data.shape}')


def load_data(data_dir, output_path=None, refresh=False):
    '''Return raw Medicare Part B Data 2012 - 2015.
      Handles normalization of column names.
      Assumes data is stored in form "<data_dir>/<year>/filename" in csv format.

      Keyword arguments:

      data_dir -- root directory of Medicare data

      output_path -- path to save aggregated results (gzip)

      refresh -- override return of existing data at <output_path>
    '''
    exists = os.path.isfile(output_path)
    if exists and not refresh:
        return pd.read_csv(output_path)

    # load 2012 Part B
    columns = {
        'National Provider Identifier': 'npi',
        'HCPCS Code': 'hcpcs',
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
    path = os.path.join(
        data_dir,
        '2016',
        'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_PUF_CY2016.csv.gz')
    df_2016 = pd.read_csv(path, usecols=columns.keys()) \
        .rename(columns=columns)
    df_2016['year'] = 2016
    print(f'Loaded 2016 with shape {df_2016.shape}')

    # 2017 Part B
    # Reuses columns from 2015
    path = os.path.join(
        data_dir,
        '2017',
        'Medicare_Provider_Utilization_and_Payment_Data__Physician_and_Other_Supplier_PUF_CY2017.csv.gz')
    df_2017 = pd.read_csv(path, usecols=columns.keys()) \
        .rename(columns=columns)
    df_2017['year'] = 2017
    print(f'Loaded 2017 with shape {df_2017.shape}')

    # Concatenate All Years
    df = pd.concat([df_2012, df_2013, df_2014, df_2015, df_2016, df_2017])
    print(f'All years concatenated, final shape is {df.shape}')

    # Save Results
    if output_path != None:
        df.to_csv(output_path, compression='gzip')
        print(f'Results saved to {output_path}')

    return df
