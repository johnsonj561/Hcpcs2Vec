import os
import sys

import numpy as np
import pandas as pd
import glob
import time
import logging


cms_dir = "/Users/jujohnson/cms-data/raw/"
proj_dir = "/Users/jujohnson/git/Drug2Vec/"


def load_partd_data(debug=False):
    nrows = 1000 if debug else None
    logging.info(f"Loading Part D data with debug = {debug}")

    partd_files = sorted(glob.glob(f"{cms_dir}/**/PartD_*.csv.gz"))
    years = ["2013", "2014", "2015", "2016", "2017", "2018"]
    columns = ["npi", "drug_name", "total_claim_count"]
    dfs = []

    for file, year in zip(partd_files, years):
        df = pd.read_csv(file, nrows=1000, usecols=columns)
        df["year"] = year
        dfs.append(df)

    return pd.concat(dfs).dropna()


def load_drug_name_corpus(debug=False):
    corpus_file = "debug-corpus.npy" if debug else "corpus.npy"
    corpus_output = os.path.join(proj_dir, "data", corpus_file)

    # load from disk if exists
    if os.path.isfile(corpus_output):
        print(f"Loading corpus from disk {corpus_output}")
        corpus = np.load(corpus_output, allow_pickle=True)
        return corpus

    # load Medicare Data
    t0 = time.time()
    data = load_partd_data(debug)
    print(f"Loaded data in {time.time() - t0}")

    # generate sequences of Drug names
    # that occur in the same context
    t0 = time.time()
    grouped_drug_names = (
        data.sort_values(by="total_claim_count")
        .groupby(by=["year", "npi"])["drug_name"]
        .agg(list)
    )
    grouped_drug_names = pd.DataFrame(grouped_drug_names)
    print(f"Generated drug name sequences in {time.time() - t0}")

    # drop top 1 percent longest sequences
    quantile = 0.99
    grouped_drug_names["seq_length"] = grouped_drug_names["drug_name"].agg(len)
    max_seq_length = grouped_drug_names["seq_length"].quantile(quantile)
    grouped_drug_names = grouped_drug_names.loc[
        grouped_drug_names["seq_length"] <= max_seq_length
    ]
    print(f"Removed sequences longer than {max_seq_length}")

    # save corpus
    np.save(corpus_output, grouped_drug_names["drug_name"].values)

    return grouped_drug_names["drug_name"].values
