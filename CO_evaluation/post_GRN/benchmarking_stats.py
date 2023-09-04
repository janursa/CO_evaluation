import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
from tqdm import tqdm
import os
import numpy as np

from CO_evaluation.imports import EXTERNAL_CO_BENCHMARK_DIR, BENCHMARK_CO_DIR, METRICS
from CO_evaluation.utils.CO_benchmark import GRN_Evaluator, load_GRN_results_for_all_samples, method_rename_dict, sample_rename_dict, get_GT, ALL_TFS
from CO_evaluation.utils import print_output, print_df_cols


def run_stats(benchmarking_method) -> pd.DataFrame:
    '''
    For each sample, we calculate class imbalance stats. #links, #all_combinations, #GT, #positive_class
    '''
    if os.path.exists(STATS_FILE) and not force:
        stats = pd.read_csv(STATS_FILE, index_col=0)
        return stats
    LINKS_DIR = f'{EXTERNAL_CO_BENCHMARK_DIR}/data/inference_results022/'
    samples = [d for d in sorted(os.listdir(LINKS_DIR))]

    grn_name = 'celloracle_cluster_mouseAtacBaseGRN'

    stats_all = {}
    for sample in tqdm(samples, desc=f'Different samples'):
        tissue = sample.split("-")[0]

        GT = get_GT(tissue)
        # get the links
        links = pd.read_csv(f'{LINKS_DIR}/{sample}/{grn_name}/link.csv', index_col=0)
        nonzero = pd.read_csv(f'{LINKS_DIR}/{sample}/{grn_name}/genes_nonzero.csv', index_col=0).x.values

        # calculate the score
        stats_all[sample] = GRN_Evaluator(all_TFs=ALL_TFS,
                       df_ground_truth=GT,
                       links=links,
                       genes_used=nonzero,
                       benchmarking_method=benchmarking_method,
                       calculate_stats=True
                       ).benchmarking_stats
    stats_all = pd.DataFrame(stats_all).rename(columns=sample_rename_dict).transpose()
    stats_all.to_csv(STATS_FILE)
    print_output(STATS_FILE, verbose)
    return stats_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help='To print the details the run details')
    parser.add_argument('--force', action='store_true', help='To overwrite the results')

    parser.add_argument('--benchmark-method', type=str, default='CO', help='The method to calculate GRN benchmarking scores against Chip data')

    args = parser.parse_args()
    verbose = args.verbose
    force = args.force
    benchmarking_method = args.benchmark_method

    STATS_DIR = f'{BENCHMARK_CO_DIR}/stats'
    os.makedirs(STATS_DIR, exist_ok=True)
    STATS_FILE = f'{STATS_DIR}/stats_{benchmarking_method}.csv'

    run_stats(benchmarking_method)




