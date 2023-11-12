import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
from tqdm import tqdm
import os
import numpy as np

from CO_evaluation.imports import BENCHMARK_CO_DIR, METRICS
from CO_evaluation.utils.CO_benchmark import GRN_Evaluator, load_GRN_results_for_all_samples, method_rename_dict, sample_rename_dict, get_GT
from CO_evaluation.utils import print_output


def get_external_scores(benchmarking_method:str) -> dict[str, pd.DataFrame]:
    """Get the scores of the methods reported by CO"""
    # if given, recalculate the external scores. This is if the metric is different than CO
    to_save = f"{SCORES_DIR}/grn_score_summary_22.parquet"

    scores_all = load_GRN_results_for_all_samples(benchmarking_method,
                                                  f'{data_dir}/celloracle_grn_benchmark/data/inference_results022/',
                                                  tissue=None, BASE_FOLDER=data_dir,
                                                  VERBOSE_FOLDER=f'{SCORES_DIR}/verbose')

    scores_all.to_parquet(to_save)
    print_output(to_save, verbose)

    # Reads the summary of the scores obtained for all the external reported by CO
    scores_all = pd.read_parquet(f"{SCORES_DIR}/grn_score_summary_22.parquet")
    scores_all.method.unique()
    # Rename methods
    scores_all["method_renamed"] = [method_rename_dict[i] for i in scores_all["method"]]
    scores_all["sample_renamed"] = [sample_rename_dict[i] for i in scores_all["sample"]]
    scores_selected_metrics = {}

    for metric in METRICS:
        scores = scores_all[[metric, "method_renamed", "sample_renamed"]]
        scores = scores.pivot(index="method_renamed", columns="sample_renamed", values=metric)
        scores = scores.transpose()
        scores = scores[METHODS]
        scores_selected_metrics[metric] = scores

    return scores_selected_metrics
def get_shuffled_scores(benchmarking_method) -> pd.DataFrame:
    '''
    Obtains the links inferred by CO using base GRN, shuffles the values, and calculate the benchmarking score
    '''
    file_shuffled_scores = f'{SCORES_DIR}/shuffled_scores.csv'
    if os.path.exists(file_shuffled_scores):
        scores = pd.read_csv(file_shuffled_scores, index_col=0)
        return scores
    LINKS_DIR = f'{data_dir}/celloracle_grn_benchmark/data/inference_results022/'
    samples = [d for d in sorted(os.listdir(LINKS_DIR))]

    grn_name = 'celloracle_cluster_mouseAtacBaseGRN'

    scores = {}
    for sample in tqdm(samples, desc=f'Different samples shuffled method'):
        tissue = sample.split("-")[0]

        GT = get_GT(tissue)
        print(GT)
        # get the links
        links = pd.read_csv(f'{LINKS_DIR}/{sample}/{grn_name}/link.csv', index_col=0)
        nonzero = pd.read_csv(f'{LINKS_DIR}/{sample}/{grn_name}/genes_nonzero.csv', index_col=0).x.values

        # randomize the links
        links['coef_abs'] = np.random.rand(len(links))
        links['coef_mean'] = np.random.rand(len(links))
        links['-logp'] = np.random.rand(len(links))
        # calculate the score   
        ALL_TFS = np.load(f"{data_dir}/celloracle_grn_benchmark/data/ground_truth_data/chip_atlas/TFs_in_gimmev5_mouse.npy", allow_pickle=True)

        scores[sample] = GRN_Evaluator(all_TFs=ALL_TFS,
                       df_ground_truth=GT,
                       links=links,
                       genes_used=nonzero,
                       benchmarking_method=benchmarking_method
                       ).score
    scores = pd.DataFrame(scores).rename(columns=sample_rename_dict).transpose()
    scores.to_csv(file_shuffled_scores)
    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calculate-random', action='store_true', help='To recalculate the scores for the random links')
    parser.add_argument('--benchmark-method', type=str, default='CO', help='The method to calculate GRN benchmarking scores against Chip data')
    parser.add_argument('--verbose', action='store_true', help='To print the details the run details')
    parser.add_argument('--data_dir', type=str, default='../external/', help='Path to external data')


    args = parser.parse_args()
    calculate_random = args.calculate_random
    verbose = args.verbose
    benchmarking_method = args.benchmark_method
    data_dir = args.data_dir
    print(data_dir)

    METHODS = ["CellOracle\nscATAC-atlas\nBase-GRN",
               "CellOracle\npromoter\nBase-GRN",
               "CellOracle\n scrambled Base-GRN",
               "CellOracle\n no Base-GRN",
               "SCENIC",
               "GENIE3",
               "WGCNA",
               "DCOL"]

    SCORES_DIR = f'{BENCHMARK_CO_DIR}/scores_{benchmarking_method}'
    os.makedirs(SCORES_DIR, exist_ok=True)

    scores_external = get_external_scores(benchmarking_method)

    if calculate_random:
        # calculate random scores
        shuffled_scores = get_shuffled_scores(benchmarking_method)
        # merge the local and external, and save them
        df_all = {}
        for metric in METRICS:
            scores = shuffled_scores[metric].to_frame('shuffled')
            df_all[metric] = pd.merge(scores_external[metric], scores, left_index=True, right_index=True, how='inner')
    else:
        df_all = scores_external

    for metric, df in tqdm(df_all.items(), desc='Run for metrics'):
        to_save = f'{SCORES_DIR}/scores_all_{metric}.csv'
        df.to_csv(to_save)
        print_output(to_save, verbose)


