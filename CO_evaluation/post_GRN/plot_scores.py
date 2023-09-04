import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import argparse

from CO_evaluation.imports import BENCHMARK_CO_DIR, METRICS
from CO_evaluation.utils import print_output

title_relabel={'auc':'AUCROC', 'epr':'EPR'}
def plot(scores_selected_metrics, metrices):
    os.makedirs(f"{SCORES_DIR}/figures", exist_ok=True)
    for i in metrices:
        scores = scores_selected_metrics[i]
        plt.figure(figsize=[8, 5])

        if i == "epr_tfsubset":
            sns.heatmap(scores, cmap="viridis", fmt=".3g", annot=True, vmin=0.5, vmax=2, cbar=False)
        elif i == "epr":
            sns.heatmap(scores, cmap="viridis", fmt=".0f",
                        annot=True, vmin=0.5, vmax=1000, cbar=False)
        elif i == "aupr":
            sns.heatmap(scores, cmap="viridis", fmt=".4g", annot=True, vmin=0., vmax=0.6, cbar=False)
        else:
            sns.heatmap(scores, cmap="viridis", fmt=".3g", annot=True, vmin=0.5, cbar=False)

        # titles = {"epr": "EPR", "auc": "AUROC"}
        plt.title(title_relabel[i])
        plt.ylabel("scRNA-seq data")
        # plt.xlabel("GRN inference method")
        plt.xticks(rotation=45)
        # plt.subplots_adjust(left=0.15, bottom=0.35)
        to_save = f"{SCORES_DIR}/figures/{i}.png"
        plt.savefig(to_save, transparent=False, bbox_inches='tight')
        print_output(to_save, verbose)
        # plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help='To print the details the run details')
    parser.add_argument('--benchmark-method', type=str, default='CO', help='The method to calculate GRN benchmarking scores against Chip data')

    args = parser.parse_args()
    verbose = args.verbose
    benchmark_method = args.benchmark_method

    # retreive the scores
    SCORES_DIR = f'{BENCHMARK_CO_DIR}/scores_{benchmark_method}'
    scores_all: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        df = pd.read_csv(f'{SCORES_DIR}/scores_all_{metric}.csv', index_col=0)
        scores_all[metric] = df
    # plot
    plot(scores_all, METRICS)