import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import argparse

from CO_evaluation.imports import BENCHMARK_CO_DIR, METRICS
from CO_evaluation.utils import print_output

title_relabel={'auc':'AUROC', 'epr':'EPR', 'auc_pr':'AUPR', 'f1':'F1'}
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
        elif i == "auc_pr":
            sns.heatmap(scores, cmap="viridis", fmt=".4g", annot=True, vmin=1, cbar=False)
        elif i == 'f1':
            sns.heatmap(scores, cmap="viridis", fmt=".4g", annot=True, vmin=1, cbar=False)
        else:
            sns.heatmap(scores, cmap="viridis", fmt=".3g", annot=True, vmin=1, cbar=False)

        plt.title(title_relabel[i])
        plt.ylabel("scRNA-seq data")
        plt.xticks(rotation=45)
        to_save = f"{SCORES_DIR}/figures/{i}.png"
        plt.savefig(to_save, transparent=False, bbox_inches='tight')
        print_output(to_save, verbose)
        # print(scores.index)
        # aa
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
    metrics_left = []
    for metric in METRICS:
        if 'random' in metric:
            print(metric)
            continue
        df = pd.read_csv(f'{SCORES_DIR}/scores_all_{metric}.csv', index_col=0)
        if metric  in ['auc_pr', 'f1', 'auc']:
        # if metric  in ['auc_pr']:
            # get the random scores for aupr and f1 to normalize
            baseline = pd.read_csv(f'{BENCHMARK_CO_DIR}/scores_M1/scores_all_{metric}_random.csv', index_col=0)
            df = df/baseline
            # print(df['shuffled'])
        df = df.rename(columns={'shuffled':'CellOracle-shuffled'})
        scores_all[metric] = df
        metrics_left.append(metric)
    plot(scores_all, metrics_left)



# step 1
all_combinations = pd.DataFrame(permutations(genes_used, 2), columns==["regulatory_gene", "target_gene"])
# step 2 
tfs_no_data = [i for i in all_TFs if i not in tissue_TFs]
all_combinations = all_combinations[~all_combinations.regulatory_gene.isin(tfs_no_data)]
