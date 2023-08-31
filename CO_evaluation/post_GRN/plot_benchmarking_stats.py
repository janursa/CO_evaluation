import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import argparse

from CO_evaluation.imports import BENCHMARK_CO_DIR
from CO_evaluation.utils import print_output

def plot_only_ratioGT(df):
    plt.figure(figsize=(12, 6))
    plt.bar(df.index, df['ratio_GT'], color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel('Ratio of positives (%)')
    plt.title(benchmarking_method)
    plt.tight_layout()

    plt.savefig(PLOT_FILE, transparent=False)
    print_output(PLOT_FILE, verbose)
def plot_both(df):
    # Set up the figure and axes
    plt.figure(figsize=(6, 4))
    bar_width = 0.35
    index = range(len(df))

    # Plot the bars
    bar1 = plt.bar(index, df['ratio_GT'], bar_width, color='skyblue', label='y_true')
    bar2 = plt.bar([i + bar_width for i in index], df['ratio_links'], bar_width, color='coral', label='y_score')

    # Adjust the plot settings
    plt.xlabel('Samples')
    plt.ylabel('Ratio of non-zero elements (%)')
    plt.title(benchmarking_method)
    plt.xticks([i + bar_width / 2 for i in index], df.index, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_FILE, transparent=False)
    print_output(PLOT_FILE, verbose)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help='To print the details the run details')
    parser.add_argument('--benchmark-method', type=str, default='CO',
                        help='The method to calculate GRN benchmarking scores against Chip data')

    args = parser.parse_args()
    verbose = args.verbose
    benchmarking_method = args.benchmark_method

    STATS_DIR = f'{BENCHMARK_CO_DIR}/stats'
    STATS_FILE = f'{STATS_DIR}/stats_{benchmarking_method}.csv'
    PLOT_FILE = f'{STATS_DIR}/plot_{benchmarking_method}.png'

    stats = pd.read_csv(STATS_FILE, index_col=0)
    # print(stats)
    # ratio of links to all_combinations
    stats['ratio_GT'] = round(stats['ratio_GT']*100,1)
    # ratio of GT to all_combinations
    stats['ratio_links'] = round(stats['ratio_links'] * 100, 1)


    plot_both(stats)