import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, shutil, importlib, glob
from itertools import permutations
from tqdm import tqdm
from CO_evaluation.utils import print_output, print_df_cols
import itertools
import decoupler as dc

from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, f1_score
from sklearn.metrics import average_precision_score

from CO_evaluation.imports import  EXTERNAL_CO_BENCHMARK_DIR


try:
    ALL_TFS = np.load(
        f"{EXTERNAL_CO_BENCHMARK_DIR}/data/ground_truth_data/chip_atlas/TFs_in_gimmev5_mouse.npy", allow_pickle=True)

except:
    pass

sample_rename_dict = \
        {'Heart-10X_P7_4': "Heart_0",
         'Kidney-10X_P4_5': "Kidney_0",
         'Kidney-10X_P4_6': "Kidney_1",
         'Kidney-10X_P7_5': "Kidney_2",
         'Liver-10X_P4_2': "Liver_0",
         'Liver-10X_P7_0': "Liver_1",
         'Liver-10X_P7_1': "Liver_2",
         'Lung-10X_P7_8': "Lung_0",
         'Lung-10X_P7_9': "Lung_1",
         'Lung-10X_P8_12': "Lung_2",
         'Lung-10X_P8_13': "Lung_3",
         'Spleen-10X_P4_7': "Spleen_0",
         'Spleen-10X_P7_6': "Spleen_1"}
method_rename_dict = \
        {'WGCNA': "WGCNA",
         'GENIE3': "GENIE3",
         "DCOL": "DCOL",

         'SCENIC_10kb': "SCENIC",
         'celloracle_cluster_NoBaseGRN': "CellOracle\n no Base-GRN",
         'celloracle_whole_promoterBaseGRN': "CellOracle\n without CLR",
         'celloracle_cluster_promoterBaseGRN': "CellOracle\npromoter\nBase-GRN",
         'celloracle_cluster_mouseAtacBaseGRN': "CellOracle\nscATAC-atlas\nBase-GRN",
         'celloracle_cluster_scrambledPromoterBaseGRN': "CellOracle\n scrambled Base-GRN"}

def return_specificity_and_sensitivity(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (fn + tp)

    return specificity, sensitivity


def calculate_tf(y_true, y_predicted_score, verbose=False):


    # Calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_true, y_predicted_score)

    # Store roc curve as a data frame
    roc = pd.DataFrame({'fpr' : fpr,
                        'tpr' : tpr,
                        '1-fpr' : 1-fpr,
                        'tf' : tpr - fpr,
                        'thresholds' : thresholds})

    threshold = roc.loc[roc.tf.argmax(), "thresholds"]
    # Calculate Specificity and sensitivity

    y_pred = (y_predicted_score >= threshold)
    specificity, sensitivity = return_specificity_and_sensitivity(y_true, y_pred)

    return specificity, sensitivity, roc




def calculate_J(y_true, y_predicted_score, verbose=False):


    # Calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_true, y_predicted_score)

    # Store roc curve as a data frame
    roc = pd.DataFrame({'fpr' : fpr,
                        'tpr' : tpr,
                        '1-fpr' : 1-fpr,
                        'tf' : tpr - fpr,
                        'thresholds' : thresholds})

    # Calculate Specificity and sensitivity for each threshold value
    results = []
    if verbose:
        loop = tqdm(roc.thresholds)
    else:
        loop = roc.thresholds
    for threshold in loop:
        y_pred = (y_predicted_score >= threshold)
        results.append(return_specificity_and_sensitivity(y_true, y_pred))


    # Organize data
    df = pd.DataFrame(np.array(results),
                 columns=["specificity", "sensitivity"],
                 index=roc.index)
    roc = pd.concat([roc, df], axis=1)

    # Calculate Youden's J stativ
    roc["YoudensJ"] = roc["sensitivity"] + roc["specificity"] -1

    return roc


def calculate_(y_true, y_predicted_score, verbose=False):


    # Calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_true, y_predicted_score)

    # Store roc curve as a data frame
    roc = pd.DataFrame({'fpr' : fpr,
                        'tpr' : tpr,
                        '1-fpr' : 1-fpr,
                        'tf' : tpr - fpr,
                        'thresholds' : thresholds})

    # Calculate Specificity and sensitivity for each threshold value
    results = []
    if verbose:
        loop = tqdm(roc.thresholds)
    else:
        loop = roc.thresholds
    for threshold in loop:
        y_pred = (y_predicted_score >= threshold)
        results.append(return_specificity_and_sensitivity(y_true, y_pred))


    # Organize data
    df = pd.DataFrame(np.array(results),
                 columns=["specificity", "sensitivity"],
                 index=roc.index)
    roc = pd.concat([roc, df], axis=1)

    # Calculate Youden's J stativ
    roc["YoudensJ"] = roc["sensitivity"] + roc["specificity"] -1

    return roc

class GRN_Evaluator:
    metrices = ["epr", "auc", "auc_pr", "f1"]
    def __init__(self, all_TFs, df_ground_truth=None, path=None, links=None, genes_used=None,
                 benchmarking_method='CO', calculate_stats:bool=False):
        '''
        Inputs:
        benchmarking_method: whether to runs the original implementation of CO or the updated ones
                            CO: original implemenation
                            M1: our implementation
        stats: whether to calculate stats of benchmarking data such as class imbalance
        '''
        assert (benchmarking_method in ['CO', 'M1'])

        self.all_TFs = all_TFs
        self.score = {}
        self.benchmarking_method = benchmarking_method
        self.calculate_stats = calculate_stats
        self.benchmarking_stats = None

        assert (df_ground_truth is not None)
        self.load_ground_truth(df_ground_truth=df_ground_truth)

        if (links is not None) & (genes_used is not None):
            pass
        elif path is not None:
            links = pd.read_csv(os.path.join(path, "link.csv"), index_col=0)
            genes_used = pd.read_csv(os.path.join(path, "genes_nonzero.csv"), index_col=0).x.values # genes after filtering with chip
        else:
            raise ValueError('Entries are missing')
        # to make the semantics similar across different methods
        links = links.rename(columns={"regulatoryGene": "regulatory_gene",
                                      "TF": "regulatory_gene",
                                      "tf": "regulatory_gene",
                                      "gene": "target_gene",
                                      "targetGene": "target_gene",
                                      "target": "target_gene",
                                      'weight': "inference_result",
                                      'coef_abs': "inference_result",
                                      'CoexWeight': "inference_result",
                                      'dcol': "inference_result",
                                      'score': "inference_result",
                                      })
        assert ('inference_result' in links.columns)
        assert ('regulatory_gene' in links.columns)
        assert ('target_gene' in links.columns)
        links["key"] = links["regulatory_gene"] + "_" + links["target_gene"]
        self.links = links[["regulatory_gene", "target_gene", "inference_result", "key"]]
        self.genes_used = genes_used

        assert (self.df_ground_truth is not None)
        assert (self.genes_used is not None)

        self.process_data()
        self.calculate_scores()
    def load_ground_truth(self, df_ground_truth):
        """
        df_ground_truth should have three columns: ["tf", "target", "inference_result"]

        """
        self.df_ground_truth = df_ground_truth.copy()

        if "key" not in self.df_ground_truth.columns:
            self.df_ground_truth["key"] = \
                self.df_ground_truth["tf"] + "_" + self.df_ground_truth["target"]
        self.all_genes_in_ground_truth = np.unique(list(self.df_ground_truth["tf"].unique()) + \
                                                   list(self.df_ground_truth["target"].unique()))
    def process_data(self):
        assert (np.all(np.isin(self.genes_used, self.all_genes_in_ground_truth)))
        genes_used = self.genes_used
        GT = self.df_ground_truth
        np.random.seed(123)
        # 1. Make data frame for all possible connections
        if self.benchmarking_method == 'CO':
            # 1
            all_combinations = pd.DataFrame(permutations(genes_used, 2),
                                            columns=["regulatory_gene", "target_gene"])
            # 2 Remove tfs from evaluation list if TF does not exist in ground truth data.
            tfs_no_data = [i for i in self.all_TFs if i not in self.df_ground_truth.tf.unique()]
            all_combinations = all_combinations[~all_combinations.regulatory_gene.isin(tfs_no_data)]
        else:
            # 1
            unique_TFs = self.df_ground_truth.tf.unique()
            tfs_left = np.intersect1d(genes_used, unique_TFs)
            all_combinations = pd.DataFrame(itertools.product(tfs_left, genes_used),
                                            columns=["regulatory_gene", "target_gene"])
            # 2 remove those TFs from the GT if not present in genes
            GT = GT[GT.tf.isin(tfs_left)]
            # 3 remove if targets from GT are not present in genes
            GT = GT[GT.target.isin(genes_used)]
        n_all_combinations = len(all_combinations),

        all_combinations["key"] = all_combinations["regulatory_gene"] + "_" + all_combinations["target_gene"]
        # 2. Remove tfs from evaluation list if TF does not exist in ground truth data.

        # 3. Add ground truth data.
        all_combinations["ground_truth"] = all_combinations.key.isin(GT.key)

        # 4. Add inference data
        all_combinations = pd.merge(all_combinations, self.links[["key", "inference_result"]], on="key", how="left")
        all_combinations["na_in_raw_result"] = all_combinations.inference_result.isna()
        all_combinations["inference_result"] = all_combinations.inference_result.fillna(0)

        # 5. Add randomized data
        inferenced = all_combinations.inference_result.values.copy()
        np.random.shuffle(inferenced)
        all_combinations["randomized"] = inferenced

        self.ref_table = all_combinations

        # create a summary of stats
        TFs_1 = set(all_combinations.regulatory_gene.unique())
        TFs_2 = set(GT.tf.unique())
        shared_TFs = TFs_1.intersection(TFs_2)
        if self.calculate_stats:
            self.benchmarking_stats = dict(
                n_TFs_GT=len(shared_TFs),
                n_genes_used=len(self.genes_used),
                n_links=len(self.links),
                n_GT=len(GT),
                ratio_GT=all_combinations["ground_truth"].sum()/len(all_combinations),
                ratio_links = (all_combinations['inference_result'] != 0).sum() / len(all_combinations)
            )

    def calculate_scores(self):

        # 1. fp, tp, auc
        self.fp, self.tp, _ = roc_curve(y_true=self.ref_table.ground_truth,
                                        y_score=self.ref_table.inference_result)
        self.score["auc"] = auc(self.fp, self.tp)

        self.fp_random, self.tp_random, _ = roc_curve(y_true=self.ref_table.ground_truth,
                                                      y_score=self.ref_table.randomized)
        self.score["auc_random"] = auc(self.fp_random, self.tp_random)

        # AUC-PR
        auc_pr = average_precision_score(y_true=self.ref_table.ground_truth,
                                         y_score=self.ref_table.inference_result)
        self.score["auc_pr"] = auc_pr
        auc_pr_random = average_precision_score(y_true=self.ref_table.ground_truth,
                                                y_score=self.ref_table.randomized)
        self.score["auc_pr_random"] = auc_pr_random



        # 2. early precision
        self.k = self.ref_table.ground_truth.sum() # Number of positice edge in ground truth data

        sorted_ref_table = self.ref_table.sort_values(by="inference_result", ascending=False)
        self.score["ep"] = sorted_ref_table.ground_truth[:self.k].mean() # Top k precision

        self.score["ep_random"] = self.ref_table.ground_truth.mean() # ground_truth_ratio

        self.score["epr"] = self.score["ep"] / self.score["ep_random"]



        # f1
        def f1_calculater(y_true, y_scores):
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

            # Compute F1 score for each threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-15)

            # Get the threshold that gives the maximum F1 score
            best_threshold = thresholds[np.argmax(f1_scores)]
            best_f1 = np.max(f1_scores)
            return best_f1

        self.score["f1"] = f1_calculater(y_true=self.ref_table.ground_truth,
                                    y_scores=self.ref_table.inference_result)
        self.score["f1_random"] = f1_calculater(y_true=self.ref_table.ground_truth,
                                           y_scores=self.ref_table.randomized
                                           )
    def calculate_J(self, verbose=True):
        return calculate_J(self.ref_table.ground_truth,
                         self.ref_table.inference_result, verbose)

    def plot_auc(self):

        plt.plot(self.fp_random, self.tp_random, lw=2, color="lightgray",
                 label=f"randomized GRN auc: {self.score['auc_random']:.2g}")

        plt.plot(self.fp, self.tp, lw=2, color="C0",
                 label=f"GRN auc: {self.score['auc']:.2g}")

        plt.legend()
        sns.despine()

def get_base_name(path):
    if path.endswith("/"):
        return path.split("/")[-2]
    else:
        return path.split("/")[-1]


def calculate_tp_fp_auc(df):

    score = {}


    fp, tp, _ = roc_curve(y_true=df.ground_truth, y_score=df.inference_result)
    auc_score = auc(fp, tp)
    score["result"] = {"tp": tp, "fp": fp, "auc": auc_score}

    fp, tp, _ = roc_curve(y_true=df.ground_truth, y_score=df.randomized)
    auc_score = auc(fp, tp)
    score["randomized"] = {"tp": tp, "fp": fp, "auc": auc_score}

    return score

#####

def load_GRN_results_for_all_samples(benchmarking_method, analysis_parent_dir, tissue=None, BASE_FOLDER=None, VERBOSE_FOLDER=None):
    """
    BASE_FOLDER: where CO benchmarking data located
    """
    os.makedirs(VERBOSE_FOLDER, exist_ok=True) # to save individual scores for recovery
    samples = [d for d in sorted(os.listdir(analysis_parent_dir)) if os.path.isdir(os.path.join(analysis_parent_dir, d))]

    if tissue is not None:
        samples = [i for i in samples if tissue in i]

    li = []
    for sample in tqdm(samples, desc=f'Different samples'):
        sample_path = os.path.join(analysis_parent_dir, sample)
        scores = load_GRN_results(benchmarking_method, sample_path=sample_path, BASE_FOLDER=BASE_FOLDER, VERBOSE_FOLDER=f'{VERBOSE_FOLDER}/{sample}')
        li.append(scores)
    return pd.concat(li, axis=0)

def load_GRN_results(benchmarking_method, sample_path, BASE_FOLDER=None, VERBOSE_FOLDER=None, force=None):

    sample = sample_path.split("/")[-1]
    tissue = sample.split("-")[0]

    df_ground_truth = pd.read_csv(f"{BASE_FOLDER}/data/ground_truth_data/chip_atlas/data/{tissue}/chip_GT_links.csv", index_col=0)

    grn_names = [d for d in os.listdir(sample_path) if os.path.isdir(os.path.join(sample_path, d))]
    # grn_names = ['celloracle_cluster_mouseAtacBaseGRN', 'SCENIC_10kb']

    scores: list[dict[str,float]] = []

    os.makedirs(VERBOSE_FOLDER, exist_ok=True)
    for grn_name in tqdm(grn_names, desc=f'Different GRN methods {sample}'):
        file_to_save = f'{VERBOSE_FOLDER}/scores_{grn_name}.json'
        if os.path.exists(file_to_save):
            with open(file_to_save, 'r') as f:
                score = json.load(f)
            scores.append(score)
            continue
        path = os.path.join(sample_path, grn_name)
        ge = GRN_Evaluator(all_TFs=ALL_TFS,
                           df_ground_truth=df_ground_truth,
                           path=path,
                           benchmarking_method=benchmarking_method)
        score = ge.score
        scores.append(score)
        with open(file_to_save, 'w') as f:
            json.dump(score, f)

    scores = pd.concat([pd.Series(score_dict) for score_dict in scores],axis=1).transpose()
    scores["method"] = grn_names
    scores["sample"] = sample
    scores["tissue"] = tissue
    scores = scores.iloc[np.argsort(scores.method.str.len().values)].reset_index(drop=True)

    return scores


def plot_bar(scores_df, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 5, figsize=[20,7])


    axs[0].bar(scores_df.method, scores_df.epr_tfsubset)
    axs[0].tick_params(labelrotation=90)
    axs[0].set_title("epr_tfsubset")
    axs[1].bar(scores_df.method, scores_df.auc)
    axs[1].tick_params(labelrotation=90)
    axs[1].set_title("auroc")
    axs[2].bar(scores_df.method, scores_df.auc_tfsubset)
    axs[2].tick_params(labelrotation=90)
    axs[2].set_title("auroc_tfsubset")
    axs[3].bar(scores_df.method, scores_df.epr)
    axs[3].tick_params(labelrotation=90)
    axs[3].set_title("epr")
    axs[4].bar(scores_df.method, scores_df.aupr)
    axs[4].tick_params(labelrotation=90)
    axs[4].set_title("aupr")

def plot_bar_all(scores_all):

    samples = scores_all["sample"].unique()
    for sample in samples:
        scores = scores_all[scores_all["sample"] == sample]
        plot_bar(scores)
        plt.suptitle(sample)
        plt.show()




def plot_heatmap_all(scores_all, metric):
    tissues = scores_all["tissue"].unique()
    for tissue in tissues:
        scores = scores_all[scores_all["tissue"] == tissue]
        df = scores[[metric,"sample", "method"]]
        df = df.pivot(index="sample", columns="method", values=metric)
        sns.heatmap(data=df, annot=True, fmt=".4g", cmap="viridis")
        plt.title(tissue)
        plt.show()

def plot_heatmap_all(scores_all, metric):
    tissues = scores_all["tissue"].unique()
    for tissue in tissues:
        scores = scores_all[scores_all["tissue"] == tissue]
        df = scores[[metric,"sample", "method"]]
        df = df.pivot(index="sample", columns="method", values=metric)
        sns.heatmap(data=df, annot=True, fmt=".4g", cmap="viridis")
        plt.title(tissue)
        plt.show()



####

def box_plot(data, x, y, ax, palette, y_lim=None):
    fig, ax = plt.subplots(figsize=[2, 5])
    if y_lim is not None:
        ax.set_ylim(y_lim)
    sns.boxplot(data=data, x=x, y=y, fliersize=0, ax=ax, palette=palette)
    ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    ax.set_ylabel("edge weight in inferred GRN")
    sns.despine()

def violin_plot(data, x, y, palette, y_lim=None):
    fig, ax = plt.subplots(figsize=[2, 5])
    if y_lim is not None:
        ax.set_ylim(y_lim)
    sns.violinplot(data=data, x=x, y=y, ax=ax, palette=palette, scale="width",cut=0)
    ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    ax.set_ylabel("edge weight in inferred GRN")
    sns.despine()

def base_GRN_to_linklist(df):

    tt = df.groupby("gene_short_name").max()
    li = []

    for target, tfs in tqdm(tt.iterrows()):
        tfs = tfs[tfs != 0].index[1:]
        #li.append(pd.DataFrame({"tf": tfs, "target": [target]*len(tfs)}))
        li.append(np.stack([tfs, [target]*len(tfs)], axis=1))

    li = np.concatenate(li, axis=0)
    li = pd.DataFrame(li, columns=["tf", "target"])
    li["key"] = li["tf"] + "_" + li["target"]

    return li
def get_GT(tissue: str) -> pd.DataFrame:
    """
    get ground truth

    """
    GT_csv_path = f"{EXTERNAL_CO_BENCHMARK_DIR}/data/ground_truth_data/chip_atlas/data/{tissue}/chip_GT_links.csv"
    return pd.read_csv(GT_csv_path, index_col=0)
