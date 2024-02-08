import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, shutil, importlib, glob
from tqdm.notebook import tqdm
from itertools import permutations

from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from sklearn.metrics import average_precision_score


try:
    ALL_TFS = np.load("data/ground_truth_data/chip_atlas/TFs_in_gimmev5_mouse.npy", allow_pickle=True)

except:
    print('couldnt load ALL_TFS')
    pass


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

    def __init__(self, all_TFs, df_ground_truth=None, path=None, links=None, genes_used=None, annot=None, k=None, calculate_score_immediately=True):

        self.all_TFs = all_TFs
        annot = annot
        self.score = {}
        self.k = k

        if df_ground_truth is not None:
            self.load_ground_truth(df_ground_truth=df_ground_truth)
        else:
            self.df_ground_truth = None

        if path is not None:
            self.load_inference_result(path=path)

        elif (links is not None) & (genes_used is not None):
            self.load_links(links=links)
            self.load_genes_used(genes_used=genes_used)
        else:
            self.links = None
            self.genes_used = None
        if (self.df_ground_truth is not None) & (self.links is not None) & (self.genes_used is not None) & calculate_score_immediately:
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

    def load_links(self, links):
        """
        Link should have three columns: ["regulatory_gene", "target_gene", "inference_result"]

        """
        self.links = links.copy()

        if "key" not in self.links.columns:
            self.links["key"] = self.links["regulatory_gene"] + "_" + self.links["target_gene"]

    def load_genes_used(self, genes_used):
        self.genes_used = genes_used


    def load_inference_result(self, path):

        links, genes_used = load_inference_result(path=path)

        self.links = links
        self.genes_used = genes_used
        self.path = path


    def process_data(self):

        all_genes = np.intersect1d(self.genes_used, self.all_genes_in_ground_truth)
        self.ref_table = process_data(genes_used=all_genes,
                                      genes_tf=self.all_TFs,
                                      df_ground_truth=self.df_ground_truth,
                                      df_inference=self.links)


    def calculate_scores(self):

        # 1. fp, tp, auc
        self.fp, self.tp, _ = roc_curve(y_true=self.ref_table.ground_truth,
                                        y_score=self.ref_table.inference_result)
        self.score["auc"] = auc(self.fp, self.tp)

        self.fp_random, self.tp_random, _ = roc_curve(y_true=self.ref_table.ground_truth,
                                                      y_score=self.ref_table.randomized)
        self.score["auc_random"] = auc(self.fp_random, self.tp_random)

        # 2. early precision
        if self.k is None:
            self.k = self.ref_table.ground_truth.sum() # Number of positice edge in ground truth data
        #self.k = int(self.ref_table.shape[0]*self.k_ratio)


        sorted_ref_table = self.ref_table.sort_values(by="inference_result", ascending=False)
        self.score["ep"] = sorted_ref_table.ground_truth[:self.k].mean() # Top k precision

        self.score["ep_random"] = self.ref_table.ground_truth.mean() # ground_truth_ratio

        self.score["epr"] = self.score["ep"] / self.score["ep_random"]


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

    

def load_inference_result(path):

    grn_type = get_base_name(path)

    if "celloracle" in grn_type.lower():
        weight = "coef_abs"
        #weight = "-logp"
    elif "genie3" in grn_type.lower():
        weight = "weight"
    elif "wgcna" in grn_type.lower():
        weight = "weight"
    elif "scenic" in grn_type.lower():
        weight = "CoexWeight"
    elif "dcol" in grn_type.lower():
        weight = "weight"
    else:
        raise ValueError("unknown type")


    link = pd.read_csv(os.path.join(path, "link.csv"), index_col=0)
    #print(link)

    link = link.rename(columns={"regulatoryGene": "regulatory_gene",
                                "TF": "regulatory_gene",
                                "gene": "target_gene",
                                "targetGene": "target_gene",
                                weight: "inference_result"})

    link["key"] = link["regulatory_gene"] + "_" + link["target_gene"]


    link = link[["regulatory_gene", "target_gene", "inference_result", "key"]]

    path_genes = os.path.join(path, "genes_nonzero.csv")
    genes_used = pd.read_csv(path_genes, index_col=0).x.values



    return link, genes_used

def get_base_name(path):
    if path.endswith("/"):
        return path.split("/")[-2]
    else:
        return path.split("/")[-1]

def process_data(genes_used, genes_tf, df_ground_truth, df_inference):

    np.random.seed(123)

    # 1. Make data frame for all possible connections
    all_combinations = pd.DataFrame(permutations(genes_used, 2),
                                    columns=["regulatory_gene", "target_gene"])
    all_combinations["key"] = all_combinations["regulatory_gene"] + "_" + all_combinations["target_gene"]

    # 2. Remove tfs from evaluation list if TF does not exist in ground truth data.
    #    We cannot judge about such connections.
    tfs_no_data = [i for i in genes_tf if i not in df_ground_truth.tf.unique()]
    all_combinations = all_combinations[~all_combinations.regulatory_gene.isin(tfs_no_data)]

    # 3. Add ground truth data.
    all_combinations["ground_truth"] = all_combinations.key.isin(df_ground_truth.key)

    # 4. Add inference data
    all_combinations = pd.merge(all_combinations, df_inference[["key", "inference_result"]], on="key", how="left")
    all_combinations["na_in_raw_result"] = all_combinations.inference_result.isna()
    all_combinations["inference_result"] = all_combinations.inference_result.fillna(0)
    print(len(all_combinations))
    aa

    # 5. Add randomized data
    inferenced = all_combinations.inference_result.values.copy()
    np.random.shuffle(inferenced)
    all_combinations["randomized"] = inferenced

    return all_combinations


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

def load_GRN_results_for_all_samples(analysis_parent_dir, tissue=None):

    samples = sorted(os.listdir(analysis_parent_dir))

    if tissue is not None:
        samples = [i for i in samples if tissue in i]

    li = []
    for sample in tqdm(samples):
        sample_path = os.path.join(analysis_parent_dir, sample)
        scores = load_GRN_results(sample_path=sample_path, return_ge=False)
        li.append(scores)
    return pd.concat(li, axis=0)

def load_GRN_results(sample_path, return_ge=True, calculate_score=True):

    sample = sample_path.split("/")[-1]
    tissue = sample.split("-")[0]

    df_ground_truth = pd.read_csv(f"data/ground_truth_data/chip_atlas/data/{tissue}/chip_GT_links.csv", index_col=0)

    grn_names = os.listdir(sample_path)

    ge_objects = []
    scores = []
    for i in tqdm(grn_names):
        path = os.path.join(sample_path, i)
        if calculate_score:
            ge = GRN_Evaluator(all_TFs=ALL_TFS,
                               df_ground_truth=df_ground_truth,
                               path=path)
            ge.method = i
            if return_ge:
                ge_objects.append(ge)
            scores.append(ge.score)


        else:
            ge = GRN_Evaluator(all_TFs=ALL_TFS,
                               df_ground_truth=df_ground_truth,
                               path=path,
                              calculate_score_immediately=False)
            ge_objects.append(ge)
            ge.method = i


    if (return_ge == True) & (calculate_score == False):
        return ge_objects
    else:
        scores = pd.concat([pd.Series(i) for i in scores],axis=1).transpose()
        scores["method"] = grn_names
        scores["sample"] = sample
        scores["tissue"] = tissue
        scores = scores.iloc[np.argsort(scores.method.str.len().values)].reset_index(drop=True)

        if return_ge:
            return ge_objects, scores
        else:
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
