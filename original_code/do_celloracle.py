import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import seaborn as sns
import time


import os, sys, shutil, importlib, glob

import scipy
import scanpy as sc
import celloracle as co



# Define custom function
def read_data(folder, filter_by_vargenes=False, n_cells_down_sampling=-1, downsampling_column="random_idx"):
    gem_path = os.path.join(folder, "log_data.mtx")
    gene_path = os.path.join(folder, "all_genes.csv")
    meta_path = os.path.join(folder, "meta_data.csv")

    mm = sc.read_mtx(gem_path)
    meta = pd.read_csv(meta_path, index_col=0)
    genes = pd.read_csv(gene_path).x.values

    adata = sc.AnnData(mm.X.transpose(),
                             obs=meta,
                             var=pd.DataFrame(index=genes))
    
    if filter_by_vargenes:
        vargene_path = os.path.join(folder, "var_genes.csv")
        var_genes = pd.read_csv(vargene_path).x.values
        adata = adata[:, var_genes]

    if ((n_cells_down_sampling > 0 ) & (meta.shape[0] > n_cells_down_sampling)):
        
        #cells_use = meta.index[meta[downsampling_column] <= n_cells_down_sampling].values
        
        np.random.seed(123)
        cells_use = np.random.choice(meta.index.values, size=n_cells_down_sampling, replace=False)
        adata = adata[cells_use, :]
        
    return adata

def filter_gem_by_gene(adata, all_genes_in_GT):
    genes = adata.var.index.values
    genes_intersect = np.intersect1d(genes, all_genes_in_GT)
    
    return adata[:, genes_intersect]


def filter_gem_by_genecount(adata, non_zero_ratio_threshold=0.01):
    
    non_zero_ratio = (adata.X > 0).mean(axis=0).A
    non_zero_ratio = non_zero_ratio.flatten()
    genes_ = adata.var.index[non_zero_ratio >= non_zero_ratio_threshold]

    return adata[:, genes_]

def get_merged_link(links_object):
    
    li = pd.concat([i.drop("p", axis=1) for i in links_object.links_dict.values()], axis=0)
    merged_link = li.groupby(["source", "target"]).max()
    merged_link = merged_link.reset_index(drop=False)
    merged_link = merged_link.rename(columns={"source": "regulatoryGene", "target": "targetGene"})
    
    return merged_link

def run_CellOracle(adata, tfdata, GRN_unit):
    
    if GRN_unit not in ["cluster", "whole"]:
        raise ValueError("GRN_unit should be either cluster or whole")
        
    
    # Preprocessing adata
    sc.tl.pca(adata)
    adata.obs["whole"] = "cl0"
    sc.pl.pca(adata, color=["whole", "cluster"])
    
    
    oracle = co.Oracle()
    oracle.import_anndata_as_normalized_count(adata=adata, 
                                              cluster_column_name=GRN_unit,
                                              embedding_name="X_pca", 
                                              test_mode=True)
    
    oracle.adata.layers["imputed_count"] = oracle.adata.layers["normalized_count"].copy()

    # You can load TF info dataframe with the following code.
    if isinstance(tfdata, pd.core.frame.DataFrame):
        oracle.import_TF_data(TF_info_matrix=tfdata)
    else:
        oracle.import_TF_data(TFdict=tfdata)
    
    print("start GRN calculation")
    # Calculate GRN for each population in "louvain_annot" clustering unit.
    # This step may take long time.
    links = oracle.get_links(cluster_name_for_GRN_unit=GRN_unit, alpha=1,
                             verbose_level=0, test_mode=False, ignore_warning=True)
    
    print("Finished GRN calculation")
    link = get_merged_link(links)
    
    return link

def test():
    for i in sys.argv:
        print(i)

def main():
    
    folder = sys.argv[1]
    GT_csv_path = sys.argv[2]
    base_GRN_path = sys.argv[3]
    name = sys.argv[4]
    output_folder = sys.argv[5]
    GRN_unit = sys.argv[6]
    
    try:
        downsampling = int(sys.argv[7])
    except:
        downsampling = -1
   
    
    
    run_and_save_celloracle(folder, GT_csv_path, base_GRN_path, name, output_folder, GRN_unit, downsampling)
    
    
def run_and_save_celloracle(folder, GT_csv_path, base_GRN_path, name, output_folder, GRN_unit, downsampling):
    print(folder)

    # Process Ground Truth data
    GT = pd.read_csv(GT_csv_path, index_col=0)
    # Get gene list
    GT_tf = GT.tf.unique()
    GT_target = GT.target.unique()
    GT_all_genes = np.unique(np.concatenate([GT_tf, GT_target]))
    
    
        
    # 1. load data
    adata = read_data(folder=folder, 
                      filter_by_vargenes=True,  # Use variable gene only
                      n_cells_down_sampling=downsampling, # Pick up 2000 cells if cell number is larger than this.
                      downsampling_column="random_idx")
    
    if downsampling == -1:
        pass
    else:
        n_cells = adata.shape[0]
        name = name + f"_ds{min(n_cells, downsampling)}"
    
    
    # 2. Filter gene expression matrix
    print("gene x cell ", adata.shape[1], adata.shape[0])

    print("Intersecting genes with GT genes")
    adata = filter_gem_by_gene(adata=adata, all_genes_in_GT=GT_all_genes)
    genes_intersected = adata.var.index.values
    print("gene x cell ", adata.shape[1], adata.shape[0])

    print("Filtering out genes based on non-zero ratio")
    adata = filter_gem_by_genecount(adata, non_zero_ratio_threshold=0.01) # Filter out genes based on non-zero ratio
    genes_nonzero = adata.var.index.values
    print("gene x cell ", adata.shape[1], adata.shape[0])
    
    
    # 2.1 Base GRN prepareation
    
    if base_GRN_path == "NO_base_GRN": # Do not use base GRN. CellOracle will do regressions without any prior-filtering
        # Make dummy base GRN
        pseudo_tfdict = {}
        for i in genes_nonzero:
            regs = genes_nonzero[genes_nonzero != i]
            pseudo_tfdict[i] = regs
        tfdata = pseudo_tfdict 
        #print(tfdata)
    else:
        tfdata = pd.read_parquet(base_GRN_path)


    # Calculation
    link = run_CellOracle(adata=adata, tfdata=tfdata, GRN_unit=GRN_unit)
    
    # 4. Save results
    sample_name = get_sample_name(folder)
    
    save_folder = os.path.join(output_folder, sample_name, f"celloracle_{GRN_unit}_{name}")
    os.makedirs(save_folder, exist_ok=True)
    link.to_csv(os.path.join(save_folder, "link.csv"))
    
    genes_intersected = pd.DataFrame({"x": genes_intersected})
    genes_nonzero = pd.DataFrame({"x": genes_nonzero})
    
    genes_intersected.to_csv(os.path.join(save_folder, "genes_intersected.csv"))
    genes_nonzero.to_csv(os.path.join(save_folder, "genes_nonzero.csv"))
    print("Finished")
    
    #return link

def get_sample_name(path):
    if path.endswith("/"):
        return path.split("/")[-2]
    else:
        return path.split("/")[-1]
    
if __name__ == "__main__":
    main()