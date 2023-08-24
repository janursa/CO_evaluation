import portia as pt
import numpy as np
import pandas as pd
def print_output(path, verbose):
    if verbose:
        print("\033[91m" + f'output -> {path}' + "\033[0m")
def print_df_cols(df, n=100):
    pd.set_option('display.max_columns', n)
    print(df)
def run_portia(data, gene_names):
    """Runs GRN inference using Portia

    """
    # - create portia dataset
    portia_dataset = pt.GeneExpressionDataset()
    for exp_id, data_i in enumerate(data):

        if np.all(data_i==0):
            raise ValueError('all values of the given data is zero')
        portia_dataset.add(pt.Experiment(exp_id, data_i))
    # - GRN inference
    M_bar = pt.run(portia_dataset, method='fast', verbose=False, normalize=True)
    # Convert the matrix to a DataFrame
    df_matrix = pd.DataFrame(M_bar, index=gene_names, columns=gene_names)

    # Melt the DataFrame to long format and reset the index
    df_long = df_matrix.reset_index().melt(id_vars='index', var_name='target', value_name='score')

    # Rename the 'index' column to 'TF'
    df_long = df_long.rename(columns={'index': 'tf'})


    return df_long