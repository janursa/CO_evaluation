import numpy as np
import pandas as pd
def print_output(path, verbose):
    if verbose:
        print("\033[91m" + f'output -> {path}' + "\033[0m")
def print_df_cols(df, n=100):
    pd.set_option('display.max_columns', n)
    print(df)
