import os
import sys
import json
import numpy as np
import pandas as pd
from glob import glob
from typing import Final
current_folder = os.path.dirname(os.path.realpath(__file__))
MAIN_DIR = f"{current_folder}/.."
sys.path.insert(0, MAIN_DIR)
MAIN_DIR = os.path.normpath(MAIN_DIR).replace('\\', '/')


OUTPUT_DIR = f"{MAIN_DIR}/results"
BENCHMARK_CO_DIR = f"{OUTPUT_DIR}/CO/benchmark"
tissues = ["Liver", "Lung", "Heart", "Kidney", "Spleen"]

METRICS: Final = ['auc', 'auc_random', 'epr', 'auc_pr', 'auc_pr_random', 'f1', 'f1_random']