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

EXTERNAL_CO_BENCHMARK_DIR = f'{MAIN_DIR}/external/'
OUTPUT_DIR = f"{MAIN_DIR}/results"
BENCHMARK_CO_DIR = f"{OUTPUT_DIR}/CO/benchmark"
tissues = ["Liver", "Lung", "Heart", "Kidney", "Spleen"]
GEM_folders = sorted(glob(f"{EXTERNAL_CO_BENCHMARK_DIR}/data/processed_GEM/*/"))

METRICS: Final = ['auc', 'epr']