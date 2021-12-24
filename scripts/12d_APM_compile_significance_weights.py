#### Master Script 12d: Summarise aggregation weights across trained APM set ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Compile significance weights across trained APMs
# III. Summarise significance weights

### I. Initialisation
# Fundamental libraries
import os
import re
import sys
import time
import glob
import random
import datetime
import warnings
import itertools
import numpy as np
import pandas as pd
import pickle as cp
import seaborn as sns
import multiprocessing
from scipy import stats
from pathlib import Path
from ast import literal_eval
import matplotlib.pyplot as plt
from collections import Counter
from argparse import ArgumentParser
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# PyTorch, PyTorch.Text, and Lightning-PyTorch methods
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# SciKit-Learn methods
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample, shuffle
from sklearn.utils.class_weight import compute_class_weight

# TQDM for progress tracking
from tqdm import tqdm

# Import SHAP
import shap
from shap import DeepExplainer

# Custom methods
from classes.datasets import ALL_PREDICTOR_SET
from models.APM import APM_deep, shap_APM_deep
from functions.model_building import collect_agg_weights

# Set version code
VERSION = 'DEEP_v1-0'

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/APM_outputs/'+VERSION

# Load cross-validation split information to extract testing resamples
cv_splits = pd.read_csv('../cross_validation_splits.csv')
test_splits = cv_splits[cv_splits.test_or_train == 'test'].reset_index(drop=True)
uniq_GUPIs = test_splits.GUPI.unique()

# Define a directory for the storage of SHAP values
shap_dir = os.path.join(model_dir,'SHAP_values')

# Load model checkpoint information dataframe
APM_ckpt_info = pd.read_pickle(os.path.join(shap_dir,'APM_ckpt_info.pkl'))

# Set number of cores for parallel processing
NUM_CORES = multiprocessing.cpu_count()

### II. Compile significance weights across trained APMs
# Split up APM checkpoint dataframe among cores 
sizes = [APM_ckpt_info.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
sizes[:(APM_ckpt_info.shape[0] - sum(sizes))] = [val+1 for val in sizes[:(APM_ckpt_info.shape[0] - sum(sizes))]]    
end_indices = np.cumsum(sizes)
start_indices = np.insert(end_indices[:-1],0,0)
arg_iterable = [(APM_ckpt_info.iloc[start_indices[idx]:end_indices[idx]].reset_index(drop=True),True,'Compiling aggregation weights') for idx in range(len(start_indices))]

# Collect APM significance weights in parallel
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_agg_weights = pd.concat(pool.starmap(collect_agg_weights, arg_iterable),ignore_index = True)
    
# Store compiled significance weights of the optimally performing models
compiled_agg_weights.to_pickle(os.path.join(model_dir,'compiled_aggregation_weights.pkl'))

### III. Summarise significance weights
# Group by tuning index and token, and calculate agg_weight measures
grouped_agg_weights = compiled_agg_weights.groupby(['TUNE_IDX','Token'],as_index=False)['AggWeight'].aggregate('describe').unstack(0)

# Store descriptive statistics on significance weights 
grouped_agg_weights.to_csv(os.path.join(model_dir,'summarised_aggregation_weights.csv'),index=False)