#### Master Script 12c: Summarise SHAP values across study set ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Find all files storing GUPI-specific SHAP values
# III. Compile and save summary SHAP values across study patient set

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

### II. Find all files storing GUPI-specific SHAP values
# Either create or load all GUPI-specific SHAP file information
if not os.path.exists(os.path.join(shap_dir,'gupi_shap_file_info.pkl')):
    
    # Find all GUPI-specific SHAP files
    gupi_shap_files = []
    for path in Path(shap_dir).rglob('shap_*.pkl'):
        if ('info.pkl' not in str(path.resolve())) & ('combos.pkl' not in str(path.resolve())):
            gupi_shap_files.append(str(path.resolve()))
        
    # Categorize model checkpoint files based on name
    gupi_shap_info = pd.DataFrame({'file':gupi_shap_files,
                                   'GUPI':[re.search('SHAP_values/(.*)/shap_', curr_file).group(1) for curr_file in gupi_shap_files],\
                                   'VERSION':[re.search('APM_outputs/(.*)/SHAP_values', curr_file).group(1) for curr_file in gupi_shap_files],
                                   'output_type':[re.search('/shap_(.*).pkl', curr_file).group(1) for curr_file in gupi_shap_files]
                                  }).sort_values(by=['output_type','GUPI']).reset_index(drop=True)
    
    # Save GUPI-specific SHAP file information
    gupi_shap_info.to_pickle(os.path.join(shap_dir,'gupi_shap_file_info.pkl'))
    
else:
    
    # Read dataframe characterising GUPI-specific SHAP files
    gupi_shap_info = pd.read_pickle(os.path.join(shap_dir,'gupi_shap_file_info.pkl'))
    
### III. Compile and save summary SHAP values across study patient set
# Iterate through unique output types
for curr_output in tqdm(gupi_shap_info.output_type.unique()):
    
    # Extract GUPI SHAP info pertaining to current output type
    curr_output_info = gupi_shap_info[gupi_shap_info.output_type == curr_output].reset_index(drop=True)
    
    # Load list of SHAP dataframes
    shap_list = [pd.read_pickle(f) for f in curr_output_info.file]
    
    # Compile and save list of mean absolute values of SHAP
    mean_absolute_values = pd.concat([df[(df.Transformation == 'Abs') & (df.METRIC == 'mean')].reset_index(drop=True) for df in shap_list],ignore_index=True)
    mean_absolute_values = mean_absolute_values[['output_type','label','GUPI','Token','VALUE']].sort_values(by=['label','GUPI','Token','VALUE']).reset_index(drop=True)
    mean_absolute_values.to_csv(os.path.join(shap_dir,curr_output+'_mean_absolute_values.csv'),index=False)
    
    # Compile and save list of max absolute values of SHAP
    max_absolute_values = pd.concat([df[(df.Transformation == 'Abs') & (df.METRIC == 'max')].reset_index(drop=True) for df in shap_list],ignore_index=True)
    max_absolute_values = max_absolute_values[['output_type','label','GUPI','Token','VALUE']].sort_values(by=['label','GUPI','Token','VALUE']).reset_index(drop=True)
    max_absolute_values.to_csv(os.path.join(shap_dir,curr_output+'_max_absolute_values.csv'),index=False)

    # Compile and save list of mean raw values of SHAP
    mean_raw_values = pd.concat([df[(df.Transformation == 'Raw') & (df.METRIC == 'mean')].reset_index(drop=True) for df in shap_list],ignore_index=True)
    mean_raw_values = mean_raw_values[['output_type','label','GUPI','Token','VALUE']].sort_values(by=['label','GUPI','Token','VALUE']).reset_index(drop=True)
    mean_raw_values.to_csv(os.path.join(shap_dir,curr_output+'_mean_raw_values.csv'),index=False)