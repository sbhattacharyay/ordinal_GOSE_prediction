#### Master Script 12b: Compile SHAP values for each GUPI-output type combination from APM_DeepMN ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Find all files storing calculated SHAP values and create combinations with study GUPIs
# III. Compile SHAP values for the given GUPI and output type combination

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

### II. Find all files storing calculated SHAP values and create combinations with study GUPIs
# Either create or load calculating SHAP file information
if not os.path.exists(os.path.join(shap_dir,'shap_file_info.pkl')):
    
    # Find all calculated SHAP files in model directory
    shap_files = []
    for path in Path(model_dir).rglob('shap_dataframe_*'):
        shap_files.append(str(path.resolve()))

    # Characterise model SHAP files
    shap_file_info = pd.DataFrame({'file':shap_files,
                                   'TUNE_IDX':[re.search('tune_(.*)/shap_dataframe_', curr_file).group(1) for curr_file in shap_files],
                                   'VERSION':[re.search('APM_outputs/(.*)/repeat', curr_file).group(1) for curr_file in shap_files],
                                   'repeat':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in shap_files],
                                   'fold':[int(re.search('/fold(.*)/tune_', curr_file).group(1)) for curr_file in shap_files],
                                   'output_type':[re.search('/shap_dataframe_(.*).pkl', curr_file).group(1) for curr_file in shap_files]
                                  }).sort_values(by=['repeat','fold','TUNE_IDX','VERSION']).reset_index(drop=True)
    
    # Save dataframe characterising storage of SHAP values
    shap_file_info.to_pickle(os.path.join(shap_dir,'shap_file_info.pkl'))

else:
    
    # Read dataframe characterising storage of SHAP values
    shap_file_info = pd.read_pickle(os.path.join(shap_dir,'shap_file_info.pkl'))

# Either create or load SHAP file and study GUPI combinations
if not os.path.exists(os.path.join(shap_dir,'shap_summarisation_combos.pkl')):
    
    # Create SHAP file and study GUPI combinations
    shap_summarisation_combos = pd.DataFrame(list(itertools.product(uniq_GUPIs, shap_file_info.output_type.unique())), columns=['GUPI', 'output_type'])
    
    # Save dataframe characterising SHAP file and study GUPI combinations
    shap_summarisation_combos.to_pickle(os.path.join(shap_dir,'shap_summarisation_combos.pkl'))
    
else:
    
    # Read dataframe characterising SHAP file and study GUPI combinations
    shap_summarisation_combos = pd.read_pickle(os.path.join(shap_dir,'shap_summarisation_combos.pkl'))

### III. Compile SHAP values for the given GUPI and output type combination
def main(array_task_id):
    
    # Get current GUPI and output type based on array task ID
    curr_GUPI = shap_summarisation_combos.GUPI[array_task_id]
    curr_output = shap_summarisation_combos.output_type[array_task_id]
    
    # Create directory to store SHAP values of current GUPI
    gupi_dir = os.path.join(shap_dir,curr_GUPI)
    os.makedirs(gupi_dir,exist_ok=True)
    
    # Load testing set information for current GUPI
    curr_test_splits = test_splits[test_splits.GUPI == curr_GUPI].reset_index(drop=True)
    
    # Filter SHAP values that correspond to current GUPI in testing set and current output type
    shap_file_info = shap_file_info.merge(curr_test_splits[['repeat','fold']],how='inner',on=['repeat','fold'])
    shap_file_info = shap_file_info[shap_file_info.output_type == curr_output].reset_index(drop=True)
    
    # Load current SHAP dataframes and filer out SHAP values of current GUPI
    shap_dfs = [pd.read_pickle(f).drop(columns='Indicator') for f in shap_file_info.file]
    shap_dfs = [df[df.GUPI == curr_GUPI].reset_index(drop=True) for df in shap_dfs]
    shap_dfs = pd.concat(shap_dfs,ignore_index=True)
    
    # Calculate absolute SHAP values from filtered dataframe
    shap_dfs['AbsSHAP'] = shap_dfs.SHAP.abs()
    
    # Calculate SHAP value summaries across repeated cross-validation partitions
    grouped_shap_values = shap_dfs.groupby(['GUPI','label','Token'],as_index=False)['SHAP'].aggregate('describe').unstack(0)
    grouped_shap_values = grouped_shap_values.melt(id_vars=['GUPI','label','Token'],var_name='METRIC',value_name='VALUE')
    grouped_shap_values['Transformation'] = 'Raw'

    # Calculate absolute SHAP value summaries across repeated cross-validation partitions
    grouped_abs_shap_values = shap_dfs.groupby(['GUPI','label','Token'],as_index=False)['AbsSHAP'].aggregate('describe').unstack(0)
    grouped_abs_shap_values = grouped_abs_shap_values.melt(id_vars=['GUPI','label','Token'],var_name='METRIC',value_name='VALUE')
    grouped_abs_shap_values['Transformation'] = 'Abs'
    
    # Concatenate SHAP and absolute SHAP values summarised across partitions
    grouped_shap_values = pd.concat([grouped_shap_values,grouped_abs_shap_values],ignore_index=True)
    grouped_shap_values['output_type'] = curr_output
    
    # Save SHAP values for current output type-GUPI combination
    grouped_shap_values.to_pickle(os.path.join(gupi_dir,'shap_'+curr_output+'.pkl'))

if __name__ == '__main__':

    array_task_id = int(sys.argv[1])    
    main(array_task_id)