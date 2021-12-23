#### Master Script 15b: Perform interrepeat hyperparameter configuration dropout on deep learning extended concise-predictor-based models (eCPM) ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Calculate ORC of extant validation predictions
# III. Prepare bootstrapping resamples for configuration dropout  
# IV. Dropout configurations that are consistently (a = .05) inferior in performance
# V. Compile and save validation and testing set predictions across partitions

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
from shutil import rmtree
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
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

# TQDM for progress tracking
from tqdm import tqdm

# Custom methods
from functions.model_building import calc_orc

# Set the last repeat completed
REPEAT = 20

# Set version code
VERSION = 'DEEP_v1-0'

# Define model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/eCPM_outputs/'+VERSION

# Load cross-validation information to get GUPI and outcomes
cv_splits = pd.read_csv('../cross_validation_splits.csv')
study_GUPI_GOSE = cv_splits[['GUPI','GOSE']].drop_duplicates()

# Define repeat directory based on last completed repeat
repeat_dir = os.path.join(model_dir,'repeat'+str(REPEAT).zfill(int(np.log10(cv_splits.repeat.max()))+1))

# Set number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count()

# Set number of resamples for bootstrapping
NUM_RESAMP = 1000

# Load tuning grid based on the last completed repeat
if REPEAT == 1:

    tuning_grid = pd.read_csv(os.path.join(model_dir,'eCPM_deep_tuning_grid.csv'))
    tuning_grid['TUNE_IDX'] = tuning_grid['TUNE_IDX'].astype(str).str.zfill(4)
    tuning_grid['NEURONS'] = tuning_grid['NEURONS'].apply(eval)
    
else:
    
    tuning_grid = pd.read_csv(os.path.join(model_dir,'eCPM_post_repeat_'+str(REPEAT).zfill(2)+'_deep_tuning_grid.csv'))
    tuning_grid['TUNE_IDX'] = tuning_grid['TUNE_IDX'].astype(str).str.zfill(4)
    tuning_grid['NEURONS'] = tuning_grid['NEURONS'].apply(eval)
    
### II. Calculate ORC of extant validation predictions
# Perform validation prediction file search
val_pred_files = []
for path in Path(model_dir).rglob('*/val_predictions.csv'):
    val_pred_files.append(str(path.resolve()))
    
# Characterise validation prediction file information
val_pred_file_info_df = pd.DataFrame({'file':val_pred_files,
                                      'TUNE_IDX':[re.search('tune_(.*)/', curr_file).group(1) for curr_file in val_pred_files],
                                      'VERSION':[re.search('eCPM_outputs/(.*)/repeat', curr_file).group(1) for curr_file in val_pred_files],
                                      'repeat':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in val_pred_files],
                                      'fold':[int(re.search('/fold(.*)/tune_', curr_file).group(1)) for curr_file in val_pred_files]
                                     }).sort_values(by=['repeat','fold','TUNE_IDX','VERSION']).reset_index(drop=True)

# Merge output activation information to validation prediction info dataframe
val_pred_file_info_df = pd.merge(val_pred_file_info_df,tuning_grid[['TUNE_IDX','OUTPUT_ACTIVATION']],how='left',on='TUNE_IDX').reset_index(drop=True)

# Partition validation files across number of available cores for parallel processing
npc = [val_pred_file_info_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
npc[:(val_pred_file_info_df.shape[0] - sum(npc))] = [val+1 for val in npc[:(val_pred_file_info_df.shape[0] - sum(npc))]]
end_indices = np.cumsum(npc)
start_indices = np.insert(end_indices[:-1],0,0)

# Build arguments for parallelisation function
arg_iterable = [(val_pred_file_info_df.iloc[start_indices[idx]:end_indices[idx]].reset_index(drop=True),True,'Calculating validation set ORC') for idx in range(len(start_indices))]

# Calculate ORC of each validation prediction file in parallel
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_val_ORC = pd.concat(pool.starmap(calc_orc, arg_iterable),ignore_index = True)
    
# Save validation prediction ORC in the repeat directory
compiled_val_ORC.to_csv(os.path.join(repeat_dir,'validation_performance.csv'),index=False)

# Group by tuning index and average validation ORC
across_cv_perf = compiled_val_ORC.groupby(['TUNE_IDX','OUTPUT_ACTIVATION'],as_index=False)['val_ORC'].mean()

# Determine 'optimal' tuning indices based on validation performance
opt_tune_idx = across_cv_perf[across_cv_perf.groupby('OUTPUT_ACTIVATION')['val_ORC'].transform(max) == across_cv_perf['val_ORC']].reset_index(drop=True)

### III. Prepare bootstrapping resamples for configuration dropout  
# Create directory for storing dropout bootstrapping information
dropout_dir = os.path.join('/home/sb2406/rds/hpc-work/interrepeat_dropout','eCPM_deep',VERSION)
os.makedirs(dropout_dir,exist_ok=True)

# Create stratified resamples for bootstrapping
bs_rs_GUPIs = [resample(study_GUPI_GOSE.GUPI.values,replace=True,n_samples=study_GUPI_GOSE.shape[0],stratify=study_GUPI_GOSE.GOSE.values) for _ in range(NUM_RESAMP)]
bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resmaples 
bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':bs_rs_GUPIs})

# Create Data Frame of output activation - resample combos
output_resample_combos = pd.DataFrame(list(itertools.product(compiled_val_ORC.OUTPUT_ACTIVATION.unique(), bs_resamples.RESAMPLE_IDX.unique())), columns=['OUTPUT_ACTIVATION', 'RESAMPLE_IDX'])

# Merge the two dataframes
bs_resamples = pd.merge(bs_resamples,output_resample_combos,how='outer',on='RESAMPLE_IDX')

# Append information of optimal tuning index
bs_resamples = pd.merge(bs_resamples,opt_tune_idx[['TUNE_IDX','OUTPUT_ACTIVATION']],how='left',on='OUTPUT_ACTIVATION')
bs_resamples = bs_resamples.rename(columns={'TUNE_IDX':'OPT_TUNE_IDX'})

# Save current resample information for parallelised hypothesis testing
bs_resamples.to_pickle(os.path.join(dropout_dir,'post_repeat_'+str(REPEAT).zfill(2)+'_resamples.pkl'))

# NOTE: at this point, run the scripts for 6c

### IV. Dropout configurations that are consistently (a = .05) inferior in performance
# Find all bootstrapped ORC results
bs_orc_files = []
for path in Path(os.path.join(dropout_dir,'repeat'+str(REPEAT).zfill(2))).rglob('*.pkl'):
    curr_path = str(path.resolve())
    if ('softmax_dropout' in curr_path) | ('sigmoid_dropout' in curr_path):
        bs_orc_files.append(curr_path)

# Characterise file information
bs_orc_info_df = pd.DataFrame({'file':bs_orc_files,
                               'REPEAT':[int(re.search('/repeat(.*)/', curr_file).group(1)) for curr_file in bs_orc_files],
                                'OUTPUT_ACTIVATION':[re.search('/repeat(.*)_dropout', curr_file).group(1) for curr_file in bs_orc_files],
                                'RESAMPLE_IDX':[int(re.search('resample_idx_(.*).pkl', curr_file).group(1)) for curr_file in bs_orc_files]}).sort_values(by=['OUTPUT_ACTIVATION','RESAMPLE_IDX']).reset_index(drop=True)
bs_orc_info_df['OUTPUT_ACTIVATION'] = bs_orc_info_df['OUTPUT_ACTIVATION'].str[3:]

# Initialise an empty list to store dropped out tuning configurations
dropped_tis = []

# Iterate through output activation options
for curr_OUTPUT_ACTIVATION in bs_orc_info_df.OUTPUT_ACTIVATION.unique():
    
    # Load and compile all files within current output activation
    curr_output_bs = pd.concat([pd.read_pickle(f) for f in bs_orc_info_df.file[bs_orc_info_df.OUTPUT_ACTIVATION == curr_OUTPUT_ACTIVATION]],ignore_index=True)
    
    # Calculate p-value for each tuning index
    p_val_df = curr_output_bs.groupby('TUNE_IDX',as_index=False)['trial_win'].apply(lambda x: x.sum()/len(x))
    p_val_df = p_val_df.rename(columns={'trial_win':'p_val'})
    
    # Find significantly poor configurations for dropout
    sig_df = p_val_df[p_val_df.p_val <= 0.05].reset_index(drop=True)
    
    # Add dropped tuning indices to list
    dropped_tis += sig_df.TUNE_IDX.to_list()
    
# Print out how many configurations have been dropped
print(str(len(dropped_tis))+' tuning indices out of '+str(tuning_grid.shape[0])+' dropped after repeat '+str(REPEAT))

# Update viable tuning grid and save
viable_tuning_grid = tuning_grid[~tuning_grid.TUNE_IDX.isin(dropped_tis)].reset_index(drop=True)
viable_tuning_grid.to_csv(os.path.join(model_dir,'eCPM_post_repeat_'+str(REPEAT).zfill(2)+'_deep_tuning_grid.csv'),index=False)

# Clear disk space by deleting folders of dropped out models
dropped_configs = val_pred_file_info_df[~val_pred_file_info_df.TUNE_IDX.isin(viable_tuning_grid.TUNE_IDX)].reset_index(drop=True)
dropped_configs['directory'] = dropped_configs['file'].str.replace("/val_predictions.csv", "", regex=False)
for d in dropped_configs.directory:
    rmtree(d)
    
# NOTE: at this point, train models on subsequent cross-validation partitions and repeat until all CV-partitions have been trained over

### V. Compile and save validation and testing set predictions across partitions
# Search for all prediction files
pred_files = []
for path in Path(model_dir).rglob('*_predictions.csv'):
    pred_files.append(str(path.resolve()))

# Characterise the prediction files found
pred_file_info_df = pd.DataFrame({'file':pred_files,
                                  'TUNE_IDX':[re.search('tune_(.*)/', curr_file).group(1) for curr_file in pred_files],
                                  'VERSION':[re.search('_outputs/(.*)/repeat', curr_file).group(1) for curr_file in pred_files],
                                  'repeat':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in pred_files],
                                  'fold':[int(re.search('/fold(.*)/tune_', curr_file).group(1)) for curr_file in pred_files],
                                  'test_or_val':[re.search('/tune_(.*)_predictions', curr_file).group(1) for curr_file in pred_files]
                                 }).sort_values(by=['repeat','fold','TUNE_IDX','VERSION']).reset_index(drop=True)
pred_file_info_df['test_or_val'] = pred_file_info_df['test_or_val'].str.split('/').str[-1]
pred_file_info_df = pred_file_info_df[pred_file_info_df.TUNE_IDX.isin(tuning_grid.TUNE_IDX)].reset_index(drop=True)
pred_file_info_df = pd.merge(pred_file_info_df,tuning_grid[['TUNE_IDX','OUTPUT_ACTIVATION']],how='left',on='TUNE_IDX')

# Separate prediction files by outcome encoding and testing vs. validation
softmax_val_info_df = pred_file_info_df[(pred_file_info_df.OUTPUT_ACTIVATION == 'softmax') & (pred_file_info_df.test_or_val == 'val')].reset_index(drop=True)
softmax_test_info_df = pred_file_info_df[(pred_file_info_df.OUTPUT_ACTIVATION == 'softmax') & (pred_file_info_df.test_or_val == 'test')].reset_index(drop=True)

sigmoid_val_info_df = pred_file_info_df[(pred_file_info_df.OUTPUT_ACTIVATION == 'sigmoid') & (pred_file_info_df.test_or_val == 'val')].reset_index(drop=True)
sigmoid_test_info_df = pred_file_info_df[(pred_file_info_df.OUTPUT_ACTIVATION == 'sigmoid') & (pred_file_info_df.test_or_val == 'test')].reset_index(drop=True)

# Compile predictions into single dataframes
softmax_val_preds = pd.concat([pd.read_csv(curr_file) for curr_file in softmax_val_info_df.file.to_list()],ignore_index=True)
softmax_test_preds = pd.concat([pd.read_csv(curr_file) for curr_file in softmax_test_info_df.file.to_list()],ignore_index = True)

sigmoid_val_preds = pd.concat([pd.read_csv(curr_file) for curr_file in sigmoid_val_info_df.file.to_list()],ignore_index = True)
sigmoid_test_preds = pd.concat([pd.read_csv(curr_file) for curr_file in sigmoid_test_info_df.file.to_list()],ignore_index = True)

# Correct formatting of tuning index
softmax_val_preds['TUNE_IDX'] = softmax_val_preds['TUNE_IDX'].astype(str).str.zfill(4)
softmax_test_preds['TUNE_IDX'] = softmax_test_preds['TUNE_IDX'].astype(str).str.zfill(4)

sigmoid_val_preds['TUNE_IDX'] = sigmoid_val_preds['TUNE_IDX'].astype(str).str.zfill(4)
sigmoid_test_preds['TUNE_IDX'] = sigmoid_test_preds['TUNE_IDX'].astype(str).str.zfill(4)

# Save prediction files appropriately
softmax_val_preds.to_csv(os.path.join(model_dir,'eCPM_deepMN_compiled_val_predictions.csv'),index=True)
softmax_test_preds.to_csv(os.path.join(model_dir,'eCPM_deepMN_compiled_test_predictions.csv'),index=True)

sigmoid_val_preds.to_csv(os.path.join(model_dir,'eCPM_deepOR_compiled_val_predictions.csv'),index=True)
sigmoid_test_preds.to_csv(os.path.join(model_dir,'eCPM_deepOR_compiled_test_predictions.csv'),index=True)