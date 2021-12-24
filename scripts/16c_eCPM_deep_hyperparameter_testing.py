#### Master Script 16c: Calculate ORC in bootstrapping resamples to determine dropout configurations ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Calculate ORC of extant validation predictions
# III. Prepare bootstrapping resamples for configuration dropout  

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

# SciKit-Learn methods
from sklearn.metrics import roc_auc_score

# TQDM for progress tracking
from tqdm import tqdm

# Set version code
VERSION = 'DEEP_v1-0'

# Define last completed repeat
REPEAT = 20

# Based on VERSION, determine current dropout directory
dropout_dir = os.path.join('/home/sb2406/rds/hpc-work/interrepeat_dropout','eCPM_deep',VERSION)
os.makedirs(os.path.join(dropout_dir,'repeat'+str(REPEAT).zfill(2)),exist_ok=True)

# Based on REPEAT, load current resamples
bs_resamples = pd.read_pickle(os.path.join(dropout_dir,'post_repeat_'+str(REPEAT).zfill(2)+'_resamples.pkl'))

# Load cross-validation information
cv_splits = pd.read_csv('../cross_validation_splits.csv')

# Based on VERSION and REPEAT, define the last trained repeat directory
repeat_dir = os.path.join('/home/sb2406/rds/hpc-work/eCPM_outputs/'+VERSION,'repeat'+str(REPEAT).zfill(int(np.log10(cv_splits.repeat.max()))+1))

# Load compiled validation performance dataframe based on current repeat directory
compiled_val_ORC = pd.read_csv(os.path.join(repeat_dir,'validation_performance.csv'))
compiled_val_ORC['TUNE_IDX'] = compiled_val_ORC['TUNE_IDX'].astype(str).str.zfill(4)

def main(array_task_id):
    
    # Isolate bootstrapping resample information of current trial
    curr_GUPIs = bs_resamples.GUPIs[array_task_id]
    curr_OUTPUT_ACTIVATION = bs_resamples.OUTPUT_ACTIVATION[array_task_id]
    curr_opt_ti = bs_resamples.OPT_TUNE_IDX[array_task_id]
    curr_rs_idx = bs_resamples.RESAMPLE_IDX[array_task_id]
    
    # Filter validation file information of current output activation
    curr_val_info = compiled_val_ORC[compiled_val_ORC.OUTPUT_ACTIVATION == curr_OUTPUT_ACTIVATION]
    
    # Create list to store ORC results
    curr_rs_ORCs = []
    
    # Create TQDM iterator for timed tracking
    iterator = tqdm(curr_val_info.TUNE_IDX.unique(),desc='RESAMPLE '+str(curr_rs_idx))
    
    # Iterate through tuning indices of current output activation
    for curr_ti in iterator:
        
        # Extract files of current tuning index and load validation predictions
        curr_ti_info = curr_val_info[curr_val_info.TUNE_IDX == curr_ti]        
        curr_ti_preds = pd.concat([pd.read_csv(curr_file) for curr_file in curr_ti_info.file.values],ignore_index=True)
        
        # Filter in sample predictions
        curr_rs_preds = curr_ti_preds[curr_ti_preds.GUPI.isin(curr_GUPIs)]
        
        # Extract prob columns
        prob_cols = [col for col in curr_rs_preds if col.startswith('Pr(GOSE')]
        
        # Calculate ORC
        if curr_OUTPUT_ACTIVATION == 'softmax':

            aucs = []
            for ix, (a, b) in enumerate(itertools.combinations(np.sort(curr_rs_preds.TrueLabel.unique()), 2)):
                filt_preds = curr_rs_preds[curr_rs_preds.TrueLabel.isin([a,b])].reset_index(drop=True)
                filt_preds['ConditProb'] = filt_preds[prob_cols[b]]/(filt_preds[prob_cols[a]] + filt_preds[prob_cols[b]])
                filt_preds['ConditProb'] = np.nan_to_num(filt_preds['ConditProb'],nan=.5,posinf=1,neginf=0)
                filt_preds['ConditLabel'] = (filt_preds.TrueLabel == b).astype(int)
                aucs.append(roc_auc_score(filt_preds['ConditLabel'],filt_preds['ConditProb']))
            curr_orc = np.mean(aucs)
            
        elif curr_OUTPUT_ACTIVATION == 'sigmoid':

            label_cols = [col for col in curr_rs_preds if col.startswith('GOSE>')]
            curr_train_probs = curr_rs_preds[prob_cols].values
            
            train_probs = np.empty([curr_train_probs.shape[0], curr_train_probs.shape[1]+1])
            train_probs[:,0] = 1 - curr_train_probs[:,0]
            train_probs[:,-1] = curr_train_probs[:,-1]
            
            for col_idx in range(1,(curr_train_probs.shape[1])):
                train_probs[:,col_idx] = curr_train_probs[:,col_idx-1] - curr_train_probs[:,col_idx]                
            
            train_labels = curr_rs_preds[label_cols].values.sum(1).astype(int)
            aucs = []
            for ix, (a, b) in enumerate(itertools.combinations(np.sort(np.unique(train_labels)), 2)):
                a_mask = train_labels == a
                b_mask = train_labels == b
                ab_mask = np.logical_or(a_mask,b_mask)
                condit_probs = train_probs[ab_mask,b]/(train_probs[ab_mask,a]+train_probs[ab_mask,b]) 
                condit_probs = np.nan_to_num(condit_probs,nan=.5,posinf=1,neginf=0)
                condit_labels = b_mask[ab_mask].astype(int)
                aucs.append(roc_auc_score(condit_labels,condit_probs))            
            curr_orc = np.mean(aucs)
                        
        curr_rs_ORCs.append(pd.DataFrame({'RESAMPLE_IDX':curr_rs_idx,'TUNE_IDX':curr_ti,'val_ORC':curr_orc},index=[0]))
        
    # Concatenate list of results
    curr_rs_ORCs = pd.concat(curr_rs_ORCs,ignore_index=True)
    
    # Filter out optimal tuning index performance
    opt_ti_perf = curr_rs_ORCs[curr_rs_ORCs.TUNE_IDX == curr_opt_ti].reset_index(drop=True)
    
    # Add optimal val_ORC across dataframe and remove row of optimal ti
    curr_rs_ORCs['opt_val_ORC'] = opt_ti_perf.val_ORC[0]
    curr_rs_ORCs = curr_rs_ORCs[curr_rs_ORCs.TUNE_IDX != curr_opt_ti].reset_index(drop=True)
    
    # Add indicator variable signifying trial win or tie
    curr_rs_ORCs['trial_win'] = (curr_rs_ORCs['val_ORC'] >= curr_rs_ORCs['opt_val_ORC']).astype(int)
    
    # Add other information
    curr_rs_ORCs['OUTPUT_ACTIVATION'] = curr_OUTPUT_ACTIVATION
    curr_rs_ORCs['OPT_TUNE_IDX'] = curr_opt_ti
    
    # Save bootstrapping results as pickle
    curr_rs_ORCs.to_pickle(os.path.join(dropout_dir,'repeat'+str(REPEAT).zfill(2),curr_OUTPUT_ACTIVATION+'_dropout_resample_idx_'+str(curr_rs_idx).zfill(4)+'.pkl'))

if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)