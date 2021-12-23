#### Master Script #b: Compile eCPM_DeepMN and eCPM_DeepOR performance metrics and calculate confidence intervals ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Compile all eCPM_deep performance metrics
# III. Calculate confidence intervals on all eCPM performance metrics

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
from pandas.api.types import CategoricalDtype
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# TQDM for progress tracking
from tqdm import tqdm

# Custom analysis functions
from functions.analysis import collect_metrics

# Define directories in which eCPM performance metrics are saved
performance_dir = '../model_performance/eCPM'

# Define number of cores for parallel processing
NUM_CORES = multiprocessing.cpu_count()

### II. Compile all eCPM_deep performance metrics
# Search for all performance metric files in the eCPM_DeepMN directory
deepMN_metric_files = []
for path in Path(os.path.join(performance_dir,'deepMN')).rglob('*.csv'):
    deepMN_metric_files.append(str(path.resolve()))
    
# Search for all performance metric files in the eCPM_DeepOR directory
deepOR_metric_files = []
for path in Path(os.path.join(performance_dir,'deepOR')).rglob('*.csv'):
    deepOR_metric_files.append(str(path.resolve()))

# Concatenate lists of performance metric files
metric_files = deepMN_metric_files+deepOR_metric_files

# Characterise list of discovered performance metric files
metric_info_df = pd.DataFrame({'file':metric_files,
                               'MODEL':['eCPM_D'+re.search('eCPM/d(.*)/resample', curr_file).group(1) for curr_file in metric_files],
                               'RESAMPLE_IDX':[int(re.search('/resample(.*)/deep', curr_file).group(1)) for curr_file in metric_files],
                               'METRIC':[re.search('/deep_(.*).csv', curr_file).group(1) for curr_file in metric_files]
                              }).sort_values(by=['MODEL','RESAMPLE_IDX','METRIC']).reset_index(drop=True)

# Iterate through unique metric types and compile eCPM_deep results into a single dataframe
for curr_metric in metric_info_df.METRIC.unique():
    
    # Filter files of current metric
    curr_metric_info_df = metric_info_df[metric_info_df.METRIC == curr_metric].reset_index(drop=True)
    
    # Partition current metric files among cores
    s = [curr_metric_info_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
    s[:(curr_metric_info_df.shape[0] - sum(s))] = [over+1 for over in s[:(curr_metric_info_df.shape[0] - sum(s))]]    
    end_idx = np.cumsum(s)
    start_idx = np.insert(end_idx[:-1],0,0)

    # Collect current metric performance files in parallel
    curr_files_per_core = [(curr_metric_info_df.iloc[start_idx[idx]:end_idx[idx],:].reset_index(drop=True),True,'eCPM_deep metric extraction: '+curr_metric) for idx in range(len(start_idx))]
    with multiprocessing.Pool(NUM_CORES) as pool:
        compiled_curr_metric_values = pd.concat(pool.starmap(collect_metrics, curr_files_per_core),ignore_index=True)
    
    # Save compiled values of current metric type into model performance directory
    compiled_curr_metric_values.to_csv(os.path.join(performance_dir,'deep_'+curr_metric+'.csv'),index=False)

### III. Calculate confidence intervals on all eCPM performance metrics
## Threshold-level ROCs
# Load and compile ROCs
eCPM_compiled_ROCs = pd.concat([pd.read_csv(os.path.join(performance_dir,'deep_ROCs.csv')),pd.read_csv(os.path.join(performance_dir,'logreg_ROCs.csv'))],ignore_index=True)

# Calculate 95% confidence intervals
CI_eCPM_compiled_ROCs = eCPM_compiled_ROCs.groupby(['MODEL','Threshold','FPR'],as_index=False)['TPR'].aggregate({'TPR_mean':np.mean,'TPR_std':np.std,'TPR_median':np.median,'TPR_lo':lambda x: np.quantile(x,.025),'TPR_hi':lambda x: np.quantile(x,.975),'resamples':'count'}).reset_index(drop=True)
CI_eCPM_compiled_ROCs[['FPR','TPR_mean','TPR_std','TPR_median','TPR_lo','TPR_hi']] = CI_eCPM_compiled_ROCs[['FPR','TPR_mean','TPR_std','TPR_median','TPR_lo','TPR_hi']].clip(0,1)

# Save 95% confidence intervals for ROC
CI_eCPM_compiled_ROCs.to_csv(os.path.join(performance_dir,'CI_ROCs.csv'))

## Threshold-level calibration curves
# Load and compile calibration curves
eCPM_compiled_calibration = pd.concat([pd.read_csv(os.path.join(performance_dir,'deep_calibration.csv')),pd.read_csv(os.path.join(performance_dir,'logreg_calibration.csv'))],ignore_index=True)

# Calculate 95% confidence intervals
CI_eCPM_compiled_calibration = eCPM_compiled_calibration.groupby(['MODEL','Threshold','PredProb'],as_index=False)['TrueProb'].aggregate({'TrueProb_mean':np.mean,'TrueProb_std':np.std,'TrueProb_median':np.median,'TrueProb_lo':lambda x: np.quantile(x,.025),'TrueProb_hi':lambda x: np.quantile(x,.975),'resamples':'count'}).reset_index(drop=True)
CI_eCPM_compiled_calibration[['PredProb','TrueProb_mean','TrueProb_std','TrueProb_median','TrueProb_lo','TrueProb_hi']] = CI_eCPM_compiled_calibration[['PredProb','TrueProb_mean','TrueProb_std','TrueProb_median','TrueProb_lo','TrueProb_hi']].clip(0,1)

# Save 95% confidence intervals for calibration curves
CI_eCPM_compiled_calibration.to_csv(os.path.join(performance_dir,'CI_calibration.csv'))

## Classification confusion matrices
# Load and compile normalised confusion matrices
eCPM_compiled_cm = pd.concat([pd.read_csv(os.path.join(performance_dir,'deep_confusion_matrices.csv')),pd.read_csv(os.path.join(performance_dir,'logreg_confusion_matrices.csv'))],ignore_index=True)

# Calculate 95% confidence intervals
CI_eCPM_compiled_cm = eCPM_compiled_cm.groupby(['MODEL','TrueLabel','PredLabel'],as_index=False)['cm_prob'].aggregate({'cm_prob_mean':np.mean,'cm_prob_std':np.std,'cm_prob_median':np.median,'cm_prob_lo':lambda x: np.quantile(x,.025),'cm_prob_hi':lambda x: np.quantile(x,.975),'resamples':'count'}).reset_index(drop=True)
CI_eCPM_compiled_cm[['cm_prob_mean','cm_prob_std','cm_prob_median','cm_prob_lo','cm_prob_hi']] = CI_eCPM_compiled_cm[['cm_prob_mean','cm_prob_std','cm_prob_median','cm_prob_lo','cm_prob_hi']].clip(0,1)

# Save 95% confidence intervals for normalised confusion matrices
CI_eCPM_compiled_cm.to_csv(os.path.join(performance_dir,'CI_confusion_matrices.csv'))

## Overall performance metrics
# Load and compile overall performance metrics
eCPM_compiled_overall = pd.concat([pd.read_csv(os.path.join(performance_dir,'deep_overall_metrics.csv')),pd.read_csv(os.path.join(performance_dir,'logreg_overall_metrics.csv'))],ignore_index=True)

# Melt overall performance metric dataframe into long format
eCPM_compiled_overall = pd.melt(eCPM_compiled_overall,id_vars=['MODEL','RESAMPLE_IDX'],var_name='METRIC', value_name='VALUE')

# Calculate 95% confidence intervals for each metric
CI_eCPM_overall = eCPM_compiled_overall.groupby(['MODEL','METRIC'],as_index=False)['VALUE'].aggregate({'mean':np.mean,'std':np.std,'median':np.median,'lo':lambda x: np.quantile(x,.025),'hi':lambda x: np.quantile(x,.975),'resamples':'count'}).reset_index(drop=True)
CI_eCPM_overall.to_csv(os.path.join(performance_dir,'CI_overall_metrics.csv'),index=False)

## Threhsold-level performance metrics
# Load and compile threshold-level performance metrics
eCPM_compiled_threshold = pd.concat([pd.read_csv(os.path.join(performance_dir,'deep_threshold_metrics.csv')),pd.read_csv(os.path.join(performance_dir,'logreg_threshold_metrics.csv'))],ignore_index=True)

# Melt threshold-level performance metric dataframe into long format
eCPM_compiled_threshold = pd.melt(eCPM_compiled_threshold,id_vars=['MODEL','Threshold','RESAMPLE_IDX'],var_name='METRIC', value_name='VALUE')

# Calculate macro-averages for each threshold-level metric
macro_eCPM_compiled_threshold = eCPM_compiled_threshold.groupby(['MODEL','RESAMPLE_IDX','METRIC'],as_index=False)['VALUE'].mean()
macro_eCPM_compiled_threshold['Threshold'] = 'Average'

# Concatenate macr-averaged metrics to compiled threshold-level metric dataframe
eCPM_compiled_threshold = pd.concat([eCPM_compiled_threshold,macro_eCPM_compiled_threshold],ignore_index=True)

# Calculate 95% confidence intervals for each metric
CI_eCPM_threshold = eCPM_compiled_threshold.groupby(['MODEL','Threshold','METRIC'],as_index=False)['VALUE'].aggregate({'mean':np.mean,'std':np.std,'median':np.median,'lo':lambda x: np.quantile(x,.025),'hi':lambda x: np.quantile(x,.975),'resamples':'count'}).reset_index(drop=True)
CI_eCPM_threshold.to_csv(os.path.join(performance_dir,'CI_threshold_metrics.csv'),index=False)