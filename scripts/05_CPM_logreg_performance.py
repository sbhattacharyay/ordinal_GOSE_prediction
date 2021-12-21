#### Master Script 5: Assess CPM_MNLR and CPM_POLR performance ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Create bootstrapping resamples (that will be used for all model performance evaluation)
# III. Prepare compiled CPM_MNLR and CPM_POLR testing set predictions
# IV. Calculate and save performance metrics

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

# SciKit-Learn methods
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OneHotEncoder, StandardScaler, minmax_scale
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

# StatsModel methods
from statsmodels.nonparametric.smoothers_lowess import lowess

# TQDM for progress tracking
from tqdm import tqdm

# Custom methods
from functions.analysis import calc_bs_ORC, calc_bs_gen_c, calc_bs_thresh_AUC, calc_bs_cm, calc_bs_accuracy, calc_bs_thresh_accuracy, calc_bs_thresh_ROC, calc_bs_thresh_calibration

# Define version for assessment
VERSION = 'LOGREG_v1-0'
model_dir = '../CPM_outputs/'+VERSION

### II. Create bootstrapping resamples (that will be used for all model performance evaluation)
# Establish number of resamples for bootstrapping
NUM_RESAMP = 1000

# Establish number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count() - 2

# Create directory to store model performance results
os.makedirs('../model_performance',exist_ok=True)

# Load cross-validation information to get GOSE and GUPIs
cv_splits = pd.read_csv('../cross_validation_splits.csv')
study_GUPI_GOSE = cv_splits[['GUPI','GOSE']].drop_duplicates()

# Make stratified resamples for bootstrapping metrics
bs_rs_GUPIs = [resample(study_GUPI_GOSE.GUPI.values,replace=True,n_samples=study_GUPI_GOSE.shape[0],stratify=study_GUPI_GOSE.GOSE.values) for _ in range(NUM_RESAMP)]
bs_rs_GUPIs = [np.unique(curr_rs) for curr_rs in bs_rs_GUPIs]

# Create Data Frame to store bootstrapping resmaples 
bs_resamples = pd.DataFrame({'RESAMPLE_IDX':[i+1 for i in range(NUM_RESAMP)],'GUPIs':bs_rs_GUPIs})

# Save bootstrapping resample dataframe
bs_resamples.to_pickle('../model_performance/bs_resamples.pkl')

### III. Prepare compiled CPM_MNLR and CPM_POLR testing set predictions
# Load compiled CPM_MNLR and CPM_POLR testing set predictions
cpm_mnlr_test_preds = pd.read_csv(os.path.join(model_dir,'compiled_mnlr_test_predictions.csv'))
cpm_polr_test_preds = pd.read_csv(os.path.join(model_dir,'compiled_polr_test_predictions.csv'))

# Label and combine MNLR and POLR test set predictions for pre-evaluation processing
cpm_mnlr_test_preds['MODEL'] = 'CPM_MNLR'
cpm_polr_test_preds['MODEL'] = 'CPM_POLR'
cpm_test_preds = pd.concat([cpm_mnlr_test_preds,cpm_polr_test_preds],ignore_index=True)

# Encode outcome labels with python-compatible indexing
le = LabelEncoder()
cpm_test_preds['TrueLabel'] = le.fit_transform(cpm_test_preds['TrueLabel'])

# Predict highest level of functional recovery based on conservative decision rule
prob_labels = [col for col in cpm_test_preds if col.startswith('Pr(GOSE=')]
cpm_test_prob_matrix = cpm_test_preds[prob_labels]
cpm_test_prob_matrix.columns = [i for i in range(cpm_test_prob_matrix.shape[1])]
cpm_test_preds['PredLabel'] = (cpm_test_prob_matrix.cumsum(axis=1) > .5).idxmax(axis=1)

# Separate MNLR and POLR predictions again
cpm_mnlr_test_preds = cpm_test_preds[cpm_test_preds['MODEL'] == 'CPM_MNLR'].reset_index(drop=True)
cpm_polr_test_preds = cpm_test_preds[cpm_test_preds['MODEL'] == 'CPM_POLR'].reset_index(drop=True)

# Partition resamples among cores for parallel calculations
sizes = [NUM_RESAMP // NUM_CORES for _ in range(NUM_CORES)]
sizes[:(NUM_RESAMP - sum(sizes))] = [val+1 for val in sizes[:(NUM_RESAMP - sum(sizes))]]    
end_indices = np.cumsum(sizes)
start_indices = np.insert(end_indices[:-1],0,0)

cpm_mnlr_resamples_per_core = [(bs_resamples.iloc[start_indices[idx]:end_indices[idx],:].reset_index(drop=True),cpm_mnlr_test_preds,True,'CPM_MNLR') for idx in range(len(start_indices))]
cpm_polr_resamples_per_core = [(bs_resamples.iloc[start_indices[idx]:end_indices[idx],:].reset_index(drop=True),cpm_polr_test_preds,True,'CPM_POLR') for idx in range(len(start_indices))]

### IV. Calculate and save performance metrics
# Create subdirectory to store CPM metrics
os.makedirs('../model_performance/CPM',exist_ok=True)

# Calculate ordinal c-index (ORC)
with multiprocessing.Pool(NUM_CORES) as pool:
    cpm_mnlr_orc = pd.concat(pool.starmap(calc_bs_ORC, cpm_mnlr_resamples_per_core),ignore_index=True)
cpm_mnlr_orc['MODEL'] = 'CPM_MNLR'

with multiprocessing.Pool(NUM_CORES) as pool:
    cpm_polr_orc = pd.concat(pool.starmap(calc_bs_ORC, cpm_polr_resamples_per_core),ignore_index=True)
cpm_polr_orc['MODEL'] = 'CPM_POLR'

# Calculate generalised c-index
with multiprocessing.Pool(NUM_CORES) as pool:
    cpm_mnlr_gen_c = pd.concat(pool.starmap(calc_bs_gen_c, cpm_mnlr_resamples_per_core),ignore_index=True)
cpm_mnlr_gen_c['MODEL'] = 'CPM_MNLR'

with multiprocessing.Pool(NUM_CORES) as pool:
    cpm_polr_gen_c = pd.concat(pool.starmap(calc_bs_gen_c, cpm_polr_resamples_per_core),ignore_index=True)
cpm_polr_gen_c['MODEL'] = 'CPM_POLR'

# Calculate overall accuracy
with multiprocessing.Pool(NUM_CORES) as pool:
    cpm_mnlr_accuracy = pd.concat(pool.starmap(calc_bs_accuracy, cpm_mnlr_resamples_per_core),ignore_index=True)
cpm_mnlr_accuracy['MODEL'] = 'CPM_MNLR'

with multiprocessing.Pool(NUM_CORES) as pool:
    cpm_polr_accuracy = pd.concat(pool.starmap(calc_bs_accuracy, cpm_polr_resamples_per_core),ignore_index=True)
cpm_polr_accuracy['MODEL'] = 'CPM_POLR'

# Compile overall metrics across model types
cpm_logreg_orc = pd.concat([cpm_mnlr_orc,cpm_polr_orc],ignore_index=True)
cpm_logreg_gen_c = pd.concat([cpm_mnlr_gen_c,cpm_polr_gen_c],ignore_index=True)
cpm_logreg_accuracy = pd.concat([cpm_mnlr_accuracy,cpm_polr_accuracy],ignore_index=True)

# Merge overall metric dataframes
cpm_logreg_om = pd.merge(cpm_logreg_orc,cpm_logreg_gen_c,how='left',on=['MODEL','RESAMPLE_IDX']).merge(cpm_logreg_accuracy,how='left',on=['MODEL','RESAMPLE_IDX'])
cpm_logreg_om = cpm_logreg_om[['MODEL','RESAMPLE_IDX','ORC','S','Gen_C','D_xy','Accuracy']]

# Save overall metric dataframe
cpm_logreg_om.to_csv('../model_performance/CPM/logreg_overall_metrics.csv',index=False)

# Calculate dichotomous c-indices
with multiprocessing.Pool(NUM_CORES) as pool:
    cpm_mnlr_thresh_AUC = pd.concat(pool.starmap(calc_bs_thresh_AUC, cpm_mnlr_resamples_per_core),ignore_index=True)
cpm_mnlr_thresh_AUC['MODEL'] = 'CPM_MNLR'

with multiprocessing.Pool(NUM_CORES) as pool:
    cpm_polr_thresh_AUC = pd.concat(pool.starmap(calc_bs_thresh_AUC, cpm_polr_resamples_per_core),ignore_index=True)
cpm_polr_thresh_AUC['MODEL'] = 'CPM_POLR'

# Calculate accuracy at each threshold
with multiprocessing.Pool(NUM_CORES) as pool:
    cpm_mnlr_thresh_accuracy = pd.concat(pool.starmap(calc_bs_thresh_accuracy, cpm_mnlr_resamples_per_core),ignore_index=True)
cpm_mnlr_thresh_accuracy['MODEL'] = 'CPM_MNLR'

with multiprocessing.Pool(NUM_CORES) as pool:
    cpm_polr_thresh_accuracy = pd.concat(pool.starmap(calc_bs_thresh_accuracy, cpm_polr_resamples_per_core),ignore_index=True)
cpm_polr_thresh_accuracy['MODEL'] = 'CPM_POLR'

# Calculate calibration curves at each threshold
with multiprocessing.Pool(NUM_CORES) as pool:
    cpm_mnlr_calibration = pd.concat(pool.starmap(calc_bs_thresh_calibration, cpm_mnlr_resamples_per_core),ignore_index=True)
cpm_mnlr_calibration['MODEL'] = 'CPM_MNLR'

with multiprocessing.Pool(NUM_CORES) as pool:
    cpm_polr_calibration = pd.concat(pool.starmap(calc_bs_thresh_calibration, cpm_polr_resamples_per_core),ignore_index=True)
cpm_polr_calibration['MODEL'] = 'CPM_POLR'

# Compile calibration curves across model types and save
cpm_logreg_calibration = pd.concat([cpm_mnlr_calibration,cpm_polr_calibration],ignore_index=True)
cpm_logreg_calibration.to_csv('../model_performance/CPM/logreg_calibration.csv',index=False)

# Calculate calibration metrics at each threshold
cpm_mnlr_calibration['abs_diff'] = (cpm_mnlr_calibration['PredProb'] - cpm_mnlr_calibration['TrueProb']).abs()
cpm_mnlr_calib_metrics = cpm_mnlr_calibration.groupby(['MODEL','Threshold','RESAMPLE_IDX'],as_index=False)['abs_diff'].aggregate({'ICI':'mean','Emax':'max','E50':np.median,'E90':lambda x: np.quantile(x,.9)}).reset_index(drop=True)

cpm_polr_calibration['abs_diff'] = (cpm_polr_calibration['PredProb'] - cpm_polr_calibration['TrueProb']).abs()
cpm_polr_calib_metrics = cpm_polr_calibration.groupby(['MODEL','Threshold','RESAMPLE_IDX'],as_index=False)['abs_diff'].aggregate({'ICI':'mean','Emax':'max','E50':np.median,'E90':lambda x: np.quantile(x,.9)}).reset_index(drop=True)

# Compile threshold-level metrics across model types
cpm_logreg_thresh_AUC = pd.concat([cpm_mnlr_thresh_AUC,cpm_polr_thresh_AUC],ignore_index=True)
cpm_logreg_thresh_accuracy = pd.concat([cpm_mnlr_thresh_accuracy,cpm_polr_thresh_accuracy],ignore_index=True)
cpm_logreg_calib_metrics = pd.concat([cpm_mnlr_calib_metrics,cpm_polr_calib_metrics],ignore_index=True)

# Merge threshold-level metric dataframes
cpm_logreg_tlm = pd.merge(cpm_logreg_thresh_AUC,cpm_logreg_thresh_accuracy,how='left',on=['MODEL','RESAMPLE_IDX','Threshold']).merge(cpm_logreg_calib_metrics,how='left',on=['MODEL','RESAMPLE_IDX','Threshold'])
cpm_logreg_tlm = cpm_logreg_tlm[['MODEL','RESAMPLE_IDX','Threshold','AUC','Accuracy','ICI','Emax','E50','E90']]

# Save threshold-level metric dataframe
cpm_logreg_tlm.to_csv('../model_performance/CPM/logreg_threshold_metrics.csv',index=False)

# Calculate normalised confusion matrices
with multiprocessing.Pool(NUM_CORES) as pool:
    cpm_mnlr_cm = pd.concat(pool.starmap(calc_bs_cm, cpm_mnlr_resamples_per_core),ignore_index=True)
cpm_mnlr_cm['MODEL'] = 'CPM_MNLR'

with multiprocessing.Pool(NUM_CORES) as pool:
    cpm_polr_cm = pd.concat(pool.starmap(calc_bs_cm, cpm_polr_resamples_per_core),ignore_index=True)
cpm_polr_cm['MODEL'] = 'CPM_POLR'

# Compile normalised confusion matrices across model types and save
cpm_logreg_cm = pd.concat([cpm_mnlr_cm,cpm_polr_cm],ignore_index=True)
cpm_logreg_cm.to_csv('../model_performance/CPM/logreg_confusion_matrices.csv',index=False)

# Calculate ROC at each threshold
with multiprocessing.Pool(NUM_CORES) as pool:
    cpm_mnlr_thresh_ROC = pd.concat(pool.starmap(calc_bs_thresh_ROC, cpm_mnlr_resamples_per_core),ignore_index=True)
cpm_mnlr_thresh_ROC['MODEL'] = 'CPM_MNLR'

with multiprocessing.Pool(NUM_CORES) as pool:
    cpm_polr_thresh_ROC = pd.concat(pool.starmap(calc_bs_thresh_ROC, cpm_polr_resamples_per_core),ignore_index=True)
cpm_polr_thresh_ROC['MODEL'] = 'CPM_POLR'

# Compile ROCs across model types and save
cpm_logreg_thresh_ROC = pd.concat([cpm_mnlr_thresh_ROC,cpm_polr_thresh_ROC],ignore_index=True)
cpm_logreg_thresh_ROC.to_csv('../model_performance/CPM/logreg_ROCs.csv',index=False)