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
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

# StatsModel methods
from statsmodels.nonparametric.smoothers_lowess import lowess

# TQDM for progress tracking
from tqdm import tqdm

# Function to load and compile test prediction files
def collect_preds(pred_file_info,progress_bar = True, progress_bar_desc = ''):
    output_df = []
    
    if progress_bar:
        iterator = tqdm(range(pred_file_info.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(pred_file_info.shape[0])
    
    for i in iterator:
        curr_pred = pd.read_csv(pred_file_info.file[i])
        curr_pred['repeat'] = pred_file_info.repeat[i]
        curr_pred['fold'] = pred_file_info.fold[i]
        output_df.append(curr_pred)
    return pd.concat(output_df,ignore_index=True)

# Function to load and compile test performance metrics for DeepIMPACT models
def collect_metrics(metric_file_info,progress_bar = True, progress_bar_desc = ''):
    output_df = []
    
    if progress_bar:
        iterator = tqdm(metric_file_info.file,desc=progress_bar_desc)
    else:
        iterator = metric_file_info.file
    
    return pd.concat([pd.read_csv(f) for f in iterator],ignore_index=True)

# Function to calculate ordinal c-index via bootstrapping
def calc_bs_ORC(curr_resamples, compiled_test_preds, progress_bar = True, progress_bar_desc = ''):
    
    if progress_bar:
        iterator = tqdm(range(curr_resamples.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(curr_resamples.shape[0])
    
    num_classes = len(compiled_test_preds.TrueLabel.unique())
    
    compiled_orc = []
    compiled_steps = []
    compiled_rs_idx = []
    
    for curr_rs_row in iterator:
        
        compiled_rs_idx.append(curr_resamples.RESAMPLE_IDX[curr_rs_row])
        
        curr_in_sample = curr_resamples.GUPIs[curr_rs_row]
        
        curr_rs_preds = compiled_test_preds[compiled_test_preds.GUPI.isin(curr_in_sample)].reset_index(drop=True)
            
        prob_cols = [col for col in curr_rs_preds if col.startswith('Pr(GOSE=')]
        
        pairs = []
        
        aucs = []
        
        for ix, (a, b) in enumerate(itertools.combinations(np.sort(curr_rs_preds.TrueLabel.unique()), 2)):
            
            filt_rs_preds = curr_rs_preds[curr_rs_preds.TrueLabel.isin([a,b])].reset_index(drop=True)
            filt_rs_preds['ConditProb'] = filt_rs_preds[prob_cols[b]]/(filt_rs_preds[prob_cols[a]] + filt_rs_preds[prob_cols[b]])
            filt_rs_preds['ConditLabel'] = (filt_rs_preds.TrueLabel == b).astype(int)
            
            aucs.append(roc_auc_score(filt_rs_preds['ConditLabel'],filt_rs_preds['ConditProb']))
            pairs.append((a,b))
        
        compiled_orc.append(np.mean(aucs))
        compiled_steps.append((1 - np.mean(aucs))*(num_classes*(num_classes-1)/2))
        
    return pd.DataFrame({'RESAMPLE_IDX':compiled_rs_idx,'ORC':compiled_orc,'S':compiled_steps})

# Function to calculate generalised c-index via bootstrapping
def calc_bs_gen_c(curr_resamples, compiled_test_preds, progress_bar = True, progress_bar_desc = ''):
    
    if progress_bar:
        iterator = tqdm(range(curr_resamples.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(curr_resamples.shape[0])
    
    num_classes = len(compiled_test_preds.TrueLabel.unique())
    
    compiled_gen_c = []
    compiled_D = []
    compiled_rs_idx = []
    
    for curr_rs_row in iterator:
        
        compiled_rs_idx.append(curr_resamples.RESAMPLE_IDX[curr_rs_row])
        
        curr_in_sample = curr_resamples.GUPIs[curr_rs_row]
        
        curr_rs_preds = compiled_test_preds[compiled_test_preds.GUPI.isin(curr_in_sample)].reset_index(drop=True)
            
        prob_cols = [col for col in curr_rs_preds if col.startswith('Pr(GOSE=')]
        
        pairs = []
        
        aucs = []
        
        prevalence = []

        for ix, (a, b) in enumerate(itertools.combinations(np.sort(curr_rs_preds.TrueLabel.unique()), 2)):
            
            filt_rs_preds = curr_rs_preds[curr_rs_preds.TrueLabel.isin([a,b])].reset_index(drop=True)
            filt_rs_preds['ConditProb'] = filt_rs_preds[prob_cols[b]]/(filt_rs_preds[prob_cols[a]] + filt_rs_preds[prob_cols[b]])
            filt_rs_preds['ConditLabel'] = (filt_rs_preds.TrueLabel == b).astype(int)
            
            prevalence.append((filt_rs_preds.TrueLabel == a).sum()*(filt_rs_preds.TrueLabel == b).sum())
            aucs.append(roc_auc_score(filt_rs_preds['ConditLabel'],filt_rs_preds['ConditProb']))
            pairs.append((a,b))
        
        compiled_gen_c.append(np.sum(np.multiply(aucs,prevalence))/np.sum(prevalence))
        compiled_D.append(2*(np.sum(np.multiply(aucs,prevalence))/np.sum(prevalence))-1)
        
    return pd.DataFrame({'RESAMPLE_IDX':compiled_rs_idx,'Gen_C':compiled_gen_c,'D_xy':compiled_D})

# Function to calculate threshold-level AUROCs
def calc_bs_thresh_AUC(curr_resamples, compiled_test_preds, progress_bar = True, progress_bar_desc = ''):
    
    thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
    
    compiled_AUCs = []
    
    if progress_bar:
        iterator = tqdm(range(curr_resamples.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(curr_resamples.shape[0])
        
    for curr_rs_row in iterator:
                
        curr_in_sample = curr_resamples.GUPIs[curr_rs_row]
        
        curr_rs_preds = compiled_test_preds[compiled_test_preds.GUPI.isin(curr_in_sample)].reset_index(drop=True)
            
        prob_cols = [col for col in curr_rs_preds if col.startswith('Pr(GOSE=')]

        for thresh in range(1,len(prob_cols)):

            cols_gt = prob_cols[thresh:]

            prob_gt = curr_rs_preds[cols_gt].sum(1).values

            gt = (curr_rs_preds['TrueLabel'] >= thresh).astype(int).values

            curr_AUC = roc_auc_score(gt, prob_gt)

            compiled_AUCs.append(pd.DataFrame({'RESAMPLE_IDX':[curr_resamples.RESAMPLE_IDX[curr_rs_row]],'Threshold':thresh_labels[thresh-1],'AUC':curr_AUC},index=[0]))

    return pd.concat(compiled_AUCs,ignore_index = True)

# Function to calculate normalized confusion matrices
def calc_bs_cm(curr_resamples, compiled_test_preds, progress_bar = True, progress_bar_desc = ''):

    compiled_cm = []
    
    if progress_bar:
        iterator = tqdm(range(curr_resamples.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(curr_resamples.shape[0])
        
    for curr_rs_row in iterator:
                
        curr_in_sample = curr_resamples.GUPIs[curr_rs_row]
        
        curr_rs_preds = compiled_test_preds[compiled_test_preds.GUPI.isin(curr_in_sample)].reset_index(drop=True)
                
        prob_cols = [col for col in curr_rs_preds if col.startswith('Pr(GOSE=')]
                
        curr_rs_cm = confusion_matrix(curr_rs_preds.TrueLabel, curr_rs_preds.PredLabel,normalize='true')
        
        curr_rs_cm = pd.DataFrame(curr_rs_cm)
        curr_rs_cm.columns = ['GOSE: 1','GOSE: 2/3','GOSE: 4','GOSE: 5','GOSE: 6','GOSE: 7','GOSE: 8']
        curr_rs_cm = curr_rs_cm.assign(TrueLabel=['GOSE: 1','GOSE: 2/3','GOSE: 4','GOSE: 5','GOSE: 6','GOSE: 7','GOSE: 8'])
        curr_rs_cm = curr_rs_cm.melt(id_vars=['TrueLabel'],var_name='PredLabel',value_name='cm_prob')
        
        curr_rs_cm['RESAMPLE_IDX'] = curr_resamples.RESAMPLE_IDX[curr_rs_row]
        
        compiled_cm.append(curr_rs_cm)
        
    return pd.concat(compiled_cm,ignore_index = True)

# Function to calculate accuracy
def calc_bs_accuracy(curr_resamples, compiled_test_preds, progress_bar = True, progress_bar_desc = ''):
    
    if progress_bar:
        iterator = tqdm(range(curr_resamples.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(curr_resamples.shape[0])
    
    compiled_accuracy = []
    compiled_rs_idx = []
    
    for curr_rs_row in iterator:
        
        compiled_rs_idx.append(curr_resamples.RESAMPLE_IDX[curr_rs_row])

        curr_in_sample = curr_resamples.GUPIs[curr_rs_row]
        
        curr_rs_preds = compiled_test_preds[compiled_test_preds.GUPI.isin(curr_in_sample)].reset_index(drop=True)
                
        prob_cols = [col for col in curr_rs_preds if col.startswith('Pr(GOSE=')]
                
        compiled_accuracy.append(accuracy_score(curr_rs_preds.TrueLabel, curr_rs_preds.PredLabel))
        
    return pd.DataFrame({'RESAMPLE_IDX':compiled_rs_idx,'Accuracy':compiled_accuracy})

# Function to calculate threshold-level accuracy
def calc_bs_thresh_accuracy(curr_resamples, compiled_test_preds, progress_bar = True, progress_bar_desc = ''):
    
    thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
    
    compiled_accuracies = []
    
    if progress_bar:
        iterator = tqdm(range(curr_resamples.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(curr_resamples.shape[0])
        
    for curr_rs_row in iterator:
                
        curr_in_sample = curr_resamples.GUPIs[curr_rs_row]
        
        curr_rs_preds = compiled_test_preds[compiled_test_preds.GUPI.isin(curr_in_sample)].reset_index(drop=True)
            
        prob_cols = [col for col in curr_rs_preds if col.startswith('Pr(GOSE=')]

        for thresh in range(1,len(prob_cols)):

            cols_gt = prob_cols[thresh:]

            prob_gt = curr_rs_preds[cols_gt].sum(1).values
            
            pred_gt = (prob_gt > .5).astype(int)

            gt = (curr_rs_preds['TrueLabel'] >= thresh).astype(int).values

            curr_accuracy = accuracy_score(gt, pred_gt)

            compiled_accuracies.append(pd.DataFrame({'RESAMPLE_IDX':[curr_resamples.RESAMPLE_IDX[curr_rs_row]],'Threshold':thresh_labels[thresh-1],'Accuracy':curr_accuracy},index=[0]))

    return pd.concat(compiled_accuracies,ignore_index = True)

# Function to calculate threshold-level ROC
def calc_bs_thresh_ROC(curr_resamples, compiled_test_preds, progress_bar = True, progress_bar_desc = ''):
    
    thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
    
    compiled_ROCs = []
    
    if progress_bar:
        iterator = tqdm(range(curr_resamples.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(curr_resamples.shape[0])
        
    for curr_rs_row in iterator:
                
        curr_in_sample = curr_resamples.GUPIs[curr_rs_row]
        
        curr_rs_preds = compiled_test_preds[compiled_test_preds.GUPI.isin(curr_in_sample)].reset_index(drop=True)
            
        prob_cols = [col for col in curr_rs_preds if col.startswith('Pr(GOSE=')]
        
        for thresh in range(1,len(prob_cols)):
            
            cols_gt = prob_cols[thresh:]
            
            prob_gt = curr_rs_preds[cols_gt].sum(1).values
            
            gt = (curr_rs_preds['TrueLabel'] >= thresh).astype(int).values
            
            (fpr, tpr, _) = roc_curve(gt, prob_gt)
            
            interp_tpr = np.interp(np.linspace(0,1,200),fpr,tpr)
            
            roc_axes = pd.DataFrame({'RESAMPLE_IDX':curr_resamples.RESAMPLE_IDX[curr_rs_row],'Threshold':thresh_labels[thresh-1],'FPR':np.linspace(0,1,200),'TPR':interp_tpr})
                        
            compiled_ROCs.append(roc_axes)
                
    return pd.concat(compiled_ROCs,ignore_index = True)

# Function to calculate median predicted rank of correct class
def calculate_bs_med_rank(rs_GUPIs, compiled_test_preds, bbc_cv=False, progress_bar = True, progress_bar_desc = ''):
    
    if progress_bar:
        iterator = tqdm(rs_GUPIs,desc=progress_bar_desc)
    else:
        iterator = rs_GUPIs
    
    compiled_med_rank = []
        
    for curr_rs in iterator:
        
        curr_rs_preds = compiled_test_preds[compiled_test_preds.GUPI.isin(curr_rs)]
        
        prob_cols = [col for col in curr_rs_preds if col.startswith('Pr(GOSE=')]
        
        class_ranks = np.argsort(-curr_rs_preds[prob_cols].values,axis=-1)

        true_labels = curr_rs_preds['TrueLabel'].values
        
        ranks = []
        for i in range(class_ranks.shape[0]):
            curr_true_label = true_labels[i]
            ranks.append(np.where(class_ranks[i,:] == curr_true_label)[0][0] + 1)
        
        compiled_med_rank.append(int(np.median(ranks)))
        
    return compiled_med_rank

# Function to calculate threshold-level calibration curves
def calc_bs_thresh_calibration(curr_resamples, compiled_test_preds, progress_bar = True, progress_bar_desc = ''):
    
    thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
    
    compiled_calibs = []
    
    if progress_bar:
        iterator = tqdm(range(curr_resamples.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(curr_resamples.shape[0])
        
    for curr_rs_row in iterator:
                
        curr_in_sample = curr_resamples.GUPIs[curr_rs_row]
        
        curr_rs_preds = compiled_test_preds[compiled_test_preds.GUPI.isin(curr_in_sample)].reset_index(drop=True)
        
        prob_cols = [col for col in curr_rs_preds if col.startswith('Pr(GOSE')]
        
        for thresh in range(1,len(prob_cols)):
            
            cols_gt = prob_cols[thresh:]
            
            prob_gt = curr_rs_preds[cols_gt].sum(1).values
            
            gt = (curr_rs_preds['TrueLabel'] >= thresh).astype(int).values
            
            TrueProb = lowess(endog = gt, exog = prob_gt, it = 0, xvals = np.linspace(0,1,200))
                                    
            calib_axes = pd.DataFrame({'RESAMPLE_IDX':curr_resamples.RESAMPLE_IDX[curr_rs_row],'Threshold':thresh_labels[thresh-1],'PredProb':np.linspace(0,1,200),'TrueProb':TrueProb})
                        
            compiled_calibs.append(calib_axes)
                
    return pd.concat(compiled_calibs,ignore_index = True)