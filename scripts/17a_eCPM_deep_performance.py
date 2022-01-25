#### Master Script 17a: Assess eCPM_DeepMN and eCPM_DeepOR performance ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Calculate perfomance metrics on resamples

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
from scipy.special import logit
from argparse import ArgumentParser
from pandas.api.types import CategoricalDtype
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings(action="ignore")

# SciKit-Learn methods
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, recall_score
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer, OneHotEncoder, StandardScaler, minmax_scale
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

# StatsModel methods
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant

# TQDM for progress tracking
from tqdm import tqdm

# Load bootstrapping resamples used across model performance evaluations
bs_resamples = pd.read_pickle('../model_performance/bs_resamples.pkl')

# Expand resampling information by the two model choices
uniq_models = pd.DataFrame({'MODEL':['deepMN','deepOR'],'key':1})
bs_resamples['key'] = 1
rs_model_combos = pd.merge(bs_resamples,uniq_models,how='outer',on='key').drop(columns='key')

# Define model version code
VERSION = 'DEEP_v1-0'
model_dir = '/home/sb2406/rds/hpc-work/eCPM_outputs/'+VERSION

### II. Calculate perfomance metrics on resamples
# Define metric calculation function
def main(array_task_id):
    
    # Get resample information for current trial
    curr_gupis = rs_model_combos.GUPIs[array_task_id]
    curr_model = rs_model_combos.MODEL[array_task_id]
    curr_rs_idx = rs_model_combos.RESAMPLE_IDX[array_task_id]
    
    # Create directory to save current combination outputs
    metric_dir = os.path.join('../model_performance','eCPM',curr_model,'resample'+str(curr_rs_idx).zfill(4))
    os.makedirs(metric_dir,exist_ok=True)
    
    # Load compiled testing set predictions
    compiled_test_preds = pd.read_csv(os.path.join(model_dir,'eCPM_'+curr_model+'_compiled_test_predictions.csv'),index_col=0)
    compiled_test_preds['TUNE_IDX'] = compiled_test_preds['TUNE_IDX'].astype(str).str.zfill(4)
    
    if curr_model == 'deepMN':
        
        prob_cols = [col for col in compiled_test_preds if col.startswith('Pr(GOSE=')]
        thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
                    
        for thresh in range(1,len(prob_cols)):

            cols_gt = prob_cols[thresh:]
            prob_gt = compiled_test_preds[cols_gt].sum(1).values
            gt = (compiled_test_preds['TrueLabel'] >= thresh).astype(int).values
            
            compiled_test_preds['Pr('+thresh_labels[thresh-1]+')'] = prob_gt
            compiled_test_preds[thresh_labels[thresh-1]] = gt
            
    if curr_model == 'deepOR':
        
        # Add 'TrueLabel' variable
        old_label_cols = [col for col in compiled_test_preds if col.startswith('GOSE>')]
        compiled_test_preds['TrueLabel'] = compiled_test_preds[old_label_cols].sum(1).astype(int)
        
        # Calculate class-specific probabilities
        old_prob_labels = [col for col in compiled_test_preds if col.startswith('Pr(GOSE>')]
        new_prob_labels = ['Pr(GOSE=1)','Pr(GOSE=2/3)','Pr(GOSE=4)','Pr(GOSE=5)','Pr(GOSE=6)','Pr(GOSE=7)','Pr(GOSE=8)']
        prob_matrix = compiled_test_preds[old_prob_labels]
        
        for i in range(len(new_prob_labels)):
            if i == 0:
                compiled_test_preds[new_prob_labels[i]] = 1 - prob_matrix.iloc[:,0]
            elif i == (len(new_prob_labels)-1):
                compiled_test_preds[new_prob_labels[i]] = prob_matrix.iloc[:,i-1]
            else:
                compiled_test_preds[new_prob_labels[i]] = prob_matrix.iloc[:,i-1] - prob_matrix.iloc[:,i]
                
    # Predict highest level of functional recovery based on conservative decision rule
    prob_labels = [col for col in compiled_test_preds if col.startswith('Pr(GOSE=')]
    test_prob_matrix = compiled_test_preds[prob_labels]
    test_prob_matrix.columns = [i for i in range(test_prob_matrix.shape[1])]
    compiled_test_preds['PredLabel'] = (test_prob_matrix.cumsum(axis=1) > .5).idxmax(axis=1)

    ### Separate in- and out-sample predictions
    curr_is_preds = compiled_test_preds[compiled_test_preds.GUPI.isin(curr_gupis)].reset_index(drop=True)
    curr_os_preds = compiled_test_preds[~compiled_test_preds.GUPI.isin(curr_gupis)].reset_index(drop=True)

    ### ORC and S-index            
    best_is_orc = 0

    for curr_ti in curr_is_preds.TUNE_IDX.unique():

        curr_ti_preds = curr_is_preds[curr_is_preds.TUNE_IDX == curr_ti].reset_index(drop=True)
        
        aucs = []
        for ix, (a, b) in enumerate(itertools.combinations(np.sort(curr_ti_preds.TrueLabel.unique()), 2)):
            filt_rs_preds = curr_ti_preds[curr_ti_preds.TrueLabel.isin([a,b])].reset_index(drop=True)
            filt_rs_preds['ConditProb'] = filt_rs_preds[prob_labels[b]]/(filt_rs_preds[prob_labels[a]] + filt_rs_preds[prob_labels[b]])
            filt_rs_preds['ConditProb'] = np.nan_to_num(filt_rs_preds['ConditProb'],nan=.5,posinf=1,neginf=0)
            filt_rs_preds['ConditLabel'] = (filt_rs_preds.TrueLabel == b).astype(int)
            aucs.append(roc_auc_score(filt_rs_preds['ConditLabel'],filt_rs_preds['ConditProb']))
        curr_orc = np.mean(aucs)
        
        if curr_orc > best_is_orc:

            opt_ti = curr_ti
            best_is_orc = curr_orc

    curr_os_ti_preds = curr_os_preds[curr_os_preds.TUNE_IDX == opt_ti].reset_index(drop=True)
    aucs = []
    for ix, (a, b) in enumerate(itertools.combinations(np.sort(curr_os_ti_preds.TrueLabel.unique()), 2)):
        filt_rs_preds = curr_os_ti_preds[curr_os_ti_preds.TrueLabel.isin([a,b])].reset_index(drop=True)
        filt_rs_preds['ConditProb'] = filt_rs_preds[prob_labels[b]]/(filt_rs_preds[prob_labels[a]] + filt_rs_preds[prob_labels[b]])
        filt_rs_preds['ConditProb'] = np.nan_to_num(filt_rs_preds['ConditProb'],nan=.5,posinf=1,neginf=0)
        filt_rs_preds['ConditLabel'] = (filt_rs_preds.TrueLabel == b).astype(int)
        aucs.append(roc_auc_score(filt_rs_preds['ConditLabel'],filt_rs_preds['ConditProb']))
    final_orc = np.mean(aucs)
    num_classes = len(curr_os_ti_preds.TrueLabel.unique())
    final_steps = (1 - final_orc)*(num_classes*(num_classes-1)/2)
    
    ### Generalised c-index and Sommer's D_xy
    best_is_gen_c = 0

    for curr_ti in curr_is_preds.TUNE_IDX.unique():

        curr_ti_preds = curr_is_preds[curr_is_preds.TUNE_IDX == curr_ti].reset_index(drop=True)
        
        aucs = []
        prevalence = []
        for ix, (a, b) in enumerate(itertools.combinations(np.sort(curr_ti_preds.TrueLabel.unique()), 2)):
            filt_rs_preds = curr_ti_preds[curr_ti_preds.TrueLabel.isin([a,b])].reset_index(drop=True)
            filt_rs_preds['ConditProb'] = filt_rs_preds[prob_labels[b]]/(filt_rs_preds[prob_labels[a]] + filt_rs_preds[prob_labels[b]])
            filt_rs_preds['ConditProb'] = np.nan_to_num(filt_rs_preds['ConditProb'],nan=.5,posinf=1,neginf=0)
            filt_rs_preds['ConditLabel'] = (filt_rs_preds.TrueLabel == b).astype(int)
            prevalence.append((filt_rs_preds.TrueLabel == a).sum()*(filt_rs_preds.TrueLabel == b).sum())
            aucs.append(roc_auc_score(filt_rs_preds['ConditLabel'],filt_rs_preds['ConditProb']))
        curr_gen_c = np.sum(np.multiply(aucs,prevalence))/np.sum(prevalence)
        
        if curr_gen_c > best_is_gen_c:

            opt_ti = curr_ti
            best_is_gen_c = curr_gen_c

    curr_os_ti_preds = curr_os_preds[curr_os_preds.TUNE_IDX == opt_ti].reset_index(drop=True)
    aucs = []
    prevalence = []
    for ix, (a, b) in enumerate(itertools.combinations(np.sort(curr_os_ti_preds.TrueLabel.unique()), 2)):
        filt_rs_preds = curr_os_ti_preds[curr_os_ti_preds.TrueLabel.isin([a,b])].reset_index(drop=True)
        filt_rs_preds['ConditProb'] = filt_rs_preds[prob_labels[b]]/(filt_rs_preds[prob_labels[a]] + filt_rs_preds[prob_labels[b]])
        filt_rs_preds['ConditProb'] = np.nan_to_num(filt_rs_preds['ConditProb'],nan=.5,posinf=1,neginf=0)
        filt_rs_preds['ConditLabel'] = (filt_rs_preds.TrueLabel == b).astype(int)
        prevalence.append((filt_rs_preds.TrueLabel == a).sum()*(filt_rs_preds.TrueLabel == b).sum())
        aucs.append(roc_auc_score(filt_rs_preds['ConditLabel'],filt_rs_preds['ConditProb']))
    final_gen_c = np.sum(np.multiply(aucs,prevalence))/np.sum(prevalence)
    final_D_xy = 2*(final_gen_c)-1
    
    ### Accuracy
    best_is_accuracy = 0

    for curr_ti in curr_is_preds.TUNE_IDX.unique():

        curr_ti_preds = curr_is_preds[curr_is_preds.TUNE_IDX == curr_ti].reset_index(drop=True)
        curr_accuracy = accuracy_score(curr_ti_preds.TrueLabel, curr_ti_preds.PredLabel)
        
        if curr_accuracy > best_is_accuracy:
            opt_ti = curr_ti
            best_is_accuracy = curr_accuracy

    curr_os_ti_preds = curr_os_preds[curr_os_preds.TUNE_IDX == opt_ti].reset_index(drop=True)
    final_accuracy = accuracy_score(curr_os_ti_preds.TrueLabel, curr_os_ti_preds.PredLabel)
    
    ### Compile overall metrics into a single dataframe
    pd.DataFrame({'MODEL':'eCPM_D'+curr_model[1:],'RESAMPLE_IDX':curr_rs_idx,'ORC':final_orc,'S':final_steps,'Gen_C':final_gen_c,'D_xy':final_D_xy,'Accuracy':final_accuracy},index=[0]).to_csv(os.path.join(metric_dir,'deep_overall_metrics.csv'),index=False)
    
    ### Threshold-level AUC and ROC
    thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
    best_is_auc = 0

    for curr_ti in curr_is_preds.TUNE_IDX.unique():

        curr_ti_preds = curr_is_preds[curr_is_preds.TUNE_IDX == curr_ti].reset_index(drop=True)
            
        thresh_prob_labels = [col for col in curr_ti_preds if col.startswith('Pr(GOSE>')]
        curr_auc = roc_auc_score(curr_ti_preds[thresh_labels],curr_ti_preds[thresh_prob_labels])

        if curr_auc > best_is_auc:

            opt_ti = curr_ti
            best_is_auc = curr_auc
    
    curr_os_ti_preds = curr_os_preds[curr_os_preds.TUNE_IDX == opt_ti].reset_index(drop=True)

    thresh_prob_labels = [col for col in curr_os_ti_preds if col.startswith('Pr(GOSE>')]
    final_auc = pd.DataFrame({'MODEL':'eCPM_D'+curr_model[1:],
                              'RESAMPLE_IDX':curr_rs_idx,
                              'Threshold':thresh_labels,
                              'AUC':roc_auc_score(curr_os_ti_preds[thresh_labels],curr_os_ti_preds[thresh_prob_labels],average=None)})

    final_rocs = []
    for thresh in thresh_labels:
        prob_name = 'Pr('+thresh+')'
        (fpr, tpr, _) = roc_curve(curr_os_ti_preds[thresh], curr_os_ti_preds[prob_name])
        interp_tpr = np.interp(np.linspace(0,1,200),fpr,tpr)
        roc_axes = pd.DataFrame({'Threshold':thresh,'FPR':np.linspace(0,1,200),'TPR':interp_tpr})
        final_rocs.append(roc_axes)
    final_rocs = pd.concat(final_rocs,ignore_index = True)
    final_rocs['MODEL']='eCPM_D'+curr_model[1:]
    final_rocs['RESAMPLE_IDX'] = curr_rs_idx
    final_rocs.to_csv(os.path.join(metric_dir,'deep_ROCs.csv'),index=False)
    
    ### Calculate accuracy at each threshold
    thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
    best_is_thresh_accuracy = 0

    for curr_ti in curr_is_preds.TUNE_IDX.unique():

        curr_ti_preds = curr_is_preds[curr_is_preds.TUNE_IDX == curr_ti].reset_index(drop=True)
        thresh_prob_labels = np.sort([col for col in curr_ti_preds if col.startswith('Pr(GOSE>')])
        
        accuracies = []
        for idx in range(len(thresh_prob_labels)):
            curr_thresh_preds = (curr_ti_preds[thresh_prob_labels[idx]]>.5).astype(int)
            curr_thresh_label = curr_ti_preds[thresh_labels[idx]].astype(int)
            accuracies.append(accuracy_score(curr_thresh_label,curr_thresh_preds))
        curr_thresh_accuracy = np.mean(accuracies)
        
        if curr_thresh_accuracy > best_is_thresh_accuracy:

            opt_ti = curr_ti
            best_is_thresh_accuracy = curr_thresh_accuracy
    
    curr_os_ti_preds = curr_os_preds[curr_os_preds.TUNE_IDX == opt_ti].reset_index(drop=True)
    thresh_prob_labels = np.sort([col for col in curr_os_ti_preds if col.startswith('Pr(GOSE>')])
    
    accuracies = []
    for idx in range(len(thresh_prob_labels)):
        curr_thresh_preds = (curr_os_ti_preds[thresh_prob_labels[idx]]>.5).astype(int)
        curr_thresh_label = curr_os_ti_preds[thresh_labels[idx]].astype(int)
        accuracies.append(accuracy_score(curr_thresh_label,curr_thresh_preds))
    final_thresh_accuracy = pd.DataFrame({'MODEL':'eCPM_D'+curr_model[1:],
                                          'RESAMPLE_IDX':curr_rs_idx,
                                          'Threshold':thresh_labels,
                                          'Accuracy':accuracies})
    
    ### Threshold-level calibration curves and associated metrics
    thresh_labels = ['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7']
    best_is_slope_error = 100000
    for curr_ti in curr_is_preds.TUNE_IDX.unique():
        curr_ti_preds = curr_is_preds[curr_is_preds.TUNE_IDX == curr_ti].reset_index(drop=True)
        thresh_slopes = []

        for thresh in thresh_labels:
            thresh_prob_name = 'Pr('+thresh+')'
            
            logit_gt = np.nan_to_num(logit(curr_ti_preds[thresh_prob_name]))
            
            calib_glm = Logit(curr_ti_preds[thresh], add_constant(logit_gt))
            
            calib_glm_res = calib_glm.fit(disp=False)
            
            thresh_slopes.append(np.abs(1-calib_glm_res.params[1]))
        curr_slope_error = np.mean(thresh_slopes)
        
        if curr_slope_error < best_is_slope_error:
            opt_ti = curr_ti
            best_is_slope_error = curr_slope_error
            
    curr_os_ti_preds = curr_os_preds[curr_os_preds.TUNE_IDX == opt_ti].reset_index(drop=True)
    
    # Calculate threshold-level calibration curves
    final_calibs = []
    for thresh in thresh_labels:
        thresh_prob_name = 'Pr('+thresh+')'
        TrueProb = lowess(endog = curr_os_ti_preds[thresh], exog = curr_os_ti_preds[thresh_prob_name], it = 0, xvals = np.linspace(0,1,200))
        calib_axes = pd.DataFrame({'Threshold':thresh,'PredProb':np.linspace(0,1,200),'TrueProb':TrueProb})
        final_calibs.append(calib_axes)
    final_calibs = pd.concat(final_calibs,ignore_index = True)
    final_calibs['MODEL']='eCPM_D'+curr_model[1:]
    final_calibs['RESAMPLE_IDX'] = curr_rs_idx
    final_calibs.to_csv(os.path.join(metric_dir,'deep_calibration.csv'),index=False)

    # Calculate threshold-level calibration metrics
    final_calib_metrics = []
    for thresh in thresh_labels:
        thresh_prob_name = 'Pr('+thresh+')'
        logit_gt = np.nan_to_num(logit(curr_os_ti_preds[thresh_prob_name]))
        calib_glm = Logit(curr_os_ti_preds[thresh], add_constant(logit_gt))
        calib_glm_res = calib_glm.fit(disp=False)
        final_calib_metrics.append(pd.DataFrame({'Threshold':thresh,'Predictor':['Calib_Intercept','Calib_Slope'],'COEF':calib_glm_res.params}))
    final_calib_metrics = pd.concat(final_calib_metrics,ignore_index = True).pivot(index="Threshold", columns="Predictor", values="COEF").reset_index()
    final_calib_metrics.columns.name = None
    final_calib_metrics['MODEL']='eCPM_D'+curr_model[1:]
    final_calib_metrics['RESAMPLE_IDX'] = curr_rs_idx

    #### Compile and save threshold-level metrics
    ecpm_deep_tlm = pd.merge(final_auc,final_thresh_accuracy,how='left',on=['MODEL','RESAMPLE_IDX','Threshold']).merge(final_calib_metrics,how='left',on=['MODEL','RESAMPLE_IDX','Threshold'])
    ecpm_deep_tlm = ecpm_deep_tlm[['MODEL','RESAMPLE_IDX','Threshold','AUC','Accuracy','Calib_Intercept','Calib_Slope']]
    ecpm_deep_tlm.to_csv(os.path.join(metric_dir,'deep_threshold_metrics.csv'),index=False)

    ### Normalized confusion matrix
    best_is_recall = 0

    for curr_ti in curr_is_preds.TUNE_IDX.unique():

        curr_ti_preds = curr_is_preds[curr_is_preds.TUNE_IDX == curr_ti].reset_index(drop=True)
        curr_recall = recall_score(curr_ti_preds.TrueLabel, curr_ti_preds.PredLabel,average = 'macro')

        if curr_recall > best_is_recall:

            opt_ti = curr_ti
            best_is_recall = curr_recall

    curr_os_ti_preds = curr_os_preds[curr_os_preds.TUNE_IDX == opt_ti].reset_index(drop=True)
    
    final_cm = confusion_matrix(curr_os_ti_preds.TrueLabel, curr_os_ti_preds.PredLabel,normalize='true')  
    final_cm = pd.DataFrame(final_cm)
    final_cm.columns = ['GOSE: 1','GOSE: 2/3','GOSE: 4','GOSE: 5','GOSE: 6','GOSE: 7','GOSE: 8']
    final_cm = final_cm.assign(TrueLabel=['GOSE: 1','GOSE: 2/3','GOSE: 4','GOSE: 5','GOSE: 6','GOSE: 7','GOSE: 8'])
    final_cm = final_cm.melt(id_vars=['TrueLabel'],var_name='PredLabel',value_name='cm_prob') 
    final_cm['MODEL']='eCPM_D'+curr_model[1:]
    final_cm['RESAMPLE_IDX'] = curr_rs_idx
    final_cm.to_csv(os.path.join(metric_dir,'deep_confusion_matrices.csv'),index=False)
    
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)