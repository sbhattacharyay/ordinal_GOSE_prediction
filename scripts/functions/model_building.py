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
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

# TQDM for progress tracking
from tqdm import tqdm

# Custom methods
from classes.datasets import CONCISE_PREDICTOR_SET
from models.CPM import CPM_deep

# Train CPM_deep
def train_CPM_deep(TRAINING_SET,
                   VAL_SET,
                   TESTING_SET,
                   TUNE_IDX,
                   REPEAT,
                   FOLD,
                   OUTPUT_DIR,
                   BATCH_SIZE,
                   LEARNING_RATE,
                   LAYERS,
                   NEURONS,
                   DROPOUT,
                   ES_PATIENCE,
                   EPOCHS,
                   CLASS_WEIGHTS,
                   OUTPUT_ACTIVATION):
    """
    Args:
        TRAINING_SET (pd.DataFrame)
        VAL_SET (pd.DataFrame)
        TESTING_SET (pd.DataFrame)
        TUNE_IDX (str)
        REPEAT (int)
        FOLD (int)
        OUTPUT_DIR (str): directory to save model outputs
        BATCH_SIZE (int): size of minibatches during training
        LEARNING_RATE (float): Learning rate for ADAM optimizer
        LAYERS (int): number of hidden layers in feed forward neural network
        NEURONS (list of length layers): the number of neurons in each layer
        DROPOUT (flaot): the proportion of each dense layer dropped out during training
        ES_PATIENCE (int): patience during early stopping
        EPOCHS (int): maximum epochs during training
        CLASS_WEIGHTS (boolean): identifies whether loss should be weighted against class frequency
        OUTPUT_ACTIVATION (string): 'softmax' for DeepMN or 'sigmoid' for DeepOR
    """    
    
    # Create a directory within current repeat/fold combination to store outputs of current tuning configuration
    tune_model_dir = os.path.join(OUTPUT_DIR,'tune_'+TUNE_IDX)
    os.makedirs(tune_model_dir,exist_ok = True)
    
    # Create PyTorch Dataset objects
    train_Dataset = CONCISE_PREDICTOR_SET(TRAINING_SET,OUTPUT_ACTIVATION)
    val_Dataset = CONCISE_PREDICTOR_SET(VAL_SET,OUTPUT_ACTIVATION)
    test_Dataset = CONCISE_PREDICTOR_SET(TESTING_SET,OUTPUT_ACTIVATION)
    
    # Create PyTorch DataLoader objects
    curr_train_DL = DataLoader(train_Dataset,
                               batch_size=int(BATCH_SIZE),
                               shuffle=True)
    
    curr_val_DL = DataLoader(val_Dataset,
                             batch_size=len(val_Dataset), 
                             shuffle=False)
    
    curr_test_DL = DataLoader(test_Dataset,
                             batch_size=len(test_Dataset),
                             shuffle=False)
    
    # Initialize current model class based on hyperparameter selections
    model = CPM_deep(train_Dataset.X.shape[1],
                     LAYERS,
                     NEURONS,
                     DROPOUT,
                     OUTPUT_ACTIVATION,
                     LEARNING_RATE,
                     CLASS_WEIGHTS,
                     train_Dataset.y)
    
    early_stop_callback = EarlyStopping(
        monitor='val_AUROC',
        patience=ES_PATIENCE,
        mode='max'
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_AUROC',
        dirpath=tune_model_dir,
        filename='{epoch:02d}-{val_AUROC:.2f}',
        save_top_k=1,
        mode='max'
    )
    
    csv_logger = pl.loggers.CSVLogger(save_dir=OUTPUT_DIR,name='tune_'+TUNE_IDX)

    trainer = pl.Trainer(logger = csv_logger,
                         max_epochs = EPOCHS,
                         enable_progress_bar = False,
                         enable_model_summary = False,
                         callbacks=[early_stop_callback,checkpoint_callback])
    
    trainer.fit(model,curr_train_DL,curr_val_DL)
    
    best_model = CPM_deep.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.eval()
    
    # Save validation set probabilities
    
    for i, (x,y) in enumerate(curr_val_DL):

        yhat = best_model(x)
        val_true_y = y.cpu().numpy()

        if OUTPUT_ACTIVATION == 'softmax': 

            curr_val_probs = F.softmax(yhat.detach()).cpu().numpy()
            curr_val_preds = pd.DataFrame(curr_val_probs,columns=['Pr(GOSE=1)','Pr(GOSE=2/3)','Pr(GOSE=4)','Pr(GOSE=5)','Pr(GOSE=6)','Pr(GOSE=7)','Pr(GOSE=8)'])
            curr_val_preds['TrueLabel'] = val_true_y

        elif OUTPUT_ACTIVATION == 'sigmoid': 

            curr_val_probs = F.sigmoid(yhat.detach()).cpu().numpy()
            curr_val_probs = pd.DataFrame(curr_val_probs,columns=['Pr(GOSE>1)','Pr(GOSE>3)','Pr(GOSE>4)','Pr(GOSE>5)','Pr(GOSE>6)','Pr(GOSE>7)'])
            curr_val_labels = pd.DataFrame(val_true_y,columns=['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7'])
            curr_val_preds = pd.concat([curr_val_probs,curr_val_labels],axis = 1)

        else:
            raise ValueError("Invalid output layer type. Must be 'softmax' or 'sigmoid'")

        curr_val_preds.insert(loc=0, column='GUPI', value=VAL_SET.GUPI.values)        
        curr_val_preds['TUNE_IDX'] = TUNE_IDX

        curr_val_preds.to_csv(os.path.join(tune_model_dir,'val_predictions.csv'),index=False)
        
    best_model.eval()
        
    # Save testing set probabilities
    
    for i, (x,y) in enumerate(curr_test_DL):

        yhat = best_model(x)
        test_true_y = y.cpu().numpy()

        if OUTPUT_ACTIVATION == 'softmax': 

            curr_test_probs = F.softmax(yhat.detach()).cpu().numpy()
            curr_test_preds = pd.DataFrame(curr_test_probs,columns=['Pr(GOSE=1)','Pr(GOSE=2/3)','Pr(GOSE=4)','Pr(GOSE=5)','Pr(GOSE=6)','Pr(GOSE=7)','Pr(GOSE=8)'])
            curr_test_preds['TrueLabel'] = test_true_y

        elif OUTPUT_ACTIVATION == 'sigmoid': 

            curr_test_probs = F.sigmoid(yhat.detach()).cpu().numpy()
            curr_test_probs = pd.DataFrame(curr_test_probs,columns=['Pr(GOSE>1)','Pr(GOSE>3)','Pr(GOSE>4)','Pr(GOSE>5)','Pr(GOSE>6)','Pr(GOSE>7)'])
            curr_test_labels = pd.DataFrame(test_true_y,columns=['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7'])
            curr_test_preds = pd.concat([curr_test_probs,curr_test_labels],axis = 1)

        else:
            raise ValueError("Invalid output layer type. Must be 'softmax' or 'sigmoid'")

        curr_test_preds.insert(loc=0, column='GUPI', value=TESTING_SET.GUPI.values)        
        curr_test_preds['TUNE_IDX'] = TUNE_IDX

        curr_test_preds.to_csv(os.path.join(tune_model_dir,'test_predictions.csv'),index=False)

# Functions to collect validation metrics from files in parallel
def collect_val_metrics(
    csv_file_list,
    n_cores,
    progress_bar=True,
    progress_bar_desc=''):

    # Establish sizes of files for each core
    sizes = [len(csv_file_list) // n_cores for _ in range(n_cores)]
    sizes[:(len(csv_file_list) - sum(sizes))] = [val+1 for val in sizes[:(len(csv_file_list) - sum(sizes))]]
    end_indices = np.cumsum(sizes)
    start_indices = np.insert(end_indices[:-1],0,0)
    
    # Build arguments for metric sub-functions

    arg_iterable = [(
        csv_file_list[start_indices[idx]:end_indices[idx]],
        progress_bar,
        progress_bar_desc) 
        for idx in range(len(start_indices))]
    
    # Run metric sub-function in parallel
    with multiprocessing.Pool(n_cores) as pool:
        result = pool.starmap(_val_metric_par, arg_iterable)  
        
    return pd.concat(result, ignore_index=True).sort_values(by=['fold','TUNE_IDX']).reset_index(drop=True)

def _val_metric_par(sub_csv_file_list,
                    progress_bar=True,
                    progress_bar_desc=''):
    
    if progress_bar:
        iterator = tqdm(range(len(sub_csv_file_list)), desc=progress_bar_desc)
    else:
        iterator = range(len(sub_csv_file_list))
    return pd.concat([val_metric_extraction(sub_csv_file_list[sub_file_idx]) for sub_file_idx in iterator], ignore_index=True)

def val_metric_extraction(chosen_file):
    curr_metric_df = pd.read_csv(chosen_file)
    curr_metric_df = curr_metric_df.groupby(['epoch','step'],as_index=False)[curr_metric_df.columns[~curr_metric_df.columns.isin(['epoch', 'step'])]].max()
    curr_metric_df['TUNE_IDX'] = re.search('/tune_(.*)/version_', chosen_file).group(1)
    curr_metric_df['repeat'] = int(re.search('/repeat(.*)/fold', chosen_file).group(1))
    curr_metric_df['fold'] = int(re.search('/fold(.*)/tune', chosen_file).group(1))
    return(curr_metric_df)

### SURFACE FUNCTION
def generate_resamples(
    pred_df,
    output_activation,
    num_resamples,
    n_cores,
    progress_bar=True,
    progress_bar_desc=''):

    # Establish number of resamples per each core
    sizes = [num_resamples // n_cores for _ in range(n_cores)]
    sizes[:(num_resamples - sum(sizes))] = [val+1 for val in sizes[:(num_resamples - sum(sizes))]]
    
    # Build arguments for metric sub-functions
    arg_iterable = [(pred_df,
                     output_activation,
                     s,
                     progress_bar,
                     progress_bar_desc)
                   for s in sizes]
    
    # Run metric sub-function in parallel
    with multiprocessing.Pool(n_cores) as pool:
        result = pool.starmap(_bs_rs_generator, arg_iterable)  
        
    # Add a core index to each resample dataframe
    for i in range(len(result)):
        curr_df = result[i]
        curr_df['core_idx'] = i
        result[i] = curr_df
    
    result = pd.concat(result)
    
    # From unique combinations of core_idx and core_sub_idx, assign resample IDs
    rs_combos = result[['core_sub_idx','core_idx']].drop_duplicates(ignore_index = True)
    rs_combos = rs_combos.sort_values(by=['core_idx','core_sub_idx']).reset_index(drop=True)
    rs_combos['rs_idx'] = [i for i in range(rs_combos.shape[0])]
    
    # Merge resample IDs
    result = pd.merge(result,rs_combos,how='left',on=['core_idx','core_sub_idx'])
    
    return result

## SUB-SURFACE and ACTIVE FUNCTION
def _bs_rs_generator(
    pred_df,
    output_activation,
    size,
    progress_bar,
    progress_bar_desc):
    
    # Create random generator instance
    rs = np.random.RandomState()
    
    if output_activation == 'softmax':
        curr_outcomes = pred_df[['GUPI','TrueLabel']].drop_duplicates(ignore_index = True)
        
    elif output_activation == 'sigmoid':
        label_col = [col for col in pred_df if col.startswith('GOSE>')]
        label_col.insert(0,'GUPI')
        curr_outcomes = pred_df[label_col].drop_duplicates(ignore_index = True)
        curr_outcomes['TrueLabel'] = curr_outcomes[[col for col in curr_outcomes if col.startswith('GOSE>')]].sum(axis = 1)
        curr_outcomes = curr_outcomes.drop(columns = [col for col in curr_outcomes if col.startswith('GOSE>')])
        
    else:
        raise ValueError("Invalid output layer type. Must be 'softmax' or 'sigmoid'")
    
    if progress_bar:
        iterator = tqdm(range(size), desc=progress_bar_desc)
    else:
        iterator = range(size)
        
    bs_subcore_resamples = []
    for i in iterator:
        curr_bs_rs = resample(curr_outcomes,replace=True, random_state = rs, stratify = curr_outcomes.TrueLabel).drop_duplicates(ignore_index = True)
        curr_bs_rs['core_sub_idx'] = i
        bs_subcore_resamples.append(curr_bs_rs)
            
    return pd.concat(bs_subcore_resamples)

# Function to bootstrap AUROC on optimal validation configuration
def bootstrap_opt_val_metric(
    opt_preds_df,
    output_activation,
    bs_resamples,
    n_cores,
    progress_bar=True,
    progress_bar_desc=''):
    
    # Establish number of resamples for each core

    sizes = [len(bs_resamples.rs_idx.unique()) // n_cores for _ in range(n_cores)]
    sizes[:(len(bs_resamples.rs_idx.unique()) - sum(sizes))] = [val+1 for val in sizes[:(len(bs_resamples.rs_idx.unique()) - sum(sizes))]]
    end_indices = np.cumsum(sizes)
    start_indices = np.insert(end_indices[:-1],0,0)
    
    # Build arguments for metric sub-functions

    arg_iterable = [(
        opt_preds_df,        
        output_activation,        
        bs_resamples,
        bs_resamples.rs_idx.unique()[start_indices[idx]:end_indices[idx]],
        progress_bar,
        progress_bar_desc)
        for idx in range(len(start_indices))]
    
    # Run metric sub-function in parallel
    with multiprocessing.Pool(n_cores) as pool:
        result = pool.starmap(_opt_val_bs_par, arg_iterable)  
        
    return pd.concat(result)
    
### SUB-SURFACE AND ACTIVE FUNCTOINS:
def _opt_val_bs_par(
    opt_preds_df,
    output_activation,
    bs_resamples,
    curr_resamples,
    progress_bar,
    progress_bar_desc):
    
    # Initiate dataframe to store AUROCs from current core
    curr_rs_test_results = pd.DataFrame(columns=['TUNE_IDX','RESAMPLE_IDX','opt_AUROC'])
    
    if progress_bar:
        iterator = tqdm(curr_resamples, desc=progress_bar_desc)
    else:
        iterator = curr_resamples
        
    # Iterate through assigned resampling indices
    for curr_rs_idx in iterator:
        
        # Extract current bootstrapping resamples
        curr_bs_rs = bs_resamples[bs_resamples.rs_idx == curr_rs_idx]

        # Extract current optimal configuration in-sample predictions
        curr_rs_preds = opt_preds_df[opt_preds_df.GUPI.isin(curr_bs_rs.GUPI)]
        
        if output_activation == 'softmax':

            prob_cols = [col for col in curr_rs_preds if col.startswith('Pr(GOSE')]
            curr_config_auroc = roc_auc_score(curr_rs_preds.TrueLabel.values, curr_rs_preds[prob_cols].values, multi_class='ovo')

        elif output_activation == 'sigmoid':

            prob_cols = [col for col in curr_rs_preds if col.startswith('Pr(GOSE')]
            label_cols = [col for col in curr_rs_preds if col.startswith('GOSE>')]
            curr_config_auroc = roc_auc_score(curr_rs_preds[label_cols].values, curr_rs_preds[prob_cols].values, multi_class='macro')

        curr_rs_test_results = curr_rs_test_results.append(pd.DataFrame({'TUNE_IDX':opt_preds_df.TUNE_IDX.unique()[0],
                                                                         'RESAMPLE_IDX':curr_rs_idx,
                                                                         'opt_AUROC':curr_config_auroc},index=[0]),ignore_index=True)
        
    return curr_rs_test_results

# Calculate trial configuration performance on bootstrap resamples in parallel
def bootstrap_val_metric(
    trial_configurations,
    output_activation,
    bs_resamples,
    val_pred_file_info_df,
    repeat,
    n_cores,
    split_by_tune_idx,
    progress_bar=True,
    progress_bar_desc=''):
    
    if split_by_tune_idx:

        # Establish sizes of files for each core
        sizes = [len(trial_configurations) // n_cores for _ in range(n_cores)]
        sizes[:(len(trial_configurations) - sum(sizes))] = [val+1 for val in sizes[:(len(trial_configurations) - sum(sizes))]]
        end_indices = np.cumsum(sizes)
        start_indices = np.insert(end_indices[:-1],0,0)
        
        # Build arguments for metric sub-functions
        arg_iterable = [(
            trial_configurations[start_indices[idx]:end_indices[idx]],
            output_activation,
            bs_resamples,
            val_pred_file_info_df,
            repeat,
            progress_bar,
            progress_bar_desc)
            for idx in range(len(start_indices))]
        
    else:
        
        # Establish sizes of resamples for each core
        sizes = [len(bs_resamples.rs_idx.unique()) // n_cores for _ in range(n_cores)]
        sizes[:(len(bs_resamples.rs_idx.unique()) - sum(sizes))] = [val+1 for val in sizes[:(len(bs_resamples.rs_idx.unique()) - sum(sizes))]]
        end_indices = np.cumsum(sizes)
        start_indices = np.insert(end_indices[:-1],0,0)
        
        # Build arguments for metric sub-functions
        arg_iterable = [(
            trial_configurations,
            output_activation,
            bs_resamples[bs_resamples.rs_idx.isin([i for i in range(start_indices[idx],end_indices[idx])])],
            val_pred_file_info_df,
            repeat,
            progress_bar,
            progress_bar_desc)
            for idx in range(len(start_indices))]        
        
    # Run metric sub-function in parallel
    with multiprocessing.Pool(n_cores) as pool:
        result = pool.starmap(_val_bs_par, arg_iterable)  
        
    return pd.concat(result, ignore_index=True).sort_values(by=['TUNE_IDX','RESAMPLE_IDX']).reset_index(drop=True)
    
### SUB-SURFACE AND ACTIVE FUNCTOINS:
def _val_bs_par(
    curr_trial_configs,
    output_activation,
    bs_resamples,
    val_pred_file_info_df,
    repeat,
    progress_bar,
    progress_bar_desc):
    
    # Initiate dataframe to store AUROCs from current core
    curr_rs_test_results = pd.DataFrame(columns=['TUNE_IDX','RESAMPLE_IDX','trial_AUROC'])
    
    if progress_bar:
        iterator = tqdm(curr_trial_configs.TUNE_IDX.values, desc=progress_bar_desc)
    else:
        iterator = curr_trial_configs.TUNE_IDX.values
        
    # Iterate through trial configurations
    for curr_trial_tune_idx in iterator:
        
        # Find available validation prediction files for current under-trial configuration
        curr_trial_candidate_dirs = val_pred_file_info_df.file[(val_pred_file_info_df.TUNE_IDX == curr_trial_tune_idx) & (val_pred_file_info_df.repeat <= repeat)].values
        curr_trial_candidate_dirs = [curr_cand_dir for curr_cand_dir in curr_trial_candidate_dirs if os.path.isfile(curr_cand_dir)]
        
        # Load and concatenate validation predictions for current trial tuning index
        curr_trial_val_preds_df = pd.concat([pd.read_csv(curr_dir) for curr_dir in curr_trial_candidate_dirs])
        curr_trial_val_preds_df['TUNE_IDX'] = curr_trial_tune_idx
        
        # Iterate through bootstrapping resamples and calculate AUROC for current trial configuration tuning index
        for curr_rs_idx in bs_resamples.rs_idx.unique():
            
            # Extract current bootstrapping resamples
            curr_bs_rs = bs_resamples[bs_resamples.rs_idx == curr_rs_idx]
            
            # Extract current trial configuration in-sample predictions
            curr_rs_preds = curr_trial_val_preds_df[curr_trial_val_preds_df.GUPI.isin(curr_bs_rs.GUPI)]

            if output_activation == 'softmax':
                
                prob_cols = [col for col in curr_rs_preds if col.startswith('Pr(GOSE')]
                
                curr_config_auroc = roc_auc_score(curr_rs_preds[curr_rs_preds.TUNE_IDX == curr_trial_tune_idx].TrueLabel.values, curr_rs_preds[curr_rs_preds.TUNE_IDX == curr_trial_tune_idx][prob_cols].values, multi_class='ovo')
                    
            elif output_activation == 'sigmoid':

                prob_cols = [col for col in curr_rs_preds if col.startswith('Pr(GOSE')]
                label_cols = [col for col in curr_rs_preds if col.startswith('GOSE>')]
                
                curr_config_auroc = roc_auc_score(curr_rs_preds[curr_rs_preds.TUNE_IDX == curr_trial_tune_idx][label_cols].values, curr_rs_preds[curr_rs_preds.TUNE_IDX == curr_trial_tune_idx][prob_cols].values, multi_class='macro')

            curr_rs_test_results = curr_rs_test_results.append(pd.DataFrame({'TUNE_IDX':curr_trial_tune_idx,
                                                                             'RESAMPLE_IDX':curr_rs_idx,
                                                                             'trial_AUROC':curr_config_auroc},index=[0]),ignore_index=True)
    return curr_rs_test_results

# Functions for Bootstrap Bias Corrected with Dropping CV (BBCD-CV) doi: 10.1007/s10994-018-5714-4
def interrepeat_dropout(
    REPEAT,
    MODEL_DIR,
    validation_perf,
    tuning_grid,
    grouping_vars,
    num_resamples,
    num_cores,
    save_perf_metrics,
    progress_bars
):
    
    # Find all validation prediction files within the current model directory
    val_pred_files = []
    for path in Path(MODEL_DIR).rglob('*val_predictions.csv'):
        val_pred_files.append(str(path.resolve()))

    val_pred_file_info_df = pd.DataFrame({'file':val_pred_files,
                                          'TUNE_IDX':[re.search('tune_(.*)/', curr_file).group(1) for curr_file in val_pred_files],
                                          'VERSION':[re.search('IMPACT_model_outputs/v(.*)/repeat', curr_file).group(1) for curr_file in val_pred_files],
                                          'repeat':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in val_pred_files],
                                          'fold':[int(re.search('/fold(.*)/tune_', curr_file).group(1)) for curr_file in val_pred_files]
                                         }).sort_values(by=['repeat','fold','TUNE_IDX','VERSION']).reset_index(drop=True)
    
    # Filter only validation prediction files available to current repeat and viable tuning indices
    val_pred_file_info_df = val_pred_file_info_df[val_pred_file_info_df.repeat <= REPEAT]
    val_pred_file_info_df = val_pred_file_info_df[val_pred_file_info_df.TUNE_IDX.isin(tuning_grid.TUNE_IDX)]

    # Filter only viable tuning indices
    validation_perf = validation_perf[validation_perf.TUNE_IDX.isin(tuning_grid.TUNE_IDX)]

    max_validation_perf = validation_perf.groupby(['TUNE_IDX','repeat','fold'],as_index=False)['val_AUROC'].max()
    max_validation_perf = pd.merge(max_validation_perf, tuning_grid, how="left", on=['TUNE_IDX'])

    if save_perf_metrics:
        max_validation_perf.to_csv(os.path.join(MODEL_DIR,'repeat'+str(REPEAT).zfill(2),'validation_performance.csv'),index=False)
        
    # Group by tuning index and average validation AUROC
    across_cv_perf = max_validation_perf.groupby(max_validation_perf.columns[~max_validation_perf.columns.isin(['repeat','fold','val_AUROC'])].values.tolist(),as_index=False)['val_AUROC'].mean()

    opt_tune_idx = across_cv_perf[across_cv_perf.groupby(grouping_vars)['val_AUROC'].transform(max) == across_cv_perf['val_AUROC']].reset_index(drop=True)

    # Non-ideal tuning configurations under trial
    trial_configs = across_cv_perf[~across_cv_perf.TUNE_IDX.isin(opt_tune_idx.TUNE_IDX)]

    # Initialize empty list to store droppped configurations
    dropped_idx = []

    # Iterate through each unique combination of the grouping variables
    for combo_idx in range(opt_tune_idx.shape[0]):

        # Acquire current output activation from optimal configuration dataframe
        curr_output = opt_tune_idx.OUTPUT_ACTIVATION.values[combo_idx]

        # Acquire current optimal configuration tuning index
        curr_opt_tune_idx = opt_tune_idx.TUNE_IDX.values[combo_idx]

        # Set current progress bar label
        pb_label = ', '.join([name+': '+str(opt_tune_idx[name].values[combo_idx]) for name in grouping_vars])
        
        # Find available validation prediction files for current optimal tuning index
        curr_opt_candidate_dirs = val_pred_file_info_df.file[(val_pred_file_info_df.TUNE_IDX == curr_opt_tune_idx) & (val_pred_file_info_df.repeat <= REPEAT)].values
        curr_opt_candidate_dirs = [curr_cand_dir for curr_cand_dir in curr_opt_candidate_dirs if os.path.isfile(curr_cand_dir)]

        # Compile current optimal configuration validation predictions
        curr_opt_val_preds_df = pd.concat([pd.read_csv(curr_cand_dir) for curr_cand_dir in curr_opt_candidate_dirs])
        
        # Filter out under-trial configurations that match the current grouping combination
        curr_trial_configs = pd.merge(trial_configs,opt_tune_idx[opt_tune_idx.TUNE_IDX == curr_opt_tune_idx][grouping_vars],how='inner',on=grouping_vars).reset_index(drop=True)
        
        # Generate bootstrapping resamples
        bs_resamples = generate_resamples(curr_opt_val_preds_df,
                                          curr_output,
                                          num_resamples,
                                          num_cores,
                                          progress_bar=progress_bars,
                                          progress_bar_desc='Generating resamples '+pb_label)
        
        # Calculate optimal configuration performance on generated resamples
        opt_bs_AUC = bootstrap_opt_val_metric(curr_opt_val_preds_df,
                                              curr_output,
                                              bs_resamples,
                                              num_cores,
                                              progress_bar=progress_bars,
                                              progress_bar_desc='Bootstrapping optimal config AUC '+pb_label)
                
        bootstrapped_AUC = bootstrap_val_metric(curr_trial_configs,
                                                curr_output,
                                                bs_resamples,
                                                val_pred_file_info_df,
                                                REPEAT,
                                                num_cores,
                                                len(curr_trial_configs) >= num_cores,
                                                progress_bars,
                                                'Bootstrapping AUC '+pb_label)
        
        bootstrapped_AUC = pd.merge(bootstrapped_AUC,opt_bs_AUC[['RESAMPLE_IDX','opt_AUROC']],on='RESAMPLE_IDX',how='left')
        bootstrapped_AUC['trial_win'] = bootstrapped_AUC['trial_AUROC'] >= bootstrapped_AUC['opt_AUROC']
        bootstrap_pvals = bootstrapped_AUC.groupby(['TUNE_IDX'],as_index=False)['trial_win'].agg(['sum','size']).reset_index()
        bootstrap_pvals['pval'] = bootstrap_pvals['sum']/bootstrap_pvals['size']

        dropped_idx.append(bootstrap_pvals[bootstrap_pvals['pval'] < 0.01].TUNE_IDX.values)

        print('Tuning indices droppped from '+pb_label+':')
        print(bootstrap_pvals[bootstrap_pvals['pval'] < 0.01].TUNE_IDX.values)

    # Concatenate arrays of dropped indices
    dropped_tune_idx = np.concatenate(dropped_idx)
    viable_tuning_grid = tuning_grid[~tuning_grid.TUNE_IDX.isin(dropped_tune_idx)]
    
    return (dropped_tune_idx,viable_tuning_grid)

def calc_auroc(pred_file_info, progress_bar = True, progress_bar_desc = ''):
    
    if progress_bar:
        iterator = tqdm(range(pred_file_info.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(pred_file_info.shape[0])
    
    compiled_auroc = []
        
    for idx in iterator:
        
        curr_file = pred_file_info.file[idx]
        
        curr_output = pred_file_info.OUTPUT_ACTIVATION[idx]
        
        curr_preds = pd.read_csv(curr_file)
                
        prob_cols = [col for col in curr_preds if col.startswith('Pr(GOSE')]
        
        if curr_output == 'softmax':
            
            curr_auroc = roc_auc_score(curr_preds.TrueLabel.values, curr_preds[prob_cols].values, multi_class='ovo')

        elif curr_output == 'sigmoid':

            label_cols = [col for col in curr_preds if col.startswith('GOSE>')]
            curr_train_probs = curr_preds[prob_cols].values
            
            train_probs = np.empty([curr_train_probs.shape[0], curr_train_probs.shape[1]+1])
            train_probs[:,0] = 1 - curr_train_probs[:,0]
            train_probs[:,-1] = curr_train_probs[:,-1]
            
            for col_idx in range(1,(curr_train_probs.shape[1])):
                train_probs[:,col_idx] = curr_train_probs[:,col_idx-1] - curr_train_probs[:,col_idx]                
            
            curr_auroc = roc_auc_score(curr_preds[label_cols].values.sum(1).astype(int), train_probs, multi_class='ovo')
        
        curr_info_row = pred_file_info[pred_file_info.file == curr_file].reset_index(drop=True)
        curr_info_row['val_AUROC'] = curr_auroc
        
        compiled_auroc.append(curr_info_row)
    
    return pd.concat(compiled_auroc,ignore_index = True)

def bs_dropout_auroc(bs_combos, val_file_info, bs_rs_GUPIs, curr_output, progress_bar = True, progress_bar_desc = ''):

    compiled_auroc = []
        
    curr_tis = bs_combos.TUNE_IDX.unique()
    
    if progress_bar:
        iterator = tqdm(curr_tis,desc=progress_bar_desc)
    else:
        iterator = curr_tis
    
    for ti in curr_tis:
        
        ti_preds = pd.concat([pd.read_csv(curr_file) for curr_file in val_file_info.file[val_file_info.TUNE_IDX == ti].values],ignore_index=True)
        
        curr_ti_combos = bs_combos[bs_combos.TUNE_IDX == ti]
        
        for curr_rs_index in curr_ti_combos.RESAMPLE:
            
            curr_rs_GUPIs = bs_rs_GUPIs[curr_rs_index - 1]
            
            curr_in_sample_preds = ti_preds[ti_preds.GUPI.isin(curr_rs_GUPIs)].reset_index(drop=True)
            
            prob_cols = [col for col in curr_in_sample_preds if col.startswith('Pr(GOSE')]

            if curr_output == 'softmax':

                curr_auroc = roc_auc_score(curr_in_sample_preds.TrueLabel.values, curr_in_sample_preds[prob_cols].values, multi_class='ovo')

            elif curr_output == 'sigmoid':

                label_cols = [col for col in curr_in_sample_preds if col.startswith('GOSE>')]

                curr_auroc = roc_auc_score(curr_in_sample_preds[label_cols].values, curr_in_sample_preds[prob_cols].values, multi_class='macro')
                
            compiled_auroc.append(pd.DataFrame({'TUNE_IDX':ti,'RESAMPLE':curr_rs_index,'val_AUROC':curr_auroc},index=[0]))
            
    return pd.concat(compiled_auroc,ignore_index = True)

def collate_batch(batch):

    (gupis, idx_list, y_list, pt_offsets) = ([], [], [], [0])
    
    for (curr_GUPI, curr_Indices, curr_y) in batch:
        
        gupis.append(curr_GUPI)
        idx_list.append(torch.tensor(curr_Indices,dtype=torch.int64))
        y_list.append(curr_y)
        pt_offsets.append(len(curr_Indices))
      
    idx_list = torch.cat(idx_list)
    y_list = torch.tensor(y_list, dtype=torch.int64)
    pt_offsets = torch.tensor(pt_offsets[:-1]).cumsum(dim=0)
    return (gupis, idx_list, y_list, pt_offsets)

def collect_agg_weights(model_ckpt_info, progress_bar = True, progress_bar_desc = ''):
    
    if progress_bar:
        iterator = tqdm(range(model_ckpt_info.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(model_ckpt_info.shape[0])
        
    compiled_df = []
        
    for i in iterator:
        
        # Extract current checkpoint information
        curr_file = model_ckpt_info.file[i]
        curr_TUNE_IDX = model_ckpt_info.TUNE_IDX[i]
        curr_repeat = model_ckpt_info.repeat[i]
        curr_fold = model_ckpt_info.fold[i]

        # Load current token dictionary
        curr_vocab = cp.load(open('/home/sb2406/rds/hpc-work/CENTER-TBI_tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/token_dictionary.pkl',"rb"))

        # Load current model weights from checkpoint file
        model = deepCENTERTBI.load_from_checkpoint(curr_file)
        model.eval()
        with torch.no_grad():
            #curr_embedX = model.embedX.weight.numpy()
            curr_embedW = np.exp(model.embedW.weight.numpy())

        # Get validation performance of current tuning index, repeat, and fold
        curr_val_ORC = val_performance[(val_performance.TUNE_IDX==curr_TUNE_IDX)&(val_performance.repeat==curr_repeat)&(val_performance.fold==curr_fold)].val_AUROC.values[0]

        compiled_df.append(pd.DataFrame({'TUNE_IDX':curr_TUNE_IDX, 'Token':curr_vocab.get_itos(),'AggWeight':curr_embedW.reshape(-1), 'repeat':curr_repeat, 'fold':curr_fold, 'val_ORC':curr_val_ORC}))
        
    return pd.concat(compiled_df,ignore_index=True)

def format_shap(shap_matrix,idx,token_labels,testing_set):
    shap_df = pd.DataFrame(shap_matrix,columns=token_labels)
    shap_df['GUPI'] = testing_set.GUPI
    shap_df = shap_df.melt(id_vars = 'GUPI', var_name = 'Token', value_name = 'SHAP')
    shap_df['label'] = idx
    return shap_df

def collect_shap_values(shap_info_df, model_dir, progress_bar = True, progress_bar_desc = ''):
    
    #shap_dfs = []
    
    if progress_bar:
        iterator = tqdm(range(shap_info_df.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(shap_info_df.shape[0])
        
    for i in iterator:
        # Extract current file, repeat, and fold information
        curr_file = shap_info_df.file[i]
        curr_repeat = shap_info_df.repeat[i]
        curr_fold = shap_info_df.fold[i]
        curr_output_type = shap_info_df.output_type[i]

        # Define current fold directory based on current information
        tune_dir = os.path.join(model_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold),'tune_0008')

        # Load current token dictionary
        curr_vocab = cp.load(open('/home/sb2406/rds/hpc-work/CENTER-TBI_tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/token_dictionary.pkl',"rb"))

        # Extract current testing set for current repeat and fold combination
        testing_set = pd.read_pickle('/home/sb2406/rds/hpc-work/CENTER-TBI_tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/testing_indices.pkl')
        testing_set['seq_len'] = testing_set.Index.apply(len)
        testing_set['unknowns'] = testing_set.Index.apply(lambda x: x.count(0))

        # Number of columns to add
        cols_to_add = testing_set['unknowns'].max() - 1

        # Initialize empty dataframe for multihot encoding of testing set
        multihot_matrix = np.zeros([testing_set.shape[0],len(curr_vocab)+cols_to_add])

        # Encode testing set into multihot encoded matrix
        for i in range(testing_set.shape[0]):
            curr_indices = np.array(testing_set.Index[i])
            if sum(curr_indices == 0) > 1:
                zero_indices = np.where(curr_indices == 0)[0]
                curr_indices[zero_indices[1:]] = [len(curr_vocab) + i for i in range(sum(curr_indices == 0)-1)]
            multihot_matrix[i,curr_indices] = 1
            
        # Define token labels
        token_labels = curr_vocab.get_itos() + [curr_vocab.get_itos()[0]+'_'+str(i+1).zfill(3) for i in range(cols_to_add)]
        token_labels[0] = token_labels[0]+'_000'

        # Load current shap value matrix
        shap_values = cp.load(open(os.path.join(tune_dir,'shap_arrays_'+curr_output_type+'.pkl'),"rb"))
        
        # Convert each SHAP matrix to formatted dataframe and concatenate across labels
        shap_df = pd.concat([format_shap(curr_matrix,idx,token_labels,testing_set) for idx,curr_matrix in enumerate(shap_values)],ignore_index=True)
        shap_df['repeat'] = curr_repeat
        shap_df['fold'] = curr_fold
        
        # Convert multihot encoded matrix into formatted dataframe for token indicators
        indicator_df = pd.DataFrame(multihot_matrix,columns=token_labels)
        indicator_df['GUPI'] = testing_set.GUPI
        indicator_df = indicator_df.melt(id_vars = 'GUPI', var_name = 'Token', value_name = 'Indicator')
        indicator_df['Indicator'] = indicator_df['Indicator'].astype(int)

        # Merge indicator dataframe with SHAP values
        shap_df = pd.merge(shap_df,indicator_df,how='left',on=['GUPI','Token'])
        
        # Remove rows which correspond to non-existent or unknown tokens
        shap_df = shap_df[shap_df.Indicator == 1]
        shap_df = shap_df[~shap_df.Token.str.startswith('<unk>_')].reset_index(drop=True)
        shap_df.to_pickle(os.path.join(tune_dir,'shap_dataframe_'+curr_output_type+'.pkl'))
        
        # Append current dataframe to list
        #shap_dfs.append(shap_df)

    #return pd.concat(shap_dfs,ignore_index=True)
    
def collect_shap_dfs(shap_info_df, progress_bar = True, progress_bar_desc = ''):
    
    shap_dfs = []
    
    if progress_bar:
        iterator = tqdm(range(shap_info_df.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(shap_info_df.shape[0])
        
    for i in iterator:
        
        shap_dfs.append(pd.read_pickle(shap_info_df.file[i]))
        
    return pd.concat(shap_dfs,ignore_index=True)