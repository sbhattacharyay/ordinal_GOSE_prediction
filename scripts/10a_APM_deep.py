#### Master Script 10a: Train deep learning all-predictor-based models (APM) ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Create grid of training combinations
# III. Train APM_deep model based on provided hyperparameter row index

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
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

# TQDM for progress tracking
from tqdm import tqdm

# Custom methods
from classes.datasets import ALL_PREDICTOR_SET
from models.APM import APM_deep

# Set version code
VERSION = 'DEEP_v1-0'

# Initialise model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/APM_outputs/'+VERSION
os.makedirs(model_dir,exist_ok=True)

# Load cross-validation split information
cv_splits = pd.read_csv('../cross_validation_splits.csv')

# Load the tuning grid
tuning_grid = pd.read_csv(os.path.join(model_dir,'APM_deep_tuning_grid.csv'))
tuning_grid['TUNE_IDX'] = tuning_grid['TUNE_IDX'].astype(str).str.zfill(4)
tuning_grid['NEURONS'] = tuning_grid['NEURONS'].apply(eval)

### II. Create grid of training combinations
# Identify unique token index-CV combinations
uniq_splits = cv_splits[['repeat','fold']].drop_duplicates().reset_index(drop=True)
uniq_splits['key'] = 1
uniq_tune_idx = tuning_grid
uniq_tune_idx['key'] = 1
uniq_tune_idx = tuning_grid[['key','TUNE_IDX']]
cv_ti_combos = pd.merge(uniq_splits,uniq_tune_idx,how='outer',on='key').drop(columns='key')

# Define which repeat partitions to train in current run
REPEAT = [1]
cv_ti_combos = cv_ti_combos[cv_ti_combos.repeat.isin(REPEAT)].reset_index(drop=True)

### III. Train APM_deep model based on provided hyperparameter row index
def main(array_task_id):

    # Extract current repeat, fold, and tuning index information
    curr_repeat = cv_ti_combos.repeat[array_task_id]
    curr_fold = cv_ti_combos.fold[array_task_id]
    curr_TUNE_IDX = cv_ti_combos.TUNE_IDX[array_task_id]

    # Create a directory for the current repeat
    repeat_dir = os.path.join(model_dir,'repeat'+str(curr_repeat).zfill(int(np.log10(cv_splits.repeat.max()))+1))
    os.makedirs(repeat_dir,exist_ok=True)
    
    # Create a directory for the current fold
    fold_dir = os.path.join(repeat_dir,'fold'+str(curr_fold).zfill(int(np.log10(cv_splits.fold.max()))+1))
    os.makedirs(fold_dir,exist_ok=True)
    
    # Create a directory for the current tune index
    tune_dir = os.path.join(fold_dir,'tune_'+curr_TUNE_IDX)
    os.makedirs(tune_dir,exist_ok = True)
    
    # Load current token-indexed training and testing sets
    training_set = pd.read_pickle('/home/sb2406/rds/hpc-work/APM_tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/training_indices.pkl')
    testing_set = pd.read_pickle('/home/sb2406/rds/hpc-work/APM_tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/testing_indices.pkl')

    # Load current token dictionary
    curr_vocab = cp.load(open('/home/sb2406/rds/hpc-work/APM_tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/token_dictionary.pkl',"rb"))
    
    # Unique GUPI-GOSE combinations
    study_GUPIs = cv_splits[['GUPI','GOSE']].drop_duplicates()
    
    # Add GOSE scores to training and testing sets
    training_set = pd.merge(training_set,study_GUPIs,how='left',on='GUPI')
    testing_set = pd.merge(testing_set,study_GUPIs,how='left',on='GUPI')
    
    # Set aside 15% of the training set for validation, independent of the final testing set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15)
    for train_index, val_index in sss.split(training_set.drop(columns='GOSE'),training_set.GOSE):
        training_set, val_set = training_set.loc[train_index].reset_index(drop=True), training_set.loc[val_index].reset_index(drop=True)
    
    cp.dump(sss, open(os.path.join(tune_dir,'val_set_splitter.pkl'), "wb"))
    
    # Extract current tuning configurations based on the tuning index
    curr_tune_configs = tuning_grid[tuning_grid.TUNE_IDX == curr_TUNE_IDX].reset_index(drop=True)

    # Create PyTorch Dataset objects
    train_Dataset = ALL_PREDICTOR_SET(training_set,curr_tune_configs.OUTPUT_ACTIVATION[0])
    val_Dataset = ALL_PREDICTOR_SET(val_set,curr_tune_configs.OUTPUT_ACTIVATION[0])
    test_Dataset = ALL_PREDICTOR_SET(testing_set,curr_tune_configs.OUTPUT_ACTIVATION[0])
    
    # Create PyTorch DataLoader objects
    curr_train_DL = DataLoader(train_Dataset,
                               batch_size=int(curr_tune_configs.BATCH_SIZE[0]),
                               shuffle=True,
                              collate_fn=collate_batch)
    
    curr_val_DL = DataLoader(val_Dataset,
                             batch_size=len(val_Dataset), 
                             shuffle=False,
                            collate_fn=collate_batch)
    
    curr_test_DL = DataLoader(test_Dataset,
                             batch_size=len(test_Dataset),
                             shuffle=False,
                             collate_fn=collate_batch)
    
    # Initialize current model class based on hyperparameter selections
    model = APM_deep(len(curr_vocab),
                     curr_tune_configs.LAYERS[0],
                     curr_tune_configs.NEURONS[0],
                     curr_tune_configs.DROPOUT[0],
                     curr_tune_configs.OUTPUT_ACTIVATION[0],
                     curr_tune_configs.LEARNING_RATE[0],
                     curr_tune_configs.CLASS_WEIGHTS[0],
                     train_Dataset.y)
    
    early_stop_callback = EarlyStopping(
        monitor='val_AUROC',
        patience=curr_tune_configs.ES_PATIENCE[0],
        mode='max'
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_AUROC',
        dirpath=tune_dir,
        filename='{epoch:02d}-{val_AUROC:.2f}',
        save_top_k=1,
        mode='max'
    )
    
    csv_logger = pl.loggers.CSVLogger(save_dir=fold_dir,name='tune_'+curr_TUNE_IDX)

    trainer = pl.Trainer(gpus = 1,
                         logger = csv_logger,
                         max_epochs = curr_tune_configs.EPOCHS[0],
                         enable_progress_bar = True,
                         enable_model_summary = True,
                         callbacks=[early_stop_callback,checkpoint_callback])
    
    trainer.fit(model,curr_train_DL,curr_val_DL)
    
    best_model = APM_deep.load_from_checkpoint(checkpoint_callback.best_model_path)
    best_model.eval()
    
    # Save validation set probabilities
    for i, (gupis, idx_list, y_list, pt_offsets) in enumerate(curr_val_DL):

        yhat = best_model(idx_list, pt_offsets)
        val_true_y = y_list.cpu().numpy()

        if curr_tune_configs.OUTPUT_ACTIVATION[0] == 'softmax': 

            curr_val_probs = F.softmax(yhat.detach()).cpu().numpy()
            curr_val_preds = pd.DataFrame(curr_val_probs,columns=['Pr(GOSE=1)','Pr(GOSE=2/3)','Pr(GOSE=4)','Pr(GOSE=5)','Pr(GOSE=6)','Pr(GOSE=7)','Pr(GOSE=8)'])
            curr_val_preds['TrueLabel'] = val_true_y

        elif curr_tune_configs.OUTPUT_ACTIVATION[0] == 'sigmoid': 

            curr_val_probs = F.sigmoid(yhat.detach()).cpu().numpy()
            curr_val_probs = pd.DataFrame(curr_val_probs,columns=['Pr(GOSE>1)','Pr(GOSE>3)','Pr(GOSE>4)','Pr(GOSE>5)','Pr(GOSE>6)','Pr(GOSE>7)'])
            curr_val_labels = pd.DataFrame(val_true_y,columns=['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7'])
            curr_val_preds = pd.concat([curr_val_probs,curr_val_labels],axis = 1)

        else:
            raise ValueError("Invalid output layer type. Must be 'softmax' or 'sigmoid'")

        curr_val_preds.insert(loc=0, column='GUPI', value=gupis)        
        curr_val_preds['TUNE_IDX'] = curr_TUNE_IDX

        curr_val_preds.to_csv(os.path.join(tune_dir,'val_predictions.csv'),index=False)
        
    best_model.eval()
        
    # Save testing set probabilities
    for i, (gupis, idx_list, y_list, pt_offsets) in enumerate(curr_test_DL):
        
        yhat = best_model(idx_list, pt_offsets)
        test_true_y = y_list.cpu().numpy()

        if curr_tune_configs.OUTPUT_ACTIVATION[0] == 'softmax': 

            curr_test_probs = F.softmax(yhat.detach()).cpu().numpy()
            curr_test_preds = pd.DataFrame(curr_test_probs,columns=['Pr(GOSE=1)','Pr(GOSE=2/3)','Pr(GOSE=4)','Pr(GOSE=5)','Pr(GOSE=6)','Pr(GOSE=7)','Pr(GOSE=8)'])
            curr_test_preds['TrueLabel'] = test_true_y

        elif curr_tune_configs.OUTPUT_ACTIVATION[0] == 'sigmoid': 

            curr_test_probs = F.sigmoid(yhat.detach()).cpu().numpy()
            curr_test_probs = pd.DataFrame(curr_test_probs,columns=['Pr(GOSE>1)','Pr(GOSE>3)','Pr(GOSE>4)','Pr(GOSE>5)','Pr(GOSE>6)','Pr(GOSE>7)'])
            curr_test_labels = pd.DataFrame(test_true_y,columns=['GOSE>1','GOSE>3','GOSE>4','GOSE>5','GOSE>6','GOSE>7'])
            curr_test_preds = pd.concat([curr_test_probs,curr_test_labels],axis = 1)

        else:
            raise ValueError("Invalid output layer type. Must be 'softmax' or 'sigmoid'")

        curr_test_preds.insert(loc=0, column='GUPI', value=gupis)        
        curr_test_preds['TUNE_IDX'] = curr_TUNE_IDX

        curr_test_preds.to_csv(os.path.join(tune_dir,'test_predictions.csv'),index=False)
    
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)