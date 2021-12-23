#### Master Script 15a: Train deep learning extended concise-predictor-based models (eCPM) ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Create grid of training combinations
# III. Train eCPM_deep model based on provided hyperparameter row index

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
from classes.datasets import CONCISE_PREDICTOR_SET
from models.CPM import CPM_deep
from functions.model_building import train_CPM_deep

# Set version code
VERSION = 'DEEP_v1-0'

# Initialise model output directory based on version code
model_dir = '/home/sb2406/rds/hpc-work/eCPM_outputs/'+VERSION
os.makedirs(model_dir,exist_ok=True)

# Load cross-validation split information
cv_splits = pd.read_csv('../cross_validation_splits.csv')

# Load the tuning grid
# tuning_grid = pd.read_csv(os.path.join(model_dir,'eCPM_deep_tuning_grid.csv'))
# tuning_grid['TUNE_IDX'] = tuning_grid['TUNE_IDX'].astype(str).str.zfill(4)
# tuning_grid['NEURONS'] = tuning_grid['NEURONS'].apply(eval)

tuning_grid = pd.read_csv(os.path.join(model_dir,'eCPM_post_repeat_'+str(16).zfill(2)+'_deep_tuning_grid.csv'))
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
REPEAT = [i for i in range(17,21)]
cv_ti_combos = cv_ti_combos[cv_ti_combos.repeat.isin(REPEAT)].reset_index(drop=True)

### III. Train eCPM_deep model based on provided hyperparameter row index
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
    tune_dir = os.path.join(fold_dir,'tune_'+str(curr_TUNE_IDX).zfill(int(np.log10(cv_splits.fold.max()))+1))
    os.makedirs(tune_dir,exist_ok=True)
    
    # Load current imputed training and testing sets
    training_set = pd.read_csv('/home/sb2406/rds/hpc-work/imputed_eCPM_sets/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/training_set.csv')
    testing_set = pd.read_csv('/home/sb2406/rds/hpc-work/imputed_eCPM_sets/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/testing_set.csv')

    # One-hot encode categorical predictors
    cat_encoder = OneHotEncoder(drop = 'first',categories=[[1,2,3,4,5,6],[1,2,3,4,5,6],[0,1,2],[0,1,2,3,4,5],[1,2,3,4,5,6]])
    cat_column_names = ['GCSm_'+str(i+1) for i in range(1,6)] + \
    ['marshall_'+str(i+1) for i in range(1,6)] + \
    ['unreactive_pupils_'+str(i+1) for i in range(2)] + \
    ['EduLvlUSATyp_'+str(i+1) for i in range(5)] + \
    ['WorstHBCAIS_'+str(i+1) for i in range(1,6)]
    
    training_categorical = pd.DataFrame(cat_encoder.fit_transform(training_set[['GCSm','marshall','unreactive_pupils','EduLvlUSATyp','WorstHBCAIS']]).toarray(),
                                        columns=cat_column_names)
    training_set = pd.concat([training_set.drop(columns=['GCSm','marshall','unreactive_pupils','EduLvlUSATyp','WorstHBCAIS']),training_categorical],axis=1)

    testing_categorical = pd.DataFrame(cat_encoder.transform(testing_set[['GCSm','marshall','unreactive_pupils','EduLvlUSATyp','WorstHBCAIS']]).toarray(),
                                columns=cat_column_names)
    testing_set = pd.concat([testing_set.drop(columns=['GCSm','marshall','unreactive_pupils','EduLvlUSATyp','WorstHBCAIS']),testing_categorical],axis=1)

    cp.dump(cat_encoder, open(os.path.join(fold_dir,'one_hot_encoder.pkl'), "wb"))

    # Standardize numerical predictors
    num_scaler = StandardScaler()

    training_set[['age','Hb','glu','GFAP', 'Tau', 'S100B', 'NFL']] = num_scaler.fit_transform(training_set[['age','Hb','glu','GFAP', 'Tau', 'S100B', 'NFL']])
    testing_set[['age','Hb','glu','GFAP', 'Tau', 'S100B', 'NFL']] = num_scaler.transform(testing_set[['age','Hb','glu','GFAP', 'Tau', 'S100B', 'NFL']])

    cp.dump(num_scaler, open(os.path.join(fold_dir,'standardizer.pkl'), "wb"))

    # Set aside 15% of the training set for validation, independent of the final testing set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15)
    for train_index, val_index in sss.split(training_set.drop(columns='GOSE'),training_set.GOSE):
        training_set, val_set = training_set.loc[train_index].reset_index(drop=True), training_set.loc[val_index].reset_index(drop=True)

    cp.dump(sss, open(os.path.join(tune_dir,'val_set_splitter.pkl'), "wb"))

    curr_tune_configs = tuning_grid[tuning_grid.TUNE_IDX == curr_TUNE_IDX].reset_index(drop=True)
    
    train_CPM_deep(training_set,
                   val_set,
                   testing_set,
                   curr_TUNE_IDX,
                   curr_repeat,
                   curr_fold,
                   fold_dir,
                   curr_tune_configs.BATCH_SIZE.values[0],
                   curr_tune_configs.LEARNING_RATE.values[0],
                   curr_tune_configs.LAYERS.values[0],
                   curr_tune_configs.NEURONS.values[0],
                   curr_tune_configs.DROPOUT.values[0],
                   curr_tune_configs.ES_PATIENCE.values[0],
                   curr_tune_configs.EPOCHS.values[0],
                   curr_tune_configs.CLASS_WEIGHTS.values[0],
                   curr_tune_configs.OUTPUT_ACTIVATION.values[0])
    
if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])
    main(array_task_id)