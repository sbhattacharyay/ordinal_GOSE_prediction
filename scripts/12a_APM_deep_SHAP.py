#### Master Script 12a: Calculate SHAP values for APM_DeepMN ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Find all top-performing model checkpoint files for SHAP calculation
# III. Calculate SHAP values based on given parameters

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
from functions.model_building import format_shap

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
os.makedirs(shap_dir,exist_ok=True)

### II. Find all top-performing model checkpoint files for SHAP calculation
# Either create or load APM checkpoint information for SHAP value 
if not os.path.exists(os.path.join(shap_dir,'APM_ckpt_info.pkl')):
    
    # Load final validation set performance dataframe and identify optimally performing tuning configurations
    val_performance = pd.read_csv(os.path.join(model_dir,'repeat20','validation_performance.csv'))
    val_performance['TUNE_IDX'] = val_performance['TUNE_IDX'].astype(str).str.zfill(4)
    across_cv_perf = val_performance.groupby(['TUNE_IDX','OUTPUT_ACTIVATION'],as_index=False)['val_ORC'].mean()
    opt_tune_idx = across_cv_perf[across_cv_perf.groupby('OUTPUT_ACTIVATION')['val_ORC'].transform(max) == across_cv_perf['val_ORC']].reset_index(drop=True)
        
    # Find all model checkpoint files in APM output directory
    APM_ckpt_files = []
    for path in Path(model_dir).rglob('*.ckpt'):
        APM_ckpt_files.append(str(path.resolve()))

    # Categorize model checkpoint files based on name
    APM_ckpt_info = pd.DataFrame({'file':APM_ckpt_files,
                                  'TUNE_IDX':[re.search('tune_(.*)/', curr_file).group(1) for curr_file in APM_ckpt_files],
                                  'VERSION':[re.search('APM_outputs/(.*)/repeat', curr_file).group(1) for curr_file in APM_ckpt_files],
                                  'repeat':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in APM_ckpt_files],
                                  'fold':[int(re.search('/fold(.*)/tune_', curr_file).group(1)) for curr_file in APM_ckpt_files]
                                 }).sort_values(by=['repeat','fold','TUNE_IDX','VERSION']).reset_index(drop=True)

    # Filter optimally performing tuning index for multinomial encoding
    APM_ckpt_info = APM_ckpt_info[APM_ckpt_info.TUNE_IDX.isin(opt_tune_idx[opt_tune_idx.OUTPUT_ACTIVATION == 'softmax'].TUNE_IDX)].reset_index(drop=True)
    
    # Create combinations for each possible output type
    output_types = pd.DataFrame({'output_type':['logit','thresh_logit','prob','thresh_prob'],'key':1})
    
    # Merge output type information to checkpoint info dataframe
    APM_ckpt_info['key'] = 1
    APM_ckpt_info = pd.merge(APM_ckpt_info,output_types,how='outer',on='key').drop(columns='key')
    
    # Save model checkpoint information dataframe
    APM_ckpt_info.to_pickle(os.path.join(shap_dir,'APM_ckpt_info.pkl'))
    
else:
    
    # Read model checkpoint information dataframe
    APM_ckpt_info = pd.read_pickle(os.path.join(shap_dir,'APM_ckpt_info.pkl'))
    
### III. Calculate SHAP values based on given parameters
def main(array_task_id):

    # Extract current file, repeat, and fold information
    curr_file = APM_ckpt_info.file[array_task_id]
    curr_repeat = APM_ckpt_info.repeat[array_task_id]
    curr_fold = APM_ckpt_info.fold[array_task_id]
    curr_output_type = APM_ckpt_info.output_type[array_task_id]
    curr_TUNE_IDX = APM_ckpt_info.TUNE_IDX[array_task_id]
    
    # Define current fold directory based on current information
    tune_dir = os.path.join(model_dir,'repeat'+str(curr_repeat).zfill(2),'fold'+str(curr_fold),'tune_'+curr_TUNE_IDX)
    
    # Extract current testing set for current repeat and fold combination
    testing_set = pd.read_pickle('/home/sb2406/rds/hpc-work/APM_tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/testing_indices.pkl')
    testing_set['seq_len'] = testing_set.Index.apply(len)
    testing_set['unknowns'] = testing_set.Index.apply(lambda x: x.count(0))

    # Number of columns to add
    cols_to_add = testing_set['unknowns'].max() - 1

    # Load current token dictionary
    curr_vocab = cp.load(open('/home/sb2406/rds/hpc-work/APM_tokens/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/token_dictionary.pkl',"rb"))

    # Initialize empty dataframe for multihot encoding of testing set
    multihot_matrix = np.zeros([testing_set.shape[0],len(curr_vocab)+cols_to_add])

    # Encode testing set into multihot encoded matrix
    for i in range(testing_set.shape[0]):
        curr_indices = np.array(testing_set.Index[i])
        if sum(curr_indices == 0) > 1:
            zero_indices = np.where(curr_indices == 0)[0]
            curr_indices[zero_indices[1:]] = [len(curr_vocab) + i for i in range(sum(curr_indices == 0)-1)]
        multihot_matrix[i,curr_indices] = 1

    # Load current pretrained model
    model = APM_deep.load_from_checkpoint(curr_file)
    model.eval()
    
    # Extract learned weights from model checkpoint file
    vocab_embed_matrix = model.embedX.weight.detach().numpy()
    vocab_embed_matrix = np.append(vocab_embed_matrix,np.tile(np.expand_dims(vocab_embed_matrix[0,:], axis=0),(cols_to_add,1)),axis=0)
    vocab_embed_weights = np.exp(model.embedW.weight.detach().numpy())
    vocab_embed_weights = np.append(vocab_embed_weights,np.tile(np.expand_dims(vocab_embed_weights[0], axis=0),(cols_to_add,1)),axis=0)
    vocab_embed_matrix = torch.tensor(vocab_embed_matrix*vocab_embed_weights).float()

    # Load modified APM_deep instance based on trained weights and current output type
    if curr_output_type == 'logit':
        shap_model = shap_APM_deep(vocab_embed_matrix,model.hidden2gose,prob=False,thresh=False)
    elif curr_output_type == 'thresh_logit':
        shap_model = shap_APM_deep(vocab_embed_matrix,model.hidden2gose,prob=False,thresh=True)
    elif curr_output_type == 'prob':
        shap_model = shap_APM_deep(vocab_embed_matrix,model.hidden2gose,prob=True,thresh=False)
    elif curr_output_type == 'thresh_prob':
        shap_model = shap_APM_deep(vocab_embed_matrix,model.hidden2gose,prob=True,thresh=True)

    # Initialize deep explainer explanation object
    e = DeepExplainer(shap_model, torch.tensor(multihot_matrix).float())
    
    # Calculate SHAP values and save both explainer object and shap matrices 
    shap_values = e.shap_values(torch.tensor(multihot_matrix).float())
    cp.dump(e, open(os.path.join(tune_dir,'deep_explainer_'+curr_output_type+'.pkl'), "wb"))
    cp.dump(shap_values, open(os.path.join(tune_dir,'shap_arrays_'+curr_output_type+'.pkl'), "wb"))
    
    # Define token labels
    token_labels = curr_vocab.get_itos() + [curr_vocab.get_itos()[0]+'_'+str(i+1).zfill(3) for i in range(cols_to_add)]
    token_labels[0] = token_labels[0]+'_000'
    
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

    # Remove rows which correspond to non-existent or unknown tokens and save formatted dataframe
    shap_df = shap_df[shap_df.Indicator == 1]
    shap_df = shap_df[~shap_df.Token.str.startswith('<unk>_')].reset_index(drop=True)
    shap_df.to_pickle(os.path.join(tune_dir,'shap_dataframe_'+curr_output_type+'.pkl'))
    
    # Calculate correlation among tokens if it does not already exist
    if curr_output_type == 'logit':
        
        corr_matrix = multihot_matrix.copy()
        corr_matrix[corr_matrix == 0] = -1
        corr_matrix = np.matmul(corr_matrix.transpose(),corr_matrix)
        corr_matrix = corr_matrix/multihot_matrix.shape[0]
        corr_matrix = np.triu(corr_matrix,1)
        corr_matrix[np.tril_indices(corr_matrix.shape[0], 1)] = np.nan
        
        corr_df = pd.DataFrame(corr_matrix,columns=token_labels)
        corr_df['Token1'] = token_labels
        corr_df = corr_df.melt(id_vars = 'Token1', var_name = 'Token2', value_name = 'correlation')
        corr_df = corr_df.dropna().reset_index(drop=True)
        corr_df.to_pickle(os.path.join(tune_dir,'correlation_dataframe.xz'),compression='xz')

if __name__ == '__main__':
    
    array_task_id = int(sys.argv[1])    
    main(array_task_id)