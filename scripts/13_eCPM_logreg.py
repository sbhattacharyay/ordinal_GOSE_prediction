#### Master Script 13: Train logistic regression extended concise-predictor-based models (eCPM) ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Define function to train logistic regression eCPMs given repeated cross-validation dataframe
# III. Parallelised training of logistic regression eCPMs and testing set prediction
# IV. Compile testing set predictions

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

# StatsModels methods
from statsmodels.iolib.smpickle import load_pickle
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.discrete.discrete_model import MNLogit

# SciKit-Learn methods
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# TQDM for progress tracking
from tqdm import tqdm

# Custom function to collect testing predictions
from functions.analysis import collect_preds

### II. Define function to train logistic regression eCPMs given repeated cross-validation dataframe 
def train_eCPM_logreg(split_df,model_dir,progress_bar=True,progress_bar_desc = 'Training eCPM_logreg'):
    
    if progress_bar:
        iterator = tqdm(range(split_df.shape[0]),desc=progress_bar_desc)
    else:
        iterator = range(split_df.shape[0])
    
    for curr_split_row in iterator:
        
        # Get current fold and repeat
        curr_repeat = split_df.repeat[curr_split_row]
        curr_fold = split_df.fold[curr_split_row]
        
        # Create directories for current repeat and fold
        repeat_dir = os.path.join(model_dir,'repeat'+str(curr_repeat).zfill(int(np.log10(cv_splits.repeat.max()))+1))
        os.makedirs(repeat_dir,exist_ok=True)
        
        fold_dir = os.path.join(repeat_dir,'fold'+str(curr_fold).zfill(int(np.log10(cv_splits.fold.max()))+1))
        os.makedirs(fold_dir,exist_ok=True)

        # Load current imputed training and testing sets
        training_set = pd.read_csv('../imputed_eCPM_sets/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/training_set.csv')
        testing_set = pd.read_csv('../imputed_eCPM_sets/repeat'+str(curr_repeat).zfill(2)+'/fold'+str(curr_fold)+'/testing_set.csv')

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

        # Convert GOSE to ordered category type
        training_set['GOSE'] = training_set['GOSE'].astype(CategoricalDtype(categories=['1', '2_or_3', '4', '5', '6', '7', '8'],
                                                                            ordered=True))

        # Train ordered logit model and save coefficients
        polr_dir = os.path.join(fold_dir,'polr')
        os.makedirs(polr_dir,exist_ok=True)

        POLR = OrderedModel(training_set['GOSE'],
                            training_set[training_set.columns[~training_set.columns.isin(['GUPI','GOSE'])]],
                            distr='logit')
        res_POLR = POLR.fit(method='bfgs',
                            maxiter = 1000,
                            disp=False)
        res_POLR.save(os.path.join(polr_dir,'polr.pkl'))

        polr_test_probs = res_POLR.model.predict(res_POLR.params,
                                                 exog=testing_set[testing_set.columns[~testing_set.columns.isin(['GUPI','GOSE'])]],
                                                 which='prob')
        polr_test_preds = pd.DataFrame(polr_test_probs,
                                       columns=['Pr(GOSE=1)','Pr(GOSE=2/3)','Pr(GOSE=4)','Pr(GOSE=5)','Pr(GOSE=6)','Pr(GOSE=7)','Pr(GOSE=8)'])
        polr_test_preds['TrueLabel'] = testing_set.GOSE
        polr_test_preds.insert(loc=0, column='GUPI', value=testing_set.GUPI)        

        # Save testing set predictions
        polr_test_preds.to_csv(os.path.join(polr_dir,'test_predictions.csv'),index=False)

        # Train multinomial logit model and save coefficients
        mnlr_dir = os.path.join(fold_dir,'mnlr')
        os.makedirs(mnlr_dir,exist_ok=True)

        MNLR = MNLogit(training_set['GOSE'],
                       training_set[training_set.columns[~training_set.columns.isin(['GUPI','GOSE'])]],
                       distr='logit')
        res_MNLR = MNLR.fit(method='bfgs',
                           maxiter = 1000,
                           disp = False)
        res_MNLR.save(os.path.join(mnlr_dir,'mnlr.pkl'))

        mnlr_test_probs = res_MNLR.model.predict(res_MNLR.params,
                                                 exog=testing_set[testing_set.columns[~testing_set.columns.isin(['GUPI','GOSE'])]])
        mnlr_test_preds = pd.DataFrame(mnlr_test_probs,
                                       columns=['Pr(GOSE=1)','Pr(GOSE=2/3)','Pr(GOSE=4)','Pr(GOSE=5)','Pr(GOSE=6)','Pr(GOSE=7)','Pr(GOSE=8)'])
        mnlr_test_preds['TrueLabel'] = testing_set.GOSE
        mnlr_test_preds.insert(loc=0, column='GUPI', value=testing_set.GUPI)        

        # Save testing set predictions
        mnlr_test_preds.to_csv(os.path.join(mnlr_dir,'test_predictions.csv'),index=False)

### III. Parallelised training of logistic regression eCPMs and testing set prediction
# Set version number and model output directory
VERSION = 'LOGREG_v1-0'
model_dir = '../eCPM_outputs/'+VERSION
os.makedirs(model_dir,exist_ok=True)

# Establish number of cores for all parallel processing
NUM_CORES = multiprocessing.cpu_count() - 2

# Load cross-validation information
cv_splits = pd.read_csv('../cross_validation_splits.csv')

# Unique repeat-fold combinations
uniq_splits = cv_splits[['repeat','fold']].drop_duplicates(ignore_index=True)

# Split up partitions among cores
sizes = [uniq_splits.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
sizes[:(uniq_splits.shape[0] - sum(sizes))] = [val+1 for val in sizes[:(uniq_splits.shape[0] - sum(sizes))]]    
end_indices = np.cumsum(sizes)
start_indices = np.insert(end_indices[:-1],0,0)
core_splits = [(uniq_splits.iloc[start_indices[idx]:end_indices[idx],:].reset_index(drop=True),model_dir) for idx in range(len(start_indices))]

# Run training on repeated-CV partition splits in parallel
with multiprocessing.Pool(NUM_CORES) as pool:
    pool.starmap(train_eCPM_logreg, core_splits)
    
### IV. Compile testing set predictions
## Compile all test predictions from eCPM_MNLR
mnlr_test_pred_files = []
for path in Path(model_dir).rglob('*mnlr/test_predictions.csv'):
    mnlr_test_pred_files.append(str(path.resolve()))

mnlr_testpred_info_df = pd.DataFrame({'file':mnlr_test_pred_files,
                                      'VERSION':[re.search('outputs/(.*)/repeat', curr_file).group(1) for curr_file in mnlr_test_pred_files],
                                      'repeat':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in mnlr_test_pred_files],
                                      'fold':[int(re.search('/fold(.*)/mnlr', curr_file).group(1)) for curr_file in mnlr_test_pred_files]
                                     }).sort_values(by=['repeat','fold','VERSION']).reset_index(drop=True)

# Split up files among cores
sizes = [mnlr_testpred_info_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
sizes[:(mnlr_testpred_info_df.shape[0] - sum(sizes))] = [val+1 for val in sizes[:(mnlr_testpred_info_df.shape[0] - sum(sizes))]]    
end_indices = np.cumsum(sizes)
start_indices = np.insert(end_indices[:-1],0,0)

test_files_per_core = [(mnlr_testpred_info_df.iloc[start_indices[idx]:end_indices[idx],:].reset_index(drop=True),True,'Collecting MNLR predictions') for idx in range(len(start_indices))]

with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_mnlr_test_preds = pd.concat(pool.starmap(collect_preds, test_files_per_core),ignore_index=True)

compiled_mnlr_test_preds.to_csv(os.path.join(model_dir,'compiled_mnlr_test_predictions.csv'),index = False)

## Compile all test predictions from eCPM_POLR
polr_test_pred_files = []
for path in Path(model_dir).rglob('*polr/test_predictions.csv'):
    polr_test_pred_files.append(str(path.resolve()))

polr_testpred_info_df = pd.DataFrame({'file':polr_test_pred_files,
                                      'VERSION':[re.search('outputs/(.*)/repeat', curr_file).group(1) for curr_file in polr_test_pred_files],
                                      'repeat':[int(re.search('/repeat(.*)/fold', curr_file).group(1)) for curr_file in polr_test_pred_files],
                                      'fold':[int(re.search('/fold(.*)/polr', curr_file).group(1)) for curr_file in polr_test_pred_files]
                                     }).sort_values(by=['repeat','fold','VERSION']).reset_index(drop=True)

# Split up files among cores
sizes = [polr_testpred_info_df.shape[0] // NUM_CORES for _ in range(NUM_CORES)]
sizes[:(polr_testpred_info_df.shape[0] - sum(sizes))] = [val+1 for val in sizes[:(polr_testpred_info_df.shape[0] - sum(sizes))]]    
end_indices = np.cumsum(sizes)
start_indices = np.insert(end_indices[:-1],0,0)

test_files_per_core = [(polr_testpred_info_df.iloc[start_indices[idx]:end_indices[idx],:].reset_index(drop=True),True,'Collecting POLR predictions') for idx in range(len(start_indices))]

with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_polr_test_preds = pd.concat(pool.starmap(collect_preds, test_files_per_core),ignore_index=True)

compiled_polr_test_preds.to_csv(os.path.join(model_dir,'compiled_polr_test_predictions.csv'),index = False)