#### Master Script 18: Perform ordinal regression analysis on study characteristics and predictors ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Perform ordinal regression analysis on summary characteristics
# III. Perform ordinal regression analysis on CPM characteristics
# IV. Perform ordinal regression analysis on eCPM characteristics

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

# Custom methods
from functions.analysis import ordinal_analysis

# Define and intialise ordinal regression analysis
ordinal_dir = '../ordinal_analysis'
os.makedirs(ordinal_dir,exist_ok=True)

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

### II. Perform ordinal regression analysis on summary characteristics
# Initialise subdirectory for storing summary characterisitic analysis outputs
summary_dir = os.path.join(ordinal_dir,'summary')
os.makedirs(summary_dir,exist_ok=True)

# Load summary characteristics from CENTER-TBI dataset
summary_characteristics = pd.read_csv('../CENTER-TBI/DemoInjHospMedHx/data.csv',na_values = ["NA","NaN"," ", ""])
summary_characteristics = summary_characteristics[summary_characteristics.GUPI.isin(cv_splits.GUPI.unique())].reset_index(drop=True)
summary_characteristics = summary_characteristics[['GUPI','Age','Sex','Race','GCSScoreBaselineDerived','GOSE6monthEndpointDerived']]
summary_characteristics['Severity'] = pd.cut(summary_characteristics['GCSScoreBaselineDerived'],bins=[2,8,12,15],labels=['Severe','Moderate','Mild'],ordered=False)
summary_characteristics['Race'][summary_characteristics['Race'] == 'Unknown'] = np.nan
summary_characteristics['Race'][summary_characteristics['Race'] == 'NotAllowed'] = np.nan

# Convert GOSE to ordered category type
summary_characteristics['GOSE6monthEndpointDerived'] = summary_characteristics['GOSE6monthEndpointDerived'].astype(CategoricalDtype(categories=['1', '2_or_3', '4', '5', '6', '7', '8'],ordered=True))

# Remove redundant columns and rename GOSE column
summary_characteristics = summary_characteristics.drop(columns='GCSScoreBaselineDerived').rename(columns={'GOSE6monthEndpointDerived':'GOSE'})

# Remove rows with missing values
summary_characteristics = summary_characteristics.dropna().reset_index(drop=True)

# Convert categorical characteristics to proper type
cat_encoder = OneHotEncoder(drop = 'first',categories=[['M','F'],['White','Black','Asian'],['Mild','Moderate','Severe']])
cat_column_names = ['Female','Race_Black','Race_Asian','Severity_Moderate','Severity_Severe']

summary_categorical = pd.DataFrame(cat_encoder.fit_transform(summary_characteristics[['Sex','Race','Severity']]).toarray(),
                                   columns=cat_column_names)
summary_characteristics = pd.concat([summary_characteristics.drop(columns=['Sex','Race','Severity']),summary_categorical],axis=1)

summary_POLR = OrderedModel(summary_characteristics['GOSE'],
                            summary_characteristics[summary_characteristics.columns[~summary_characteristics.columns.isin(['GUPI','GOSE'])]],
                            distr='logit')
res_summary_POLR = summary_POLR.fit(method='bfgs',
                                    maxiter = 1000,
                                    disp=False)
res_summary_POLR.save(os.path.join(summary_dir,'summary_polr.pkl'))

### III. Perform ordinal regression analysis on CPM characteristics
# Initialise subdirectory for storing CPM characterisitic analysis outputs
CPM_dir = os.path.join(ordinal_dir,'CPM')
os.makedirs(CPM_dir,exist_ok=True)

# Split repeated cross-validation folds across cores for parallel processing
CPM_core_splits = [(uniq_splits.iloc[start_indices[idx]:end_indices[idx],:].reset_index(drop=True),CPM_dir,'CPM') for idx in range(len(start_indices))]

# Perform ordinal analysis in parallel and collect coefficients and p-values
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_CPM_analysis = pd.concat(pool.starmap(ordinal_analysis, CPM_core_splits),ignore_index=True)
    
# Save analysis outputs for CPM
compiled_CPM_analysis.to_csv(os.path.join(CPM_dir,'compiled_analysis_outputs.csv'),index=False)

# Convert standard error to variance
compiled_CPM_analysis['STDVAR'] = compiled_CPM_analysis['STDERR'] ** 2

# For each predictor, calculate necessary mean statistics and append to compiled analyssi dataframe
mean_CPM_stats = compiled_CPM_analysis.groupby('Predictor').aggregate({'Z':['mean','count'],'COEF':['mean','count'],'STDVAR':['mean','count']})
mean_CPM_stats.columns = ['_'.join(col).strip() for col in mean_CPM_stats.columns.values]
mean_CPM_stats = mean_CPM_stats.reset_index()
compiled_CPM_analysis = compiled_CPM_analysis.merge(mean_CPM_stats,how='left',on='Predictor')

# Calculate between-imputation variation
compiled_CPM_analysis['Z_DIFF_SQUARED_DIV'] = ((compiled_CPM_analysis['Z']-compiled_CPM_analysis['Z_mean'])**2)/(compiled_CPM_analysis['Z_count']-1)
compiled_CPM_analysis['COEF_DIFF_SQUARED_DIV'] = ((compiled_CPM_analysis['COEF']-compiled_CPM_analysis['COEF_mean'])**2)/(compiled_CPM_analysis['COEF_count']-1)
var_b_CPM_stats = compiled_CPM_analysis.groupby('Predictor',as_index=False).aggregate({'Z_DIFF_SQUARED_DIV':'sum','COEF_DIFF_SQUARED_DIV':'sum'})
var_b_CPM_stats.columns = ['Predictor','Z_var_b','COEF_var_b']

# Compile mean and variance information
CPM_stats = pd.merge(mean_CPM_stats,var_b_CPM_stats,how='left',on='Predictor')

# Calculate degrees of freedom for p values
B_m = CPM_stats['Z_var_b']
m = CPM_stats['Z_count']
z_df = (m-1)*((( (m*B_m) + B_m + m )/( (m*B_m) + B_m ))**2)
var_T = 1 + (((m+1)/(m))*B_m)
CPM_stats['Z_df'] = z_df
CPM_stats['Z_var_T'] = var_T

# Calculate total variance for coefficients
V_w = CPM_stats['STDVAR_mean']
V_b = CPM_stats['COEF_var_b']
m = CPM_stats['COEF_count']
V_T = V_w + (((m+1)/(m))*V_b)
CPM_stats['COEF_var_T'] = V_T

# Calculate p-values from degrees of freedom, total variance, and mean
z_df = CPM_stats['Z_df']
z_scales = np.sqrt(CPM_stats['Z_var_T'])
z_means = CPM_stats['Z_mean']
CPM_stats['pvals'] = stats.t.sf(x=z_means,df=z_df,scale=z_scales)

# Extract meaningful columns and save analysis dataframe
CPM_stats = CPM_stats[['Predictor','COEF_mean','COEF_var_T','pvals']]
CPM_stats['Significant'] = (CPM_stats['pvals'] <= .05).astype(int)
CPM_stats.to_csv(os.path.join(CPM_dir,'analysis_statistics.csv'),index=False)

CPM_stats['pvals'] = CPM_stats['pvals'].apply("{:.04f}".format)

### IV. Perform ordinal regression analysis on eCPM characteristics
# Initialise subdirectory for storing eCPM characterisitic analysis outputs
eCPM_dir = os.path.join(ordinal_dir,'eCPM')
os.makedirs(eCPM_dir,exist_ok=True)

# Split repeated cross-validation folds across cores for parallel processing
eCPM_core_splits = [(uniq_splits.iloc[start_indices[idx]:end_indices[idx],:].reset_index(drop=True),eCPM_dir,'eCPM') for idx in range(len(start_indices))]

# Perform ordinal analysis in parallel and collect coefficients and p-values
with multiprocessing.Pool(NUM_CORES) as pool:
    compiled_eCPM_analysis = pd.concat(pool.starmap(ordinal_analysis, eCPM_core_splits),ignore_index=True)
    
# Save analysis outputs for eCPM
compiled_eCPM_analysis.to_csv(os.path.join(eCPM_dir,'compiled_analysis_outputs.csv'),index=False)

# Convert standard error to variance
compiled_eCPM_analysis['STDVAR'] = compiled_eCPM_analysis['STDERR'] ** 2

# For each predictor, calculate necessary mean statistics and append to compiled analyssi dataframe
mean_eCPM_stats = compiled_eCPM_analysis.groupby('Predictor').aggregate({'Z':['mean','count'],'COEF':['mean','count'],'STDVAR':['mean','count']})
mean_eCPM_stats.columns = ['_'.join(col).strip() for col in mean_eCPM_stats.columns.values]
mean_eCPM_stats = mean_eCPM_stats.reset_index()
compiled_eCPM_analysis = compiled_eCPM_analysis.merge(mean_eCPM_stats,how='left',on='Predictor')

# Calculate between-imputation variation
compiled_eCPM_analysis['Z_DIFF_SQUARED_DIV'] = ((compiled_eCPM_analysis['Z']-compiled_eCPM_analysis['Z_mean'])**2)/(compiled_eCPM_analysis['Z_count']-1)
compiled_eCPM_analysis['COEF_DIFF_SQUARED_DIV'] = ((compiled_eCPM_analysis['COEF']-compiled_eCPM_analysis['COEF_mean'])**2)/(compiled_eCPM_analysis['COEF_count']-1)
var_b_eCPM_stats = compiled_eCPM_analysis.groupby('Predictor',as_index=False).aggregate({'Z_DIFF_SQUARED_DIV':'sum','COEF_DIFF_SQUARED_DIV':'sum'})
var_b_eCPM_stats.columns = ['Predictor','Z_var_b','COEF_var_b']

# Compile mean and variance information
eCPM_stats = pd.merge(mean_eCPM_stats,var_b_eCPM_stats,how='left',on='Predictor')

# Calculate degrees of freedom for p values
B_m = eCPM_stats['Z_var_b']
m = eCPM_stats['Z_count']
z_df = (m-1)*((( (m*B_m) + B_m + m )/( (m*B_m) + B_m ))**2)
var_T = 1 + (((m+1)/(m))*B_m)
eCPM_stats['Z_df'] = z_df
eCPM_stats['Z_var_T'] = var_T

# Calculate total variance for coefficients
V_w = eCPM_stats['STDVAR_mean']
V_b = eCPM_stats['COEF_var_b']
m = eCPM_stats['COEF_count']
V_T = V_w + (((m+1)/(m))*V_b)
eCPM_stats['COEF_var_T'] = V_T

# Calculate p-values from degrees of freedom, total variance, and mean
z_df = eCPM_stats['Z_df']
z_scales = np.sqrt(eCPM_stats['Z_var_T'])
z_means = eCPM_stats['Z_mean']
eCPM_stats['pvals'] = stats.t.sf(x=z_means,df=z_df,scale=z_scales)

# Extract meaningful columns and save analysis dataframe
eCPM_stats = eCPM_stats[['Predictor','COEF_mean','COEF_var_T','pvals']]
eCPM_stats['Significant'] = (eCPM_stats['pvals'] <= .05).astype(int)
eCPM_stats.to_csv(os.path.join(eCPM_dir,'analysis_statistics.csv'),index=False)

eCPM_stats['pvals'] = eCPM_stats['pvals'].apply("{:.04f}".format)