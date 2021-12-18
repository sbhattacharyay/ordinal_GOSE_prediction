#### Master Script 1: Extract study sample from CENTER-TBI dataset ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialization
# II. Load and filter CENTER-TBI dataset
# III. Characterise ICU stay timestamps

### I. Initialization
import os
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
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings(action="ignore")

### II. Load and filter CENTER-TBI dataset
# Load CENTER-TBI dataset to access ICU discharge date/times
CENTER_TBI_demo_info = pd.read_csv('../CENTER-TBI/DemoInjHospMedHx/data.csv',na_values = ["NA","NaN"," ", ""])

# Filter patients who were enrolled in the ICU
CENTER_TBI_demo_info = CENTER_TBI_demo_info[CENTER_TBI_demo_info.PatientType == 3]

# Filter patients who are or are above 16 years of age
CENTER_TBI_demo_info = CENTER_TBI_demo_info[CENTER_TBI_demo_info.Age >= 16]

# Filter patients who have non-missing GOSE scores
CENTER_TBI_demo_info = CENTER_TBI_demo_info[~CENTER_TBI_demo_info.GOSE6monthEndpointDerived.isna()]

### III. Characterise ICU stay timestamps
# Select columns that indicate ICU admission and discharge times
CENTER_TBI_ICU_datetime = CENTER_TBI_demo_info[['GUPI','ICUAdmDate','ICUAdmTime','ICUDischDate','ICUDischTime']]

# Compile date and time information and convert to datetime
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'] = CENTER_TBI_ICU_datetime[['ICUAdmDate', 'ICUAdmTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'][CENTER_TBI_ICU_datetime.ICUAdmDate.isna() | CENTER_TBI_ICU_datetime.ICUAdmTime.isna()] = np.nan
CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = CENTER_TBI_ICU_datetime[['ICUDischDate', 'ICUDischTime']].astype(str).agg(' '.join, axis=1)
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'][CENTER_TBI_ICU_datetime.ICUDischDate.isna() | CENTER_TBI_ICU_datetime.ICUDischTime.isna()] = np.nan
CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] = pd.to_datetime(CENTER_TBI_ICU_datetime['ICUDischTimeStamp'],format = '%Y-%m-%d %H:%M:%S' )

CENTER_TBI_ICU_datetime['ICUDurationHours'] = (CENTER_TBI_ICU_datetime['ICUDischTimeStamp'] - CENTER_TBI_ICU_datetime['ICUAdmTimeStamp']).astype('timedelta64[h]')

# For missing timestamps, cross-check with information available in other study
missing_timestamp_GUPIs = CENTER_TBI_ICU_datetime.GUPI[CENTER_TBI_ICU_datetime['ICUAdmTimeStamp'].isna() | CENTER_TBI_ICU_datetime['ICUDischTimeStamp'].isna()].values

dynamic_study_icu_timestamps = pd.read_csv('../../dynamic_ts_pred/timestamps/ICU_adm_disch_timestamps.csv')
dynamic_study_icu_timestamps = dynamic_study_icu_timestamps[dynamic_study_icu_timestamps.GUPI.isin(missing_timestamp_GUPIs)]

CENTER_TBI_ICU_datetime = CENTER_TBI_ICU_datetime[~CENTER_TBI_ICU_datetime.GUPI.isin(dynamic_study_icu_timestamps.GUPI)]
CENTER_TBI_ICU_datetime = CENTER_TBI_ICU_datetime.append(dynamic_study_icu_timestamps,ignore_index=True)

# Sort timestamps by GUPI
CENTER_TBI_ICU_datetime = CENTER_TBI_ICU_datetime.sort_values(by='GUPI',ignore_index = True)

# Filter out patients with less than 24 hours of ICU stay
CENTER_TBI_ICU_datetime = CENTER_TBI_ICU_datetime[CENTER_TBI_ICU_datetime.ICUDurationHours >= 24]

# Save timestamps as CSV
CENTER_TBI_ICU_datetime.to_csv('../ICU_adm_disch_timestamps.csv',index = False)