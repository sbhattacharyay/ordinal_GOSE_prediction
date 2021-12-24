#### Master Script 13: Prepare extended concise predictor set for ordinal prediction ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Load IMPACT variables from CENTER-TBI
# III. Load and prepare added variables from CENTER-TBI
# IV. Multiply impute concise predictor set in parallel

### I. Initialisation
# Import necessary libraries
library(tidyverse)
library(lubridate)
library(mice)
library(doParallel)
library(foreach)

# Load cross-validation splits
cv_splits = read.csv('../cross_validation_splits.csv')

# Create directory to store imputed extended concise predictor sets
dir.create('../imputed_eCPM_sets',showWarnings = F)

### II. Load IMPACT variables from CENTER-TBI
# Load IMPACT-specific variables from CENTER-TBI dataset
CENTER_TBI_IMPACT = read.csv('../CENTER-TBI/IMPACT/data.csv',
                             na.strings=c("NA","NaN","", " ")) %>%
  rename(GUPI = entity_id)

# Load ICU admission and discharge timestamps for study patients
CENTER_TBI_ICU_datetime = read.csv('../ICU_adm_disch_timestamps.csv') %>%
  mutate(ICUAdmTimeStamp = as.POSIXct(ICUAdmTimeStamp,
                                      format = '%Y-%m-%d %H:%M:%S',
                                      tz = 'GMT')) %>%
  mutate(LastTimeStamp = ICUAdmTimeStamp+hours(24))
study_GUPIs = unique(CENTER_TBI_ICU_datetime$GUPI)

# Filter patients in study and drop unused columns
CENTER_TBI_IMPACT = CENTER_TBI_IMPACT %>%
  filter(GUPI %in% study_GUPIs) %>%
  select(-c(SiteCode,GCS,PatientType))

### III. Load and prepare added variables from CENTER-TBI
# Extract employment status, education status, post-traumatic amnesia, and best of HBC ISS
demo_info = read.csv('../CENTER-TBI/DemoInjHospMedHx/data.csv',
                     na.strings=c("NA","NaN","", " ")) %>%
  select(GUPI,EmplmtStatus,EduLvlUSATyp,BestOfHeadBrainCervicalISS,LOCPTA) %>%
  filter(GUPI %in% study_GUPIs)

# Convert ISS == 0 to 1 (0 not feasible for brain injured patients)
demo_info$BestOfHeadBrainCervicalISS[demo_info$BestOfHeadBrainCervicalISS == 0] = 1

# Convert coding of employment status to retirement indicator, post-traumatic amnesia to ongoing indicator, and ISS to categorical
demo_info = demo_info %>%
  mutate(Retired = as.integer(EmplmtStatus == 8),
         PTA = as.integer(LOCPTA == 1),
         BestOfHeadBrainCervicalISS = factor(BestOfHeadBrainCervicalISS)) %>%
  mutate(WorstHBCAIS = as.integer(BestOfHeadBrainCervicalISS)) %>%
  select(-c(EmplmtStatus,LOCPTA,BestOfHeadBrainCervicalISS))

# Extract protein biomarker information from CENTER-TBI
biomarkers = read.csv('../CENTER-TBI/Biomarkers/data.csv',
                      na.strings=c("NA","NaN","", " ")) %>%
  filter(GUPI %in% study_GUPIs) %>%
  filter(!is.na(CollectionDate) | !is.na(CentrifugationDate) | !is.na(FreezerMinusTwentyDate)
         | !is.na(FreezerMinusEightyDate))

# For timestamps that have dates but no times, replace with median time
medianCollectionTime <- sort(biomarkers$CollectionTime)[ceiling(length(sort(biomarkers$CollectionTime))/2)]
biomarkers[is.na(biomarkers$CollectionTime) & !is.na(biomarkers$CollectionDate),'CollectionTime'] <- medianCollectionTime
biomarkers[is.na(biomarkers$CentrifugationTime) & !is.na(biomarkers$CentrifugationDate),'CentrifugationTime'] <- biomarkers[is.na(biomarkers$CentrifugationTime) & !is.na(biomarkers$CentrifugationDate),'CollectionTime']

# Format timestamps as datetime variables
biomarkers <- biomarkers %>%
  mutate(CollectionTS = as.POSIXct(paste(CollectionDate,CollectionTime),format = '%Y-%m-%d %H:%M:%S',tz = 'GMT'),
         CentrifugationTS = as.POSIXct(paste(CentrifugationDate,CentrifugationTime),format = '%Y-%m-%d %H:%M:%S',tz = 'GMT'),
         FreezerMinusTwentyTS = as.POSIXct(paste(FreezerMinusTwentyDate,FreezerMinusTwentyTime),format = '%Y-%m-%d %H:%M:%S',tz = 'GMT'),
         FreezerMinusEightyTS = as.POSIXct(paste(FreezerMinusEightyDate,FreezerMinusEightyTime),format = '%Y-%m-%d %H:%M:%S',tz = 'GMT')) %>%
  select(-contains('Date'),-contains('Time')) %>%
  mutate(Timestamp = CentrifugationTS)

# Set best approximate timestamp of sample if centrifugation timestamp is missing
biomarkers[is.na(biomarkers$Timestamp),'Timestamp'] <- biomarkers[is.na(biomarkers$Timestamp),'CollectionTS'] 
biomarkers[is.na(biomarkers$Timestamp),'Timestamp'] <- biomarkers[is.na(biomarkers$Timestamp),'FreezerMinusTwentyTS'] 
biomarkers[is.na(biomarkers$Timestamp),'Timestamp'] <- biomarkers[is.na(biomarkers$Timestamp),'FreezerMinusEightyTS'] 

# Filter out biomarker results available within 24 hours of ICU admission and select GFAP, Tau, S100B, and NFL
biomarkers <- biomarkers %>%
  relocate(Timestamp, .after = GUPI) %>%
  select(-contains('TS')) %>%
  .[which(rowSums(!is.na(.[,3:7])) != 0),] %>%
  left_join(CENTER_TBI_ICU_datetime %>% select(GUPI,LastTimeStamp),by = 'GUPI') %>%
  filter(Timestamp <= LastTimeStamp) %>%
  select(GUPI,GFAP,Tau,S100B,NFL)

# Replace units mismatch of S100B
biomarkers$S100B[(biomarkers$S100B >= 1000)&(!is.na(biomarkers$S100B))] <- biomarkers$S100B[(biomarkers$S100B >= 1000)&(!is.na(biomarkers$S100B))]/1000

# If multiple biomarker samples exist for a patient in the first 24 hours, take the maximum value
biomarkers <- biomarkers %>%
  group_by(GUPI) %>%
  summarise(GFAP = max(GFAP,na.rm=T),
            Tau = max(Tau,na.rm=T),
            S100B = max(S100B,na.rm=T),
            NFL = max(NFL,na.rm=T))

# Replace missing value markers with NA
biomarkers[biomarkers == -Inf] <- NA

# Combine demographic information and biomarker dataframes into single added predictor dataframe
added_vars <- demo_info %>%
  left_join(biomarkers,by = 'GUPI')

# Merge added predictors with original concise predictor set
extended_CENTER_TBI_IMPACT <- CENTER_TBI_IMPACT %>%
  left_join(added_vars,by='GUPI')

### III. Multiply impute concise predictor set in parallel
# Set the number of parallel cores
NUM.CORES <- detectCores() - 2

# Initialize local cluster for parallel processing
registerDoParallel(cores = NUM.CORES)

# Iterate through cross-validation repeats
foreach(curr_repeat = unique(cv_splits$repeat.), .inorder = F) %dopar% {
  
  # Create directory to store imputed IMPACT datasets for current repeat
  dir.create(paste0('../imputed_eCPM_sets/repeat',sprintf('%02d',curr_repeat)),showWarnings = F)
  
  for (curr_fold in unique(cv_splits$fold)){
    
    # Create directory to store imputed IMPACT datasets for current repeat-fold combination 
    dir.create(paste0('../imputed_eCPM_sets/repeat',sprintf('%02d',curr_repeat),'/fold',curr_fold),showWarnings = F)
    
    # Get current training GUPIs
    curr_train_set = cv_splits %>%
      filter(repeat. == curr_repeat,
             fold == curr_fold,
             test_or_train == 'train')
    
    # Get current testing GUPIs
    curr_test_set = cv_splits %>%
      filter(repeat. == curr_repeat,
             fold == curr_fold,
             test_or_train == 'test')
    
    # Train multiple imputation object on the training set
    mi.impact <-
      mice(
        data = extended_CENTER_TBI_IMPACT %>% select(-GOSE),
        ignore = extended_CENTER_TBI_IMPACT$GUPI %in% curr_test_set$GUPI,
        m = 1,
        seed = (curr_repeat - 1)*length(unique(cv_splits$fold)) + curr_fold,
        maxit = 30,
        method = 'pmm',
        printFlag = TRUE
      )
    
    # Save multiple imputation object
    saveRDS(mi.impact,paste0('../imputed_eCPM_sets/repeat',sprintf('%02d',curr_repeat),'/fold',curr_fold,'/mice_object.rds'))
    
    # Extract imputed dataset
    curr.imp <- complete(mi.impact, action = 1)
    
    # Split into training and testing sets, and save files
    curr.training.imp <- curr.imp %>%
      filter(GUPI %in% curr_train_set$GUPI) %>%
      left_join(curr_train_set %>% select(GUPI,GOSE), by = 'GUPI')
    
    write.csv(curr.training.imp,
              paste0('../imputed_eCPM_sets/repeat',sprintf('%02d',curr_repeat),'/fold',curr_fold,'/training_set.csv'),
              row.names = F)
    
    curr.testing.imp <- curr.imp %>%
      filter(GUPI %in% curr_test_set$GUPI) %>%
      left_join(curr_test_set %>% select(GUPI,GOSE), by = 'GUPI')
    
    write.csv(curr.testing.imp,
              paste0('../imputed_eCPM_sets/repeat',sprintf('%02d',curr_repeat),'/fold',curr_fold,'/testing_set.csv'),
              row.names = F)
    
  }
}