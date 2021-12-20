#### Master Script 3: Prepare concise predictor set for ordinal prediction ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Load IMPACT variables from CENTER-TBI
# III. Multiply impute concise predictor set in parallel

### I. Initialisation
# Import necessary libraries
library(tidyverse)
library(lubridate)
library(mice)
library(doParallel)
library(foreach)

# Load cross-validation splits
cv_splits = read.csv('../cross_validation_splits.csv')

# Create directory to store imputed concise predictor sets
dir.create('../imputed_CPM_sets',showWarnings = F)

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

### III. Multiply impute concise predictor set in parallel
# Set the number of parallel cores
NUM.CORES <- detectCores() - 2

# Initialize local cluster for parallel processing
registerDoParallel(cores = NUM.CORES)

# Iterate through cross-validation repeats
foreach(curr_repeat = unique(cv_splits$repeat.), .inorder = F) %dopar% {

  # Create directory to store imputed IMPACT datasets for current repeat
  dir.create(paste0('../imputed_CPM_sets/repeat',sprintf('%02d',curr_repeat)),showWarnings = F)

  for (curr_fold in unique(cv_splits$fold)){

    # Create directory to store imputed IMPACT datasets for current repeat-fold combination
    dir.create(paste0('../imputed_CPM_sets/repeat',sprintf('%02d',curr_repeat),'/fold',curr_fold),showWarnings = F)

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
        data = CENTER_TBI_IMPACT %>% select(-GOSE),
        ignore = CENTER_TBI_IMPACT$GUPI %in% curr_test_set$GUPI,
        m = 1,
        seed = (curr_repeat - 1)*length(unique(cv_splits$fold)) + curr_fold,
        maxit = 30,
        method = 'pmm',
        printFlag = TRUE
      )

    # Save multiple imputation object
    saveRDS(mi.impact,paste0('../imputed_CPM_sets/repeat',sprintf('%02d',curr_repeat),'/fold',curr_fold,'/mice_object.rds'))

    # Extract imputed dataset
    curr.imp <- complete(mi.impact, action = 1)

    # Split into training and testing sets, and save files
    curr.training.imp <- curr.imp %>%
      filter(GUPI %in% curr_train_set$GUPI) %>%
      left_join(curr_train_set %>% select(GUPI,GOSE), by = 'GUPI')

    write.csv(curr.training.imp,
              paste0('../imputed_CPM_sets/repeat',sprintf('%02d',curr_repeat),'/fold',curr_fold,'/training_set.csv'),
              row.names = F)

    curr.testing.imp <- curr.imp %>%
      filter(GUPI %in% curr_test_set$GUPI) %>%
      left_join(curr_test_set %>% select(GUPI,GOSE), by = 'GUPI')

    write.csv(curr.testing.imp,
              paste0('../imputed_CPM_sets/repeat',sprintf('%02d',curr_repeat),'/fold',curr_fold,'/testing_set.csv'),
              row.names = F)

  }
}