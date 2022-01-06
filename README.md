# Baseline ordinal prediction of functional outcomes after traumatic brain injury (TBI) in the ICU
The leap to ordinal: An artificially intelligent approach to predict nuanced functional outcomes in critically ill patients with traumatic brain injury

## Contents

- [Overview](#overview)
- [Abstract](#abstract)
- [Code](#code)
- [License](./LICENSE)
- [Citation](#citation)

## Overview

This repository contains the code underlying the article entitled **The leap to ordinal: An artificially intelligent approach to predict nuanced functional outcomes in critically ill patients with traumatic brain injury** from the Collaborative European NeuroTrauma Effectiveness Research in TBI ([CENTER-TBI](https://www.center-tbi.eu/)) consortium. In this file, we present the abstract, to outline the motivation for the work and the findings, and then a brief description of the code with which we generate these finding and achieve this objective.\
\
The code on this repository is commented throughout to provide a description of each step alongside the code which achieves it.

## Abstract

## Code 
All of the code used in this work can be found in the `./scripts` directory as Python (`.py`), R (`.R`), or bash (`.sh`) scripts. Moreover, custom classes have been saved in the `./scripts/classes` sub-directory, custom functions have been saved in the `./scripts/functions` sub-directory, and custom PyTorch models have been saved in the `./scripts/models` sub-directory.

### 1. [Extract study sample from CENTER-TBI dataset](scripts/01_extract_study_sample.py)

### 2. [Partition CENTER-TBI for stratified, repeated k-fold cross-validation](scripts/02_partition_for_repeated_cv.py)

### 3. [Prepare concise predictor set for ordinal prediction](scripts/03_prepare_concise_predictor_set.R)

### 4. [Train logistic regression concise-predictor-based models (CPM)](scripts/04_CPM_logreg.py)

### 5. [Assess CPM_MNLR and CPM_POLR performance](scripts/05_CPM_logreg_performance.py)

### 6. Train and optimise CPM_DeepMN and CPM_DeepOR

<ol type="a">
  <h4><li><a href="scripts/06a_CPM_deep.py">Train deep learning concise-predictor-based models (CPM)</a></li></h4>
  <h4><li><a href="scripts/06b_CPM_deep_interrepeat_dropout.py">Perform interrepeat hyperparameter configuration dropout on deep learning concise-predictor-based models (CPM)</a></li></h4>
  <h4><li><a href="scripts/06c_CPM_deep_hyperparameter_testing.py">Calculate ORC in bootstrapping resamples to determine dropout configurations</a></li></h4>
</ol>

### 7. Calculate and compile CPM_DeepMN and CPM_DeepOR metrics

<ol type="a">
  <h4><li><a href="scripts/07a_CPM_deep_performance.py">Assess CPM_DeepMN and CPM_DeepOR performance</a></li></h4>
  <h4><li><a href="scripts/07b_CPM_compile_metrics.py">Compile CPM_DeepMN and CPM_DeepOR performance metrics and calculate confidence intervals</a></li></h4>
</ol>

### 8. [Prepare predictor tokens for the training of all-predictor-based models (APMs)](scripts/08_prepare_APM_tokens.R)

### 9. [Train APM dictionaries and convert tokens to embedding layer indices](scripts/09_prepare_APM_dictionaries.py)

### 10. Train and optimise APM_MN and APM_OR

<ol type="a">
  <h4><li><a href="scripts/10a_APM_deep.py">Train deep learning all-predictor-based models (APM)</a></li></h4>
  <h4><li><a href="scripts/10b_APM_deep_interrepeat_dropout.py">Perform interrepeat hyperparameter configuration dropout on deep learning all-predictor-based models (APM)</a></li></h4>
  <h4><li><a href="scripts/10c_APM_deep_hyperparameter_testing.py">Calculate ORC in bootstrapping resamples to determine dropout configurations</a></li></h4>
</ol>

### 11. Calculate and compile APM_MN and APM_OR metrics

<ol type="a">
  <h4><li><a href="scripts/11a_APM_deep_performance.py">Assess APM_DeepMN and APM_DeepOR performance</a></li></h4>
  <h4><li><a href="scripts/11b_APM_compile_metrics.py">Compile APM_DeepMN and APM_DeepOR performance metrics and calculate confidence intervals</a></li></h4>
</ol>

### 12. Assess feature significance in APM_MN

<ol type="a">
  <h4><li><a href="scripts/12a_APM_deep_SHAP.py">Calculate SHAP values for APM_DeepMN</a></li></h4>
  <h4><li><a href="scripts/12b_APM_compile_SHAP.py">Compile SHAP values for each GUPI-output type combination from APM_DeepMN</a></li></h4>
  <h4><li><a href="scripts/12c_APM_summarise_SHAP.py">Summarise SHAP values across study set</a></li></h4>
  <h4><li><a href="scripts/12d_APM_compile_significance_weights.py">Summarise aggregation weights across trained APM set</a></li></h4>
</ol>

### 13. [Prepare extended concise predictor set for ordinal prediction](scripts/13_prepare_extended_concise_predictor_set.R)

### 14. [Train logistic regression extended concise-predictor-based models (eCPM)](scripts/14_eCPM_logreg.py)

### 15. [Assess eCPM_MNLR and eCPM_POLR performance](scripts/15_eCPM_logreg_performance.py)

### 16. Train and optimise eCPM_DeepMN and eCPM_DeepOR

<ol type="a">
  <h4><li><a href="scripts/16a_eCPM_deep.py">Train deep learning extended concise-predictor-based models (eCPM)</a></li></h4>
  <h4><li><a href="scripts/16b_eCPM_deep_interrepeat_dropout.py">Perform interrepeat hyperparameter configuration dropout on deep learning extended concise-predictor-based models (eCPM)</a></li></h4>
  <h4><li><a href="scripts/16c_eCPM_deep_hyperparameter_testing.py">Calculate ORC in bootstrapping resamples to determine dropout configurations</a></li></h4>
</ol>

### 17. Calculate and compile eCPM_DeepMN and eCPM_DeepOR metrics

<ol type="a">
  <h4><li><a href="scripts/17a_eCPM_deep_performance.py">Assess eCPM_DeepMN and eCPM_DeepOR performance</a></li></h4>
  <h4><li><a href="scripts/17b_eCPM_compile_metrics.py">Compile eCPM_DeepMN and eCPM_DeepOR performance metrics and calculate confidence intervals</a></li></h4>
</ol>

### 18. [Perform ordinal regression analysis on study characteristics and predictors](scripts/18_ordinal_regression_analysis.py)

### 19. [Visualise study results for manuscript](scripts/19_manuscript_visualisations.R)
