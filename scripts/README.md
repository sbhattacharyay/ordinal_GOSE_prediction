## Scripts
All of the code used in this work can be found in this directory as Python (`.py`), R (`.R`), or bash (`.sh`) scripts. Moreover, custom classes have been saved in the `./classes` sub-directory, custom functions have been saved in the `./functions` sub-directory, and custom PyTorch models have been saved in the `./models` sub-directory.

### 1. [Extract study sample from CENTER-TBI dataset](01_extract_study_sample.py)
In this `.py` file, we extract the study sample from the CENTER-TBI dataset, filter patients by our study criteria, and convert ICU timestamps to machine-readable format.

### 2. [Partition CENTER-TBI for stratified, repeated k-fold cross-validation](02_partition_for_repeated_cv.py)
In this `.py` file, we create 100 partitions, stratified by 6-month GOSE, for repeated k-fold cross-validation, and save the splits into a dataframe for subsequent scripts.

### 3. [Prepare concise predictor set for ordinal prediction](03_prepare_concise_predictor_set.R)
In this `.R` file, we perform multiple imputation with chained equations (MICE, m = 100) on the concise predictor set for CPM training. The training set for each repeated k-fold CV partition is used to train an independent predictive mean matching imputation transformation for that partition. The result is 100 imputations, one for each repeated k-fold cross validation partition.

### 4. [Train logistic regression concise-predictor-based models (CPM)](04_CPM_logreg.py)
In this `.py` file, we define a function to train logistic regression CPMs given the repeated cross-validation dataframe. Then we perform parallelised training of logistic regression CPMs and testing set prediction. Finally, we compile testing set predictions.

### 5. [Assess CPM_MNLR and CPM_POLR performance](05_CPM_logreg_performance.py)
In this `.py` file, we create and save bootstrapping resamples used for all model performance evaluation. We prepare compiled CPM_MNLR and CPM_POLR testing set predictions, and calculate/save performance metrics.

### 6. Train and optimise CPM_DeepMN and CPM_DeepOR

<ol type="a">
  <li><h4><a href="06a_CPM_deep.py">Train deep learning concise-predictor-based models (CPM)</a></h4> In this <code>.py</code> file, we first create a grid of tuning configuration-cross-validation combinations and train CPM_DeepMN or CPM_DeepOR models based on provided hyperparameter row index. This is run, with multi-array indexing, on the HPC using a <a href="06a_CPM_deep.sh">bash script</a>.</li>
  <li><h4><a href="06b_CPM_deep_interrepeat_dropout.py">Perform interrepeat hyperparameter configuration dropout on deep learning concise-predictor-based models (CPM)</a></h4> In this <code>.py</code> file, we calculate ORC of extant validation predictions, prepare bootstrapping resamples for configuration dropout, and dropout configurations that are consistently (<span>&#593;</span> = .05) inferior in performance. </li>
  <li><h4><a href="06c_CPM_deep_hyperparameter_testing.py">Calculate ORC in bootstrapping resamples to determine dropout configurations</a></h4> In this <code>.py</code> file, we calculate ORC in each resample and compare to 'optimal' configuration. This is run, with multi-array indexing, on the HPC using a <a href="06c_CPM_deep_hyperparameter_testing.sh">bash script</a>.</li>
</ol>

### 7. Calculate and compile CPM_DeepMN and CPM_DeepOR metrics

<ol type="a">
  <li><h4><a href="07a_CPM_deep_performance.py">Assess CPM_DeepMN and CPM_DeepOR performance</a></h4> In this <code>.py</code> file, we calculate perfomance metrics on resamples. This is run, with multi-array indexing, on the HPC using a <a href="07a_CPM_deep_performance.sh">bash script</a>.</li>
  <li><h4><a href="07b_CPM_compile_metrics.py">Compile CPM_DeepMN and CPM_DeepOR performance metrics and calculate confidence intervals</a></h4> In this <code>.py</code> file, we compile all CPM_DeepMN and CPM_DeepOR performance metrics and calculate confidence intervals on all CPM performance metrics. </li>
</ol>

### 8. [Prepare predictor tokens for the training of all-predictor-based models (APMs)](08_prepare_APM_tokens.R)
In this `.R` file, we load and prepare formatted CENTER-TBI predictor tokens. Then, convert formatted predictors to tokens for each repeated cross-validation partition.

### 9. [Train APM dictionaries and convert tokens to embedding layer indices](09_prepare_APM_dictionaries.py)
In this `.py` file, we train APM dictionaries per repeated cross-validation partition and convert tokens to indices.

### 10. Train and optimise APM_MN and APM_OR

<ol type="a">
  <li><h4><a href="10a_APM_deep.py">Train deep learning all-predictor-based models (APM)</a></h4> In this <code>.py</code> file, we first create a grid of tuning configuration-cross-validation combinations and train APM_MN or APM_OR models based on provided hyperparameter row index. This is run, with multi-array indexing, on the HPC using a <a href="10a_APM_deep.sh">bash script</a>.</li>
  <li><h4><a href="10b_APM_deep_interrepeat_dropout.py">Perform interrepeat hyperparameter configuration dropout on deep learning all-predictor-based models (APM)</a></h4> In this <code>.py</code> file, we calculate ORC of extant validation predictions, prepare bootstrapping resamples for configuration dropout, and dropout configurations that are consistently (<span>&#593;</span> = .05) inferior in performance. </li>
  <li><h4><a href="10c_APM_deep_hyperparameter_testing.py">Calculate ORC in bootstrapping resamples to determine dropout configurations</a></h4> In this <code>.py</code> file, we calculate ORC in each resample and compare to 'optimal' configuration. This is run, with multi-array indexing, on the HPC using a <a href="10c_APM_deep_hyperparameter_testing.sh">bash script</a>.</li>
</ol>

### 11. Calculate and compile APM_MN and APM_OR metrics

<ol type="a">
  <li><h4><a href="11a_APM_deep_performance.py">Assess APM_MN and APM_OR performance</a></h4> In this <code>.py</code> file, we calculate perfomance metrics on resamples. This is run, with multi-array indexing, on the HPC using a <a href="11a_APM_deep_performance.sh">bash script</a>.</li>
  <li><h4><a href="11b_APM_compile_metrics.py">Compile APM_MN and APM_OR performance metrics and calculate confidence intervals</a></h4> In this <code>.py</code> file, we compile all APM_MN and APM_OR performance metrics and calculate confidence intervals on all CPM performance metrics. </li>
</ol>

### 12. Assess feature significance in APM_MN

<ol type="a">
  <li><h4><a href="12a_APM_deep_SHAP.py">Calculate SHAP values for APM_MN</a></h4> In this <code>.py</code> file, we find all top-performing model checkpoint files for SHAP calculation and calculate SHAP values based on given parameters. This is run, with multi-array indexing, on the HPC using a <a href="12a_APM_deep_SHAP.sh">bash script</a>.</li>
  <li><h4><a href="12b_APM_compile_SHAP.py">Compile SHAP values for each GUPI-output type combination from APM_MN</a></h4> In this <code>.py</code> file, we find all files storing calculated SHAP values and create combinations with study GUPIs and compile SHAP values for the given GUPI and output type combination. This is run, with multi-array indexing, on the HPC using a <a href="12b_APM_compile_SHAP.sh">bash script</a>.</li>
  <li><h4><a href="12c_APM_summarise_SHAP.py">Summarise SHAP values across study set</a></h4> In this <code>.py</code> file, we find all files storing GUPI-specific SHAP values and compile and save summary SHAP values across study patient set. </li>
  <li><h4><a href="12d_APM_compile_significance_weights.py">Summarise aggregation weights across trained APM set</a></h4> In this <code>.py</code> file, we compile significance weights across trained APMs and summarise significance weights. </li>
</ol>

### 13. [Prepare extended concise predictor set for ordinal prediction](13_prepare_extended_concise_predictor_set.R)
In this `.R` file, we load IMPACT variables from CENTER-TBI, load and prepare the added variables from CENTER-TBI, and multiply impute extended concise predictor set in parallel. The training set for each repeated k-fold CV partition is used to train an independent predictive mean matching imputation transformation for that partition. The result is 100 imputations, one for each repeated k-fold cross validation partition.

### 14. [Train logistic regression extended concise-predictor-based models (eCPM)](14_eCPM_logreg.py)
In this `.py` file, we define a function to train logistic regression eCPMs given the repeated cross-validation dataframe. Then we perform parallelised training of logistic regression eCPMs and testing set prediction. Finally, we compile testing set predictions.

### 15. [Assess eCPM_MNLR and eCPM_POLR performance](15_eCPM_logreg_performance.py)
In this `.py` file, we load the common bootstrapping resamples (that will be used for all model performance evaluation), prepare compiled eCPM_MNLR and eCPM_POLR testing set predictions, and calculate/save performance metrics

### 16. Train and optimise eCPM_DeepMN and eCPM_DeepOR

<ol type="a">
  <li><h4><a href="16a_eCPM_deep.py">Train deep learning extended concise-predictor-based models (eCPM)</a></h4> In this <code>.py</code> file, we first create a grid of tuning configuration-cross-validation combinations and train eCPM_DeepMN or eCPM_DeepOR models based on provided hyperparameter row index. This is run, with multi-array indexing, on the HPC using a <a href="16a_eCPM_deep.sh">bash script</a>.</li>
  <li><h4><a href="16b_eCPM_deep_interrepeat_dropout.py">Perform interrepeat hyperparameter configuration dropout on deep learning extended concise-predictor-based models (eCPM)</a></h4> In this <code>.py</code> file, we calculate ORC of extant validation predictions, prepare bootstrapping resamples for configuration dropout, and dropout configurations that are consistently (<span>&#593;</span> = .05) inferior in performance </li>
  <li><h4><a href="16c_eCPM_deep_hyperparameter_testing.py">Calculate ORC in bootstrapping resamples to determine dropout configurations</a></h4> In this <code>.py</code> file, we calculate ORC in each resample and compare to 'optimal' configuration. This is run, with multi-array indexing, on the HPC using a <a href="16c_eCPM_deep_hyperparameter_testing.sh">bash script</a>.</li>
</ol>

### 17. Calculate and compile eCPM_DeepMN and eCPM_DeepOR metrics

<ol type="a">
  <li><h4><a href="17a_eCPM_deep_performance.py">Assess eCPM_DeepMN and eCPM_DeepOR performance</a></h4> In this <code>.py</code> file, we calculate perfomance metrics on resamples. This is run, with multi-array indexing, on the HPC using a <a href="17a_eCPM_deep_performance.sh">bash script</a>.</li>
  <li><h4><a href="17b_eCPM_compile_metrics.py">Compile eCPM_DeepMN and eCPM_DeepOR performance metrics and calculate confidence intervals</a></h4> In this <code>.py</code> file, we compile all eCPM_DeepMN and eCPM_DeepOR performance metrics and calculate confidence intervals on all CPM performance metrics. </li>
</ol>

### 18. [Perform ordinal regression analysis on study characteristics and predictors](18_ordinal_regression_analysis.py)
In this `.py` file, we perform ordinal regression analysis on summary characteristics, perform ordinal regression analysis on CPM characteristics, and perform ordinal regression analysis on eCPM characteristics.

### 19. [Visualise study results for manuscript](19_manuscript_visualisations.R)
In this `.R` file, we produce the figures for the manuscript and the supplementary figures. The large majority of the quantitative figures in the manuscript are produced using the `ggplot` package.
