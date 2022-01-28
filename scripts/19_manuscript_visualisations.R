#### Master Script 19: Visualise study results for manuscript ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Figure 2
# III. Figure 3
# IV. Figure 4
# V. Supplementary Figure 2
# VI. Supplementary Figure 3
# VII. Supplementary Figure 4
# VIII. Supplementary Figure 5
# IX. Table 4
# X. Supplementary Table 2, 3, and 4
# XI. Supplementary Appendix 3
# XII. Supplementary Appendix 4

### I. Initialisation
# Import necessary libraries
library(tidyverse)
library(readxl)
library(RColorBrewer)
library(rvg)
library(svglite)
library(viridis)
library(lemon)
library(VIM)
library(latex2exp)

### II. Figure 2
# Load a sample eCPM training set
eCPM.training.set <- read.csv('../imputed_eCPM_sets/repeat01/fold1/training_set.csv')%>%
  filter(NFL < 500)

# Expand the Red-Yellow-Green palette to permit 20 quantiles
nb.cols <- 22
mycolors <- rev(colorRampPalette(brewer.pal(11, "RdYlGn"))(nb.cols))

# Calculate density curve and distribution information of NFL
dens <- density(eCPM.training.set$NFL)
df <- data.frame(x=dens$x, y=dens$y)
probs <- seq(0,1,length.out=21)
quantiles <- quantile(eCPM.training.set$NFL, prob=probs)
df$quant <- factor(findInterval(df$x,quantiles))

# Produce distribution plot as example for discretisation
nfl.dist.plot <- ggplot(df, aes(x,y)) + 
  geom_ribbon(aes(ymin=0, ymax=y, fill=quant)) + 
  scale_x_continuous(breaks=seq(0,500,100)) + 
  scale_y_continuous(expand = c(0,0))+
  scale_fill_manual(values = mycolors,
                    guide='none') +
  coord_cartesian(xlim = c(0,500)) +
  theme_classic() +
  theme(panel.grid = element_blank(),
        panel.background = element_blank(),
        axis.ticks.y = element_blank(),
        axis.ticks.length.x = unit(.05, "cm"),
        axis.text.y = element_blank(),
        axis.text.x = element_text(color = 'black',size = 3),
        axis.title.y = element_blank(),
        axis.title.x = element_blank(),
        axis.line.y = element_blank(),
        axis.line.x = element_line(size = .75/.pt))

# Create directory for current date and save NFL distribution plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'NFL_dist.svg'),nfl.dist.plot,device= svg,units='in',dpi=600,width=1.37,height = .87)

### III. Figure 3
# Load threshold_ICI confidence intervals of all model types to determine CPM_best, APM_best, and eCPM_best
CPM.CI.ICI.ave <- read.csv('../model_performance/CPM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC == 'ICI') %>% 
  arrange(mean) %>%
  head(1)

APM.CI.ICI.ave <- read.csv('../model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC == 'ICI') %>% 
  arrange(mean) %>%
  head(1)

eCPM.CI.ICI.ave <- read.csv('../model_performance/eCPM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC == 'ICI') %>% 
  arrange(mean) %>%
  head(1)

# Load and compile calibration confidence intervals of CPM_best, APM_best, and eCPM_best
CPM.CI.calibration <- read.csv('../model_performance/CPM/CI_calibration.csv') %>%
  filter(MODEL %in% CPM.CI.ICI.ave$MODEL) %>%
  mutate(MODEL = 'CPM_{Best}')

APM.CI.calibration <- read.csv('../model_performance/APM/CI_calibration.csv') %>%
  filter(MODEL %in% APM.CI.ICI.ave$MODEL) %>%
  mutate(MODEL = 'APM_{Best}')

eCPM.CI.calibration <- read.csv('../model_performance/eCPM/CI_calibration.csv') %>%
  filter(MODEL %in% eCPM.CI.ICI.ave$MODEL) %>%
  mutate(MODEL = 'eCPM_{Best}')

CI.calibration <- rbind(CPM.CI.calibration,APM.CI.calibration,eCPM.CI.calibration)

# Format calibration dataframe for plot (and decide whether to filter out eCPM)
CI.calibration <- CI.calibration %>% 
  filter(MODEL != 'eCPM_{Best}') %>%
  mutate(Threshold = plyr::mapvalues(Threshold,
                                     from = c("GOSE>1","GOSE>3","GOSE>4","GOSE>5","GOSE>6","GOSE>7"),
                                     to =c("GOSE > 1","GOSE > 3","GOSE > 4","GOSE > 5","GOSE > 6","GOSE > 7"))) %>%
  mutate(MODEL = factor(MODEL,levels =c('CPM_{Best}','APM_{Best}'))) %>%
  rename(Model = MODEL)

# Implement ggplot to visualise calibration curves
best.calibration.curves.plot <- CI.calibration %>%
  ggplot(aes(x = PredProb)) +
  facet_wrap( ~ Threshold,
              scales = 'free',
              ncol = 3) +
  xlab("Predicted Probability") +
  ylab("Observed Probability") +
  coord_cartesian(ylim = c(0,1),xlim = c(0,1))+
  geom_segment(x = 0, y = 0, xend = 1, yend = 1,alpha = 0.5,linetype = "dashed",size=.75/.pt, color = 'gray')+
  geom_ribbon(aes(ymin = TrueProb_lo, ymax = TrueProb_hi, fill = Model), alpha = 0.3,size=.75/.pt,color=NA) +
  geom_line(aes(y = TrueProb_mean, color = Model), alpha = 1, size=1.3/.pt) +
  scale_fill_discrete(breaks=levels(CI.calibration$Model),
                      labels=c(expression(CPM[Best]), expression(APM[Best])))+
  scale_color_discrete(breaks=levels(CI.calibration$Model),
                       labels=c(expression(CPM[Best]), expression(APM[Best])))+
  theme_classic()+
  theme(
    strip.text = element_text(size=6, color = "black",face = 'bold'), 
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.text.x = element_text(size = 5, color = "black"),
    axis.text.y = element_text(size = 5, color = "black"),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    strip.background = element_blank(),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    aspect.ratio = 1,
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    axis.line = element_blank(),
    legend.key.size = unit(1.3/.pt,"line")
  )

# Create directory for current date and save calibration plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'thresh_calibration.svg'),best.calibration.curves.plot,device= svg,units='in',dpi=600,width=6,height = 4.38)

# Get ICI values for CPM_best, APM_best, and eCPM_best at each threshold
CPM.CI.ICI <- read.csv('../model_performance/CPM/CI_threshold_metrics.csv') %>%
  filter(Threshold != 'Average', METRIC == 'ICI', MODEL %in% CPM.CI.ICI.ave$MODEL) %>% 
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi))

APM.CI.ICI <- read.csv('../model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(Threshold != 'Average', METRIC == 'ICI', MODEL %in% APM.CI.ICI.ave$MODEL) %>% 
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi))

eCPM.CI.ICI <- read.csv('../model_performance/eCPM/CI_threshold_metrics.csv') %>%
  filter(Threshold != 'Average', METRIC == 'ICI', MODEL %in% eCPM.CI.ICI.ave$MODEL) %>% 
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi))

# Load compiled predictions from optimally calibrated CPM_{Best} and APM_{Best}
CPM.Best.predictions <- read.csv('../CPM_outputs/LOGREG_v1-0/compiled_mnlr_test_predictions.csv') %>%
  mutate(Model = 'CPM_MNLR') %>%
  select(-c(repeat.,fold))
APM.Best.predictions <- read.csv('../APM_outputs/DEEP_v1-0/APM_deepMN_compiled_test_predictions.csv') %>%
  mutate(Model = 'APM_DeepMN') %>%
  filter(TUNE_IDX == 8) %>%
  select(-c(X,TUNE_IDX))
Best.calibrated.predictions <- rbind(CPM.Best.predictions,APM.Best.predictions) %>%
  select(starts_with('Pr.GOSE.'),Model)

# Calculate threshold-level probabilities
prob.labels <- sort(grep('Pr.GOSE',names(Best.calibrated.predictions),value = T))
Best.calibrated.thresh.probs <- 1 - t(apply(t(Best.calibrated.predictions[,prob.labels]), 2, cumsum)) %>%
  as.data.frame() %>%
  select(-last_col()) %>%
  rename(Pr.GOSE.gt.1 = Pr.GOSE.1.,
         Pr.GOSE.gt.3 = Pr.GOSE.2.3.,
         Pr.GOSE.gt.4 = Pr.GOSE.4.,
         Pr.GOSE.gt.5 = Pr.GOSE.5.,
         Pr.GOSE.gt.6 = Pr.GOSE.6.,
         Pr.GOSE.gt.7 = Pr.GOSE.7.)
Best.calibrated.thresh.probs$Model <- Best.calibrated.predictions$Model

# Convert to long format for ggplot
Best.calibrated.thresh.probs <- Best.calibrated.thresh.probs %>%
  pivot_longer(cols=-Model,names_to='Threshold',values_to='Prob') %>%
  mutate(Threshold = plyr::mapvalues(Threshold,
                                     from = c("Pr.GOSE.gt.1","Pr.GOSE.gt.3","Pr.GOSE.gt.4","Pr.GOSE.gt.5","Pr.GOSE.gt.6","Pr.GOSE.gt.7"),
                                     to =c("GOSE > 1","GOSE > 3","GOSE > 4","GOSE > 5","GOSE > 6","GOSE > 7")))

dist.segments <- data.frame(x = 0, y = 0, xend = 1, yend = 0, Threshold = c("GOSE > 1","GOSE > 3","GOSE > 4","GOSE > 5","GOSE > 6","GOSE > 7"))

# Implement ggplot to create distribution histograms for calibration plots
calibration.dist.plot <- ggplot(data = NULL) +
  geom_histogram(data = Best.calibrated.thresh.probs %>% filter(Model=='CPM_MNLR'),aes(x = Prob, y = ..density..),bins = 200,fill='#F8766D',alpha=.75) +
  geom_histogram(data = Best.calibrated.thresh.probs %>% filter(Model=='APM_DeepMN'),aes(x = Prob, y = -..density..),fill = "#00BFC4", bins = 200,alpha=.75) +
  geom_segment(data = dist.segments,aes(x = x, y = y, xend = xend, yend = yend),size=.75/.pt, color = 'black')+
  facet_wrap( ~ Threshold,
              scales = 'free',
              ncol = 3) +
  coord_cartesian(xlim = c(0,1),ylim = c(-30,30),expand = T) +
  scale_y_symmetric(mid = 0) +
  xlab("Predicted Probability") +
  ylab("Observed Probability") +
  theme_classic()+
  theme(
    strip.text = element_text(size=6, color = "black",face = 'bold'), 
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.text.x = element_text(size = 5, color = "black"),
    axis.text.y = element_text(size = 5, color = "black"),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    strip.background = element_blank(),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    aspect.ratio = 1,
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    axis.line = element_blank(),
    legend.key.size = unit(1.3/.pt,"line")
  )

# Create directory for current date and save calibration distribution plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'thresh_distributions.svg'),calibration.dist.plot,device= svg,units='in',dpi=600,width=6,height = 4.38)

### IV. Figure 4
# Load cross-validation partitions to extract study GUPIs and GOSE
study.patient.set <- read.csv('../cross_validation_splits.csv') %>%
  select(GUPI,GOSE) %>%
  distinct()

# Load token dictionary
token.dictionary <- read.csv('../APM_outputs/DEEP_v1-0/compiled_token_dictionary.csv') %>%
  select(Token,Baseline,Numeric,BaseToken,Type)

# Load mean absolute SHAP values for APM_DeepMN nodes
prob.mean.abs.values <- read.csv('../APM_outputs/DEEP_v1-0/SHAP_values/prob_mean_absolute_values.csv')

# Calculate mean of SHAP values across patient set and calculate max mean absolute SHAP value per predictor
overall.prob.mav <- prob.mean.abs.values %>%
  group_by(label,Token) %>%
  summarise(meanAbs = mean(VALUE)) %>%
  left_join(token.dictionary,by = 'Token') %>%
  group_by(label,BaseToken,Baseline,Numeric,Type) %>%
  summarise(mean_abs = mean(meanAbs),
            max_abs = max(meanAbs))

# Identify the top 15 predictors based on max mean absolute SHAP values
top.variables <- overall.prob.mav %>%
  group_by(BaseToken,Type) %>%
  summarise(sum_abs = sum(max_abs)) %>%
  arrange(-sum_abs) %>%
  ungroup() %>%
  mutate(order = row_number()) %>%
  filter(order <= 15)

# Create new dataframe to store SHAP values for plot
shap.plot.df <- overall.prob.mav %>%
  inner_join(top.variables,by = c('BaseToken','Type'))

# Calculate average of SHAP values across the non-top predictors and bind to new dataframe
remaining.df <- overall.prob.mav %>%
  anti_join(top.variables,by = c('BaseToken','Type')) %>%
  group_by(label) %>%
  summarise(max_abs = mean(max_abs),
            count = n()) %>%
  mutate(order=16,
         BaseToken = 'Average over 1,136 other predictors')
shap.plot.df <- rbind(shap.plot.df,remaining.df)

# Load figure label conversions of predictors
token.to.label.df <- read_xlsx('../APM_tokens/token_to_text.xlsx')

# Add figure labels to dataframe for plotting SHAP values
shap.plot.df <- shap.plot.df %>%
  left_join(token.to.label.df,by = 'BaseToken') %>%
  mutate(label = plyr::mapvalues(label,
                                 from = c(0:6),
                                 to=c("1","2 or 3","4","5","6","7","8")))

# Implement ggplot to visualise SHAP values of most significant predictors
top.predictor.shap.plot <- shap.plot.df %>%
  ggplot(aes(x = max_abs)) +
  geom_col(aes(fill = factor(label),
               y = factor(order)), 
           width = 0.7, 
           position = position_stack(reverse = TRUE)) +
  guides(fill = guide_legend(title = expression(bold(Output~nodes~of~APM[MN]~(GOSE))),
                             nrow = 1)) +
  scale_x_continuous(expand = c(0,0),
                     breaks = seq(0,.06,by=.01))+
  scale_y_discrete(breaks = shap.plot.df$order,
                   labels = shap.plot.df$FigureLabel,
                   expand = c(0,0), 
                   limits = rev(levels(factor(shap.plot.df$order)))) +
  xlab('Max absolute value of mean token SHAP values per predictor')+
  theme_bw()+
  theme(
    panel.grid.major.y = element_blank(), 
    panel.grid.minor.y = element_blank(),
    axis.text.x = element_text(size = 5, color = "black"),
    axis.text.y = element_text(size = 6, color = "black",face = 'bold'),
    axis.title.y = element_blank(),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    panel.border = element_blank(),
    axis.line.x = element_line(size=1/.pt),
    legend.position = 'bottom',
    axis.ticks.y = element_blank(),
    legend.key.size = unit(1.3/.pt,"line"),
    legend.title = element_text(size = 7, color = "black",face = 'bold'),
    legend.text=element_text(size=6)
  )

# Create directory for current date and save SHAP plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'shap_plot.svg'),top.predictor.shap.plot,device= svg,units='in',dpi=600,width=7.5,height = 5)

### V. Supplementary Figure 2
# Load cross-validation splits
cv_splits = read.csv('../cross_validation_splits.csv')

# Load IMPACT-specific variables from CENTER-TBI dataset
CENTER_TBI_IMPACT = read.csv('../CENTER-TBI/IMPACT/data.csv',
                             na.strings=c("NA","NaN","", " ")) %>%
  rename(GUPI = entity_id) %>%
  filter(GUPI %in% cv_splits$GUPI) %>%
  select(-c(SiteCode,GCS,PatientType,GOSE,GUPI))

# Shorten plot labels to fit
plt.labels <- names(CENTER_TBI_IMPACT)
plt.labels[1] <- "Age"
plt.labels[2] <- "U.P."
plt.labels[5] <- "Glu."
plt.labels[6] <- "Hypoxia"
plt.labels[7] <- "HoTN"
plt.labels[8] <- "Marshall"
plt.labels[9] <- "tSAH"

# Produce both barplots of missing variables and combinations plot
miss.aggr <- aggr(CENTER_TBI_IMPACT,numbers=TRUE,labels = plt.labels)

### VI. Supplementary Figure 3
# Load and compile calibration confidence intervals of CPM
CPM.CI.calibration <- read.csv('../model_performance/CPM/CI_calibration.csv')

# Format calibration dataframe for plot
CPM.CI.calibration <- CPM.CI.calibration %>% 
  mutate(Threshold = plyr::mapvalues(Threshold,
                                     from = c("GOSE>1","GOSE>3","GOSE>4","GOSE>5","GOSE>6","GOSE>7"),
                                     to =c("GOSE > 1","GOSE > 3","GOSE > 4","GOSE > 5","GOSE > 6","GOSE > 7"))) %>%
  mutate(MODEL = factor(MODEL,levels =c("CPM_MNLR","CPM_POLR","CPM_DeepMN","CPM_DeepOR"))) %>%
  rename(Model = MODEL)

# Implement ggplot to visualise calibration curves
CPM.calibration.curves.plot <- CPM.CI.calibration %>%
  ggplot(aes(x = PredProb)) +
  facet_wrap( ~ Threshold,
              scales = 'free',
              ncol = 3) +
  xlab("Predicted Probability") +
  ylab("Observed Probability") +
  coord_cartesian(ylim = c(0,1),xlim = c(0,1))+
  geom_segment(x = 0, y = 0, xend = 1, yend = 1,alpha = 0.5,linetype = "dashed",size=.75/.pt, color = 'gray')+
  geom_ribbon(aes(ymin = TrueProb_lo, ymax = TrueProb_hi, fill = Model), alpha = 0.3,size=.75/.pt,color=NA) +
  geom_line(aes(y = TrueProb_mean, color = Model), alpha = 1, size=1.3/.pt) +
  scale_fill_discrete(breaks=levels(CPM.CI.calibration$Model),
                      labels=c(expression(CPM[MNLR]), expression(CPM[POLR]),expression(CPM[DeepMN]),expression(CPM[DeepOR])))+
  scale_color_discrete(breaks=levels(CPM.CI.calibration$Model),
                       labels=c(expression(CPM[MNLR]), expression(CPM[POLR]),expression(CPM[DeepMN]),expression(CPM[DeepOR])))+
  theme_classic()+
  theme(
    strip.text = element_text(size=6, color = "black",face = 'bold'), 
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.text.x = element_text(size = 5, color = "black"),
    axis.text.y = element_text(size = 5, color = "black"),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    strip.background = element_blank(),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    aspect.ratio = 1,
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    axis.line = element_blank(),
    legend.key.size = unit(1.3/.pt,"line")
  )

# Create directory for current date and save CPM calibration plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'CPM_thresh_calibration.svg'),CPM.calibration.curves.plot,device= svg,units='in',dpi=600,width=6,height = 4.38)

# Get ICI values for CPM
CPM.CI.ICI <- read.csv('../model_performance/CPM/CI_threshold_metrics.csv') %>%
  filter(Threshold != 'Average', METRIC == 'ICI') %>% 
  mutate(MODEL = factor(MODEL,levels =c("CPM_MNLR","CPM_POLR","CPM_DeepMN","CPM_DeepOR"))) %>%
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi)) %>%
  arrange(Threshold,MODEL)

### VII. Supplementary Figure 4
# Load and compile calibration confidence intervals of APM
APM.CI.calibration <- read.csv('../model_performance/APM/CI_calibration.csv')

# Format calibration dataframe for plot
APM.CI.calibration <- APM.CI.calibration %>% 
  mutate(Threshold = plyr::mapvalues(Threshold,
                                     from = c("GOSE>1","GOSE>3","GOSE>4","GOSE>5","GOSE>6","GOSE>7"),
                                     to =c("GOSE > 1","GOSE > 3","GOSE > 4","GOSE > 5","GOSE > 6","GOSE > 7"))) %>%
  mutate(MODEL = factor(MODEL,levels =c("APM_DeepMN","APM_DeepOR"))) %>%
  rename(Model = MODEL)

# Implement ggplot to visualise calibration curves
APM.calibration.curves.plot <- APM.CI.calibration %>%
  ggplot(aes(x = PredProb)) +
  facet_wrap( ~ Threshold,
              scales = 'free',
              ncol = 3) +
  xlab("Predicted Probability") +
  ylab("Observed Probability") +
  coord_cartesian(ylim = c(0,1),xlim = c(0,1))+
  geom_segment(x = 0, y = 0, xend = 1, yend = 1,alpha = 0.5,linetype = "dashed",size=.75/.pt, color = 'gray')+
  geom_ribbon(aes(ymin = TrueProb_lo, ymax = TrueProb_hi, fill = Model), alpha = 0.3,size=.75/.pt,color=NA) +
  geom_line(aes(y = TrueProb_mean, color = Model), alpha = 1, size=1.3/.pt) +
  scale_fill_discrete(breaks=levels(APM.CI.calibration$Model),
                      labels=c(expression(APM[MN]),expression(APM[OR])))+
  scale_color_discrete(breaks=levels(APM.CI.calibration$Model),
                       labels=c(expression(APM[MN]),expression(APM[OR])))+
  theme_classic()+
  theme(
    strip.text = element_text(size=6, color = "black",face = 'bold'), 
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.text.x = element_text(size = 5, color = "black"),
    axis.text.y = element_text(size = 5, color = "black"),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    strip.background = element_blank(),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    aspect.ratio = 1,
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    axis.line = element_blank(),
    legend.key.size = unit(1.3/.pt,"line")
  )

# Create directory for current date and save APM calibration plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'APM_thresh_calibration.svg'),APM.calibration.curves.plot,device= svg,units='in',dpi=600,width=6,height = 4.38)

# Get ICI values for APM
APM.CI.ICI <- read.csv('../model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(Threshold != 'Average', METRIC == 'ICI') %>% 
  mutate(MODEL = factor(MODEL,levels =c("APM_DeepMN","APM_DeepOR"))) %>%
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi)) %>%
  arrange(Threshold,MODEL)

### VIII. Supplementary Figure 5
# Load and compile calibration confidence intervals of eCPM
eCPM.CI.calibration <- read.csv('../model_performance/eCPM/CI_calibration.csv')

# Format calibration dataframe for plot
eCPM.CI.calibration <- eCPM.CI.calibration %>% 
  mutate(Threshold = plyr::mapvalues(Threshold,
                                     from = c("GOSE>1","GOSE>3","GOSE>4","GOSE>5","GOSE>6","GOSE>7"),
                                     to =c("GOSE > 1","GOSE > 3","GOSE > 4","GOSE > 5","GOSE > 6","GOSE > 7"))) %>%
  mutate(MODEL = factor(MODEL,levels =c("eCPM_MNLR","eCPM_POLR","eCPM_DeepMN","eCPM_DeepOR"))) %>%
  rename(Model = MODEL)

# Implement ggplot to visualise calibration curves
eCPM.calibration.curves.plot <- eCPM.CI.calibration %>%
  ggplot(aes(x = PredProb)) +
  facet_wrap( ~ Threshold,
              scales = 'free',
              ncol = 3) +
  xlab("Predicted Probability") +
  ylab("Observed Probability") +
  coord_cartesian(ylim = c(0,1),xlim = c(0,1))+
  geom_segment(x = 0, y = 0, xend = 1, yend = 1,alpha = 0.5,linetype = "dashed",size=.75/.pt, color = 'gray')+
  geom_ribbon(aes(ymin = TrueProb_lo, ymax = TrueProb_hi, fill = Model), alpha = 0.3,size=.75/.pt,color=NA) +
  geom_line(aes(y = TrueProb_mean, color = Model), alpha = 1, size=1.3/.pt) +
  scale_fill_discrete(breaks=levels(eCPM.CI.calibration$Model),
                      labels=c(expression(eCPM[MNLR]), expression(eCPM[POLR]),expression(eCPM[DeepMN]),expression(eCPM[DeepOR])))+
  scale_color_discrete(breaks=levels(eCPM.CI.calibration$Model),
                       labels=c(expression(eCPM[MNLR]), expression(eCPM[POLR]),expression(eCPM[DeepMN]),expression(eCPM[DeepOR])))+
  theme_classic()+
  theme(
    strip.text = element_text(size=6, color = "black",face = 'bold'), 
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.text.x = element_text(size = 5, color = "black"),
    axis.text.y = element_text(size = 5, color = "black"),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    strip.background = element_blank(),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    aspect.ratio = 1,
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    legend.position = 'bottom',
    legend.title = element_text(size = 7, color = "black", face = 'bold'),
    legend.text=element_text(size=6),
    axis.line = element_blank(),
    legend.key.size = unit(1.3/.pt,"line")
  )

# Create directory for current date and save eCPM calibration plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'eCPM_thresh_calibration.svg'),eCPM.calibration.curves.plot,device= svg,units='in',dpi=600,width=6,height = 4.38)

# Get ICI values for eCPM
eCPM.CI.ICI <- read.csv('../model_performance/eCPM/CI_threshold_metrics.csv') %>%
  filter(Threshold != 'Average', METRIC == 'ICI') %>% 
  mutate(MODEL = factor(MODEL,levels =c("eCPM_MNLR","eCPM_POLR","eCPM_DeepMN","eCPM_DeepOR"))) %>%
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi)) %>%
  arrange(Threshold,MODEL)

### IX. Table 4
# Determine best overall performance metrics
CPM.CI.overall.best <- read.csv('../model_performance/CPM/CI_overall_metrics.csv') %>%
  mutate(mean = case_when(METRIC == "S" ~ -mean,
                          METRIC != "S" ~ mean)) %>%
  group_by(METRIC) %>%
  slice_max(order_by = mean, n = 1) %>%
  mutate(PredType = 'CPM')

APM.CI.overall.best <- read.csv('../model_performance/APM/CI_overall_metrics.csv') %>%
  mutate(mean = case_when(METRIC == "S" ~ -mean,
                          METRIC != "S" ~ mean)) %>%
  group_by(METRIC) %>%
  slice_max(order_by = mean, n = 1) %>%
  mutate(PredType = 'APM')

eCPM.CI.overall.best <- read.csv('../model_performance/eCPM/CI_overall_metrics.csv') %>%
  mutate(mean = case_when(METRIC == "S" ~ -mean,
                          METRIC != "S" ~ mean)) %>%
  group_by(METRIC) %>%
  slice_max(order_by = mean, n = 1) %>%
  mutate(PredType = 'eCPM')

best.overall.table <- rbind(CPM.CI.overall.best,
                            APM.CI.overall.best,
                            eCPM.CI.overall.best) %>%
  mutate(mean = case_when(METRIC == "S" ~ -mean,
                          METRIC != "S" ~ mean)) %>%
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi)) %>%
  select(MODEL,METRIC,PredType,formatted) %>%
  pivot_wider(names_from = PredType, id_cols = c(METRIC),values_from = formatted) %>%
  mutate(METRIC = factor(METRIC,levels = c('ORC','S','Gen_C','D_xy','Accuracy'))) %>%
  arrange(METRIC) %>%
  mutate(Threshold='Overall')

# Load threshold-level performance metrics and determine best model for each metric
CPM.CI.performance.ave <- read.csv('../model_performance/CPM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC %in% c('AUC','Accuracy')) %>% 
  group_by(METRIC) %>%
  slice_max(order_by = mean, n = 1) %>%
  mutate(PredType = 'CPM')

APM.CI.performance.ave <- read.csv('../model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC %in% c('AUC','Accuracy')) %>% 
  group_by(METRIC) %>%
  slice_max(order_by = mean, n = 1) %>%
  mutate(PredType = 'APM')

eCPM.CI.performance.ave <- read.csv('../model_performance/eCPM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC %in% c('AUC','Accuracy')) %>% 
  group_by(METRIC) %>%
  slice_max(order_by = mean, n = 1) %>%
  mutate(PredType = 'eCPM')

best.performance.configurations <- rbind(CPM.CI.performance.ave,
                                         APM.CI.performance.ave,
                                         eCPM.CI.performance.ave) %>%
  select(-Threshold)

# Load threshold-level performance metrics of CPM_best, APM_best, and eCPM_best
best.thresh.performance.table <- rbind(read.csv('../model_performance/CPM/CI_threshold_metrics.csv'),
                                       read.csv('../model_performance/APM/CI_threshold_metrics.csv'),
                                       read.csv('../model_performance/eCPM/CI_threshold_metrics.csv')) %>%
  inner_join(best.performance.configurations %>% select(MODEL,METRIC,PredType)) %>%
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi),
         METRIC = factor(METRIC,levels = c('AUC','Accuracy'))) %>%
  select(MODEL,Threshold,METRIC,PredType,formatted) %>%
  pivot_wider(names_from = PredType, id_cols = c(METRIC,Threshold),values_from = formatted) %>%
  arrange(METRIC,Threshold)

# Concatenate threshold-level performance metrics to the overall metric table
best.performance.table <- rbind(best.overall.table,
                                best.thresh.performance.table) %>%
  relocate(Threshold,.after=METRIC)

# Load threshold-level calibration metrics and determine best model for each metric
CPM.CI.calibration.ave <- read.csv('../model_performance/CPM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC %in% c('Calib_Slope')) %>% 
  mutate(error=abs(1-mean)) %>%
  group_by(METRIC) %>%
  slice_min(order_by = error, n = 1) %>%
  mutate(PredType = 'CPM')

APM.CI.calibration.ave <- read.csv('../model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC %in% c('Calib_Slope')) %>% 
  mutate(error=abs(1-mean)) %>%
  group_by(METRIC) %>%
  slice_min(order_by = error, n = 1) %>%
  mutate(PredType = 'APM')

eCPM.CI.calibration.ave <- read.csv('../model_performance/eCPM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC %in% c('Calib_Slope')) %>% 
  mutate(error=abs(1-mean)) %>%
  group_by(METRIC) %>%
  slice_min(order_by = error, n = 1) %>%
  mutate(PredType = 'eCPM')

best.calibration.configurations <- rbind(CPM.CI.calibration.ave,
                                         APM.CI.calibration.ave,
                                         eCPM.CI.calibration.ave) %>%
  select(-Threshold)

# Load threshold-level calibration metrics of CPM_best, APM_best, and eCPM_best
best.calibration.table <- rbind(read.csv('../model_performance/CPM/CI_threshold_metrics.csv'),
                                read.csv('../model_performance/APM/CI_threshold_metrics.csv'),
                                read.csv('../model_performance/eCPM/CI_threshold_metrics.csv')) %>%
  inner_join(best.calibration.configurations %>% select(MODEL,METRIC,PredType)) %>%
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi)) %>%
  select(MODEL,Threshold,METRIC,PredType,formatted) %>%
  pivot_wider(names_from = PredType, id_cols = c(METRIC,Threshold),values_from = formatted) %>%
  arrange(METRIC,Threshold)

### X. Supplementary Table 2, 3, and 4
# CPM performance table
CPM.performance.table <- rbind(read.csv('../model_performance/CPM/CI_overall_metrics.csv') %>% 
                                 mutate(METRIC = factor(METRIC,levels = c('ORC','S','Gen_C','D_xy','Accuracy'))) %>%
                                 mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi)) %>%
                                 select(MODEL,METRIC,formatted) %>%
                                 pivot_wider(names_from = MODEL, id_cols = c(METRIC),values_from = formatted) %>%
                                 arrange(METRIC) %>%
                                 mutate(Threshold = 'Overall') %>%
                                 relocate(Threshold,CPM_MNLR,CPM_POLR,CPM_DeepMN,CPM_DeepOR,.after=METRIC),
                               read.csv('../model_performance/CPM/CI_threshold_metrics.csv') %>%
                                 filter(METRIC %in% c('AUC','Accuracy')) %>% 
                                 mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi),
                                        METRIC = factor(METRIC,levels = c('AUC','Accuracy'))) %>%
                                 select(MODEL,Threshold,METRIC,formatted) %>%
                                 pivot_wider(names_from = MODEL, id_cols = c(METRIC,Threshold),values_from = formatted) %>%
                                 arrange(METRIC,Threshold) %>%
                                 relocate(CPM_MNLR,CPM_POLR,CPM_DeepMN,CPM_DeepOR,.after=Threshold))

# APM performance table
APM.performance.table <- rbind(read.csv('../model_performance/APM/CI_overall_metrics.csv') %>% 
                                 mutate(METRIC = factor(METRIC,levels = c('ORC','S','Gen_C','D_xy','Accuracy'))) %>%
                                 mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi)) %>%
                                 select(MODEL,METRIC,formatted) %>%
                                 pivot_wider(names_from = MODEL, id_cols = c(METRIC),values_from = formatted) %>%
                                 arrange(METRIC) %>%
                                 mutate(Threshold = 'Overall') %>%
                                 relocate(Threshold,APM_DeepMN,APM_DeepOR,.after=METRIC),
                               read.csv('../model_performance/APM/CI_threshold_metrics.csv') %>%
                                 filter(METRIC %in% c('AUC','Accuracy')) %>% 
                                 mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi),
                                        METRIC = factor(METRIC,levels = c('AUC','Accuracy'))) %>%
                                 select(MODEL,Threshold,METRIC,formatted) %>%
                                 pivot_wider(names_from = MODEL, id_cols = c(METRIC,Threshold),values_from = formatted) %>%
                                 arrange(METRIC,Threshold) %>%
                                 relocate(APM_DeepMN,APM_DeepOR,.after=Threshold))

# eCPM performance table
eCPM.performance.table <- rbind(read.csv('../model_performance/eCPM/CI_overall_metrics.csv') %>% 
                                  mutate(METRIC = factor(METRIC,levels = c('ORC','S','Gen_C','D_xy','Accuracy'))) %>%
                                  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi)) %>%
                                  select(MODEL,METRIC,formatted) %>%
                                  pivot_wider(names_from = MODEL, id_cols = c(METRIC),values_from = formatted) %>%
                                  arrange(METRIC) %>%
                                  mutate(Threshold = 'Overall') %>%
                                  relocate(Threshold,eCPM_MNLR,eCPM_POLR,eCPM_DeepMN,eCPM_DeepOR,.after=METRIC),
                                read.csv('../model_performance/eCPM/CI_threshold_metrics.csv') %>%
                                  filter(METRIC %in% c('AUC','Accuracy')) %>% 
                                  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi),
                                         METRIC = factor(METRIC,levels = c('AUC','Accuracy'))) %>%
                                  select(MODEL,Threshold,METRIC,formatted) %>%
                                  pivot_wider(names_from = MODEL, id_cols = c(METRIC,Threshold),values_from = formatted) %>%
                                  arrange(METRIC,Threshold) %>%
                                  relocate(eCPM_MNLR,eCPM_POLR,eCPM_DeepMN,eCPM_DeepOR,.after=Threshold))
# CPM calibration table
CPM.calibration.table <- rbind(read.csv('../model_performance/CPM/CI_threshold_metrics.csv') %>%
                                 filter(METRIC %in% c('Calib_Slope')) %>% 
                                 mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi),
                                        METRIC = factor(METRIC,levels = c('Calib_Slope'))) %>%
                                 select(MODEL,Threshold,METRIC,formatted) %>%
                                 pivot_wider(names_from = MODEL, id_cols = c(METRIC,Threshold),values_from = formatted) %>%
                                 arrange(METRIC,Threshold) %>%
                                 relocate(CPM_MNLR,CPM_POLR,CPM_DeepMN,CPM_DeepOR,.after=Threshold))

# APM calibration table
APM.calibration.table <- rbind(read.csv('../model_performance/APM/CI_threshold_metrics.csv') %>%
                                 filter(METRIC %in% c('Calib_Slope')) %>% 
                                 mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi),
                                        METRIC = factor(METRIC,levels = c('Calib_Slope'))) %>%
                                 select(MODEL,Threshold,METRIC,formatted) %>%
                                 pivot_wider(names_from = MODEL, id_cols = c(METRIC,Threshold),values_from = formatted) %>%
                                 arrange(METRIC,Threshold) %>%
                                 relocate(APM_DeepMN,APM_DeepOR,.after=Threshold))

# eCPM calibration table
eCPM.calibration.table <- rbind(read.csv('../model_performance/eCPM/CI_threshold_metrics.csv') %>%
                                  filter(METRIC %in% c('Calib_Slope')) %>% 
                                  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi),
                                         METRIC = factor(METRIC,levels = c('Calib_Slope'))) %>%
                                  select(MODEL,Threshold,METRIC,formatted) %>%
                                  pivot_wider(names_from = MODEL, id_cols = c(METRIC,Threshold),values_from = formatted) %>%
                                  arrange(METRIC,Threshold) %>%
                                  relocate(eCPM_MNLR,eCPM_POLR,eCPM_DeepMN,eCPM_DeepOR,.after=Threshold))

### XI. Supplementary Appendix 3
# Create dataframe of predicted and observed probabilities corresponding to random guessing
random.calib.df <- data.frame(Threshold='GOSE > 3',
                              PredProb = seq(0,1,length.out=1000),
                              TrueProb = .8)

# Implement ggplot to visualise random calibration curve
random.calib.plot <- random.calib.df %>%
  ggplot(aes(x=PredProb,y=TrueProb)) +
  xlab("Predicted Probability") +
  ylab("Observed Probability") +
  coord_cartesian(ylim = c(0,1),xlim = c(0,1))+
  geom_segment(x = 0, y = 0, xend = 1, yend = 1,alpha = 0.5,linetype = "dashed",size=.75/.pt, color = 'gray')+
  geom_line(alpha = 1,size=1.5/.pt, color = '#F8766D')+
  ggtitle('Random-guessing probability calibration plot')+
  theme_classic()+
  theme(
    plot.title = element_text(size=7, color = "black",face = 'bold',hjust = .5), 
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.text.x = element_text(size = 5, color = "black"),
    axis.text.y = element_text(size = 5, color = "black"),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    aspect.ratio = 1,
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    axis.line = element_blank()
  )

# Create directory for current date and save calibration plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'random_calibration.svg'),random.calib.plot,device= svg,units='in',dpi=600,width=3,height = 3)

# Create dataframe to store distribution of prediction probabilities corresponding to random guessing
random.thresh.probs <- data.frame(Threshold='GOSE > 3',
                                  Prob = seq(0,1,length.out=1000))

# Implement ggplot to create distribution histograms for calibration plots
random.calib.dist.plot <- ggplot(data = random.thresh.probs) +
  geom_histogram(aes(x = Prob, y = ..density..),bins = 1,fill='#F8766D',alpha=.75) +
  geom_segment(inherit.aes = F,x=0,y = 0, xend = 1, yend = 0,size=.75/.pt, color = 'black')+
  coord_cartesian(xlim = c(0,1),ylim = c(-30,30),expand = T) +
  scale_y_symmetric(mid = 0) +
  xlab("Predicted Probability") +
  ylab("Observed Probability") +
  ggtitle('Random-guessing probability calibration plot')+
  theme_classic()+
  theme(
    plot.title = element_text(size=7, color = "black",face = 'bold',hjust = .5), 
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.text.x = element_text(size = 5, color = "black"),
    axis.text.y = element_text(size = 5, color = "black"),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    aspect.ratio = 1,
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    axis.line = element_blank()
  )

# Create directory for current date and save calibration distribution plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'random_distribution.svg'),random.calib.dist.plot,device= svg,units='in',dpi=600,width=3,height = 3)

# Create a dataframe for calibration error distribution
random.calib.error <- data.frame(xmins = c(0,.2),
                                 xmaxes = c(.2,.8),
                                 ymins = c(0,0),
                                 ymaxes=c(2,1))

random.calib.error.line <- data.frame(x=c(0,0,.2,.2,.8,.8),y=c(0,2,2,1,1,0))

random.calib.error.dist.plot <- ggplot(random.calib.error,aes(xmin=xmins, xmax=xmaxes, ymin=ymins, ymax=ymaxes))+
  geom_rect(fill='gray',alpha=.5)+
  geom_line(data=random.calib.error.line,inherit.aes = F,aes(x=x,y=y),size=.75/.pt)+
  xlab(TeX(r'(Calibration Error ($| p_{obs} - p_{pred} |$))', bold=TRUE)) +
  ylab("Density") +
  ggtitle('Random-guessing calibration error distribution')+
  coord_cartesian(ylim = c(0,2.5),xlim = c(0,.8))+
  theme_classic()+
  theme(
    plot.title = element_text(size=7, color = "black",face = 'bold',hjust = .5), 
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.text.x = element_text(size = 5, color = "black"),
    axis.text.y = element_text(size = 5, color = "black"),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    aspect.ratio = 1,
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    axis.line = element_blank()
  )

# Create directory for current date and save calibration distribution plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'random_calib_error_distribution.svg'),random.calib.error.dist.plot,device= svg,units='in',dpi=600,width=3,height = 3)

# ICI shading of random calibration error distribution
ICI.plot <- ggplot(random.calib.error,aes(xmin=xmins, xmax=xmaxes, ymin=ymins, ymax=ymaxes))+
  geom_rect(fill='gray',alpha=.5)+
  geom_rect(inherit.aes = FALSE,xmin=0, xmax=.2, ymin=0, ymax=2,fill='#F8766D',alpha=.2)+
  geom_rect(inherit.aes = FALSE,xmin=.2, xmax=0.34, ymin=0, ymax=1,fill='#F8766D',alpha=.2)+
  geom_line(data=random.calib.error.line,inherit.aes = F,aes(x=x,y=y),size=.75/.pt)+
  xlab(TeX(r'(Calibration Error ($| p_{obs} - p_{pred} |$))', bold=TRUE)) +
  ylab("Density") +
  ggtitle('Random-guessing calibration error distribution')+
  coord_cartesian(ylim = c(0,2.5),xlim = c(0,.8))+
  theme_classic()+
  theme(
    plot.title = element_text(size=7, color = "black",face = 'bold',hjust = .5), 
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.text.x = element_text(size = 5, color = "black"),
    axis.text.y = element_text(size = 5, color = "black"),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    aspect.ratio = 1,
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    axis.line = element_blank()
  )

# Create directory for current date and save calibration distribution plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'ICI_error_distribution.svg'),ICI.plot,device= svg,units='in',dpi=600,width=3,height = 3)

# E50 shading of random calibration error distribution
E50.plot <- ggplot(random.calib.error,aes(xmin=xmins, xmax=xmaxes, ymin=ymins, ymax=ymaxes))+
  geom_rect(fill='gray',alpha=.5)+
  geom_rect(inherit.aes = FALSE,xmin=0, xmax=.2, ymin=0, ymax=2,fill='#7CAE00',alpha=.2)+
  geom_rect(inherit.aes = FALSE,xmin=.2, xmax=0.3, ymin=0, ymax=1,fill='#7CAE00',alpha=.2)+
  geom_line(data=random.calib.error.line,inherit.aes = F,aes(x=x,y=y),size=.75/.pt)+
  xlab(TeX(r'(Calibration Error ($| p_{obs} - p_{pred} |$))', bold=TRUE)) +
  ylab("Density") +
  ggtitle('Random-guessing calibration error distribution')+
  coord_cartesian(ylim = c(0,2.5),xlim = c(0,.8))+
  theme_classic()+
  theme(
    plot.title = element_text(size=7, color = "black",face = 'bold',hjust = .5), 
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.text.x = element_text(size = 5, color = "black"),
    axis.text.y = element_text(size = 5, color = "black"),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    aspect.ratio = 1,
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    axis.line = element_blank()
  )

# Create directory for current date and save calibration distribution plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'E50_error_distribution.svg'),E50.plot,device= svg,units='in',dpi=600,width=3,height = 3)

# E90 shading of random calibration error distribution
E90.plot <- ggplot(random.calib.error,aes(xmin=xmins, xmax=xmaxes, ymin=ymins, ymax=ymaxes))+
  geom_rect(fill='gray',alpha=.5)+
  geom_rect(inherit.aes = FALSE,xmin=0, xmax=.2, ymin=0, ymax=2,fill='#00BFC4',alpha=.2)+
  geom_rect(inherit.aes = FALSE,xmin=.2, xmax=0.7, ymin=0, ymax=1,fill='#00BFC4',alpha=.2)+
  geom_line(data=random.calib.error.line,inherit.aes = F,aes(x=x,y=y),size=.75/.pt)+
  xlab(TeX(r'(Calibration Error ($| p_{obs} - p_{pred} |$))', bold=TRUE)) +
  ylab("Density") +
  ggtitle('Random-guessing calibration error distribution')+
  coord_cartesian(ylim = c(0,2.5),xlim = c(0,.8))+
  theme_classic()+
  theme(
    plot.title = element_text(size=7, color = "black",face = 'bold',hjust = .5), 
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.text.x = element_text(size = 5, color = "black"),
    axis.text.y = element_text(size = 5, color = "black"),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    aspect.ratio = 1,
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    axis.line = element_blank()
  )

# Create directory for current date and save calibration distribution plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'E90_error_distribution.svg'),E90.plot,device= svg,units='in',dpi=600,width=3,height = 3)

# ICI shading of random calibration error distribution
ICI.plot <- ggplot(random.calib.error,aes(xmin=xmins, xmax=xmaxes, ymin=ymins, ymax=ymaxes))+
  geom_rect(fill='gray',alpha=.5)+
  geom_rect(inherit.aes = FALSE,xmin=0, xmax=.2, ymin=0, ymax=2,fill='#C77CFF',alpha=.2)+
  geom_rect(inherit.aes = FALSE,xmin=.2, xmax=0.8, ymin=0, ymax=1,fill='#C77CFF',alpha=.2)+
  geom_line(data=random.calib.error.line,inherit.aes = F,aes(x=x,y=y),size=.75/.pt)+
  xlab(TeX(r'(Calibration Error ($| p_{obs} - p_{pred} |$))', bold=TRUE)) +
  ylab("Density") +
  ggtitle('Random-guessing calibration error distribution')+
  coord_cartesian(ylim = c(0,2.5),xlim = c(0,.8))+
  theme_classic()+
  theme(
    plot.title = element_text(size=7, color = "black",face = 'bold',hjust = .5), 
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.text.x = element_text(size = 5, color = "black"),
    axis.text.y = element_text(size = 5, color = "black"),
    axis.title.x = element_text(size = 7, color = "black",face = 'bold'),
    axis.title.y = element_text(size = 7, color = "black",face = 'bold'),
    panel.border = element_rect(colour = "black", fill=NA, size = 1/.pt),
    aspect.ratio = 1,
    plot.margin=grid::unit(c(0,0,0,0), "mm"),
    axis.line = element_blank()
  )

# Create directory for current date and save calibration distribution plots
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'ICI_error_distribution.svg'),ICI.plot,device= svg,units='in',dpi=600,width=3,height = 3)

### XII. Supplementary Appendix 4
## CPM
# Load tuning grids
CPM.tuning.grid <- read.csv('../CPM_outputs/DEEP_v1-0/CPM_deep_tuning_grid.csv')
CPM.tuning.grid$NEURONS[CPM.tuning.grid$LAYERS == 1] <- str_remove(CPM.tuning.grid$NEURONS[CPM.tuning.grid$LAYERS == 1], ',')
CPM.tuning.grid <- CPM.tuning.grid %>%
  rowwise() %>%
  mutate(MEDIAN_NEURONS_PER_LAYER = median(eval(parse(text=paste0('c',NEURONS)))))

CPM.post.r01.tuning.grid <- read.csv('../CPM_outputs/DEEP_v1-0/CPM_post_repeat_01_deep_tuning_grid.csv')
CPM.post.r01.tuning.grid$NEURONS[CPM.post.r01.tuning.grid$LAYERS == 1] <- str_remove(CPM.post.r01.tuning.grid$NEURONS[CPM.post.r01.tuning.grid$LAYERS == 1], ',')
CPM.post.r01.tuning.grid <- CPM.post.r01.tuning.grid %>%
  rowwise() %>%
  mutate(MEDIAN_NEURONS_PER_LAYER = median(eval(parse(text=paste0('c',NEURONS)))))

CPM.post.r03.tuning.grid <- read.csv('../CPM_outputs/DEEP_v1-0/CPM_post_repeat_03_deep_tuning_grid.csv')
CPM.post.r03.tuning.grid$NEURONS[CPM.post.r03.tuning.grid$LAYERS == 1] <- str_remove(CPM.post.r03.tuning.grid$NEURONS[CPM.post.r03.tuning.grid$LAYERS == 1], ',')
CPM.post.r03.tuning.grid <- CPM.post.r03.tuning.grid %>%
  rowwise() %>%
  mutate(MEDIAN_NEURONS_PER_LAYER = median(eval(parse(text=paste0('c',NEURONS)))))

CPM.post.r15.tuning.grid <- read.csv('../CPM_outputs/DEEP_v1-0/CPM_post_repeat_15_deep_tuning_grid.csv')
CPM.post.r15.tuning.grid$NEURONS[CPM.post.r15.tuning.grid$LAYERS == 1] <- str_remove(CPM.post.r15.tuning.grid$NEURONS[CPM.post.r15.tuning.grid$LAYERS == 1], ',')
CPM.post.r15.tuning.grid <- CPM.post.r15.tuning.grid %>%
  rowwise() %>%
  mutate(MEDIAN_NEURONS_PER_LAYER = median(eval(parse(text=paste0('c',NEURONS)))))

CPM.final.DeepMN.tuning.grid <- read.csv('../CPM_outputs/DEEP_v1-0/CPM_final_deepMN_tuning_grid.csv')
CPM.final.DeepMN.tuning.grid$NEURONS[CPM.final.DeepMN.tuning.grid$LAYERS == 1] <- str_remove(CPM.final.DeepMN.tuning.grid$NEURONS[CPM.final.DeepMN.tuning.grid$LAYERS == 1], ',')
CPM.final.DeepMN.tuning.grid <- CPM.final.DeepMN.tuning.grid %>%
  rowwise() %>%
  mutate(MEDIAN_NEURONS_PER_LAYER = median(eval(parse(text=paste0('c',NEURONS)))))

CPM.final.DeepOR.tuning.grid <- read.csv('../CPM_outputs/DEEP_v1-0/CPM_final_deepOR_tuning_grid.csv')
CPM.final.DeepOR.tuning.grid$NEURONS[CPM.final.DeepOR.tuning.grid$LAYERS == 1] <- str_remove(CPM.final.DeepOR.tuning.grid$NEURONS[CPM.final.DeepOR.tuning.grid$LAYERS == 1], ',')
CPM.final.DeepOR.tuning.grid <- CPM.final.DeepOR.tuning.grid %>%
  rowwise() %>%
  mutate(MEDIAN_NEURONS_PER_LAYER = median(eval(parse(text=paste0('c',NEURONS)))))

# Create configuration count dataframe
CPM.config.df <- rbind(
  CPM.tuning.grid %>% group_by(OUTPUT_ACTIVATION) %>% count() %>% mutate(Repeat=1),
  CPM.tuning.grid %>% group_by(OUTPUT_ACTIVATION) %>% count() %>% mutate(Repeat=1.8),
  CPM.post.r01.tuning.grid %>% group_by(OUTPUT_ACTIVATION) %>% count() %>% mutate(Repeat=2),
  CPM.post.r01.tuning.grid %>% group_by(OUTPUT_ACTIVATION) %>% count() %>% mutate(Repeat=3.8),
  CPM.post.r03.tuning.grid %>% group_by(OUTPUT_ACTIVATION) %>% count() %>% mutate(Repeat=4),
  CPM.post.r03.tuning.grid %>% group_by(OUTPUT_ACTIVATION) %>% count() %>% mutate(Repeat=15.8),
  CPM.post.r15.tuning.grid %>% group_by(OUTPUT_ACTIVATION) %>% count() %>% mutate(Repeat=16),
  CPM.post.r15.tuning.grid %>% group_by(OUTPUT_ACTIVATION) %>% count() %>% mutate(Repeat=20.8)
)

CPM.config.labels <- rbind(
  CPM.post.r01.tuning.grid %>% 
    group_by(OUTPUT_ACTIVATION) %>% 
    count() %>% 
    mutate(Repeat=2) %>% 
    mutate(MODEL=plyr::mapvalues(OUTPUT_ACTIVATION,
                                 from = c('softmax','sigmoid'),
                                 to = c('CPM[DeepMN]','CPM[DeepOR]'))) %>%
    mutate(PlotLabel=paste0(MODEL,': ',n)),
  CPM.post.r03.tuning.grid %>% 
    group_by(OUTPUT_ACTIVATION) %>% 
    count() %>% 
    mutate(Repeat=4) %>%
    mutate(MODEL=plyr::mapvalues(OUTPUT_ACTIVATION,
                                 from = c('softmax','sigmoid'),
                                 to = c('CPM[DeepMN]','CPM[DeepOR]'))) %>%
    mutate(PlotLabel=paste0(MODEL,': ',n)),
  CPM.post.r15.tuning.grid %>% 
    group_by(OUTPUT_ACTIVATION) %>% 
    count() %>% 
    mutate(Repeat=16) %>%
    mutate(MODEL=plyr::mapvalues(OUTPUT_ACTIVATION,
                                 from = c('softmax','sigmoid'),
                                 to = c('CPM[DeepMN]','CPM[DeepOR]'))) %>%
    mutate(PlotLabel=paste0(MODEL,': ',n))
)

# Create configuration count plot
CPM.config.plot <-  ggplot(data=NULL,aes(x=Repeat,color=OUTPUT_ACTIVATION,y=n)) +
  geom_line(data=CPM.config.df)+ 
  geom_point(data=CPM.config.labels) +
  geom_label(data=CPM.config.labels,
             aes(label=n),
             hjust = 0, 
             nudge_x = 0.05,
             parse=TRUE,
             label.size = NA,
             show.legend = FALSE)+
  coord_trans(y="log2")+
  scale_y_continuous(breaks=2^seq(0,11),limits = c(1,2200))+
  scale_x_continuous(breaks=seq(1,20,by=1),minor_breaks = seq(1,21,by=.2),limits=c(.8,21),expand=c(0,0))+
  theme_bw()+
  xlab("Cross-validation partition (repeat as labelled major grid, fold as minor grid)")+
  ylab(expression(bold(Number~of~trained~configurations~(log[2]~scale))))+
  scale_color_discrete(breaks=c('softmax','sigmoid'),
                       limits=c('softmax','sigmoid'),
                       name = 'CPM',
                       labels=c(expression(CPM[DeepMN]),expression(CPM[DeepOR])))+
  theme(
    panel.grid.minor.y = element_blank(),
    axis.text.x = element_text(color = "black"),
    axis.text.y = element_text(color = "black"),
    axis.title.y = element_text(color = "black",face = 'bold'),
    axis.title.x = element_text(color = "black",face = 'bold'),
    legend.position = 'bottom',
    legend.title = element_text(color = "black",face = 'bold')
  )

# Combine all tuning grids into single dataframe
compiled.CPM.tuning.grids <- rbind(
  CPM.tuning.grid %>% mutate(LastCompleted = 'Base'),
  CPM.post.r01.tuning.grid %>% mutate(LastCompleted = 'Repeat 01'),
  CPM.post.r03.tuning.grid %>% mutate(LastCompleted = 'Repeat 03'),
  CPM.post.r15.tuning.grid %>% mutate(LastCompleted = 'Repeat 15')
)

# Calculate overall count of configurations by last completed partition
CPM.total.config.df <- compiled.CPM.tuning.grids %>%
  group_by(OUTPUT_ACTIVATION,LastCompleted) %>%
  summarise(TotalConfigs = n())

# Calculate count and proportion of each of the variable hyperparameters at each last completed partition
CPM.proportion.df <- compiled.CPM.tuning.grids %>%
  select(c(OUTPUT_ACTIVATION,LastCompleted,LAYERS,MEDIAN_NEURONS_PER_LAYER,DROPOUT)) %>%
  pivot_longer(cols = -c(OUTPUT_ACTIVATION,LastCompleted),names_to = 'Hyperparameter') %>%
  group_by(OUTPUT_ACTIVATION,LastCompleted,Hyperparameter,value) %>%
  count() %>%
  left_join(CPM.total.config.df,by = c('OUTPUT_ACTIVATION','LastCompleted')) %>%
  rowwise() %>%
  mutate(Percentage = 100*n/TotalConfigs) %>%
  mutate(formatted =sprintf('%d (%.01f%%)',n,Percentage)) %>%
  select(c(OUTPUT_ACTIVATION,LastCompleted,Hyperparameter,value,formatted)) %>%
  pivot_wider(id_cols = c(OUTPUT_ACTIVATION,Hyperparameter,value),names_from = LastCompleted,values_from = formatted) %>%
  replace(is.na(.), '0 (0%)')

## APM
# Load tuning grids
APM.tuning.grid <- read.csv('../APM_outputs/DEEP_v1-0/APM_deep_tuning_grid.csv')
APM.tuning.grid$NEURONS[APM.tuning.grid$LAYERS == 1] <- str_remove(APM.tuning.grid$NEURONS[APM.tuning.grid$LAYERS == 1], ',')
APM.tuning.grid <- APM.tuning.grid %>%
  rowwise() %>%
  mutate(MEDIAN_NEURONS_PER_LAYER = median(eval(parse(text=paste0('c',NEURONS)))))

APM.post.r01.f1.tuning.grid <- read.csv('../APM_outputs/DEEP_v1-0/APM_post_repeat_01_fold_1_deep_tuning_grid.csv')
APM.post.r01.f1.tuning.grid$NEURONS[APM.post.r01.f1.tuning.grid$LAYERS == 1] <- str_remove(APM.post.r01.f1.tuning.grid$NEURONS[APM.post.r01.f1.tuning.grid$LAYERS == 1], ',')
APM.post.r01.f1.tuning.grid <- APM.post.r01.f1.tuning.grid %>%
  rowwise() %>%
  mutate(MEDIAN_NEURONS_PER_LAYER = median(eval(parse(text=paste0('c',NEURONS)))))

APM.post.r10.tuning.grid <- read.csv('../APM_outputs/DEEP_v1-0/APM_post_repeat_10_deep_tuning_grid.csv')
APM.post.r10.tuning.grid$NEURONS[APM.post.r10.tuning.grid$LAYERS == 1] <- str_remove(APM.post.r10.tuning.grid$NEURONS[APM.post.r10.tuning.grid$LAYERS == 1], ',')
APM.post.r10.tuning.grid <- APM.post.r10.tuning.grid %>%
  rowwise() %>%
  mutate(MEDIAN_NEURONS_PER_LAYER = median(eval(parse(text=paste0('c',NEURONS)))))

# Create configuration count dataframe
APM.config.df <- rbind(
  APM.tuning.grid %>% group_by(OUTPUT_ACTIVATION) %>% count() %>% mutate(Repeat=1),
  APM.post.r01.f1.tuning.grid %>% group_by(OUTPUT_ACTIVATION) %>% count() %>% mutate(Repeat=1.2),
  APM.post.r01.f1.tuning.grid %>% group_by(OUTPUT_ACTIVATION) %>% count() %>% mutate(Repeat=10.8),
  APM.post.r10.tuning.grid %>% group_by(OUTPUT_ACTIVATION) %>% count() %>% mutate(Repeat=11),
  APM.post.r10.tuning.grid %>% group_by(OUTPUT_ACTIVATION) %>% count() %>% mutate(Repeat=20.8)
)

APM.config.labels <- rbind(
  APM.post.r01.f1.tuning.grid %>% 
    group_by(OUTPUT_ACTIVATION) %>% 
    count() %>% 
    mutate(Repeat=1.2) %>% 
    mutate(MODEL=plyr::mapvalues(OUTPUT_ACTIVATION,
                                 from = c('softmax','sigmoid'),
                                 to = c('APM[MN]','APM[OR]'))) %>%
    mutate(PlotLabel=paste0(MODEL,': ',n)),
  APM.post.r10.tuning.grid %>% 
    group_by(OUTPUT_ACTIVATION) %>% 
    count() %>% 
    mutate(Repeat=11) %>%
    mutate(MODEL=plyr::mapvalues(OUTPUT_ACTIVATION,
                                 from = c('softmax','sigmoid'),
                                 to = c('APM[MN]','APM[OR]'))) %>%
    mutate(PlotLabel=paste0(MODEL,': ',n))
)

# Create configuration count plot
APM.config.plot <-  ggplot(data=NULL,aes(x=Repeat,color=OUTPUT_ACTIVATION,y=n)) +
  geom_line(data=APM.config.df)+ 
  geom_point(data=APM.config.labels) +
  geom_label(data=APM.config.labels,
             aes(label=n),
             hjust = 0, 
             nudge_x = 0.05,
             parse=TRUE,
             label.size = NA,
             show.legend = FALSE)+
  coord_trans(y="log2")+
  scale_y_continuous(breaks=2^seq(0,11),limits = c(1,2200))+
  scale_x_continuous(breaks=seq(1,20,by=1),minor_breaks = seq(1,21,by=.2),limits=c(.8,21),expand=c(0,0))+
  theme_bw()+
  xlab("Cross-validation partition (repeat as labelled major grid, fold as minor grid)")+
  ylab(expression(bold(Number~of~trained~configurations~(log[2]~scale))))+
  scale_color_discrete(breaks=c('softmax','sigmoid'),
                       limits=c('softmax','sigmoid'),
                       name = 'APM',
                       labels=c(expression(APM[MN]),expression(APM[OR])))+
  theme(
    panel.grid.minor.y = element_blank(),
    axis.text.x = element_text(color = "black"),
    axis.text.y = element_text(color = "black"),
    axis.title.y = element_text(color = "black",face = 'bold'),
    axis.title.x = element_text(color = "black",face = 'bold'),
    legend.position = 'bottom',
    legend.title = element_text(color = "black",face = 'bold')
  )

# Combine all tuning grids into single dataframe
compiled.APM.tuning.grids <- rbind(
  APM.tuning.grid %>% mutate(LastCompleted = 'Base'),
  APM.post.r01.f1.tuning.grid %>% mutate(LastCompleted = 'Repeat 1, Fold 1'),
  APM.post.r10.tuning.grid %>% mutate(LastCompleted = 'Repeat 10')
  )

# Calculate overall count of configurations by last completed partition
APM.total.config.df <- compiled.APM.tuning.grids %>%
  group_by(OUTPUT_ACTIVATION,LastCompleted) %>%
  summarise(TotalConfigs = n())

# Calculate count and proportion of each of the variable hyperparameters at each last completed partition
APM.proportion.df <- compiled.APM.tuning.grids %>%
  select(c(OUTPUT_ACTIVATION,LastCompleted,LAYERS,MEDIAN_NEURONS_PER_LAYER,DROPOUT)) %>%
  pivot_longer(cols = -c(OUTPUT_ACTIVATION,LastCompleted),names_to = 'Hyperparameter') %>%
  group_by(OUTPUT_ACTIVATION,LastCompleted,Hyperparameter,value) %>%
  count() %>%
  left_join(APM.total.config.df,by = c('OUTPUT_ACTIVATION','LastCompleted')) %>%
  rowwise() %>%
  mutate(Percentage = 100*n/TotalConfigs) %>%
  mutate(formatted =sprintf('%d (%.01f%%)',n,Percentage)) %>%
  select(c(OUTPUT_ACTIVATION,LastCompleted,Hyperparameter,value,formatted)) %>%
  pivot_wider(id_cols = c(OUTPUT_ACTIVATION,Hyperparameter,value),names_from = LastCompleted,values_from = formatted) %>%
  replace(is.na(.), '0 (0%)')

## eCPM
# Load tuning grids
eCPM.tuning.grid <- read.csv('../eCPM_outputs/DEEP_v1-0/eCPM_deep_tuning_grid.csv')
eCPM.tuning.grid$NEURONS[eCPM.tuning.grid$LAYERS == 1] <- str_remove(eCPM.tuning.grid$NEURONS[eCPM.tuning.grid$LAYERS == 1], ',')
eCPM.tuning.grid <- eCPM.tuning.grid %>%
  rowwise() %>%
  mutate(MEDIAN_NEURONS_PER_LAYER = median(eval(parse(text=paste0('c',NEURONS)))))

eCPM.post.r01.tuning.grid <- read.csv('../eCPM_outputs/DEEP_v1-0/eCPM_post_repeat_01_deep_tuning_grid.csv')
eCPM.post.r01.tuning.grid$NEURONS[eCPM.post.r01.tuning.grid$LAYERS == 1] <- str_remove(eCPM.post.r01.tuning.grid$NEURONS[eCPM.post.r01.tuning.grid$LAYERS == 1], ',')
eCPM.post.r01.tuning.grid <- eCPM.post.r01.tuning.grid %>%
  rowwise() %>%
  mutate(MEDIAN_NEURONS_PER_LAYER = median(eval(parse(text=paste0('c',NEURONS)))))

eCPM.post.r16.tuning.grid <- read.csv('../eCPM_outputs/DEEP_v1-0/eCPM_post_repeat_16_deep_tuning_grid.csv')
eCPM.post.r16.tuning.grid$NEURONS[eCPM.post.r16.tuning.grid$LAYERS == 1] <- str_remove(eCPM.post.r16.tuning.grid$NEURONS[eCPM.post.r16.tuning.grid$LAYERS == 1], ',')
eCPM.post.r16.tuning.grid <- eCPM.post.r16.tuning.grid %>%
  rowwise() %>%
  mutate(MEDIAN_NEURONS_PER_LAYER = median(eval(parse(text=paste0('c',NEURONS)))))

# Create configuration count dataframe
eCPM.config.df <- rbind(
  eCPM.tuning.grid %>% group_by(OUTPUT_ACTIVATION) %>% count() %>% mutate(Repeat=1),
  eCPM.tuning.grid %>% group_by(OUTPUT_ACTIVATION) %>% count() %>% mutate(Repeat=1.8),
  eCPM.post.r01.tuning.grid %>% group_by(OUTPUT_ACTIVATION) %>% count() %>% mutate(Repeat=2),
  eCPM.post.r01.tuning.grid %>% group_by(OUTPUT_ACTIVATION) %>% count() %>% mutate(Repeat=16.8),
  eCPM.post.r16.tuning.grid %>% group_by(OUTPUT_ACTIVATION) %>% count() %>% mutate(Repeat=17),
  eCPM.post.r16.tuning.grid %>% group_by(OUTPUT_ACTIVATION) %>% count() %>% mutate(Repeat=20.8)
)

eCPM.config.labels <- rbind(
  eCPM.post.r01.tuning.grid %>% 
    group_by(OUTPUT_ACTIVATION) %>% 
    count() %>% 
    mutate(Repeat=2) %>% 
    mutate(MODEL=plyr::mapvalues(OUTPUT_ACTIVATION,
                                 from = c('softmax','sigmoid'),
                                 to = c('eCPM[DeepMN]','eCPM[DeepOR]'))) %>%
    mutate(PlotLabel=paste0(MODEL,': ',n)),
  eCPM.post.r16.tuning.grid %>% 
    group_by(OUTPUT_ACTIVATION) %>% 
    count() %>% 
    mutate(Repeat=17) %>%
    mutate(MODEL=plyr::mapvalues(OUTPUT_ACTIVATION,
                                 from = c('softmax','sigmoid'),
                                 to = c('eCPM[DeepMN]','eCPM[DeepOR]'))) %>%
    mutate(PlotLabel=paste0(MODEL,': ',n))
)

# Create configuration count plot
eCPM.config.plot <-  ggplot(data=NULL,aes(x=Repeat,color=OUTPUT_ACTIVATION,y=n)) +
  geom_line(data=eCPM.config.df)+ 
  geom_point(data=eCPM.config.labels) +
  geom_label(data=eCPM.config.labels,
             aes(label=n),
             hjust = 0, 
             nudge_x = 0.05,
             parse=TRUE,
             label.size = NA,
             show.legend = FALSE)+
  coord_trans(y="log2")+
  scale_y_continuous(breaks=2^seq(0,11),limits = c(1,2200))+
  scale_x_continuous(breaks=seq(1,20,by=1),minor_breaks = seq(1,21,by=.2),limits=c(.8,21),expand=c(0,0))+
  theme_bw()+
  xlab("Cross-validation partition (repeat as labelled major grid, fold as minor grid)")+
  ylab(expression(bold(Number~of~trained~configurations~(log[2]~scale))))+
  scale_color_discrete(breaks=c('softmax','sigmoid'),
                       limits=c('softmax','sigmoid'),
                       name = 'eCPM',
                       labels=c(expression(eCPM[DeepMN]),expression(eCPM[DeepOR])))+
  theme(
    panel.grid.minor.y = element_blank(),
    axis.text.x = element_text(color = "black"),
    axis.text.y = element_text(color = "black"),
    axis.title.y = element_text(color = "black",face = 'bold'),
    axis.title.x = element_text(color = "black",face = 'bold'),
    legend.position = 'bottom',
    legend.title = element_text(color = "black",face = 'bold')
  )

# Combine all tuning grids into single dataframe
compiled.eCPM.tuning.grids <- rbind(
  eCPM.tuning.grid %>% mutate(LastCompleted = 'Base'),
  eCPM.post.r01.tuning.grid %>% mutate(LastCompleted = 'Repeat 1'),
  eCPM.post.r16.tuning.grid %>% mutate(LastCompleted = 'Repeat 16')
)

# Calculate overall count of configurations by last completed partition
eCPM.total.config.df <- compiled.eCPM.tuning.grids %>%
  group_by(OUTPUT_ACTIVATION,LastCompleted) %>%
  summarise(TotalConfigs = n())

# Calculate count and proportion of each of the variable hyperparameters at each last completed partition
eCPM.proportion.df <- compiled.eCPM.tuning.grids %>%
  select(c(OUTPUT_ACTIVATION,LastCompleted,LAYERS,MEDIAN_NEURONS_PER_LAYER,DROPOUT)) %>%
  pivot_longer(cols = -c(OUTPUT_ACTIVATION,LastCompleted),names_to = 'Hyperparameter') %>%
  group_by(OUTPUT_ACTIVATION,LastCompleted,Hyperparameter,value) %>%
  count() %>%
  left_join(eCPM.total.config.df,by = c('OUTPUT_ACTIVATION','LastCompleted')) %>%
  rowwise() %>%
  mutate(Percentage = 100*n/TotalConfigs) %>%
  mutate(formatted =sprintf('%d (%.01f%%)',n,Percentage)) %>%
  select(c(OUTPUT_ACTIVATION,LastCompleted,Hyperparameter,value,formatted)) %>%
  pivot_wider(id_cols = c(OUTPUT_ACTIVATION,Hyperparameter,value),names_from = LastCompleted,values_from = formatted) %>%
  replace(is.na(.), '0 (0%)')
