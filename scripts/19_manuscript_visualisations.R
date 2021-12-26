#### Master Script #: Visualise study results for manuscript ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Figure 2
# III. Figure 3
# IV. Figure 4a
# V. Figure 4b
# VI. Figure 5
# Supplementary Figure 1

### I. Initialisation
# Import necessary libraries
library(tidyverse)
library(readxl)
library(RColorBrewer)
library(rvg)
library(svglite)
library(viridis)
library(lemon)

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
# Load threshold-accuracy confidence intervals of all model types to determine CPM_best, APM_best, and eCPM_best
CPM.CI.Accuracy.ave <- read.csv('../model_performance/CPM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC == 'Accuracy') %>% 
  arrange(-mean) %>%
  head(1) %>%
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi))

APM.CI.Accuracy.ave <- read.csv('../model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC == 'Accuracy') %>% 
  arrange(-mean) %>%
  head(1) %>%
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi))

eCPM.CI.Accuracy.ave <- read.csv('../model_performance/eCPM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC == 'Accuracy') %>% 
  arrange(-mean) %>%
  head(1) %>%
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi))

# Load and compile normalised confusion matrix confidence intervals of CPM_best, APM_best, and eCPM_best
CPM.CI.confusion_matrices <- read.csv('../model_performance/CPM/CI_confusion_matrices.csv') %>%
  filter(MODEL %in% CPM.CI.Accuracy.ave$MODEL) %>%
  mutate(MODEL = 'CPM_{Best}')

APM.CI.confusion_matrices <- read.csv('../model_performance/APM/CI_confusion_matrices.csv') %>%
  filter(MODEL %in% APM.CI.Accuracy.ave$MODEL) %>%
  mutate(MODEL = 'APM_{Best}')

eCPM.CI.confusion_matrices <- read.csv('../model_performance/eCPM/CI_confusion_matrices.csv') %>%
  filter(MODEL %in% eCPM.CI.Accuracy.ave$MODEL) %>%
  mutate(MODEL = 'eCPM_{Best}')

CI.confusion_matrices <- rbind(CPM.CI.confusion_matrices,APM.CI.confusion_matrices,eCPM.CI.confusion_matrices)

# Format normalised confusion matrix dataframe for plot (and decide whether to filter out eCPM)
CI.confusion_matrices <- CI.confusion_matrices %>% 
  filter(MODEL != 'eCPM_{Best}') %>%
  mutate(TrueLabel = plyr::mapvalues(TrueLabel,
                                     from = c("GOSE: 1","GOSE: 2/3","GOSE: 4","GOSE: 5","GOSE: 6","GOSE: 7","GOSE: 8"),
                                     to =c("1","2 or 3","4","5","6","7","8")),
         PredLabel = plyr::mapvalues(PredLabel,
                                     from = c("GOSE: 1","GOSE: 2/3","GOSE: 4","GOSE: 5","GOSE: 6","GOSE: 7","GOSE: 8"),
                                     to =c("1","2 or 3","4","5","6","7","8"))) %>%
  mutate(MODEL = factor(MODEL,levels =c('CPM_{Best}','APM_{Best}'))) %>%
  rename(Model = MODEL) %>%
  mutate(formatted = sprintf('%.2f \n (%.2f–%.2f)',cm_prob_mean,cm_prob_lo,cm_prob_hi))

levels(CI.confusion_matrices$Model)= c("CPM_{Best}"=expression(bold(CPM[Best])),
                                       "APM_{Best}"=expression(bold(APM[Best])))

#  Implement ggplot to visualise normalised confusion matrices
best.normalised.cm.plot <- CI.confusion_matrices %>%
  ggplot(aes(x = PredLabel,y = TrueLabel,fill = cm_prob_mean))+
  geom_tile() +
  scale_fill_viridis(discrete=FALSE,
                     breaks = seq(0,.6,.1),
                     limits = c(0,.6)) +
  geom_text(aes(label = formatted,color = as.factor(as.integer(cm_prob_mean>0.36))),
            show.legend = F,
            size = 5/.pt)+
  guides(fill = guide_colourbar(title = 'Proportion of classifications per true outcome',
                                barwidth = grid::unit(5.5,'inches'),
                                barheight = grid::unit(.15,'inches'),
                                direction="horizontal",
                                title.position = 'top',
                                frame.colour=c("black"),
                                frame.linewidth = 1.5/.pt,
                                title.hjust = .5)) +
  facet_rep_wrap(~Model,ncol = 2, nrow = 1, scales='free',labeller = label_parsed) +
  xlab('Predicted functional outcome (GOSE) at 6 months')+
  ylab('True functional outcome (GOSE) at 6 months')+
  scale_color_manual(values = c('white','black')) +
  scale_y_discrete(limits = rev(c("1","2 or 3","4","5","6","7","8")),expand=c(0,0))+
  scale_x_discrete(limits = c("1","2 or 3","4","5","6","7","8"),expand=c(0,0))+
  theme_classic()+
  theme(
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",vjust = 1.5),
    axis.text.y = element_text(size = 6, color = "black",angle=90,vjust = -1.5,hjust = .5),
    axis.title.y = element_text(size=7, color = "black",face = 'bold'),
    axis.title.x = element_text(size=7, color = "black",face = 'bold'),
    strip.background = element_blank(),
    strip.text = element_text(size=7, color = "black",face = 'bold'),
    panel.border = element_blank(), 
    legend.position = 'bottom',
    legend.title = element_text(size = 7, color = "black",face = 'bold'),
    legend.text=element_text(size=6),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    aspect.ratio = 1
  )

# Create directory for current date and save confusion matrices
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'confusion_matrices.svg'),best.normalised.cm.plot,device= svg,units='in',dpi=600,width=6.5,height = 4.15)

### IV. Figure 4a
# Load threshold-AUC confidence intervals of all model types to determine CPM_best, APM_best, and eCPM_best
CPM.CI.AUC.ave <- read.csv('../model_performance/CPM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC == 'AUC') %>% 
  arrange(-mean) %>%
  head(1) %>%
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi))

APM.CI.AUC.ave <- read.csv('../model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC == 'AUC') %>% 
  arrange(-mean) %>%
  head(1) %>%
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi))

eCPM.CI.AUC.ave <- read.csv('../model_performance/eCPM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC == 'AUC') %>% 
  arrange(-mean) %>%
  head(1) %>%
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi))

# Load and compile ROC confidence intervals of CPM_best, APM_best, and eCPM_best
CPM.CI.ROCs <- read.csv('../model_performance/CPM/CI_ROCs.csv') %>%
  filter(MODEL %in% CPM.CI.AUC.ave$MODEL) %>%
  mutate(MODEL = 'CPM_{Best}')

APM.CI.ROCs <- read.csv('../model_performance/APM/CI_ROCs.csv') %>%
  filter(MODEL %in% APM.CI.AUC.ave$MODEL) %>%
  mutate(MODEL = 'APM_{Best}')

eCPM.CI.ROCs <- read.csv('../model_performance/eCPM/CI_ROCs.csv') %>%
  filter(MODEL %in% eCPM.CI.AUC.ave$MODEL) %>%
  mutate(MODEL = 'eCPM_{Best}')

CI.ROCs <- rbind(CPM.CI.ROCs,APM.CI.ROCs,eCPM.CI.ROCs)

# Fix endpoints to corners of AUC plot
CI.ROCs[CI.ROCs$FPR == 0,c('TPR_mean','TPR_median','TPR_lo','TPR_hi')] <- 0
CI.ROCs[CI.ROCs$FPR == 1,c('TPR_mean','TPR_median','TPR_lo','TPR_hi')] <- 1

# Format ROC dataframe for plot (and decide whether to filter out eCPM)
CI.ROCs <- CI.ROCs %>% 
  filter(MODEL != 'eCPM_{Best}') %>%
  mutate(Threshold = plyr::mapvalues(Threshold,
                                     from = c("GOSE>1","GOSE>3","GOSE>4","GOSE>5","GOSE>6","GOSE>7"),
                                     to =c("GOSE > 1","GOSE > 3","GOSE > 4","GOSE > 5","GOSE > 6","GOSE > 7"))) %>%
  mutate(MODEL = factor(MODEL,levels =c('CPM_{Best}','APM_{Best}'))) %>%
  rename(Model = MODEL)

# Implement ggplot to visualise ROCs
best.ROC.curves.plot <- CI.ROCs %>%
  ggplot(aes(x = FPR)) +
  facet_wrap( ~ Threshold,
              scales = 'free',
              ncol = 3) +
  xlab("False Positive Rate") +
  ylab("True Positive Rate") +
  coord_cartesian(ylim = c(0,1),xlim = c(0,1))+
  geom_segment(x = 0, y = 0, xend = 1, yend = 1,alpha = 0.5,linetype = "dashed",size=.75/.pt, color = 'gray')+
  geom_ribbon(aes(ymin = TPR_lo, ymax = TPR_hi, fill = Model), alpha = 0.3,size=.75/.pt,color=NA) +
  geom_line(aes(y = TPR_mean, color = Model), alpha = 1, size=1.3/.pt) +
  scale_fill_discrete(breaks=levels(CI.ROCs$Model),
                      labels=c(expression(CPM[Best]), expression(APM[Best])))+
  scale_color_discrete(breaks=levels(CI.ROCs$Model),
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

# Create directory for current date and save ROC plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'thresh_ROC.svg'),best.ROC.curves.plot,device= svg,units='in',dpi=600,width=6,height = 4.38)

# Get AUC values for CPM_best, APM_best, and eCPM_best at each threshold
CPM.CI.AUC <- read.csv('../model_performance/CPM/CI_threshold_metrics.csv') %>%
  filter(Threshold != 'Average', METRIC == 'AUC', MODEL %in% CPM.CI.AUC.ave$MODEL) %>% 
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi))

APM.CI.AUC <- read.csv('../model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(Threshold != 'Average', METRIC == 'AUC', MODEL %in% APM.CI.AUC.ave$MODEL) %>% 
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi))

eCPM.CI.AUC <- read.csv('../model_performance/eCPM/CI_threshold_metrics.csv') %>%
  filter(Threshold != 'Average', METRIC == 'AUC', MODEL %in% eCPM.CI.AUC.ave$MODEL) %>% 
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi))

### V. Figure 4b
# Load threshold_Emax confidence intervals of all model types to determine CPM_best, APM_best, and eCPM_best
CPM.CI.Emax.ave <- read.csv('../model_performance/CPM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC == 'Emax') %>% 
  arrange(mean) %>%
  head(1)

APM.CI.Emax.ave <- read.csv('../model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC == 'Emax') %>% 
  arrange(mean) %>%
  head(1)

eCPM.CI.Emax.ave <- read.csv('../model_performance/eCPM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC == 'Emax') %>% 
  arrange(mean) %>%
  head(1)

# Load and compile calibration confidence intervals of CPM_best, APM_best, and eCPM_best
CPM.CI.calibration <- read.csv('../model_performance/CPM/CI_calibration.csv') %>%
  filter(MODEL %in% CPM.CI.Emax.ave$MODEL) %>%
  mutate(MODEL = 'CPM_{Best}')

APM.CI.calibration <- read.csv('../model_performance/APM/CI_calibration.csv') %>%
  filter(MODEL %in% APM.CI.Emax.ave$MODEL) %>%
  mutate(MODEL = 'APM_{Best}')

eCPM.CI.calibration <- read.csv('../model_performance/eCPM/CI_calibration.csv') %>%
  filter(MODEL %in% eCPM.CI.Emax.ave$MODEL) %>%
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

# Get Emax values for CPM_best, APM_best, and eCPM_best at each threshold
CPM.CI.Emax <- read.csv('../model_performance/CPM/CI_threshold_metrics.csv') %>%
  filter(Threshold != 'Average', METRIC == 'Emax', MODEL %in% CPM.CI.Emax.ave$MODEL) %>% 
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi))

APM.CI.Emax <- read.csv('../model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(Threshold != 'Average', METRIC == 'Emax', MODEL %in% APM.CI.Emax.ave$MODEL) %>% 
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi))

eCPM.CI.Emax <- read.csv('../model_performance/eCPM/CI_threshold_metrics.csv') %>%
  filter(Threshold != 'Average', METRIC == 'Emax', MODEL %in% eCPM.CI.Emax.ave$MODEL) %>% 
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

### VI. Figure 5
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
  guides(fill = guide_legend(title = expression(bold(Output~nodes~of~APM[DeepMN]~(GOSE))),
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

### Supplementary Figure 1
# Load normalised confusion matrix confidence intervals of CPM
CPM.CI.confusion_matrices <- read.csv('../model_performance/CPM/CI_confusion_matrices.csv')

# Format normalised confusion matrix dataframe for plot
CPM.CI.confusion_matrices <- CPM.CI.confusion_matrices %>% 
  mutate(TrueLabel = plyr::mapvalues(TrueLabel,
                                     from = c("GOSE: 1","GOSE: 2/3","GOSE: 4","GOSE: 5","GOSE: 6","GOSE: 7","GOSE: 8"),
                                     to =c("1","2 or 3","4","5","6","7","8")),
         PredLabel = plyr::mapvalues(PredLabel,
                                     from = c("GOSE: 1","GOSE: 2/3","GOSE: 4","GOSE: 5","GOSE: 6","GOSE: 7","GOSE: 8"),
                                     to =c("1","2 or 3","4","5","6","7","8"))) %>%
  mutate(MODEL = factor(MODEL,levels =c('CPM_MNLR','CPM_POLR','CPM_DeepMN','CPM_DeepOR'))) %>%
  rename(Model = MODEL) %>%
  mutate(formatted = sprintf('%.2f \n (%.2f–%.2f)',cm_prob_mean,cm_prob_lo,cm_prob_hi))

levels(CPM.CI.confusion_matrices$Model)= c("CPM_MNLR"=expression(bold(CPM[MNLR])),
                                           "CPM_POLR"=expression(bold(CPM[POLR])),
                                           "CPM_DeepMN"=expression(bold(CPM[DeepMN])),
                                           "CPM_DeepOR"=expression(bold(CPM[DeepOR])))

#  Implement ggplot to visualise normalised confusion matrices
CPM.normalised.cm.plot <- CPM.CI.confusion_matrices %>%
  ggplot(aes(x = PredLabel,y = TrueLabel,fill = cm_prob_mean))+
  geom_tile() +
  scale_fill_viridis(discrete=FALSE,
                     breaks = seq(0,.8,.1),
                     limits = c(0,.8)) +
  geom_text(aes(label = formatted,color = as.factor(as.integer(cm_prob_mean>0.48))),
            show.legend = F,
            size = 5/.pt)+
  guides(fill = guide_colourbar(title = 'Proportion of classifications per true outcome',
                                barwidth = grid::unit(5.5,'inches'),
                                barheight = grid::unit(.15,'inches'),
                                direction="horizontal",
                                title.position = 'top',
                                frame.colour=c("black"),
                                frame.linewidth = 1.5/.pt,
                                title.hjust = .5)) +
  facet_rep_wrap(~Model,ncol = 2, nrow = 2, scales='free',labeller = label_parsed) +
  xlab('Predicted functional outcome (GOSE) at 6 months')+
  ylab('True functional outcome (GOSE) at 6 months')+
  scale_color_manual(values = c('white','black')) +
  scale_y_discrete(limits = rev(c("1","2 or 3","4","5","6","7","8")),expand=c(0,0))+
  scale_x_discrete(limits = c("1","2 or 3","4","5","6","7","8"),expand=c(0,0))+
  theme_classic()+
  theme(
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",vjust = 1.5),
    axis.text.y = element_text(size = 6, color = "black",angle=90,vjust = -1.5,hjust = .5),
    axis.title.y = element_text(size=7, color = "black",face = 'bold'),
    axis.title.x = element_text(size=7, color = "black",face = 'bold'),
    strip.background = element_blank(),
    strip.text = element_text(size=7, color = "black",face = 'bold'),
    panel.border = element_blank(), 
    legend.position = 'bottom',
    legend.title = element_text(size = 7, color = "black",face = 'bold'),
    legend.text=element_text(size=6),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    aspect.ratio = 1
  )

# Create directory for current date and save confusion matrices
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'CPM_confusion_matrices.svg'),CPM.normalised.cm.plot,device= svg,units='in',dpi=600,width=6.5,height = 8.30)

### Supplementary Figure 2
# Load normalised confusion matrix confidence intervals of APM
APM.CI.confusion_matrices <- read.csv('../model_performance/APM/CI_confusion_matrices.csv')

# Format normalised confusion matrix dataframe for plot
APM.CI.confusion_matrices <- APM.CI.confusion_matrices %>% 
  mutate(TrueLabel = plyr::mapvalues(TrueLabel,
                                     from = c("GOSE: 1","GOSE: 2/3","GOSE: 4","GOSE: 5","GOSE: 6","GOSE: 7","GOSE: 8"),
                                     to =c("1","2 or 3","4","5","6","7","8")),
         PredLabel = plyr::mapvalues(PredLabel,
                                     from = c("GOSE: 1","GOSE: 2/3","GOSE: 4","GOSE: 5","GOSE: 6","GOSE: 7","GOSE: 8"),
                                     to =c("1","2 or 3","4","5","6","7","8"))) %>%
  mutate(MODEL = factor(MODEL,levels =c('APM_DeepMN','APM_DeepOR'))) %>%
  rename(Model = MODEL) %>%
  mutate(formatted = sprintf('%.2f \n (%.2f–%.2f)',cm_prob_mean,cm_prob_lo,cm_prob_hi))

levels(APM.CI.confusion_matrices$Model)= c("APM_DeepMN"=expression(bold(APM[DeepMN])),
                                           "APM_DeepOR"=expression(bold(APM[DeepOR])))

#  Implement ggplot to visualise normalised confusion matrices
APM.normalised.cm.plot <- APM.CI.confusion_matrices %>%
  ggplot(aes(x = PredLabel,y = TrueLabel,fill = cm_prob_mean))+
  geom_tile() +
  scale_fill_viridis(discrete=FALSE,
                     breaks = seq(0,.6,.1),
                     limits = c(0,.6)) +
  geom_text(aes(label = formatted,color = as.factor(as.integer(cm_prob_mean>0.36))),
            show.legend = F,
            size = 5/.pt)+
  guides(fill = guide_colourbar(title = 'Proportion of classifications per true outcome',
                                barwidth = grid::unit(5.5,'inches'),
                                barheight = grid::unit(.15,'inches'),
                                direction="horizontal",
                                title.position = 'top',
                                frame.colour=c("black"),
                                frame.linewidth = 1.5/.pt,
                                title.hjust = .5)) +
  facet_rep_wrap(~Model,ncol = 2, nrow = 2, scales='free',labeller = label_parsed) +
  xlab('Predicted functional outcome (GOSE) at 6 months')+
  ylab('True functional outcome (GOSE) at 6 months')+
  scale_color_manual(values = c('white','black')) +
  scale_y_discrete(limits = rev(c("1","2 or 3","4","5","6","7","8")),expand=c(0,0))+
  scale_x_discrete(limits = c("1","2 or 3","4","5","6","7","8"),expand=c(0,0))+
  theme_classic()+
  theme(
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",vjust = 1.5),
    axis.text.y = element_text(size = 6, color = "black",angle=90,vjust = -1.5,hjust = .5),
    axis.title.y = element_text(size=7, color = "black",face = 'bold'),
    axis.title.x = element_text(size=7, color = "black",face = 'bold'),
    strip.background = element_blank(),
    strip.text = element_text(size=7, color = "black",face = 'bold'),
    panel.border = element_blank(), 
    legend.position = 'bottom',
    legend.title = element_text(size = 7, color = "black",face = 'bold'),
    legend.text=element_text(size=6),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    aspect.ratio = 1
  )

# Create directory for current date and save confusion matrices
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'APM_confusion_matrices.svg'),APM.normalised.cm.plot,device= svg,units='in',dpi=600,width=6.5,height = 4.15)

### Supplementary Figure 3
# Load normalised confusion matrix confidence intervals of eCPM
eCPM.CI.confusion_matrices <- read.csv('../model_performance/eCPM/CI_confusion_matrices.csv')

# Format normalised confusion matrix dataframe for plot
eCPM.CI.confusion_matrices <- eCPM.CI.confusion_matrices %>% 
  mutate(TrueLabel = plyr::mapvalues(TrueLabel,
                                     from = c("GOSE: 1","GOSE: 2/3","GOSE: 4","GOSE: 5","GOSE: 6","GOSE: 7","GOSE: 8"),
                                     to =c("1","2 or 3","4","5","6","7","8")),
         PredLabel = plyr::mapvalues(PredLabel,
                                     from = c("GOSE: 1","GOSE: 2/3","GOSE: 4","GOSE: 5","GOSE: 6","GOSE: 7","GOSE: 8"),
                                     to =c("1","2 or 3","4","5","6","7","8"))) %>%
  mutate(MODEL = factor(MODEL,levels =c('eCPM_MNLR','eCPM_POLR','eCPM_DeepMN','eCPM_DeepOR'))) %>%
  rename(Model = MODEL) %>%
  mutate(formatted = sprintf('%.2f \n (%.2f–%.2f)',cm_prob_mean,cm_prob_lo,cm_prob_hi))

levels(eCPM.CI.confusion_matrices$Model)= c("eCPM_MNLR"=expression(bold(eCPM[MNLR])),
                                            "eCPM_POLR"=expression(bold(eCPM[POLR])),
                                            "eCPM_DeepMN"=expression(bold(eCPM[DeepMN])),
                                            "eCPM_DeepOR"=expression(bold(eCPM[DeepOR])))

#  Implement ggplot to visualise normalised confusion matrices
eCPM.normalised.cm.plot <- eCPM.CI.confusion_matrices %>%
  ggplot(aes(x = PredLabel,y = TrueLabel,fill = cm_prob_mean))+
  geom_tile() +
  scale_fill_viridis(discrete=FALSE,
                     breaks = seq(0,.7,.1),
                     limits = c(0,.7)) +
  geom_text(aes(label = formatted,color = as.factor(as.integer(cm_prob_mean>0.42))),
            show.legend = F,
            size = 5/.pt)+
  guides(fill = guide_colourbar(title = 'Proportion of classifications per true outcome',
                                barwidth = grid::unit(5.5,'inches'),
                                barheight = grid::unit(.15,'inches'),
                                direction="horizontal",
                                title.position = 'top',
                                frame.colour=c("black"),
                                frame.linewidth = 1.5/.pt,
                                title.hjust = .5)) +
  facet_rep_wrap(~Model,ncol = 2, nrow = 2, scales='free',labeller = label_parsed) +
  xlab('Predicted functional outcome (GOSE) at 6 months')+
  ylab('True functional outcome (GOSE) at 6 months')+
  scale_color_manual(values = c('white','black')) +
  scale_y_discrete(limits = rev(c("1","2 or 3","4","5","6","7","8")),expand=c(0,0))+
  scale_x_discrete(limits = c("1","2 or 3","4","5","6","7","8"),expand=c(0,0))+
  theme_classic()+
  theme(
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.text.x = element_text(size = 6, color = "black",vjust = 1.5),
    axis.text.y = element_text(size = 6, color = "black",angle=90,vjust = -1.5,hjust = .5),
    axis.title.y = element_text(size=7, color = "black",face = 'bold'),
    axis.title.x = element_text(size=7, color = "black",face = 'bold'),
    strip.background = element_blank(),
    strip.text = element_text(size=7, color = "black",face = 'bold'),
    panel.border = element_blank(), 
    legend.position = 'bottom',
    legend.title = element_text(size = 7, color = "black",face = 'bold'),
    legend.text=element_text(size=6),
    axis.line = element_blank(),
    axis.ticks = element_blank(),
    aspect.ratio = 1
  )

# Create directory for current date and save confusion matrices
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'eCPM_confusion_matrices.svg'),eCPM.normalised.cm.plot,device= svg,units='in',dpi=600,width=6.5,height = 8.30)

### Supplementary Figure 4
# Load ROC confidence intervals of CPM
CPM.CI.ROCs <- read.csv('../model_performance/CPM/CI_ROCs.csv')

# Fix endpoints to corners of AUC plot
CPM.CI.ROCs[CPM.CI.ROCs$FPR == 0,c('TPR_mean','TPR_median','TPR_lo','TPR_hi')] <- 0
CPM.CI.ROCs[CPM.CI.ROCs$FPR == 1,c('TPR_mean','TPR_median','TPR_lo','TPR_hi')] <- 1

# Format ROC dataframe for plot
CPM.CI.ROCs <- CPM.CI.ROCs %>% 
  mutate(Threshold = plyr::mapvalues(Threshold,
                                     from = c("GOSE>1","GOSE>3","GOSE>4","GOSE>5","GOSE>6","GOSE>7"),
                                     to =c("GOSE > 1","GOSE > 3","GOSE > 4","GOSE > 5","GOSE > 6","GOSE > 7"))) %>%
  mutate(MODEL = factor(MODEL,levels =c("CPM_MNLR","CPM_POLR","CPM_DeepMN","CPM_DeepOR"))) %>%
  rename(Model = MODEL)

# Implement ggplot to visualise ROCs
CPM.ROC.curves.plot <- CPM.CI.ROCs %>%
  ggplot(aes(x = FPR)) +
  facet_wrap( ~ Threshold,
              scales = 'free',
              ncol = 3) +
  xlab("False Positive Rate") +
  ylab("True Positive Rate") +
  coord_cartesian(ylim = c(0,1),xlim = c(0,1))+
  geom_segment(x = 0, y = 0, xend = 1, yend = 1,alpha = 0.5,linetype = "dashed",size=.75/.pt, color = 'gray')+
  geom_ribbon(aes(ymin = TPR_lo, ymax = TPR_hi, fill = Model), alpha = 0.3,size=.75/.pt,color=NA) +
  geom_line(aes(y = TPR_mean, color = Model), alpha = 1, size=1.3/.pt) +
  scale_fill_discrete(breaks=levels(CPM.CI.ROCs$Model),
                      labels=c(expression(CPM[MNLR]), expression(CPM[POLR]),expression(CPM[DeepMN]),expression(CPM[DeepOR])))+
  scale_color_discrete(breaks=levels(CPM.CI.ROCs$Model),
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

# Create directory for current date and save CPM ROC plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'CPM_thresh_ROC.svg'),CPM.ROC.curves.plot,device= svg,units='in',dpi=600,width=6,height = 4.38)

# Get AUC values for CPM
CPM.CI.AUC <- read.csv('../model_performance/CPM/CI_threshold_metrics.csv') %>%
  filter(Threshold != 'Average', METRIC == 'AUC') %>% 
  mutate(MODEL = factor(MODEL,levels =c("CPM_MNLR","CPM_POLR","CPM_DeepMN","CPM_DeepOR"))) %>%
  mutate(formatted = sprintf('%s: %.2f (%.2f–%.2f)',MODEL,mean,lo,hi)) %>%
  arrange(Threshold,MODEL)

### Supplementary Figure 5
# Load ROC confidence intervals of APM
APM.CI.ROCs <- read.csv('../model_performance/APM/CI_ROCs.csv')

# Fix endpoints to corners of AUC plot
APM.CI.ROCs[APM.CI.ROCs$FPR == 0,c('TPR_mean','TPR_median','TPR_lo','TPR_hi')] <- 0
APM.CI.ROCs[APM.CI.ROCs$FPR == 1,c('TPR_mean','TPR_median','TPR_lo','TPR_hi')] <- 1

# Format ROC dataframe for plot
APM.CI.ROCs <- APM.CI.ROCs %>% 
  mutate(Threshold = plyr::mapvalues(Threshold,
                                     from = c("GOSE>1","GOSE>3","GOSE>4","GOSE>5","GOSE>6","GOSE>7"),
                                     to =c("GOSE > 1","GOSE > 3","GOSE > 4","GOSE > 5","GOSE > 6","GOSE > 7"))) %>%
  mutate(MODEL = factor(MODEL,levels =c("APM_DeepMN","APM_DeepOR"))) %>%
  rename(Model = MODEL)

# Implement ggplot to visualise ROCs
APM.ROC.curves.plot <- APM.CI.ROCs %>%
  ggplot(aes(x = FPR)) +
  facet_wrap( ~ Threshold,
              scales = 'free',
              ncol = 3) +
  xlab("False Positive Rate") +
  ylab("True Positive Rate") +
  coord_cartesian(ylim = c(0,1),xlim = c(0,1))+
  geom_segment(x = 0, y = 0, xend = 1, yend = 1,alpha = 0.5,linetype = "dashed",size=.75/.pt, color = 'gray')+
  geom_ribbon(aes(ymin = TPR_lo, ymax = TPR_hi, fill = Model), alpha = 0.3,size=.75/.pt,color=NA) +
  geom_line(aes(y = TPR_mean, color = Model), alpha = 1, size=1.3/.pt) +
  scale_fill_discrete(breaks=levels(APM.CI.ROCs$Model),
                      labels=c(expression(APM[DeepMN]),expression(APM[DeepOR])))+
  scale_color_discrete(breaks=levels(APM.CI.ROCs$Model),
                       labels=c(expression(APM[DeepMN]),expression(APM[DeepOR])))+
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

# Create directory for current date and save APM ROC plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'APM_thresh_ROC.svg'),APM.ROC.curves.plot,device= svg,units='in',dpi=600,width=6,height = 4.38)

# Get AUC values for APM
APM.CI.AUC <- read.csv('../model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(Threshold != 'Average', METRIC == 'AUC') %>% 
  mutate(MODEL = factor(MODEL,levels =c("APM_DeepMN","APM_DeepOR"))) %>%
  mutate(formatted = sprintf('%s: %.2f (%.2f–%.2f)',MODEL,mean,lo,hi)) %>%
  arrange(Threshold,MODEL)

### Supplementary Figure 6
# Load ROC confidence intervals of eCPM
eCPM.CI.ROCs <- read.csv('../model_performance/eCPM/CI_ROCs.csv')

# Fix endpoints to corners of AUC plot
eCPM.CI.ROCs[eCPM.CI.ROCs$FPR == 0,c('TPR_mean','TPR_median','TPR_lo','TPR_hi')] <- 0
eCPM.CI.ROCs[eCPM.CI.ROCs$FPR == 1,c('TPR_mean','TPR_median','TPR_lo','TPR_hi')] <- 1

# Format ROC dataframe for plot
eCPM.CI.ROCs <- eCPM.CI.ROCs %>% 
  mutate(Threshold = plyr::mapvalues(Threshold,
                                     from = c("GOSE>1","GOSE>3","GOSE>4","GOSE>5","GOSE>6","GOSE>7"),
                                     to =c("GOSE > 1","GOSE > 3","GOSE > 4","GOSE > 5","GOSE > 6","GOSE > 7"))) %>%
  mutate(MODEL = factor(MODEL,levels =c("eCPM_MNLR","eCPM_POLR","eCPM_DeepMN","eCPM_DeepOR"))) %>%
  rename(Model = MODEL)

# Implement ggplot to visualise ROCs
eCPM.ROC.curves.plot <- eCPM.CI.ROCs %>%
  ggplot(aes(x = FPR)) +
  facet_wrap( ~ Threshold,
              scales = 'free',
              ncol = 3) +
  xlab("False Positive Rate") +
  ylab("True Positive Rate") +
  coord_cartesian(ylim = c(0,1),xlim = c(0,1))+
  geom_segment(x = 0, y = 0, xend = 1, yend = 1,alpha = 0.5,linetype = "dashed",size=.75/.pt, color = 'gray')+
  geom_ribbon(aes(ymin = TPR_lo, ymax = TPR_hi, fill = Model), alpha = 0.3,size=.75/.pt,color=NA) +
  geom_line(aes(y = TPR_mean, color = Model), alpha = 1, size=1.3/.pt) +
  scale_fill_discrete(breaks=levels(eCPM.CI.ROCs$Model),
                      labels=c(expression(eCPM[MNLR]), expression(eCPM[POLR]),expression(eCPM[DeepMN]),expression(eCPM[DeepOR])))+
  scale_color_discrete(breaks=levels(eCPM.CI.ROCs$Model),
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

# Create directory for current date and save eCPM ROC plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'eCPM_thresh_ROC.svg'),eCPM.ROC.curves.plot,device= svg,units='in',dpi=600,width=6,height = 4.38)

# Get AUC values for eCPM
eCPM.CI.AUC <- read.csv('../model_performance/eCPM/CI_threshold_metrics.csv') %>%
  filter(Threshold != 'Average', METRIC == 'AUC') %>% 
  mutate(MODEL = factor(MODEL,levels =c("eCPM_MNLR","eCPM_POLR","eCPM_DeepMN","eCPM_DeepOR"))) %>%
  mutate(formatted = sprintf('%s: %.2f (%.2f–%.2f)',MODEL,mean,lo,hi)) %>%
  arrange(Threshold,MODEL)

### Supplementary Figure 7
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

# Get Emax values for CPM
CPM.CI.Emax <- read.csv('../model_performance/CPM/CI_threshold_metrics.csv') %>%
  filter(Threshold != 'Average', METRIC == 'Emax') %>% 
  mutate(MODEL = factor(MODEL,levels =c("CPM_MNLR","CPM_POLR","CPM_DeepMN","CPM_DeepOR"))) %>%
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi)) %>%
  arrange(Threshold,MODEL)

### Supplementary Figure 8
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
                      labels=c(expression(APM[DeepMN]),expression(APM[DeepOR])))+
  scale_color_discrete(breaks=levels(APM.CI.calibration$Model),
                       labels=c(expression(APM[DeepMN]),expression(APM[DeepOR])))+
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

# Get Emax values for APM
APM.CI.Emax <- read.csv('../model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(Threshold != 'Average', METRIC == 'Emax') %>% 
  mutate(MODEL = factor(MODEL,levels =c("APM_DeepMN","APM_DeepOR"))) %>%
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi)) %>%
  arrange(Threshold,MODEL)

### Supplementary Figure 9
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

# Get Emax values for eCPM
eCPM.CI.Emax <- read.csv('../model_performance/eCPM/CI_threshold_metrics.csv') %>%
  filter(Threshold != 'Average', METRIC == 'Emax') %>% 
  mutate(MODEL = factor(MODEL,levels =c("eCPM_MNLR","eCPM_POLR","eCPM_DeepMN","eCPM_DeepOR"))) %>%
  mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi)) %>%
  arrange(Threshold,MODEL)

### Performance metrics table for CPM_best, APM_best, and eCPM_best
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

### Calibration metrics table for CPM_best, APM_best, and eCPM_best
# Load threshold-level calibration metrics and determine best model for each metric
CPM.CI.calibration.ave <- read.csv('../model_performance/CPM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC %in% c('ICI','E50','E90','Emax')) %>% 
  group_by(METRIC) %>%
  slice_min(order_by = mean, n = 1) %>%
  mutate(PredType = 'CPM')

APM.CI.calibration.ave <- read.csv('../model_performance/APM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC %in% c('ICI','E50','E90','Emax')) %>% 
  group_by(METRIC) %>%
  slice_min(order_by = mean, n = 1) %>%
  mutate(PredType = 'APM')

eCPM.CI.calibration.ave <- read.csv('../model_performance/eCPM/CI_threshold_metrics.csv') %>%
  filter(Threshold == 'Average', METRIC %in% c('ICI','E50','E90','Emax')) %>% 
  group_by(METRIC) %>%
  slice_min(order_by = mean, n = 1) %>%
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

### Performance metrics table for CPM, APM, and eCPM
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

### Calibration metrics table for CPM, APM, and eCPM
# CPM calibration table
CPM.calibration.table <- rbind(read.csv('../model_performance/CPM/CI_threshold_metrics.csv') %>%
                                 filter(METRIC %in% c('ICI','E50','E90','Emax')) %>% 
                                 mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi),
                                        METRIC = factor(METRIC,levels = c('ICI','E50','E90','Emax'))) %>%
                                 select(MODEL,Threshold,METRIC,formatted) %>%
                                 pivot_wider(names_from = MODEL, id_cols = c(METRIC,Threshold),values_from = formatted) %>%
                                 arrange(METRIC,Threshold) %>%
                                 relocate(CPM_MNLR,CPM_POLR,CPM_DeepMN,CPM_DeepOR,.after=Threshold))

# APM calibration table
APM.calibration.table <- rbind(read.csv('../model_performance/APM/CI_threshold_metrics.csv') %>%
                                 filter(METRIC %in% c('ICI','E50','E90','Emax')) %>% 
                                 mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi),
                                        METRIC = factor(METRIC,levels = c('ICI','E50','E90','Emax'))) %>%
                                 select(MODEL,Threshold,METRIC,formatted) %>%
                                 pivot_wider(names_from = MODEL, id_cols = c(METRIC,Threshold),values_from = formatted) %>%
                                 arrange(METRIC,Threshold) %>%
                                 relocate(APM_DeepMN,APM_DeepOR,.after=Threshold))

# eCPM calibration table
eCPM.calibration.table <- rbind(read.csv('../model_performance/eCPM/CI_threshold_metrics.csv') %>%
                                 filter(METRIC %in% c('ICI','E50','E90','Emax')) %>% 
                                 mutate(formatted = sprintf('%.2f (%.2f–%.2f)',mean,lo,hi),
                                        METRIC = factor(METRIC,levels = c('ICI','E50','E90','Emax'))) %>%
                                 select(MODEL,Threshold,METRIC,formatted) %>%
                                 pivot_wider(names_from = MODEL, id_cols = c(METRIC,Threshold),values_from = formatted) %>%
                                 arrange(METRIC,Threshold) %>%
                                 relocate(eCPM_MNLR,eCPM_POLR,eCPM_DeepMN,eCPM_DeepOR,.after=Threshold))