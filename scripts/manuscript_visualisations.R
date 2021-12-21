#### Master Script #: Visualise study results for manuscript ####
#
# Shubhayu Bhattacharyay
# University of Cambridge
# email address: sb2406@cam.ac.uk
#
### Contents:
# I. Initialisation
# II. Figure 2

### I. Initialisation
# Import necessary libraries
library(tidyverse)
library(RColorBrewer)
library(rvg)
library(svglite)

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

# Create directory for current date and save GCSm ROC plot
dir.create(file.path('../plots',Sys.Date()),showWarnings = F,recursive = T)
ggsave(file.path('../plots',Sys.Date(),'NFL_dist.svg'),nfl.dist.plot,device= svg,units='in',dpi=600,width=1.37,height = .87)
