#!/usr/bin/env Rscript

library('ape')
library('OUwie')

phylogeny <- ape::read.tree('./ScratchData/FRED4_1301.tre')
data <- read.csv("./ScratchData/name_matched_FRED4_1301.csv")[, c('binominal', 'state', 'F00709')]
stopifnot(all(phylogeny$tip.label == data$binominal));

# SYMOUM_F00709_CID_100.Rds
model <- OUwie::hOUwie(phy = phylogeny,
                       data = data,
                       rate.cat = 2,
                       discrete_model = 'SYM',
                       continuous_model = 'OUM',
                       nSim = 100,
                       null.model = TRUE,
                       lb_discrete_model = 1e-15,
                       ub_discrete_model = 10)

saveRDS(object = model, file = "./ScratchData/RTD/SYMOUM_F00709_CID_100.Rds")
