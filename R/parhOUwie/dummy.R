#!/usr/bin/env R

# run a trial fit with the big phylogeny, just for 25 sims

library("ape")
library("OUwie")

phylogeny <- ape::read.tree("./ScratchData/FRED4_1301/FRED4_1301.tre")
data <- read.csv("./ScratchData/FRED4_1301/name_matched_FRED4_1301.csv")[, c("binominal", "state", "F00679")]
stopifnot(all(phylogeny$tip.label == data$binominal))

model <- OUwie::hOUwie(phy = phylogeny,
                       data = data,
                       rate.cat = 1,
                       discrete_model = "ARD",
                       continuous_model = "OUMVA",
                       nSim = 25,
                       null.model = FALSE)

saveRDS(object = model, file = "./ScratchData/FRED4_1301/dummy.Rds")
