#!/usr/bin/env Rscript

library("ape")
library("brms")
library("readxl")
library("geiger")

fred4 <- read.csv("./ScratchData/continuous_raw.csv") # continuous trait data with all the raw records
fred4$taxa <- fred4$binominal # duplicated column to be fitted as the random effect
states <- readxl::read_xlsx(path = "./ScratchData/final.xlsx", sheet = "final")[, c("binominal", "state")]
states$binominal <- gsub(states$binominal, pattern = ' ', replacement = '_')

fred4 <- merge(x = fred4, y = states, by = "binominal", all.x = TRUE) # merge the trait data with mycorrhizal state data

# log transformation and standardization of the continuous traits
fred4$F00679 <- scale(log(fred4$F00679))[, 1] # RD
fred4$F00727 <- scale(log(fred4$F00727))[, 1] # SRL
fred4$F00709 <- scale(log(fred4$F00709))[, 1] # RTD


tree <- ape::read.tree("./ScratchData/FRED4_1301.tre") # the phylogeny
if(!ape::is.binary(tree)) tree <- ape::multi2di(tree) # if not binary, make it binary
stopifnot(all(tree$tip.label %in% fred4$binominal))

# since lambda was the best fitting model in our model comparison - branch transform the phylogeny first before fitting the model
tree <- geiger:::rescale.phylo(tree, model = "lambda", lambda = 0.831091127072489) # this lambda was the average of the lambda estimates of RD, SRL and RTD

corrmat <- ape::vcv.phylo(phy = tree, corr = TRUE) # turn the phylogeny into a variance-covariance matrix based on branch lengths

model_0 <- brms::brm(brms::brmsformula(mvbind(F00727, F00679, F00709) ~ state + (1|p|gr(binominal, cov = corrmat)) + (1|q|taxa)) + set_rescor(TRUE),
                     data = fred4, data2 = list(corrmat = corrmat), chains = 4, cores = 4, threads = 4, iter = 10000, warmup = 5000, backend = "cmdstanr")
saveRDS(object = model_0, file = "./ScratchData/brms_model_lambda.Rds")
