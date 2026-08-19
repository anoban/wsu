#!/usr/bin/env Rscript

library("ape")
library("brms")
library("cmdstanr")
cmdstanr::set_cmdstan_path(cmdstanr::cmdstan_default_path())

fred4 <- read.csv("./ScratchData/continuous_raw.csv") # continuous trait data with all the raw records
fred4$taxa <- fred4$binominal # duplicated column to be fitted as the random effect

# log transformation and standardization of the continuous traits
fred4$F00679 <- scale(log(fred4$F00679))[, 1] # RD
fred4$F00727 <- scale(log(fred4$F00727))[, 1] # SRL
fred4$F00709 <- scale(log(fred4$F00709))[, 1] # RTD

tree <- ape::read.tree("./ScratchData/FRED4_1301.tre") # the phylogeny
if(!ape::is.binary(tree)) tree <- ape::multi2di(tree) # if not binary, make it binary
stopifnot(all(tree$tip.label %in% fred4$binominal))

corrmat <- ape::vcv.phylo(phy = tree, corr = TRUE) # turn the phylogeny into a variance-covariance matrix based on branch lengths

# null models - without mycorrhizal states as the fixed effect
M2 <- brms::brm(brms::brmsformula(mvbind(F00727, F00679, F00709) ~ 1 + (1|taxa)) + set_rescor(TRUE), data = fred4, data2 = list(corrmat = corrmat),
          chains = 8, cores = 8, threads = 4, iter = 5000, warmup = 2500, backend = "cmdstanr")
saveRDS(object = M2, file = "./ScratchData/nullmods/M2_null.Rds")

M4 <- brms::brm(brms::brmsformula(mvbind(F00727, F00679, F00709) ~ 1 + (1|q|taxa)) + set_rescor(TRUE), data = fred4, data2 = list(corrmat = corrmat),
          chains = 8, cores = 8, threads = 4, iter = 5000, warmup = 2500)
saveRDS(object = M4, file = "./ScratchData/nullmods/M4_null.Rds")

M1 <- brms::brm(brms::brmsformula(mvbind(F00727, F00679, F00709) ~ 1 + (1|gr(binominal, cov = corrmat)) + (1|taxa)) + set_rescor(TRUE),
                data = fred4, data2 = list(corrmat = corrmat), chains = 8, cores = 8, threads = 4, iter = 5000, warmup = 2500, backend = "cmdstanr")
saveRDS(object = M1, file = "./ScratchData/nullmods/M1_null.Rds")

# will probably be the slowest - moving to the last
M3 <- brms::brm(brms::brmsformula(mvbind(F00727, F00679, F00709) ~ 1 + (1|p|gr(binominal, cov = corrmat)) + (1|q|taxa)) + set_rescor(TRUE),
                data = fred4, data2 = list(corrmat = corrmat), chains = 8, cores = 8, threads = 4, iter = 5000, warmup = 2500, backend = "cmdstanr")
saveRDS(object = M3, file = "./ScratchData/nullmods/M3_null.Rds")

fred4 <- read.csv("../data/chapter2/FRED/subsets/continuous_raw.csv")
fred4$taxa <- fred4$binominal
tree <- ape::read.tree("../data/chapter2/uphylomaker/FRED4_1301.tre")
