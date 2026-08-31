#-------------------------
# loo with sub sampling
#-------------------------

library("brms")
library("rstan")
library("cmdstanr")
library("loo")

set.seed(1)

# https://discourse.mc-stan.org/t/memory-requirements-to-run-loo-brmsfit/27696/9
# https://mc-stan.org/loo/articles/loo2-large-data.html
# https://paulbuerkner.com/brms/reference/loo_subsample.brmsfit.html
# https://paulbuerkner.com/brms/reference/prepare_predictions.html

#-------------------------------------------------
# models with mycorrhizal states as fixed effect
#-------------------------------------------------

M1 <- loo::loo_subsample(readRDS("./ScratchData/brms_phylo.Rds"), compare = TRUE, cores = 8)
gc(full = TRUE)

M2 <- loo::loo_subsample(readRDS("./ScratchData/brms_taxa.Rds"), compare = TRUE, cores = 8)
gc(full = TRUE)

# M3 <- readRDS("./ScratchData/brms_phylo_corr.Rds") # not finished yet

M4 <- loo::loo_subsample(readRDS("./ScratchData/brms_taxa_corr.Rds"), compare = TRUE, cores = 8)
gc(full = TRUE)

#-----------------------------------
# null models with no fixed effect
#-----------------------------------

N1 <- loo::loo_subsample(readRDS("./ScratchData/nullmods/brms_phylo_null.Rds"), compare = TRUE, cores = 8)
gc(full = TRUE)

N2 <- loo::loo_subsample(readRDS("./ScratchData/nullmods/brms_taxa_null.Rds"), compare = TRUE, cores = 8)
gc(full = TRUE)

# N3 <- readRDS("./ScratchData/nullmods/brms_phylo_corr_null.Rds") # not finished yet

N4 <- loo::loo_subsample(readRDS("./ScratchData/nullmods/brms_taxa_corr_null.Rds"), compare = TRUE, cores = 8)
gc(full = TRUE)

compres <- loo::loo_compare(M1, M2, M4, N1, N2, N4)
print(compres)
