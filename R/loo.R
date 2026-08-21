# https://mc-stan.org/loo/articles/loo2-example.html
#https://paulbuerkner.com/brms/reference/loo.brmsfit.html

library("brms")
library("rstan")
library("cmdstanr")
library("loo")

options(mc.cores = parallel::detectCores() / 2)
options(loo.cores = parallel::detectCores() / 2)

#----------------------------------------------------
# models with mycorrhizal state as the fixed effect
#----------------------------------------------------

M1 <- readRDS("./ScratchData/brms_phylo.Rds")
M2 <- readRDS("./ScratchData/brms_taxa.Rds")
# M3 <- readRDS("./ScratchData/brms_phylo_corr.Rds") # not finished yet
M4 <- readRDS("./ScratchData/brms_taxa_corr.Rds")

#-----------------------------------
# null models with no fixed effect
#-----------------------------------

N1 <- readRDS("./ScratchData/nullmods/brms_phylo_null.Rds")
N2 <- readRDS("./ScratchData/nullmods/brms_taxa_null.Rds")
# N3 <- readRDS("./ScratchData/nullmods/brms_phylo_corr_null.Rds") # not finished yet
N4 <- readRDS("./ScratchData/nullmods/brms_taxa_corr_null.Rds")

# in order to use moment_match = TRUE, the models must've been fit with save_pars = save_pars(all = TRUE)

# M models
M1 <- brms::add_criterion(M1, criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8)
M2 <- brms::add_criterion(M2, criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8)
M4 <- brms::add_criterion(M4, criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8)

# N models
N1 <- brms::add_criterion(N1, criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8)
N2 <- brms::add_criterion(N2, criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8)
N4 <- brms::add_criterion(N4, criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8)

loo::loo_compare(M1, M2, M4, N1, N2, N4)

# https://paulbuerkner.com/brms/reference/loo.brmsfit.html
# https://paulbuerkner.com/brms/reference/loo_moment_match.brmsfit.html
# https://mc-stan.org/loo/reference/loo-glossary.html
# also look up the help page of loo_compare for details on result interpretation
