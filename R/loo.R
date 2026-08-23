# https://mc-stan.org/loo/articles/loo2-example.html
# https://paulbuerkner.com/brms/reference/loo.brmsfit.html

library("brms")
library("rstan")
library("cmdstanr")
library("loo")

options(mc.cores = 2) # parallel::detectCores() / 2)
options(loo.cores = 2) # parallel::detectCores() / 2)

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
M1 <- brms::add_criterion(M1, criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8, pointwise = FALSE)
# R session crashes after this, try again with setting pointwise=TRUE as this will use less memory
# https://discourse.mc-stan.org/t/loo-add-criterion-aborts-r-session-for-cmdstanr-model/23224/6
# setting pointwise to TRUE actually uses a LOT MORE CPU (WITH MULTITHREADING) but the memory seems to be shared across the cores (processess)
# resulting in an overall less RAM use

# at the end of add_criterion with the set number of cores (processes) brms spins up mc.cores number of new processes
# to do something - with pointwise set to FALSE (the default), this will result in a hefty RAM use - probably why the R session gets killed
# take this into account when setting the mc.cores (especially when pointwise is set to FALSE)

M2 <- brms::add_criterion(M2, criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8, pointwise = FALSE)
M4 <- brms::add_criterion(M4, criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8, pointwise = FALSE)

# N models
N1 <- brms::add_criterion(N1, criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8, pointwise = FALSE)
N2 <- brms::add_criterion(N2, criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8, pointwise = FALSE)
N4 <- brms::add_criterion(N4, criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8, pointwise = FALSE)

compres <- loo::loo_compare(M1, M2, M4, N1, N2, N4) # loo::loo_compare can handle more than two models
print(compres)

# https://paulbuerkner.com/brms/reference/loo.brmsfit.html
# https://paulbuerkner.com/brms/reference/loo_moment_match.brmsfit.html
# https://mc-stan.org/loo/reference/loo-glossary.html
# also look up the help page of loo_compare for details on result interpretation
