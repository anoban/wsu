#----------------------
# moment matched loo
#----------------------

library("brms")
library("rstan")
library("cmdstanr")
library("loo")

# we can go up to 16 here (on a 32 core vCPU) which we speed up LOO immensely but we don't have enough memory to accomodate that many processes (even with a 126 GiB RAM)
options(mc.cores = 4) # keep these at a reasonable minimum (the more processes we spin up the more RAM will be needed)
options(loo.cores = 4)

# https://github.com/stan-dev/loo/issues/222

#----------------------------------------------------
# models with mycorrhizal state as the fixed effect
#----------------------------------------------------

# the following only applies to moment matched LOO, with moment_match set to FALSE, none of this hoop jumping is necessary
# but we get a lot of warnings advising to set it to TRUE
# "Found 267 observations with a pareto_k > 0.7 in model 'M1'. We recommend to set 'moment_match = TRUE' in order to perform moment matching for problematic observations."

# the main issue with moment matched LOO is that our models are pretty huge - not in terms of model complexity but in terms of records used to fit the model
# hence, the serialized models are pretty huge too (becomes even bigger with save_pars = save_pars(all = TRUE))
# 2.3G ./ScratchData/brms_phylo.Rds
# 1.2G ./ScratchData/brms_taxa_corr.Rds
# 1.2G ./ScratchData/brms_taxa.Rds
# 2.3G ./ScratchData/nullmods/brms_phylo_null.Rds
# 1.2G ./ScratchData/nullmods/brms_taxa_corr_null.Rds
# 1.2G ./ScratchData/nullmods/brms_taxa_null.Rds

# hence loading in all the models first and then doing LOO becomes EXTREMELY taxing on the RAM - causing the OS to kill the R process
# LOO for M1 alone uses about 64 GiBs of RAM (with pointwise = FALSE) - that's just one of our 8 models
# so do the LOO sequentially, one model after another

# the cores argument of add_criterion() is not the issue here, it's just used to parallelize the first step of LOO which isn't as demanding on the RAM
# this is followed by recompilation of the brms model using rstan (this step invoked g++, as and ld)
# what follows the model recompilation is the most memory demanding step
# this step uses parallelization (honouring mc.cores and loo.cores) and with this is where the process gets killed by the OS

# setting pointwise to TRUE actually uses a LOT MORE CPU (WITH MULTITHREADING) but the memory seems to be shared across the cores (processess)
# resulting in an overall less RAM use
# even when the cores argument is set to n, the function takes the liberty to use multithreading if more cores are available on the system
# not strictly limiting the used cores to n

#-------------------------------------------------
# models with mycorrhizal states as fixed effect
#-------------------------------------------------

# in R everything is pass by value, so making the Rds reads inline could potentially lower the memory use by avoiding copying????
# i.e.
# M1 <- readRDS("./ScratchData/brms_phylo.Rds")
# M1 <- brms::add_criterion(M1, criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8, pointwise = TRUE)
# this could double (???) the memory use - first for creating M1 by reading in the Rds and then when it's copied and passed to brms::add_criterion() 

M1 <- brms::add_criterion(readRDS("./ScratchData/brms_phylo.Rds"), criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8, pointwise = TRUE)
gc(full = TRUE)
M2 <- brms::add_criterion(readRDS("./ScratchData/brms_taxa.Rds"), criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8, pointwise = TRUE)
gc(full = TRUE)
# M3 <- readRDS("./ScratchData/brms_phylo_corr.Rds") # not finished yet
M4 <- brms::add_criterion(readRDS("./ScratchData/brms_taxa_corr.Rds"), criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8, pointwise = TRUE)
gc(full = TRUE)

#-----------------------------------
# null models with no fixed effect
#-----------------------------------

N1 <- brms::add_criterion(readRDS("./ScratchData/nullmods/brms_phylo_null.Rds"), criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8, pointwise = TRUE)
gc(full = TRUE)
N2 <- brms::add_criterion(readRDS("./ScratchData/nullmods/brms_taxa_null.Rds"), criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8, pointwise = TRUE)
gc(full = TRUE)
# N3 <- readRDS("./ScratchData/nullmods/brms_phylo_corr_null.Rds") # not finished yet
N4 <- brms::add_criterion(readRDS("./ScratchData/nullmods/brms_taxa_corr_null.Rds"), criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8, pointwise = TRUE)
gc(full = TRUE)

compres <- loo::loo_compare(M1, M2, M4, N1, N2, N4)
print(compres)
