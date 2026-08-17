# https://mc-stan.org/loo/articles/loo2-example.html

library("brms")
library("rstan")
library("cmdstanr")
library("loo")

options(mc.cores = parallel::detectCores() / 2)
rstan::rstan_options(auto_write = TRUE)
rstan::rstan_options(threads_per_chain = 4)

M2 <- readRDS("./ScratchData/brms_taxa.Rds")
M4 <- readRDS("./ScratchData/brms_taxa_corr.Rds")

# to compare M2 and M4
M2 <- brms::add_criterion(M2, criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8)
M4 <- brms::add_criterion(M4, criterion = "loo", moment_match = TRUE, save_psis = TRUE, cores = 8)

loo::loo_compare(M2, M4)
