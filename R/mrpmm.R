library("ape")
library("brms")
library("readxl")
library("cmdstanr")
library("rstan")

fred4 <- read.csv("./ScratchData/continuous_raw.csv") # continuous trait data with all the raw records
fred4$taxa <- fred4$binominal # duplicated column to be fitted as the random effect
states <- readxl::read_xlsx(path = "./ScratchData/final.xlsx", sheet = "final")[, c("binominal", "state")]
states$binominal <- gsub(states$binominal, pattern = ' ', replacement = '_') # replace the space in the binominal names with underscores

fred4 <- merge(x = fred4, y = states, by = "binominal", all.x = TRUE) # merge the trait data with mycorrhizal state data

# log transformation and standardization of the continuous traits
fred4$F00679 <- scale(log(fred4$F00679))[, 1] # RD
fred4$F00727 <- scale(log(fred4$F00727))[, 1] # SRL
fred4$F00709 <- scale(log(fred4$F00709))[, 1] # RTD

tree <- ape::read.tree("./ScratchData/FRED4_1301.tre") # the phylogeny
if(!ape::is.binary(tree)) tree <- ape::multi2di(tree) # if not binary, make it binary
stopifnot(all(tree$tip.label %in% fred4$binominal)) # make sure all the species in the phylogeny exist in the dataset
corrmat <- ape::vcv.phylo(phy = tree, corr = TRUE)

# models with mycorrhizal states as the fixed effect
# https://paulbuerkner.com/brms/reference/loo_moment_match.brmsfit.html
# we need to set save_pars = save_pars(all = TRUE) to be able to moment matched LOO model comparisons

tryCatch(
    expr = {
        M2 <- brms::brm(brms::brmsformula(mvbind(F00727, F00679, F00709) ~ state + (1|taxa)) + set_rescor(TRUE), data = fred4, chains = 8, cores = 8, threads = 4, iter = 5000, warmup = 2500, backend = "cmdstanr", save_pars = save_pars(all = TRUE))
        saveRDS(object = M2, file = "./ScratchData/brms_taxa.Rds")
    },
    error = function(err){ print(err) }
)

tryCatch(
    expr = {
        M4 <- brms::brm(brms::brmsformula(mvbind(F00727, F00679, F00709) ~ state + (1|q|taxa)) + set_rescor(TRUE), data = fred4, chains = 8, cores = 8, threads = 4, iter = 5000, warmup = 2500, backend = "cmdstanr", save_pars = save_pars(all = TRUE))
        saveRDS(object = M4, file = "./ScratchData/brms_taxa_corr.Rds")
    },
    error = function(err){ print(err) }
)

tryCatch(
    expr = {
        M1 <- brms::brm(brms::brmsformula(mvbind(F00727, F00679, F00709) ~ state + (1|gr(binominal, cov = corrmat)) + (1|taxa)) + set_rescor(TRUE), data = fred4, data2 = list(corrmat = corrmat), chains = 8, cores = 8, threads = 4, iter = 5000, warmup = 2500, backend = "cmdstanr", save_pars = save_pars(all = TRUE))
        saveRDS(object = M1, file = "./ScratchData/brms_phylo.Rds")
    },
    error = function(err){ print(err) }
)

# this will be the slowest fit :/
tryCatch(
    expr = {
        M3 <- brms::brm(brms::brmsformula(mvbind(F00727, F00679, F00709) ~ state + (1|p|gr(binominal, cov = corrmat)) + (1|q|taxa)) + set_rescor(TRUE), data = fred4, data2 = list(corrmat = corrmat), chains = 8, cores = 8, threads = 4, iter = 5000, warmup = 2500, backend = "cmdstanr", save_pars = save_pars(all = TRUE))
        saveRDS(object = M3, file = "./ScratchData/brms_phylo_corr.Rds")
    },
    error = function(err){ print(err) }
)
