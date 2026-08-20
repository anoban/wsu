library("ape")
library("brms")
library("cmdstanr")

fred4 <- read.csv("./ScratchData/continuous_raw.csv")
fred4$taxa <- fred4$binominal

fred4$F00679 <- scale(log(fred4$F00679))[, 1]
fred4$F00727 <- scale(log(fred4$F00727))[, 1]
fred4$F00709 <- scale(log(fred4$F00709))[, 1]

tree <- ape::read.tree("./ScratchData/FRED4_1301.tre")
if(!ape::is.binary(tree)) tree <- ape::multi2di(tree)
stopifnot(all(tree$tip.label %in% fred4$binominal))

corrmat <- ape::vcv.phylo(phy = tree, corr = TRUE)

# first change the TMPDIR env variable to somewhere with enough space, don't just use export
# do this using a .Renviron file
# /data/mounts/scratch1/u90963425/
# /run/user/6190

# let's see if the try catch blocks help with unexpected chain terminations
tryCatch(
    expr = {
        M2 <- brms::brm(brms::brmsformula(mvbind(F00727, F00679, F00709) ~ 1 + (1|taxa)) + set_rescor(TRUE), data = fred4, chains = 8, cores = 8, threads = 4, iter = 5000, warmup = 2500, backend = "cmdstanr")
        saveRDS(object = M2, file = "./ScratchData/nullmods/brms_taxa_null.Rds")
    },
    error = function(err){ print(err) }
)

tryCatch(
    expr = {
        M4 <- brms::brm(brms::brmsformula(mvbind(F00727, F00679, F00709) ~ 1 + (1|q|taxa)) + set_rescor(TRUE), data = fred4, chains = 8, cores = 8, threads = 4, iter = 5000, warmup = 2500, backend = "cmdstanr")
        saveRDS(object = M4, file = "./ScratchData/nullmods/brms_taxa_corr_null.Rds")
    },
    error = function(err){ print(err) }
)

tryCatch(
    expr = {
        M1 <- brms::brm(brms::brmsformula(mvbind(F00727, F00679, F00709) ~ 1 + (1|gr(binominal, cov = corrmat)) + (1|taxa)) + set_rescor(TRUE), data = fred4, data2 = list(corrmat = corrmat), chains = 8, cores = 8, threads = 4, iter = 5000, warmup = 2500, backend = "cmdstanr")
        saveRDS(object = M1, file = "./ScratchData/nullmods/brms_phylo_null.Rds")
    },
    error = function(err){ print(err) }
)

tryCatch(
    expr = {
        M3 <- brms::brm(brms::brmsformula(mvbind(F00727, F00679, F00709) ~ 1 + (1|p|gr(binominal, cov = corrmat)) + (1|q|taxa)) + set_rescor(TRUE), data = fred4, data2 = list(corrmat = corrmat), chains = 8, cores = 8, threads = 4, iter = 5000, warmup = 2500, backend = "cmdstanr")
        saveRDS(object = M3, file = "./ScratchData/nullmods/brms_phylo_corr_null.Rds")
    },
    error = function(err){ print(err) }
)
