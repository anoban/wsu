#!/usr/bin/env Rscript

library("ape")
library("brms")
library("cmdstanr")
library("readxl")

fred4 <- read.csv("./ScratchData/continuous_raw.csv")
fred4$taxa <- fred4$binominal
states <- readxl::read_xlsx(path = "./ScratchData/final.xlsx", sheet = "final")[, c("binominal", "state")]
states$binominal <- gsub(states$binominal, pattern = ' ', replacement = '_')

fred4 <- merge(x = fred4, y = states, by = "binominal", all.x = TRUE)


fred4$F00679 <- scale(log(fred4$F00679))[, 1]
fred4$F00727 <- scale(log(fred4$F00727))[, 1]
fred4$F00709 <- scale(log(fred4$F00709))[, 1]


tree <- ape::read.tree("./ScratchData/FRED4_1301.tre")
if(!ape::is.binary(tree)) tree <- ape::multi2di(tree)
stopifnot(all(tree$tip.label %in% fred4$binominal))

corrmat <- ape::vcv.phylo(phy = tree, corr = TRUE)

model <- brms::brm(brms::brmsformula(mvbind(F00727, F00679, F00709) ~ state + (1|q|taxa)) + set_rescor(TRUE), data = fred4, data2 = list(corrmat = corrmat), chains = 8, cores = 8, threads = 2, iter = 5000, warmup = 2500, backend = "cmdstanr")
saveRDS(object = model, file = "./ScratchData/brms_taxa_corr.Rds")
